"""基于 MJX 的物理参数辨识入口。"""

from __future__ import annotations

from gimbal_sysid import config

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path

from gimbal_sysid.data import (
    _ask_column_mapping,
    _read_csv_with_fallback,
    ask_locked_joints,
    get_model_joint_names,
    load_csv,
    select_recorded_joints,
    TrajectoryData,
)
from gimbal_sysid.dynamics import rollout_from_trajectory
from gimbal_sysid.losses import compute_loss, loss_components
from gimbal_sysid.model import load_model




def ask_sample_dt() -> float:
    """输入固定采样周期；没有时间列时必须由采集设置确认。"""
    while True:
        value = input("请输入 CSV 采样周期 dt（秒，例如 250 Hz 输入 0.004）：").strip()
        try:
            dt = float(value)
        except ValueError:
            print("请输入数字，单位为秒。")
            continue
        if np.isfinite(dt) and dt > 0:
            return dt
        print("采样周期必须大于零，且不能是 NaN 或无穷大。")


def choose_identification_mode() -> str:
    """选择本次需要辨识的物理参数。"""
    modes = {
        "1": "friction",
        "2": "inertia",
        "3": "all",
    }
    while True:
        print("请选择参数辨识模式：")
        print("  1: 只辨识摩擦参数（B、Fc）")
        print("  2: 只辨识惯量参数（J）")
        print("  3: 同时辨识摩擦和惯量参数（J、B、Fc）")

        mode_input = input("请输入 1、2 或 3：").strip()
        if mode_input in modes:
            return modes[mode_input]
        print("输入无效，请重新输入 1、2 或 3。\n")


def load_experiment_interactively():
    """交互式读取模型和一组实验数据。"""
    while True:
        try:
            xml_input = input("请输入 MJCF 模型绝对路径：").strip()
            if not xml_input:
                raise ValueError("MJCF 模型路径不能为空")

            model_joint_names = get_model_joint_names(xml_input)
            selected_joints = select_recorded_joints(xml_input)
            active_joint_ids = [
                model_joint_names.index(name)
                for name in selected_joints
            ]
            locked_joint_ids, locked_angles = ask_locked_joints(
                model_joint_names,
                selected_joints,
            )

            csv_input = input("请输入 CSV 绝对路径：").strip()
            if not csv_input:
                raise ValueError("CSV 路径不能为空")

            frame = _read_csv_with_fallback(csv_input)
            column_mapping, motor_constants = _ask_column_mapping(
                frame,
                selected_joints,
            )
            data = load_csv(
                csv_input,
                motor_constants=motor_constants,
                column_mapping=column_mapping,
            )
            data = data.with_model_configuration(
                model_joint_names=model_joint_names,
                active_joint_ids=active_joint_ids,
                locked_joint_ids=locked_joint_ids,
                locked_angles=locked_angles,
            )
            return xml_input, data
        except (ValueError, FileNotFoundError, OSError, UnicodeError, RuntimeError) as exc:
            print(f"输入或文件有问题：{exc}")
            print("请重新输入本次实验配置。\n")


def make_physical_parameters(
    raw_parameters,
    active_joint_ids,
    num_joints,
    initial_inertia,
    initial_viscous,
    initial_coulomb,
    mode,
    lower_bound=config.FRICTION_LOWER_BOUND,
    upper_bound=config.FRICTION_UPPER_BOUND,
):
    """将无约束优化变量转换为物理参数。"""
    active_joint_ids = jnp.asarray(active_joint_ids, dtype=jnp.int32)
    values = {
        "inertia": jnp.asarray(initial_inertia, dtype=jnp.float32),
        "viscous": jnp.asarray(initial_viscous, dtype=jnp.float32),
        "coulomb": jnp.asarray(initial_coulomb, dtype=jnp.float32),
    }

    if mode in {"inertia", "all"}:
        values["inertia"] = jnp.clip(
            jnp.exp(raw_parameters["inertia"]),
            config.INERTIA_LOWER_BOUND,
            config.INERTIA_UPPER_BOUND,
        )
    if mode in {"friction", "all"}:
        values["viscous"] = values["viscous"].at[active_joint_ids].set(
            jnp.clip(jnp.exp(raw_parameters["viscous"]), lower_bound, upper_bound)
        )
        values["coulomb"] = values["coulomb"].at[active_joint_ids].set(
            jnp.clip(jnp.exp(raw_parameters["coulomb"]), lower_bound, upper_bound)
        )

    return values


def make_training_windows(
    real_data: TrajectoryData,
    limit_count: bool = True,
) -> list[TrajectoryData]:
    """按照训练设置切分数据，可选择是否限制窗口数量。"""
    window_size = max(2, int(config.TRAINING_WINDOW_STEPS))
    windows = []
    for start in range(0, real_data.num_steps - 1, window_size):
        end = min(start + window_size, real_data.num_steps)
        if end - start < 2:
            continue
        windows.append(
            TrajectoryData(
                pos_real=real_data.pos_real[start:end],
                vel_real=real_data.vel_real[start:end],
                tor_real=real_data.tor_real[start:end],
                active_joint_ids=real_data.active_joint_ids,
                model_joint_names=real_data.model_joint_names,
                locked_joint_ids=real_data.locked_joint_ids,
                locked_angles=real_data.locked_angles,
            )
        )

    if not windows:
        raise ValueError("训练数据至少需要两个采样点")
    if limit_count and len(windows) > config.TRAINING_WINDOW_COUNT:
        selected_indices = np.linspace(
            0,
            len(windows) - 1,
            num=config.TRAINING_WINDOW_COUNT,
            dtype=np.int32,
        )
        windows = [windows[index] for index in selected_indices]
    return windows


def make_objective(model, mjx_model, real_data, mode, with_components=False):
    """创建当前辨识模式对应的可求导目标函数。

    长实验轨迹按窗口计算损失。每个窗口都从该窗口的真实状态开始，
    这样仍然使用全部采样数据，但不会对整段长轨迹一次性反向传播。
    """
    active_joint_ids = np.asarray(
        real_data.active_joint_ids,
        dtype=np.int32,
    )
    num_joints = model.nv
    body_ids = np.asarray(
        model.jnt_bodyid[active_joint_ids],
        dtype=np.int32,
    )
    initial_inertia = np.asarray(
        mjx_model.body_inertia[body_ids, 2],
        dtype=np.float32,
    )
    initial_viscous = np.full(num_joints, config.INITIAL_VISCOUS, dtype=np.float32)
    initial_coulomb = np.full(num_joints, config.INITIAL_COULOMB, dtype=np.float32)

    # 预先切分窗口。训练和最终评估共用完全相同的窗口定义。
    windows = make_training_windows(real_data)

    def objective(raw_parameters):
        parameters = make_physical_parameters(
            raw_parameters,
            active_joint_ids,
            num_joints,
            initial_inertia,
            initial_viscous,
            initial_coulomb,
            mode,
        )
        current_model = mjx_model
        if mode in {"inertia", "all"}:
            body_inertia = current_model.body_inertia.at[body_ids, 2].set(
                parameters["inertia"]
            )
            current_model = current_model.replace(body_inertia=body_inertia)

        total_loss = 0.0
        position_loss = 0.0
        velocity_loss = 0.0

        for window in windows:
            _, simulation_trajectory = rollout_from_trajectory(
                model,
                current_model,
                window,
                parameters["viscous"],
                parameters["coulomb"],
                config.FRICTION_K,
            )
            if with_components:
                components = loss_components(simulation_trajectory, model, window)
                total_loss = total_loss + components["总损失"]
                position_loss = position_loss + components["角度损失"]
                velocity_loss = velocity_loss + components["速度损失"]
            else:
                total_loss = total_loss + compute_loss(
                    simulation_trajectory,
                    model,
                    window,
                )

        window_count = float(len(windows))
        if with_components:
            position_loss = position_loss / window_count
            velocity_loss = velocity_loss / window_count
            total_loss = total_loss / window_count
            return total_loss, {
                "角度损失": position_loss,
                "速度损失": velocity_loss,
                "总损失": total_loss,
            }
        return total_loss / window_count

    return objective


def main() -> None:
    mode = choose_identification_mode()
    model_path, real_data = load_experiment_interactively()
    dt = ask_sample_dt()
    model, mjx_model = load_model(model_path, dt=dt)
    print(
        f"训练窗口 = {config.TRAINING_WINDOW_STEPS} 个采样点，"
        f"训练使用最多 {config.TRAINING_WINDOW_COUNT} 个窗口，"
        f"原始数据共 {real_data.num_steps} 个采样点"
    )

    num_active = len(real_data.active_joint_ids)
    raw_parameters = {}
    if mode in {"inertia", "all"}:
        body_ids = model.jnt_bodyid[real_data.active_joint_ids]
        initial_inertia = mjx_model.body_inertia[body_ids, 2]
        raw_parameters["inertia"] = jnp.log(initial_inertia)
    if mode in {"friction", "all"}:
        raw_parameters["viscous"] = jnp.log(jnp.full(num_active, config.INITIAL_VISCOUS))
        raw_parameters["coulomb"] = jnp.log(jnp.full(num_active, config.INITIAL_COULOMB))

    objective = make_objective(model, mjx_model, real_data, mode, with_components=True)
    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.GRADIENT_CLIP_NORM),
        optax.adam(config.LEARNING_RATE),
    )
    optimizer_state = optimizer.init(raw_parameters)

    best_loss = float("inf")
    best_raw = None
    best_epoch = -1
    history = []
    # 第 0 次为初值；第 N 次评估第 N 次更新后的参数。
    for epoch in range(config.NUM_EPOCHS + 1):
        (loss, components), gradients = value_and_grad(raw_parameters)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            print(f"第 {epoch} 次评估损失非有限，停止训练并保留此前最佳参数。")
            break
        history.append([epoch, loss_value, float(components["角度损失"]), float(components["速度损失"])])
        if loss_value < best_loss:
            best_loss, best_epoch = loss_value, epoch
            best_raw = dict(raw_parameters)

        if epoch == 0 or epoch % config.LOG_EVERY == 0:
            body_ids = model.jnt_bodyid[real_data.active_joint_ids]
            estimated = make_physical_parameters(
                raw_parameters,
                real_data.active_joint_ids,
                model.nv,
                mjx_model.body_inertia[body_ids, 2],
                np.full(model.nv, config.INITIAL_VISCOUS, dtype=np.float32),
                np.full(model.nv, config.INITIAL_COULOMB, dtype=np.float32),
                mode,
            )
            print(
                f"第 {epoch:03d} 次更新后，总损失 = {loss_value:.6e}，"
                f"角度损失 = {float(components['角度损失']):.6e}，"
                f"速度损失 = {float(components['速度损失']):.6e}"
            )
            if mode in {"inertia", "all"}:
                print(f"  惯量 = {estimated['inertia']}")
            if mode in {"friction", "all"}:
                print(f"  粘性摩擦 = {estimated['viscous']}")
                print(f"  库伦摩擦 = {estimated['coulomb']}")

        if epoch == config.NUM_EPOCHS:
            break
        if not all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradients)):
            print("梯度出现非有限值，停止更新并保留最佳参数。")
            break
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, raw_parameters)
        candidate_parameters = optax.apply_updates(raw_parameters, updates)
        if not all(
            np.isfinite(np.asarray(value)).all()
            for value in jax.tree.leaves(candidate_parameters)
        ):
            print("参数更新后出现非有限值，停止训练并保留最佳参数。")
            break
        raw_parameters = candidate_parameters

    if best_raw is None:
        raise RuntimeError("未得到有限损失，不保存无效辨识结果。")
    raw_parameters = best_raw
    print(f"训练结束，采用第 {best_epoch} 次更新后的最佳参数，总损失 = {best_loss:.6e}；尚不代表已收敛。")

    # 使用最终参数再仿真一次，并保存给 results.py 绘图。
    body_ids = model.jnt_bodyid[real_data.active_joint_ids]
    final_parameters = make_physical_parameters(
        raw_parameters,
        real_data.active_joint_ids,
        model.nv,
        mjx_model.body_inertia[body_ids, 2],
        np.full(model.nv, config.INITIAL_VISCOUS, dtype=np.float32),
        np.full(model.nv, config.INITIAL_COULOMB, dtype=np.float32),
        mode,
    )
    current_model = mjx_model
    if mode in {"inertia", "all"}:
        body_inertia = current_model.body_inertia.at[body_ids, 2].set(
            final_parameters["inertia"]
        )
        current_model = current_model.replace(body_inertia=body_inertia)

    _, final_trajectory = rollout_from_trajectory(
        model,
        current_model,
        real_data,
        final_parameters["viscous"],
        final_parameters["coulomb"],
        config.FRICTION_K,
    )

    # 与训练目标一致的多重射击轨迹：每个窗口从真实初始状态开始。
    # 该轨迹用于 results.py 的误差统计；final_trajectory 保留为连续自由滚动结果，
    # 用于观察参数模型在长时间运行时的漂移。
    windowed_trajectories = []
    windowed_pos_real = []
    windowed_vel_real = []
    # 评估阶段不限制窗口数量，覆盖完整 CSV 数据。
    evaluation_windows = make_training_windows(real_data, limit_count=False)
    for window in evaluation_windows:
        _, window_trajectory = rollout_from_trajectory(
            model,
            current_model,
            window,
            final_parameters["viscous"],
            final_parameters["coulomb"],
            config.FRICTION_K,
        )
        windowed_trajectories.append(np.asarray(window_trajectory))
        windowed_pos_real.append(window.pos_real)
        windowed_vel_real.append(window.vel_real)
    windowed_trajectory = np.concatenate(windowed_trajectories, axis=0)
    windowed_pos_real = np.concatenate(windowed_pos_real, axis=0)
    windowed_vel_real = np.concatenate(windowed_vel_real, axis=0)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = config.RESULTS_DIR / config.RESULT_FILENAME

    active_joint_ids = np.asarray(real_data.active_joint_ids, dtype=np.int32)
    qpos_addresses = np.asarray(
        model.jnt_qposadr[active_joint_ids],
        dtype=np.int32,
    )
    qvel_addresses = np.asarray(
        model.jnt_dofadr[active_joint_ids],
        dtype=np.int32,
    )

    np.savez(
        result_path,
        mode=mode,
        parameter_selection="best_loss",
        best_epoch=best_epoch,
        best_loss=best_loss,
        loss_history=np.asarray(history),
        dt=dt,
        trajectory_alignment="includes_initial_state",
        model_path=str(Path(model_path).resolve()),
        # 默认结果轨迹与训练目标一致，results.py 据此计算指标。
        simulation_trajectory=windowed_trajectory,
        # 连续自由滚动轨迹不参与默认误差统计，仅用于观察长期漂移。
        free_rollout_trajectory=np.asarray(final_trajectory),
        pos_real=real_data.pos_real,
        vel_real=real_data.vel_real,
        # 与 simulation_trajectory 一一对应的窗口真实数据。
        evaluation_pos_real=windowed_pos_real,
        evaluation_vel_real=windowed_vel_real,
        active_joint_ids=active_joint_ids,
        qpos_addresses=qpos_addresses,
        qvel_addresses=qvel_addresses,
        model_nq=model.nq,
        model_nv=model.nv,
        model_joint_names=np.asarray(real_data.model_joint_names),
        estimated_inertia=np.asarray(final_parameters["inertia"]),
        estimated_viscous=np.asarray(final_parameters["viscous"]),
        estimated_coulomb=np.asarray(final_parameters["coulomb"]),
        evaluation_mode="all_windows_multi_shooting",
        training_window_steps=config.TRAINING_WINDOW_STEPS,
        training_window_count=config.TRAINING_WINDOW_COUNT,
    )
    print(f"辨识结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
