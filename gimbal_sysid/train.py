"""基于 MJX 的物理参数辨识入口。"""

from __future__ import annotations

from gimbal_sysid import config

import jax
import jax.numpy as jnp
import numpy as np
import optax
import mujoco
from pathlib import Path
import os
import tempfile
import time

from gimbal_sysid.data import (
    _ask_column_mapping,
    _read_csv_with_fallback,
    ask_locked_joints,
    get_model_joint_names,
    load_csv,
    select_recorded_joints,
    TrajectoryData,
)
from gimbal_sysid.dynamics import (
    init_state_from_trajectory, _make_lock_projection, rollout,
)
from gimbal_sysid.losses import loss_components
from gimbal_sysid.model import load_model




def save_npz_atomic(path, **arrays):
    """同目录临时文件写完后替换，避免中断留下半个结果文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def make_evaluation_rollout(model, mjx_model, trajectory, parameters):
    """数组作为 JIT 输入，相同长度的窗口复用一次编译。"""
    @jax.jit
    def evaluate(position, velocity, torque):
        window = TrajectoryData(
            pos_real=position, vel_real=velocity, tor_real=torque,
            active_joint_ids=trajectory.active_joint_ids,
            model_joint_names=trajectory.model_joint_names,
            locked_joint_ids=trajectory.locked_joint_ids,
            locked_angles=trajectory.locked_angles,
        )
        data = init_state_from_trajectory(model, mjx_model, window)
        locks = _make_lock_projection(model, mjx_model, window)
        controls = jnp.zeros((torque.shape[0], mjx_model.nu), dtype=jnp.float32)
        controls = controls.at[:, trajectory.active_joint_ids].set(torque)
        _, predicted = rollout(
            mjx_model, data, controls[:-1], parameters["viscous"],
            parameters["coulomb"], config.FRICTION_K, *locks,
        )
        initial = jnp.concatenate([data.qpos, data.qvel])[None, :]
        return jnp.concatenate([initial, predicted], axis=0)
    return evaluate


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


def _full_mass_matrix(model, locked_joint_ids, locked_angles):
    """在给定锁定姿态下计算 MuJoCo 的完整关节空间质量矩阵。"""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    joint_ids = [] if locked_joint_ids is None else locked_joint_ids
    angles = [] if locked_angles is None else locked_angles
    for joint_id, angle in zip(joint_ids, angles):
        data.qpos[model.jnt_qposadr[int(joint_id)]] = float(angle)
    mujoco.mj_forward(model, data)
    mass_matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, mass_matrix)
    return mass_matrix


def calibrate_effective_inertia(model, active_joint_ids, locked_joint_ids, locked_angles):
    """计算固定锁定姿态下各活动铰链的等效惯量及 body 惯量换算系数。

    训练参数使用 M[joint_dof, joint_dof]，而 MJX 仍通过修改活动 body 的
    第三个主惯量来实现。质量矩阵对惯性参数是线性的，因此用有限差分得到
    的 sensitivity 可以把二者稳定地互相换算，避免把子 body 惯量手工相加。
    """
    active_joint_ids = np.asarray(active_joint_ids, dtype=np.int32)
    locked_joint_ids = [] if locked_joint_ids is None else locked_joint_ids
    locked_angles = [] if locked_angles is None else locked_angles
    body_ids = np.asarray(model.jnt_bodyid[active_joint_ids], dtype=np.int32)
    body_base = np.asarray(model.body_inertia[body_ids, 2], dtype=np.float64)

    base_mass = _full_mass_matrix(model, locked_joint_ids, locked_angles)
    effective_base = np.asarray([
        base_mass[int(model.jnt_dofadr[joint_id]), int(model.jnt_dofadr[joint_id])]
        for joint_id in active_joint_ids
    ], dtype=np.float64)

    sensitivity = []
    try:
        for body_id, joint_id, base_value in zip(body_ids, active_joint_ids, body_base):
            delta = max(abs(float(base_value)) * 1.0e-3, 1.0e-7)
            model.body_inertia[int(body_id), 2] = float(base_value + delta)
            plus = _full_mass_matrix(model, locked_joint_ids, locked_angles)
            model.body_inertia[int(body_id), 2] = float(max(base_value - delta, 1.0e-8))
            minus = _full_mass_matrix(model, locked_joint_ids, locked_angles)
            denominator = (base_value + delta) - max(base_value - delta, 1.0e-8)
            dof = int(model.jnt_dofadr[int(joint_id)])
            sensitivity.append((plus[dof, dof] - minus[dof, dof]) / denominator)
    finally:
        model.body_inertia[body_ids, 2] = body_base

    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    if not np.isfinite(effective_base).all() or not np.isfinite(sensitivity).all():
        raise ValueError("固定姿态下的等效惯量计算得到 NaN/Inf")
    if np.any(sensitivity <= 0):
        raise ValueError(f"等效惯量换算系数必须为正，实际为 {sensitivity}")
    print(
        "固定锁定姿态下的等效惯量："
        + ", ".join(f"{value:.8g} kg·m²" for value in effective_base),
        flush=True,
    )
    return {
        "body_ids": body_ids,
        "body_base": body_base.astype(np.float32),
        "effective_base": effective_base.astype(np.float32),
        "sensitivity": sensitivity.astype(np.float32),
    }


def make_training_windows(
    real_data: TrajectoryData,
    limit_count: bool = True,
) -> list[TrajectoryData]:
    """不重叠分窗；评估保留末尾单点，训练仅排除无法推进的单点窗口。

    limit_count 保留接口兼容，True 现在表示训练用途，不再固定抽取少量窗口。
    不按活动度分位数丢弃数据，低速数据也有摩擦辨识价值。
    """
    if real_data.num_steps < 2:
        raise ValueError("训练数据至少需要两个采样点")
    for signal in (real_data.pos_real, real_data.vel_real, real_data.tor_real):
        if not np.isfinite(signal).all():
            raise ValueError("输入包含 NaN/Inf，请先清理数据，不能跳过后拼接时间轴")
    size = max(2, int(config.TRAINING_WINDOW_STEPS))
    windows = []
    for start in range(0, real_data.num_steps, size):
        end = min(start + size, real_data.num_steps)
        if limit_count and end - start < 2:
            continue
        windows.append(TrajectoryData(
            pos_real=real_data.pos_real[start:end],
            vel_real=real_data.vel_real[start:end],
            tor_real=real_data.tor_real[start:end],
            active_joint_ids=real_data.active_joint_ids,
            model_joint_names=real_data.model_joint_names,
            locked_joint_ids=real_data.locked_joint_ids,
            locked_angles=real_data.locked_angles,
        ))
    return windows


def split_window_ids(windows):
    """按时间均匀留出固定窗口；验证数据不参与梯度更新。"""
    usable = np.array([i for i, w in enumerate(windows) if w.num_steps >= 2], dtype=np.int32)
    fraction = float(config.VALIDATION_FRACTION)
    if not 0 < fraction < 1:
        raise ValueError("VALIDATION_FRACTION 必须位于 (0, 1)")
    if len(usable) < 2:
        raise ValueError("至少需要两个可推进窗口才能划分训练和验证集")
    count = min(len(usable) - 1, max(1, int(np.ceil(len(usable) * fraction))))
    selected = np.floor((np.arange(count) + 0.5) * len(usable) / count).astype(int)
    validation = usable[selected]
    training = np.setdiff1d(usable, validation)
    return training, validation


def iter_window_batches(training_ids, batch_size, seed):
    """每遍无放回打乱；最后的小批量不跨遍补齐，不遗漏窗口。"""
    if batch_size < 1:
        raise ValueError("TRAINING_WINDOW_COUNT 必须大于零")
    rng = np.random.default_rng(seed)
    cycle = 0
    while True:
        cycle += 1
        order = rng.permutation(training_ids)
        for start in range(0, len(order), batch_size):
            yield cycle, order[start:start + batch_size]


def make_objective(model, mjx_model, real_data, mode, with_components=False, inertia_calibration=None):
    """单窗口目标；数据以动态数组输入，同长度窗口复用 JIT。"""
    active_ids = np.asarray(real_data.active_joint_ids, dtype=np.int32)
    body_ids = np.asarray(model.jnt_bodyid[active_ids])
    initial_inertia = (
        np.asarray(inertia_calibration["effective_base"])
        if inertia_calibration is not None
        else np.asarray(mjx_model.body_inertia[body_ids, 2])
    )
    initial_viscous = np.full(model.nv, config.INITIAL_VISCOUS, dtype=np.float32)
    initial_coulomb = np.full(model.nv, config.INITIAL_COULOMB, dtype=np.float32)
    measured_position_scale = np.maximum(
        np.std(real_data.pos_real, axis=0), config.LOSS_SCALE_EPSILON
    ).astype(np.float32)
    measured_velocity_scale = np.maximum(
        np.std(real_data.vel_real, axis=0), config.LOSS_SCALE_EPSILON
    ).astype(np.float32)
    if not config.NORMALIZE_LOSS_BY_DATA_STD:
        measured_position_scale = np.ones_like(measured_position_scale)
        measured_velocity_scale = np.ones_like(measured_velocity_scale)

    def objective(raw_parameters, position, velocity, torque):
        parameters = make_physical_parameters(
            raw_parameters, active_ids, model.nv, initial_inertia,
            initial_viscous, initial_coulomb, mode,
        )
        current_model = mjx_model
        if mode in {"inertia", "all"}:
            if inertia_calibration is None:
                body_inertia_values = parameters["inertia"]
            else:
                body_inertia_values = jnp.asarray(inertia_calibration["body_base"])
                body_inertia_values = body_inertia_values + (
                    parameters["inertia"]
                    - jnp.asarray(inertia_calibration["effective_base"])
                ) / jnp.asarray(inertia_calibration["sensitivity"])
                body_inertia_values = jnp.maximum(body_inertia_values, 1.0e-8)
            current_model = mjx_model.replace(
                body_inertia=mjx_model.body_inertia.at[body_ids, 2].set(body_inertia_values)
            )
        window = TrajectoryData(
            position, velocity, torque, real_data.active_joint_ids,
            real_data.model_joint_names, real_data.locked_joint_ids, real_data.locked_angles,
        )
        data = init_state_from_trajectory(model, current_model, window)
        controls = jnp.zeros((torque.shape[0], mjx_model.nu), dtype=jnp.float32)
        controls = controls.at[:, active_ids].set(torque)
        locks = _make_lock_projection(model, current_model, window)
        _, predicted = rollout(
            current_model, data, controls[:-1], parameters["viscous"],
            parameters["coulomb"], config.FRICTION_K, *locks,
        )
        predicted = jnp.concatenate([
            jnp.concatenate([data.qpos, data.qvel])[None, :], predicted
        ])
        components = loss_components(
            predicted, model, window,
            position_scale=measured_position_scale,
            velocity_scale=measured_velocity_scale,
        )
        return (components["总损失"], components) if with_components else components["总损失"]

    objective.loss_scales = (measured_position_scale, measured_velocity_scale)
    return objective


def evaluate_window_ids(evaluate, raw_parameters, windows, ids):
    """固定验证集按采样点数加权，不同批次训练损失不用于选择最佳参数。"""
    total = np.zeros(3)
    count = 0
    for i in ids:
        w = windows[i]
        loss, components = evaluate(raw_parameters, w.pos_real, w.vel_real, w.tor_real)
        values = np.array([float(loss), float(components["角度损失"]), float(components["速度损失"])])
        total += w.num_steps * values
        count += w.num_steps
    return total / count


def main() -> None:
    mode = choose_identification_mode()
    model_path, real_data = load_experiment_interactively()
    dt = ask_sample_dt()
    model, mjx_model = load_model(model_path, dt=dt)
    windows = make_training_windows(real_data, limit_count=False)
    training_ids, validation_ids = split_window_ids(windows)
    batch_size = int(config.TRAINING_WINDOW_COUNT)
    if batch_size < 1 or config.NUM_EPOCHS < 0:
        raise ValueError("批量大小必须为正数，更新次数不能为负数")
    print(
        f"原始数据 {real_data.num_steps} 点，全评估 {len(windows)} 个窗口；"
        f"训练 {len(training_ids)} 个，固定验证 {len(validation_ids)} 个，"
        f"每次更新最多 {batch_size} 个窗口。"
        f"每遍训练约 {int(np.ceil(len(training_ids) / batch_size))} 次更新。",
        flush=True,
    )
    num_active = len(real_data.active_joint_ids)
    inertia_calibration = None
    if mode in {"inertia", "all"}:
        inertia_calibration = calibrate_effective_inertia(
            model,
            real_data.active_joint_ids,
            real_data.locked_joint_ids,
            real_data.locked_angles,
        )
    raw_parameters = {}
    if mode in {"inertia", "all"}:
        raw_parameters["inertia"] = jnp.log(
            jnp.asarray(inertia_calibration["effective_base"])
        )
    if mode in {"friction", "all"}:
        raw_parameters["viscous"] = jnp.log(jnp.full(num_active, config.INITIAL_VISCOUS))
        raw_parameters["coulomb"] = jnp.log(jnp.full(num_active, config.INITIAL_COULOMB))

    objective = make_objective(
        model, mjx_model, real_data, mode, with_components=True,
        inertia_calibration=inertia_calibration,
    )
    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    validate = jax.jit(objective)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.GRADIENT_CLIP_NORM),
        optax.adam(config.LEARNING_RATE),
    )
    optimizer_state = optimizer.init(raw_parameters)
    batches = iter_window_batches(training_ids, batch_size, config.TRAINING_RANDOM_SEED)
    visited_ids = set()
    best_loss, best_epoch, best_raw = float("inf"), -1, None
    history, validation_history = [], []
    stop_reason = "update_limit"
    failure_update = -1
    failure_window_id = -1
    failure_details = ""
    updates_done = 0

    def check_validation(update):
        nonlocal best_loss, best_epoch, best_raw
        print(f"第 {update} 次更新：评估固定验证集（{len(validation_ids)} 个窗口）...", flush=True)
        metrics = evaluate_window_ids(validate, raw_parameters, windows, validation_ids)
        validation_history.append([update, *metrics])
        if np.isfinite(metrics).all() and metrics[0] < best_loss:
            best_loss, best_epoch = float(metrics[0]), update
            best_raw = dict(raw_parameters)
        print(f"验证损失 = {metrics[0]:.6e}，角度 = {metrics[1]:.6e}，速度 = {metrics[2]:.6e}", flush=True)
        return bool(np.isfinite(metrics).all())

    print("开始单窗口编译和初始验证；后续同长度窗口复用编译。", flush=True)
    valid = check_validation(0)
    if not valid:
        stop_reason = "nonfinite_validation"
    for update in range(1, config.NUM_EPOCHS + 1):
        if not valid:
            break
        cycle, ids = next(batches)
        gradients_sum = jax.tree.map(jnp.zeros_like, raw_parameters)
        metrics_sum = np.zeros(3)
        sample_count = 0
        batch_ok = True
        for window_id in ids:
            w = windows[window_id]
            (loss, components), gradients = value_and_grad(
                raw_parameters, w.pos_real, w.vel_real, w.tor_real,
            )
            metrics = np.array([float(loss), float(components["角度损失"]), float(components["速度损失"])])
            if not np.isfinite(metrics).all() or not all(
                np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradients)
            ):
                failure_update = update
                failure_window_id = int(window_id)
                failure_details = (
                    f"loss={metrics[0]}, angle={metrics[1]}, velocity={metrics[2]}"
                )
                print(
                    f"训练窗口 {window_id}（更新 {update}）损失/梯度非有限；"
                    f"{failure_details}；停止更新，保留验证最优参数。",
                    flush=True,
                )
                stop_reason, batch_ok = "nonfinite_training", False
                break
            visited_ids.add(int(window_id))
            gradients_sum = jax.tree.map(
                lambda a, b: a + w.num_steps * b, gradients_sum, gradients,
            )
            metrics_sum += w.num_steps * metrics
            sample_count += w.num_steps
        if not batch_ok:
            break
        gradients = jax.tree.map(lambda g: g / sample_count, gradients_sum)
        metrics = metrics_sum / sample_count
        history.append([updates_done, *metrics])
        changes, next_optimizer_state = optimizer.update(gradients, optimizer_state, raw_parameters)
        candidate = optax.apply_updates(raw_parameters, changes)
        if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(candidate)):
            stop_reason = "nonfinite_parameters"
            failure_update = update
            failure_details = "candidate_parameters contains NaN/Inf"
            break
        raw_parameters, optimizer_state = candidate, next_optimizer_state
        updates_done = update
        if update == 1 or update % max(1, config.LOG_EVERY) == 0:
            print(
                f"第 {update} 次更新，第 {cycle} 遍训练；更新前批次损失 {metrics[0]:.6e}；"
                f"累计覆盖 {len(visited_ids)}/{len(training_ids)} 个训练窗口", flush=True,
            )
        if update % max(1, config.VALIDATION_EVERY) == 0 or update == config.NUM_EPOCHS:
            valid = check_validation(update)
            if not valid:
                stop_reason = "nonfinite_validation"
    if validation_history[-1][0] != updates_done:
        check_validation(updates_done)
    if best_raw is None:
        raise RuntimeError("固定验证集未得到有限损失，不保存无效辨识结果。")
    raw_parameters = best_raw
    print(
        f"训练停止（{stop_reason}），采用第 {best_epoch} 次更新后的验证最优参数，"
        f"验证损失 = {best_loss:.6e}；不代表已收敛。", flush=True,
    )

    # 使用最终参数再仿真一次，并保存给 results.py 绘图。
    body_ids = model.jnt_bodyid[real_data.active_joint_ids]
    final_parameters = make_physical_parameters(
        raw_parameters,
        real_data.active_joint_ids,
        model.nv,
        (inertia_calibration["effective_base"] if inertia_calibration is not None
         else mjx_model.body_inertia[body_ids, 2]),
        np.full(model.nv, config.INITIAL_VISCOUS, dtype=np.float32),
        np.full(model.nv, config.INITIAL_COULOMB, dtype=np.float32),
        mode,
    )
    current_model = mjx_model
    if mode in {"inertia", "all"}:
        if inertia_calibration is None:
            body_inertia_values = final_parameters["inertia"]
        else:
            body_inertia_values = jnp.asarray(inertia_calibration["body_base"])
            body_inertia_values = body_inertia_values + (
                final_parameters["inertia"]
                - jnp.asarray(inertia_calibration["effective_base"])
            ) / jnp.asarray(inertia_calibration["sensitivity"])
            body_inertia_values = jnp.maximum(body_inertia_values, 1.0e-8)
        body_inertia = current_model.body_inertia.at[body_ids, 2].set(
            body_inertia_values
        )
        current_model = current_model.replace(body_inertia=body_inertia)

    result_path = config.RESULTS_DIR / config.RESULT_FILENAME
    checkpoint_path = result_path.with_name(result_path.stem + "_checkpoint.npz")
    active_joint_ids = np.asarray(real_data.active_joint_ids, dtype=np.int32)
    metadata = dict(
        mode=mode, parameter_selection="best_validation_loss",
        best_epoch=best_epoch, best_loss=best_loss,
        loss_history=np.asarray(history).reshape(-1, 4), dt=dt,
        loss_history_scope="changing_training_batch_before_update",
        validation_history=np.asarray(validation_history),
        training_window_ids=training_ids, validation_window_ids=validation_ids,
        visited_training_window_ids=np.asarray(sorted(visited_ids), dtype=np.int32),
        training_random_seed=config.TRAINING_RANDOM_SEED,
        validation_fraction=config.VALIDATION_FRACTION,
        normalize_loss_by_data_std=config.NORMALIZE_LOSS_BY_DATA_STD,
        position_loss_scale=objective.loss_scales[0],
        velocity_loss_scale=objective.loss_scales[1],
        stop_reason=stop_reason, updates_completed=updates_done,
        failure_update=failure_update, failure_window_id=failure_window_id,
        failure_details=failure_details,
        trajectory_alignment="includes_initial_state",
        model_path=str(Path(model_path).resolve()),
        pos_real=real_data.pos_real, vel_real=real_data.vel_real,
        tor_real=real_data.tor_real,
        active_joint_ids=active_joint_ids,
        locked_joint_ids=np.asarray(real_data.locked_joint_ids, dtype=np.int32),
        locked_angles=np.asarray(real_data.locked_angles, dtype=np.float32),
        qpos_addresses=np.asarray(model.jnt_qposadr[active_joint_ids]),
        qvel_addresses=np.asarray(model.jnt_dofadr[active_joint_ids]),
        model_nq=model.nq, model_nv=model.nv,
        model_joint_names=np.asarray(real_data.model_joint_names),
        estimated_inertia=np.asarray(final_parameters["inertia"]),
        inertia_parameterization=(
            "fixed_locked_pose_joint_effective_inertia"
            if inertia_calibration is not None else "body_inertia_z"
        ),
        effective_inertia_base=(
            np.asarray(inertia_calibration["effective_base"])
            if inertia_calibration is not None else np.asarray([])
        ),
        effective_inertia_sensitivity=(
            np.asarray(inertia_calibration["sensitivity"])
            if inertia_calibration is not None else np.asarray([])
        ),
        estimated_viscous=np.asarray(final_parameters["viscous"]),
        estimated_coulomb=np.asarray(final_parameters["coulomb"]),
        training_window_steps=config.TRAINING_WINDOW_STEPS,
        training_window_count=config.TRAINING_WINDOW_COUNT,
    )
    save_npz_atomic(
        checkpoint_path, **metadata, artifact_type="parameter_checkpoint",
        **{f"raw_{name}": np.asarray(value) for name, value in raw_parameters.items()},
    )
    print(f"参数检查点已保存到: {checkpoint_path}（不含回放轨迹）", flush=True)

    evaluation_windows = make_training_windows(real_data, limit_count=False)
    evaluate = make_evaluation_rollout(model, current_model, real_data, final_parameters)
    print(
        f"开始全数据分窗评估：{len(evaluation_windows)} 个窗口；"
        "首个窗口需要编译，相同长度的后续窗口复用编译结果。",
        flush=True,
    )
    started = time.monotonic()
    predictions = []
    for index, window in enumerate(evaluation_windows, 1):
        predicted = np.asarray(evaluate(window.pos_real, window.vel_real, window.tor_real))
        if not np.isfinite(predicted).all():
            raise RuntimeError(
                f"评估窗口 {index} 出现非有限轨迹；参数已保存至 {checkpoint_path}，"
                "未覆盖正式结果。"
            )
        predictions.append(predicted)
        if index == 1 or index % max(1, config.EVALUATION_LOG_EVERY) == 0 or index == len(evaluation_windows):
            print(
                f"评估进度 {index}/{len(evaluation_windows)}，"
                f"已用时 {time.monotonic() - started:.1f} 秒", flush=True,
            )

    result = dict(
        **metadata, artifact_type="evaluation_result",
        simulation_trajectory=np.concatenate(predictions, axis=0),
        evaluation_pos_real=np.concatenate([w.pos_real for w in evaluation_windows]),
        evaluation_vel_real=np.concatenate([w.vel_real for w in evaluation_windows]),
        evaluation_mode="all_windows_multi_shooting",
        evaluation_sample_indices=np.arange(real_data.num_steps),
        evaluation_window_starts=np.cumsum([0] + [w.num_steps for w in evaluation_windows[:-1]]),
        evaluation_window_lengths=np.asarray([w.num_steps for w in evaluation_windows]),
        free_rollout_saved=False,
    )
    save_npz_atomic(result_path, **result)
    print(f"辨识结果已保存到: {result_path}（可用于绘图和回放）", flush=True)

    if config.SAVE_FREE_ROLLOUT:
        print("开始可选的整段自由仿真；上述分窗结果已安全保存。", flush=True)
        free_trajectory = np.asarray(evaluate(real_data.pos_real, real_data.vel_real, real_data.tor_real))
        if np.isfinite(free_trajectory).all():
            result.update(free_rollout_trajectory=free_trajectory, free_rollout_saved=True)
            save_npz_atomic(result_path, **result)
            print(f"连续自由仿真已追加到: {result_path}", flush=True)
        else:
            print("连续自由仿真出现非有限值，保留已保存的分窗结果。", flush=True)


if __name__ == "__main__":
    main()
