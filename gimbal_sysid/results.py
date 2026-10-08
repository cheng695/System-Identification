"""生成参数辨识结果、残差统计和图片。"""

from __future__ import annotations

from gimbal_sysid import config

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_result(result_path: str | Path) -> dict[str, np.ndarray]:
    """读取 train.py 保存的 npz 结果文件。"""
    result_path = Path(result_path).expanduser().resolve()
    if not result_path.exists():
        raise FileNotFoundError(f"找不到辨识结果文件: {result_path}")

    with np.load(result_path, allow_pickle=False) as data:
        if "simulation_trajectory" not in data.files:
            raise ValueError("此文件是参数检查点，不含评估轨迹；绘图请使用 identification_result.npz。")
        return {name: data[name] for name in data.files}


def calculate_result(result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """从与训练目标一致的评估轨迹中提取测量关节，并计算残差。"""
    simulation_trajectory = result["simulation_trajectory"]
    pos_real = result.get("evaluation_pos_real", result["pos_real"])
    vel_real = result.get("evaluation_vel_real", result["vel_real"])
    qpos_addresses = result["qpos_addresses"]
    qvel_addresses = result["qvel_addresses"]
    nq = int(result["model_nq"])

    pos_sim = simulation_trajectory[:, qpos_addresses]
    vel_sim = simulation_trajectory[:, nq + qvel_addresses]

    if pos_sim.shape != pos_real.shape:
        raise ValueError(
            f"角度数据形状不一致：仿真为 {pos_sim.shape}，真实为 {pos_real.shape}"
        )
    if vel_sim.shape != vel_real.shape:
        raise ValueError(
            f"速度数据形状不一致：仿真为 {vel_sim.shape}，真实为 {vel_real.shape}"
        )

    return {
        "pos_sim": pos_sim,
        "vel_sim": vel_sim,
        "pos_real": pos_real,
        "vel_real": vel_real,
        "pos_residual": pos_sim - pos_real,
        "vel_residual": vel_sim - vel_real,
    }


def _joint_label(result: dict[str, np.ndarray], local_id: int) -> str:
    joint_ids = result["active_joint_ids"]
    names = result["model_joint_names"]
    joint_id = int(joint_ids[local_id])
    return str(names[joint_id])


def print_parameters(result: dict[str, np.ndarray]) -> None:
    """按保存时的数组索引约定显示参数及其是否参与优化。"""
    mode = str(result["mode"].item()) if "mode" in result else "unknown"
    labels = {"friction": "仅摩擦", "inertia": "仅惯量", "all": "摩擦和惯量"}
    print(f"辨识模式：{labels.get(mode, '未记录')}")
    if str(result.get("parameter_selection", "")) == "best_validation_loss":
        print(f"最佳参数：第 {int(result['best_epoch'])} 次更新，固定验证损失 = {float(result['best_loss']):.6e}（不代表已收敛）")
    elif "parameter_selection" in result and str(result["parameter_selection"].item()) == "best_loss":
        print(f"最佳参数：第 {int(result['best_epoch'])} 次更新，总损失 = {float(result['best_loss']):.6e}（不代表已收敛）")
    else:
        print("最终参数（保存的最后一次更新值，不代表已收敛）：")
    for local_id, joint_id in enumerate(result["active_joint_ids"]):
        print(f"  {_joint_label(result, local_id)}：")
        inertia_is_effective = (
            "inertia_parameterization" in result
            and str(result["inertia_parameterization"].item())
            == "fixed_locked_pose_joint_effective_inertia"
        )
        inertia_label = (
            "固定锁定姿态下关节等效惯量"
            if inertia_is_effective else "刚体第三主惯量"
        )
        for key, label, unit, index, fitted in (
            ("estimated_inertia", inertia_label, "kg·m²", local_id, mode in {"inertia", "all"}),
            ("estimated_viscous", "粘性摩擦 B", "N·m·s/rad", int(joint_id), mode in {"friction", "all"}),
            ("estimated_coulomb", "库伦摩擦 Fc", "N·m", int(joint_id), mode in {"friction", "all"}),
        ):
            if key not in result:
                print(f"    {label}：文件未保存")
                continue
            status = ("参与辨识" if fitted else "固定值") if mode in labels else "辨识状态未知"
            print(f"    {label} = {float(result[key][index]):.8g} {unit}（{status}）")
    if "inertia_parameterization" in result and str(result["inertia_parameterization"].item()) == "fixed_locked_pose_joint_effective_inertia":
        print("惯量字段对应锁定姿态下 M[dof, dof] 的关节等效惯量，已包含子 body 的惯性影响。")
    else:
        print("惯量字段对应当前代码的 body_inertia[:, 2]，并非整个关节的等效惯量。")


def _plot_comparison(
    values: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    signal_name: str,
    unit: str,
    output_path: Path,
) -> None:
    real = values[f"{signal_name}_real"]
    sim = values[f"{signal_name}_sim"]
    sample_ids = np.arange(real.shape[0])
    if "dt" in result:
        sample_ids = sample_ids * float(result["dt"])

    figure, axes = plt.subplots(
        real.shape[1],
        1,
        figsize=(config.FIGURE_WIDTH, max(config.FIGURE_MIN_HEIGHT, config.FIGURE_HEIGHT_PER_JOINT * real.shape[1])),
        squeeze=False,
    )
    for local_id, axis_row in enumerate(axes):
        axis = axis_row[0]
        axis.plot(sample_ids, real[:, local_id], label="real")
        axis.plot(sample_ids, sim[:, local_id], label="simulation", alpha=config.SIMULATION_LINE_ALPHA)
        axis.set_title(f"{_joint_label(result, local_id)} {signal_name}")
        axis.set_xlabel("time (s)" if "dt" in result else "sample")
        axis.set_ylabel(unit)
        axis.grid(True)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close(figure)


def _plot_residuals(
    values: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    sample_ids = np.arange(values["pos_residual"].shape[0])
    if "dt" in result:
        sample_ids = sample_ids * float(result["dt"])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(config.FIGURE_WIDTH, config.RESIDUAL_FIGURE_HEIGHT),
        sharex=True,
    )

    for local_id in range(values["pos_residual"].shape[1]):
        label = _joint_label(result, local_id)
        axes[0].plot(
            sample_ids,
            values["pos_residual"][:, local_id],
            label=label,
        )
        axes[1].plot(
            sample_ids,
            values["vel_residual"][:, local_id],
            label=label,
        )

    axes[0].set_ylabel("position residual")
    axes[1].set_ylabel("velocity residual")
    axes[1].set_xlabel("time (s)" if "dt" in result else "sample")
    axes[0].set_title("Identification residuals")
    for axis in axes:
        axis.grid(True)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close(figure)


def report_segments(values, result, output_dir):
    """前中后三段详细曲线及训练/验证误差，不把拼接窗口称作自由仿真。"""
    n = len(values["pos_real"])
    dt = float(result.get("dt", 1.0))
    time_axis = np.asarray(result.get("evaluation_sample_indices", np.arange(n))) * dt
    starts = result.get("evaluation_window_starts")
    lengths = result.get("evaluation_window_lengths")
    if starts is not None and lengths is not None:
        for label, key in (("训练窗口", "training_window_ids"), ("固定验证窗口", "validation_window_ids")):
            mask = np.zeros(n, dtype=bool)
            for i in result.get(key, []):
                mask[int(starts[i]):int(starts[i] + lengths[i])] = True
            if mask.any():
                print(f"{label}：角度 RMSE = {np.sqrt(np.mean(values['pos_residual'][mask] ** 2)):.6e}，"
                      f"速度 RMSE = {np.sqrt(np.mean(values['vel_residual'][mask] ** 2)):.6e}")
    chunks = [c for c in np.array_split(np.arange(n), 3) if len(c)]
    for i, ids in enumerate(chunks):
        print(f"时间段 {i + 1} [{time_axis[ids[0]]:.3f}, {time_axis[ids[-1]]:.3f}]："
              f"角度 RMSE = {np.sqrt(np.mean(values['pos_residual'][ids] ** 2)):.6e}，"
              f"速度 RMSE = {np.sqrt(np.mean(values['vel_residual'][ids] ** 2)):.6e}")
    for joint in range(values["pos_real"].shape[1]):
        fig, axes = plt.subplots(len(chunks), 2, figsize=(12, 3 * len(chunks)), squeeze=False)
        for row, ids in enumerate(chunks):
            # 每段中部显示两秒，避免长轨迹挤成一团。
            width = min(len(ids), max(2, int(2.0 / dt)))
            offset = (len(ids) - width) // 2
            zoom = ids[offset:offset + width]
            for col, (signal, unit) in enumerate((("pos", "rad"), ("vel", "rad/s"))):
                ax = axes[row, col]
                simulated = values[f"{signal}_sim"][zoom, joint].copy()
                if starts is not None:
                    simulated[np.isin(zoom, starts)] = np.nan
                ax.plot(time_axis[zoom], values[f"{signal}_real"][zoom, joint], label="real")
                ax.plot(time_axis[zoom], simulated, label="window simulation", alpha=0.8)
                ax.set(title=f"Segment {row + 1}: {signal}", xlabel="time (s)", ylabel=unit)
                ax.grid(True)
                ax.legend()
        fig.suptitle(f"{_joint_label(result, joint)} - reset at each window")
        fig.tight_layout()
        fig.savefig(output_dir / f"segment_comparison_{joint}.png", dpi=config.FIGURE_DPI)
        plt.close(fig)


def generate_report(result_path: str | Path, output_dir: str | Path | None = None) -> None:
    """生成对比图、残差图和终端统计信息。"""
    result_path = Path(result_path).expanduser().resolve()
    result = load_result(result_path)
    print_parameters(result)
    if "evaluation_mode" in result:
        print(f"评估方式：{str(result['evaluation_mode'].item())}")
    if "stop_reason" in result:
        reason = str(result["stop_reason"])
        print(f"训练停止原因：{reason}")
        if reason != "update_limit" and "failure_update" in result:
            print(
                f"异常位置：更新 {int(result['failure_update'])}，"
                f"窗口 {int(result['failure_window_id'])}；"
                f"{str(result.get('failure_details', ''))}"
            )
    values = calculate_result(result)

    if output_dir is None:
        output_dir = result_path.parent / config.FIGURES_DIRNAME
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_comparison(
        values,
        result,
        "pos",
        "rad",
        output_dir / config.POSITION_FIGURE,
    )
    _plot_comparison(
        values,
        result,
        "vel",
        "rad/s",
        output_dir / config.VELOCITY_FIGURE,
    )
    _plot_residuals(
        values,
        result,
        output_dir / config.RESIDUAL_FIGURE,
    )

    print(f"角度残差均方根 = {np.sqrt(np.mean(values['pos_residual'] ** 2)):.6e}")
    print(f"速度残差均方根 = {np.sqrt(np.mean(values['vel_residual'] ** 2)):.6e}")
    print(f"评估覆盖 {len(values['pos_real'])}/{len(result['pos_real'])} 个采样点；每窗从实测初态重置。")
    report_segments(values, result, output_dir)
    print(f"图片已保存到: {output_dir}")


def main() -> None:
    default_path = config.RESULTS_DIR / config.RESULT_FILENAME
    result_input = input(
        f"请输入结果文件路径（直接回车使用 {default_path}）："
    ).strip()
    result_path = result_input or default_path
    generate_report(result_path)


if __name__ == "__main__":
    main()
