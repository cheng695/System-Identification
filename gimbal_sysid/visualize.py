"""使用 MuJoCo Viewer 播放参数辨识后的仿真轨迹。"""

from __future__ import annotations

from gimbal_sysid import config

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def load_saved_trajectory(result_path: str | Path):
    """读取 train.py 保存的结果文件。"""
    result_path = Path(result_path).expanduser().resolve()
    if not result_path.exists():
        raise FileNotFoundError(f"找不到辨识结果文件: {result_path}")

    with np.load(result_path, allow_pickle=False) as result:
        trajectory = result["simulation_trajectory"]
        if "model_path" in result.files:
            saved_model_path = result["model_path"]
            model_path = (
                saved_model_path.item()
                if saved_model_path.ndim == 0
                else str(saved_model_path)
            )
        else:
            model_path = None

    if model_path is None:
        return None, trajectory
    return Path(model_path), trajectory


def play_trajectory(
    model_path: str | Path,
    trajectory: np.ndarray,
    playback_speed: float = config.PLAYBACK_SPEED,
    dt: float | None = None,
) -> None:
    """在 MuJoCo Viewer 中播放完整 qpos/qvel 轨迹。"""
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 MJCF 模型文件: {model_path}")
    if playback_speed <= 0.0:
        raise ValueError("playback_speed 必须大于 0")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    trajectory = np.asarray(trajectory, dtype=np.float64)

    expected_width = model.nq + model.nv
    if trajectory.ndim != 2 or trajectory.shape[1] != expected_width:
        raise ValueError(
            f"轨迹形状应为 (采样点数量, {expected_width})，"
            f"实际为 {trajectory.shape}"
        )

    sample_dt = float(model.opt.timestep) if dt is None else dt
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("dt 必须为有限正数")
    frame_time = sample_dt / playback_speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame in trajectory:
                if not viewer.is_running():
                    break

                data.qpos[:] = frame[:model.nq]
                data.qvel[:] = frame[model.nq:model.nq + model.nv]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(frame_time)


def main() -> None:
    default_result = config.RESULTS_DIR / config.RESULT_FILENAME
    result_input = input(
        f"请输入辨识结果文件路径（直接回车使用 {default_result}）："
    ).strip()
    result_path = Path(result_input).expanduser() if result_input else default_result

    speed_input = input(f"请输入播放速度倍数（直接回车使用 {config.PLAYBACK_SPEED}）：").strip()
    playback_speed = float(speed_input) if speed_input else config.PLAYBACK_SPEED

    model_path, trajectory = load_saved_trajectory(result_path)
    if model_path is None:
        model_input = input("旧结果文件没有保存模型路径，请输入 MJCF 绝对路径：").strip()
        if not model_input:
            raise ValueError("MJCF 模型路径不能为空")
        model_path = Path(model_input).expanduser().resolve()
    print(f"模型路径: {model_path}")
    print(f"轨迹形状: {trajectory.shape}")
    print("正在启动 MuJoCo Viewer，关闭窗口即可退出。")
    with np.load(result_path, allow_pickle=False) as result:
        dt = float(result["dt"]) if "dt" in result.files else None
    if dt is None:
        print("旧结果未保存 dt，回放使用 XML 步长；建议重新训练生成结果。")
    play_trajectory(model_path, trajectory, playback_speed, dt=dt)


if __name__ == "__main__":
    main()
