"""加载 MuJoCo 模型以及创建 MJX 模型"""

from __future__ import annotations

from gimbal_sysid import config

from pathlib import Path
import math

import mujoco


def get_model_joint_names(xml_path: str | Path) -> list[str]:
    """读取 MJCF 模型中的关节名称。"""
    xml_path = Path(xml_path).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        or f"joint_{joint_id}"
        for joint_id in range(model.njnt)
    ]


def load_model(xml_path: str | Path, dt: float | None = None):
    """加载 MuJoCo 模型 以及 创建 MJX 模型"""

    # 只有真正创建 MJX 模型时才导入 mjx，单纯查看关节名称时不需要它。
    from mujoco import mjx

    xml_path = Path(xml_path).expanduser().resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {xml_path}")

    # 从 MJCF/XML 文件读取机器人模型，创建普通 MuJoCo 模型对象。
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if dt is not None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("采样周期 dt 必须是有限的正数，单位为秒")
        model.opt.timestep = dt

    if config.DISABLE_CONTACTS_DURING_TRAINING:
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0

    # 设置约束求解器的最大迭代次数。
    model.opt.iterations = config.SOLVER_ITERATIONS
    # 设置 Newton/CG 求解器的线搜索最大迭代次数。
    model.opt.ls_iterations = config.SOLVER_LS_ITERATIONS

    # 将普通 MuJoCo 模型转换为 MJX 模型，使其可以被 JAX 使用。
    mjx_model = mjx.put_model(model)

    print(f"XML path: {xml_path}")
    print(f"nq(角度) = {model.nq}")
    print(f"nv(速度) = {model.nv}")
    print(f"nu(控制输入) = {model.nu}")
    print(f"仿真步长 = {model.opt.timestep} 秒")

    return model, mjx_model
