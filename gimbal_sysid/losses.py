"""仿真轨迹与真实采样轨迹之间的损失函数。"""

from __future__ import annotations

from gimbal_sysid import config

import jax.numpy as jnp

from gimbal_sysid.data import TrajectoryData


def split_simulation_trajectory(
    simulation_trajectory,
    nq: int,
    nv: int,
):
    """将 dynamics.rollout 返回的轨迹拆成角度和速度。

    ``dynamics.rollout`` 的每一行是：

        [qpos0, qpos1, ..., qvel0, qvel1, ...]
    """
    simulation_trajectory = jnp.asarray(simulation_trajectory, dtype=jnp.float32)
    expected_shape = nq + nv
    if simulation_trajectory.ndim != 2:
        raise ValueError("仿真轨迹必须是二维数组")
    if simulation_trajectory.shape[1] != expected_shape:
        raise ValueError(
            f"仿真轨迹第二维应为 {expected_shape}，"
            f"实际为 {simulation_trajectory.shape[1]}"
        )

    pos_sim = simulation_trajectory[:, :nq]
    vel_sim = simulation_trajectory[:, nq:nq + nv]
    return pos_sim, vel_sim


def _mse(error):
    """计算均方误差。"""
    return jnp.mean(jnp.square(error))


def compute_loss(
    simulation_trajectory,
    model,
    real_data: TrajectoryData,
    position_weight: float = config.POSITION_LOSS_WEIGHT,
    velocity_weight: float = config.VELOCITY_LOSS_WEIGHT,
):
    """只对有真实测量数据的关节计算角度和速度损失。

    参数：
        simulation_trajectory: ``dynamics.rollout`` 返回的完整模型轨迹。
        model: 普通 MuJoCo ``MjModel``，用于查询关节的 qpos/qvel 地址。
        real_data: CSV 读取后的 ``TrajectoryData``。
        position_weight: 角度误差权重。
        velocity_weight: 速度误差权重。

    返回：
        一个可以直接用于 JAX 求导的标量损失。
    """
    if real_data.active_joint_ids is None:
        raise ValueError("真实数据缺少 active_joint_ids")

    pos_sim, vel_sim = split_simulation_trajectory(
        simulation_trajectory,
        model.nq,
        model.nv,
    )

    qpos_addresses = [
        model.jnt_qposadr[joint_id]
        for joint_id in real_data.active_joint_ids
    ]
    qvel_addresses = [
        model.jnt_dofadr[joint_id]
        for joint_id in real_data.active_joint_ids
    ]

    pos_sim_measured = pos_sim[:, qpos_addresses]
    vel_sim_measured = vel_sim[:, qvel_addresses]

    pos_real = jnp.asarray(real_data.pos_real, dtype=jnp.float32)
    vel_real = jnp.asarray(real_data.vel_real, dtype=jnp.float32)

    if pos_sim_measured.shape != pos_real.shape:
        raise ValueError(
            f"角度数据形状不一致：仿真为 {pos_sim_measured.shape}，"
            f"真实数据为 {pos_real.shape}"
        )
    if vel_sim_measured.shape != vel_real.shape:
        raise ValueError(
            f"速度数据形状不一致：仿真为 {vel_sim_measured.shape}，"
            f"真实数据为 {vel_real.shape}"
        )

    position_loss = _mse(pos_sim_measured - pos_real)
    velocity_loss = _mse(vel_sim_measured - vel_real)
    return position_weight * position_loss + velocity_weight * velocity_loss


def loss_components(
    simulation_trajectory,
    model,
    real_data: TrajectoryData,
):
    """返回角度损失、速度损失和总损失，方便调试和观察。"""
    if real_data.active_joint_ids is None:
        raise ValueError("真实数据缺少 active_joint_ids")

    pos_sim, vel_sim = split_simulation_trajectory(
        simulation_trajectory,
        model.nq,
        model.nv,
    )
    qpos_addresses = [
        model.jnt_qposadr[joint_id]
        for joint_id in real_data.active_joint_ids
    ]
    qvel_addresses = [
        model.jnt_dofadr[joint_id]
        for joint_id in real_data.active_joint_ids
    ]

    position_loss = _mse(
        pos_sim[:, qpos_addresses]
        - jnp.asarray(real_data.pos_real, dtype=jnp.float32)
    )
    velocity_loss = _mse(
        vel_sim[:, qvel_addresses]
        - jnp.asarray(real_data.vel_real, dtype=jnp.float32)
    )

    return {
        "角度损失": position_loss,
        "速度损失": velocity_loss,
        "总损失": config.POSITION_LOSS_WEIGHT * position_loss + config.VELOCITY_LOSS_WEIGHT * velocity_loss,
    }
