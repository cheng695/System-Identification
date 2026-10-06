"""MJX 动力学仿真。"""

from __future__ import annotations

from gimbal_sysid import config

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from mujoco import mjx

if TYPE_CHECKING:
    import mujoco

    from gimbal_sysid.data import TrajectoryData


def init_state(mjx_model, pos_sim=None, vel_sim=None):
    """创建 MJX 仿真状态，并设置初始位置和速度。"""
    data = mjx.make_data(mjx_model)
    pos_sim = jnp.zeros(mjx_model.nq, dtype=jnp.float32) if pos_sim is None else jnp.asarray(pos_sim, dtype=jnp.float32)
    vel_sim = jnp.zeros(mjx_model.nv, dtype=jnp.float32) if vel_sim is None else jnp.asarray(vel_sim, dtype=jnp.float32)

    if pos_sim.shape != (mjx_model.nq,):
        raise ValueError(f"初始位置形状错误，应为 {(mjx_model.nq,)}，实际为 {pos_sim.shape}")
    if vel_sim.shape != (mjx_model.nv,):
        raise ValueError(f"初始速度形状错误，应为 {(mjx_model.nv,)}，实际为 {vel_sim.shape}")
    return data.replace(qpos=pos_sim, qvel=vel_sim)


def init_state_from_trajectory(model: "mujoco.MjModel", mjx_model, trajectory: "TrajectoryData"):
    """根据 CSV 第一帧和锁定角度创建完整仿真初始状态。"""
    if trajectory.active_joint_ids is None or not trajectory.model_joint_names:
        raise ValueError("TrajectoryData 缺少完整模型关节配置")

    pos_sim = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
    vel_sim = jnp.zeros(mjx_model.nv, dtype=jnp.float32)

    for local_id, joint_id in enumerate(trajectory.active_joint_ids):
        pos_sim = pos_sim.at[model.jnt_qposadr[joint_id]].set(trajectory.pos_real[0, local_id])
        vel_sim = vel_sim.at[model.jnt_dofadr[joint_id]].set(trajectory.vel_real[0, local_id])

    if trajectory.locked_joint_ids is not None:
        if trajectory.locked_angles is None:
            raise ValueError("存在 locked_joint_ids，但缺少 locked_angles")
        for joint_id, angle in zip(trajectory.locked_joint_ids, trajectory.locked_angles):
            pos_sim = pos_sim.at[model.jnt_qposadr[joint_id]].set(angle)
            vel_sim = vel_sim.at[model.jnt_dofadr[joint_id]].set(0.0)

    return init_state(mjx_model, pos_sim, vel_sim)


def _make_lock_projection(model: "mujoco.MjModel", mjx_model, trajectory: "TrajectoryData"):
    """创建每一步恢复锁定关节状态所需的掩码和目标值。"""
    qpos_mask = jnp.zeros(mjx_model.nq, dtype=bool)
    locked_qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
    qvel_mask = jnp.zeros(mjx_model.nv, dtype=bool)

    if trajectory.locked_joint_ids is None:
        return qpos_mask, locked_qpos, qvel_mask
    if trajectory.locked_angles is None:
        raise ValueError("存在 locked_joint_ids，但缺少 locked_angles")

    for joint_id, angle in zip(trajectory.locked_joint_ids, trajectory.locked_angles):
        qpos_mask = qpos_mask.at[model.jnt_qposadr[joint_id]].set(True)
        locked_qpos = locked_qpos.at[model.jnt_qposadr[joint_id]].set(angle)
        qvel_mask = qvel_mask.at[model.jnt_dofadr[joint_id]].set(True)
    return qpos_mask, locked_qpos, qvel_mask


def _project_locked_state(data, qpos_mask, locked_qpos, qvel_mask):
    """将锁定关节恢复到指定角度和零速度。"""
    qpos = jnp.where(qpos_mask, locked_qpos, data.qpos)
    qvel = jnp.where(qvel_mask, 0.0, data.qvel)
    return data.replace(qpos=qpos, qvel=qvel)


def step(mjx_model, data, control, viscous_friction, coulomb_friction, friction_k=config.FRICTION_K, locked_qpos_mask=None, locked_qpos=None, locked_qvel_mask=None):
    """执行一步 MJX 仿真，并加入可微库伦摩擦。"""
    control = jnp.asarray(control, dtype=jnp.float32)
    viscous_friction = jnp.asarray(viscous_friction, dtype=jnp.float32)
    coulomb_friction = jnp.asarray(coulomb_friction, dtype=jnp.float32)
    if control.shape != (mjx_model.nu,):
        raise ValueError(f"控制输入形状错误，应为 {(mjx_model.nu,)}，实际为 {control.shape}")

    friction_torque = viscous_friction * data.qvel + coulomb_friction * jnp.tanh(friction_k * data.qvel)
    data = data.replace(ctrl=control, qfrc_applied=-friction_torque)
    next_data = mjx.step(mjx_model, data)

    if locked_qpos_mask is not None:
        next_data = _project_locked_state(next_data, locked_qpos_mask, locked_qpos, locked_qvel_mask)
    return next_data


def rollout(mjx_model, data, controls, viscous_friction, coulomb_friction, friction_k=config.FRICTION_K, locked_qpos_mask=None, locked_qpos=None, locked_qvel_mask=None):
    """使用一整段控制输入进行 MJX 仿真。"""
    controls = jnp.asarray(controls, dtype=jnp.float32)
    if controls.ndim != 2 or controls.shape[1] != mjx_model.nu:
        raise ValueError("controls 应为 (采样点数量, 控制输入数量)，" f"实际形状为 {controls.shape}")

    def scan_fn(current_data, control):
        next_data = step(mjx_model, current_data, control, viscous_friction, coulomb_friction, friction_k, locked_qpos_mask, locked_qpos, locked_qvel_mask)
        observation = jnp.concatenate([next_data.qpos, next_data.qvel])
        return next_data, observation

    return jax.lax.scan(scan_fn, data, controls)


def rollout_from_trajectory(model: "mujoco.MjModel", mjx_model, trajectory: "TrajectoryData", viscous_friction, coulomb_friction, friction_k=config.FRICTION_K):
    """返回与 CSV 同时刻的 N 帧：初始状态 + N-1 次推进。

    假设第 k 行力矩在 [k*dt, (k+1)*dt) 内保持不变。
    最后一行力矩没有对应的下一帧测量，因此不用于推进。
    """
    if trajectory.num_steps < 2:
        raise ValueError("仿真至少需要两个采样点")
    controls = trajectory.make_full_torque()
    data = init_state_from_trajectory(model, mjx_model, trajectory)
    lock_projection = _make_lock_projection(model, mjx_model, trajectory)
    final_data, predicted = rollout(
        mjx_model, data, controls[:-1], viscous_friction,
        coulomb_friction, friction_k, *lock_projection,
    )
    initial = jnp.concatenate([data.qpos, data.qvel])[None, :]
    return final_data, jnp.concatenate([initial, predicted], axis=0)
