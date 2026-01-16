# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pouring 环境观测函数

定义流体倾倒任务的观测函数。
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rheo.env_base import RheoEnvBase


def particle_positions_sampled(
    env: "RheoEnvBase",
    n_samples: int = 200,
    env_id: int = 0,
) -> np.ndarray:
    """采样粒子位置作为观测
    
    Args:
        env: 环境实例
        n_samples: 采样粒子数量
        env_id: 环境 ID
        
    Returns:
        采样的粒子位置，shape (n_samples * 3,)
    """
    state = env.taichi_env.simulator.get_state_RL(env_id=env_id)
    
    if 'x' not in state:
        return np.zeros(n_samples * 3, dtype=np.float32)
    
    positions = state['x']
    used = state['used']
    
    # 只选择活跃的粒子
    active_positions = positions[used > 0]
    
    if len(active_positions) == 0:
        return np.zeros(n_samples * 3, dtype=np.float32)
    
    # 均匀采样
    step_size = max(1, len(active_positions) // n_samples)
    sampled = active_positions[::step_size][:n_samples]
    
    # 如果采样不足，用零填充
    if len(sampled) < n_samples:
        padding = np.zeros((n_samples - len(sampled), 3), dtype=np.float32)
        sampled = np.vstack([sampled, padding])
    
    return sampled.flatten().astype(np.float32)


def particle_velocities_sampled(
    env: "RheoEnvBase",
    n_samples: int = 200,
    env_id: int = 0,
) -> np.ndarray:
    """采样粒子速度作为观测
    
    Args:
        env: 环境实例
        n_samples: 采样粒子数量
        env_id: 环境 ID
        
    Returns:
        采样的粒子速度，shape (n_samples * 3,)
    """
    state = env.taichi_env.simulator.get_state_RL(env_id=env_id)
    
    if 'v' not in state:
        return np.zeros(n_samples * 3, dtype=np.float32)
    
    velocities = state['v']
    used = state['used']
    
    active_velocities = velocities[used > 0]
    
    if len(active_velocities) == 0:
        return np.zeros(n_samples * 3, dtype=np.float32)
    
    step_size = max(1, len(active_velocities) // n_samples)
    sampled = active_velocities[::step_size][:n_samples]
    
    if len(sampled) < n_samples:
        padding = np.zeros((n_samples - len(sampled), 3), dtype=np.float32)
        sampled = np.vstack([sampled, padding])
    
    return sampled.flatten().astype(np.float32)


def agent_state(
    env: "RheoEnvBase",
    env_id: int = 0,
) -> np.ndarray:
    """获取 agent 状态作为观测
    
    Args:
        env: 环境实例
        env_id: 环境 ID
        
    Returns:
        Agent 状态，包含位置和旋转
    """
    state = env.taichi_env.simulator.get_state_RL(env_id=env_id)
    
    if 'agent' not in state or len(state['agent']) == 0:
        return np.zeros(7, dtype=np.float32)  # pos(3) + quat(4)
    
    agent_states = state['agent']
    return np.concatenate(agent_states).astype(np.float32)


def get_observation(
    env: "RheoEnvBase",
    env_id: int = 0,
    n_particle_samples: int = 200,
) -> np.ndarray:
    """获取完整观测
    
    Args:
        env: 环境实例
        env_id: 环境 ID
        n_particle_samples: 粒子采样数量
        
    Returns:
        观测向量
    """
    obs_parts = []
    
    # 粒子位置
    obs_parts.append(particle_positions_sampled(env, n_particle_samples, env_id))
    
    # 粒子速度
    obs_parts.append(particle_velocities_sampled(env, n_particle_samples, env_id))
    
    # Agent 状态
    obs_parts.append(agent_state(env, env_id))
    
    return np.concatenate(obs_parts)

