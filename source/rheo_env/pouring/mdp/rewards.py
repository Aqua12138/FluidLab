# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pouring 环境奖励函数

定义流体倾倒任务的奖励函数。
"""

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from rheo.env_base import RheoEnvBase


def particles_in_target(
    env: "RheoEnvBase",
    target_pos: Tuple[float, float, float] = (0.5, 0.15, 0.5),
    target_radius: float = 0.1,
) -> float:
    """计算目标区域内的粒子数量作为奖励
    
    Args:
        env: 环境实例
        target_pos: 目标位置 (x, y, z)
        target_radius: 目标区域半径
        
    Returns:
        奖励值（目标区域内的粒子比例）
    """
    state = env.taichi_env.simulator.get_state_RL()
    if 'x' not in state:
        return 0.0
    
    particles = state['x']
    used = state['used']
    
    # 只考虑活跃的粒子
    active_particles = particles[used > 0]
    if len(active_particles) == 0:
        return 0.0
    
    # 计算粒子到目标的距离（只考虑 xz 平面）
    target = np.array(target_pos)
    distances_xz = np.sqrt(
        (active_particles[:, 0] - target[0]) ** 2 +
        (active_particles[:, 2] - target[2]) ** 2
    )
    
    # 计算在目标区域内的粒子比例
    in_target = (distances_xz < target_radius).sum()
    reward = in_target / len(active_particles)
    
    return float(reward)


def pour_progress(
    env: "RheoEnvBase",
    initial_height: float = 0.55,
) -> float:
    """计算倾倒进度作为奖励
    
    奖励流体从高处流向低处的进度。
    
    Args:
        env: 环境实例
        initial_height: 初始流体高度
        
    Returns:
        奖励值
    """
    state = env.taichi_env.simulator.get_state_RL()
    if 'x' not in state:
        return 0.0
    
    particles = state['x']
    used = state['used']
    
    active_particles = particles[used > 0]
    if len(active_particles) == 0:
        return 0.0
    
    # 计算平均高度下降
    current_avg_height = active_particles[:, 1].mean()
    progress = (initial_height - current_avg_height) / initial_height
    
    return float(max(0.0, progress))


def spillage_penalty(
    env: "RheoEnvBase",
    floor_height: float = 0.05,
) -> float:
    """计算溢出惩罚
    
    惩罚流体溢出到地面。
    
    Args:
        env: 环境实例
        floor_height: 地面高度
        
    Returns:
        惩罚值（负数）
    """
    state = env.taichi_env.simulator.get_state_RL()
    if 'x' not in state:
        return 0.0
    
    particles = state['x']
    used = state['used']
    
    active_particles = particles[used > 0]
    if len(active_particles) == 0:
        return 0.0
    
    # 计算地面以下的粒子比例
    on_floor = (active_particles[:, 1] < floor_height).sum()
    penalty = -on_floor / len(active_particles)
    
    return float(penalty)


def compute_reward(env: "RheoEnvBase") -> float:
    """计算总奖励
    
    组合所有奖励分量。
    
    Args:
        env: 环境实例
        
    Returns:
        总奖励值
    """
    reward = 0.0
    
    # 目标区域奖励 (权重 1.0)
    reward += 1.0 * particles_in_target(env)
    
    # 倾倒进度奖励 (权重 0.5)
    reward += 0.5 * pour_progress(env)
    
    # 溢出惩罚 (权重 0.5)
    reward += 0.5 * spillage_penalty(env)
    
    return reward

