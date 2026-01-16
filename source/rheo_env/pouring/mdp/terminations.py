# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pouring 环境终止条件

定义流体倾倒任务的终止条件。
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rheo.env_base import RheoEnvBase


def time_out(env: "RheoEnvBase") -> bool:
    """超时终止
    
    Args:
        env: 环境实例
        
    Returns:
        是否终止
    """
    return env.t >= env.cfg.horizon


def nan_detected(env: "RheoEnvBase", reward: float) -> bool:
    """NaN 检测终止
    
    当奖励值为 NaN 时终止。
    
    Args:
        env: 环境实例
        reward: 当前奖励值
        
    Returns:
        是否终止
    """
    return np.isnan(reward)


def all_particles_lost(env: "RheoEnvBase", threshold: float = 0.1) -> bool:
    """所有粒子丢失终止
    
    当活跃粒子比例低于阈值时终止。
    
    Args:
        env: 环境实例
        threshold: 粒子丢失阈值
        
    Returns:
        是否终止
    """
    state = env.taichi_env.simulator.get_state_RL()
    
    if 'used' not in state:
        return False
    
    used = state['used']
    active_ratio = used.sum() / len(used)
    
    return active_ratio < threshold


def agent_out_of_bounds(
    env: "RheoEnvBase",
    bounds_lower: tuple = (0.0, 0.0, 0.0),
    bounds_upper: tuple = (1.0, 1.0, 1.0),
) -> bool:
    """Agent 越界终止
    
    当 Agent 超出指定边界时终止。
    
    Args:
        env: 环境实例
        bounds_lower: 下边界
        bounds_upper: 上边界
        
    Returns:
        是否终止
    """
    state = env.taichi_env.simulator.get_state_RL()
    
    if 'agent' not in state or len(state['agent']) == 0:
        return False
    
    # 获取 agent 位置（假设第一个元素是位置）
    agent_pos = state['agent'][0][:3]
    
    for i in range(3):
        if agent_pos[i] < bounds_lower[i] or agent_pos[i] > bounds_upper[i]:
            return True
    
    return False


def check_termination(env: "RheoEnvBase", reward: float) -> bool:
    """检查所有终止条件
    
    Args:
        env: 环境实例
        reward: 当前奖励值
        
    Returns:
        是否终止
    """
    # 超时
    if time_out(env):
        return True
    
    # NaN 检测
    if nan_detected(env, reward):
        return True
    
    # 粒子丢失
    if all_particles_lost(env):
        return True
    
    return False

