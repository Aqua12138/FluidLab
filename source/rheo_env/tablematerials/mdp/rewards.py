# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TableMaterials 环境奖励函数"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rheo.env_base import RheoEnvBase


def mixing_uniformity(env: "RheoEnvBase") -> float:
    """计算混合均匀度作为奖励
    
    Args:
        env: 环境实例
        
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
    
    # 计算空间分布的均匀性
    std_x = np.std(active_particles[:, 0])
    std_z = np.std(active_particles[:, 2])
    
    # 均匀分布时标准差较大
    uniformity = (std_x + std_z) / 2.0
    
    return float(uniformity)


def compute_reward(env: "RheoEnvBase") -> float:
    """计算总奖励"""
    return mixing_uniformity(env)

