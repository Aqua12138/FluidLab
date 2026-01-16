# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TableMaterials 环境观测函数"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rheo.env_base import RheoEnvBase


def particle_positions_sampled(
    env: "RheoEnvBase",
    n_samples: int = 200,
    env_id: int = 0,
) -> np.ndarray:
    """采样粒子位置作为观测"""
    state = env.taichi_env.simulator.get_state_RL(env_id=env_id)
    
    if 'x' not in state:
        return np.zeros(n_samples * 3, dtype=np.float32)
    
    positions = state['x']
    used = state['used']
    active_positions = positions[used > 0]
    
    if len(active_positions) == 0:
        return np.zeros(n_samples * 3, dtype=np.float32)
    
    step_size = max(1, len(active_positions) // n_samples)
    sampled = active_positions[::step_size][:n_samples]
    
    if len(sampled) < n_samples:
        padding = np.zeros((n_samples - len(sampled), 3), dtype=np.float32)
        sampled = np.vstack([sampled, padding])
    
    return sampled.flatten().astype(np.float32)


def get_observation(
    env: "RheoEnvBase",
    env_id: int = 0,
    n_particle_samples: int = 200,
) -> np.ndarray:
    """获取完整观测"""
    return particle_positions_sampled(env, n_particle_samples, env_id)

