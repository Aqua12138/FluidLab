# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TableMaterials 环境终止条件"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rheo.env_base import RheoEnvBase


def time_out(env: "RheoEnvBase") -> bool:
    """超时终止"""
    return env.t >= env.cfg.horizon


def nan_detected(env: "RheoEnvBase", reward: float) -> bool:
    """NaN 检测终止"""
    return np.isnan(reward)


def check_termination(env: "RheoEnvBase", reward: float) -> bool:
    """检查所有终止条件"""
    if time_out(env):
        return True
    if nan_detected(env, reward):
        return True
    return False

