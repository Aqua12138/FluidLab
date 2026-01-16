# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pouring 环境 - 流体倾倒任务"""

import gymnasium as gym
from . import agents

gym.register(
    id="Pouring-v0",
    entry_point=f"{__name__}.pouring_env:PouringEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pouring_env_cfg:PouringEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "shac_cfg_entry_point": f"{agents.__name__}:shac_cfg.yaml",
    },
)

