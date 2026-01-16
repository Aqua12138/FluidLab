# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TableMaterials 环境 - 多材料展示任务"""

import gymnasium as gym
from . import agents

gym.register(
    id="TableMaterials-v0",
    entry_point=f"{__name__}.tablematerials_env:TableMaterialsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tablematerials_env_cfg:TableMaterialsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "shac_cfg_entry_point": f"{agents.__name__}:shac_cfg.yaml",
    },
)

