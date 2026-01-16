# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pouring 环境配置"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from rheo.env_cfg import (
    RheoEnvCfg, 
    AgentCfg, 
    EffectorCfg, 
    BodyCfg, 
    StaticCfg,
    BoundaryCfg,
    RendererCfg,
)


@dataclass
class PouringEnvCfg(RheoEnvCfg):
    """Pouring 环境配置
    
    流体倾倒任务：控制玻璃杯将流体倒入目标容器。
    """
    # 基础参数
    horizon: int = 3300
    horizon_action: int = 3300
    action_range: Tuple[float, float] = (-0.02, 0.02)
    n_obs_ptcls_per_body: int = 500
    
    # 物理参数
    gravity: Tuple[float, float, float] = (0.0, -9.8, 0.0)
    max_substeps_local: int = 20
    
    # 材料类型
    material: str = "WATER"
    
    # Agent 配置 - 手持玻璃杯
    agent: AgentCfg = field(default_factory=lambda: AgentCfg(
        type="AgentRigid",
        collide_type="particle",
        effectors=[EffectorCfg(
            type="Rigid",
            init_pos=(0.5, 0.75, 0.5),
            init_euler=(0.0, 0.0, 0.0),
            action_dim=6,
            action_scale_p=(1.0, 1.0, 1.0),
            action_scale_v=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            mesh_file="glass.obj",
            mesh_file_vis="glass_vis.obj",
            mesh_pos=(0.0, 0.0, 0.0),
            mesh_scale=(0.3, 0.3, 0.3),
            mesh_material="BOTTLE",
            mesh_softness=666.0,
            boundary_type="cube",
            boundary_lower=(0.05, 0.05, 0.05),
            boundary_upper=(0.95, 0.95, 0.95),
        )]
    ))
    
    # 流体配置
    bodies: List[BodyCfg] = field(default_factory=lambda: [
        BodyCfg(
            type="cylinder",
            material="WATER",  # 将在运行时被 self.material 覆盖
            center=(0.5, 0.55, 0.5),
            height=0.4,
            radius=0.07,
        )
    ])
    
    # 静态物体 - 目标杯子和桌子
    statics: List[StaticCfg] = field(default_factory=lambda: [
        StaticCfg(
            file="glass.obj",
            file_vis="glass_vis.obj",
            pos=(0.5, 0.15, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(0.3, 0.3, 0.3),
            material="BOTTLE",
            has_dynamics=True,
            sdf_res=256,
        ),
        StaticCfg(
            file="table.obj",
            pos=(0.5, 0.0, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            material="TABLE",
            has_dynamics=True,
        ),
    ])
    
    # 边界配置
    boundary: BoundaryCfg = field(default_factory=lambda: BoundaryCfg(
        type="cube",
        lower=(0.05, 0.05, 0.05),
        upper=(0.95, 0.95, 0.95),
    ))
    
    # 渲染器配置 (GL 渲染器)
    renderer: RendererCfg = field(default_factory=lambda: RendererCfg(
        camera_pos=(-0.15, 2.82, 2.5),
        camera_lookat=(0.5, 0.5, 0.5),
        fov=30.0,
    ))

