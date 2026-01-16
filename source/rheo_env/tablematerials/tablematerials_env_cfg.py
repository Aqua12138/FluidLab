# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TableMaterials 环境配置"""

from dataclasses import dataclass, field
from typing import Tuple, List

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
class TableMaterialsEnvCfg(RheoEnvCfg):
    """TableMaterials 环境配置
    
    多材料展示任务：在桌面上展示多种 Herschel-Bulkley 流体材料。
    """
    # 基础参数
    horizon: int = 3000
    horizon_action: int = 3000
    action_range: Tuple[float, float] = (-0.01, 0.01)
    n_obs_ptcls_per_body: int = 500
    
    # 物理参数
    gravity: Tuple[float, float, float] = (0.0, -9.8, 0.0)
    max_substeps_local: int = 20
    quality: int = 2  # 提高网格分辨率
    
    # Agent 配置 - 搅拌器
    agent: AgentCfg = field(default_factory=lambda: AgentCfg(
        type="AgentRigid",
        collide_type="particle",
        effectors=[EffectorCfg(
            type="Rigid",
            init_pos=(0.6, 0.5, 0.5),
            init_euler=(0.0, 0.0, 0.0),
            action_dim=6,
            mesh_file="glass.obj",
            mesh_file_vis="glass_vis.obj",
            mesh_pos=(0.0, 0.0, 0.0),
            mesh_scale=(0.3, 0.3, 0.3),
            mesh_material="BOTTLE",
            mesh_softness=100.0,
            boundary_type="cylinder",
            boundary_xz_radius=0.42,
            boundary_xz_center=(0.5, 0.5),
            boundary_y_range=(0.05, 0.95),
        )]
    ))
    
    # 流体配置 - 默认使用单一材料
    bodies: List[BodyCfg] = field(default_factory=lambda: [
        BodyCfg(
            type="cylinder",
            material="TOOTHPASTE",
            center=(0.6, 0.7, 0.5),
            height=0.4,
            radius=0.07,
        )
    ])
    
    # 静态物体 - 桌子
    statics: List[StaticCfg] = field(default_factory=lambda: [
        StaticCfg(
            file="table.obj",
            pos=(0.5, 0.15, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            material="TABLE",
            has_dynamics=True,
            sdf_res=256,
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
        camera_pos=(0.5, 0.8, 3.0),
        camera_lookat=(0.5, 0.3, 0.5),
        fov=30.0,
        particle_radius=0.004,
    ))

