# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent 环境配置基类 (Environment Configuration Base Classes)

使用 dataclass 定义环境配置，参考 Isaac Lab 的配置风格。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any


@dataclass
class EffectorCfg:
    """执行器配置"""
    type: str = "Rigid"
    init_pos: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    init_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_dim: int = 6
    # action_scale_p 和 action_scale_v 必须与 action_dim 匹配
    action_scale_p: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    action_scale_v: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    
    # Mesh 配置
    mesh_file: str = ""
    mesh_file_vis: Optional[str] = None  # 可视化模型（可选）
    mesh_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_material: str = "STIRRER"
    mesh_softness: float = 100.0
    
    # Boundary 配置
    boundary_type: str = "cube"
    boundary_lower: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    boundary_upper: Tuple[float, float, float] = (0.95, 0.95, 0.95)
    # Cylinder boundary
    boundary_xz_radius: float = 0.42
    boundary_xz_center: Tuple[float, float] = (0.5, 0.5)
    boundary_y_range: Tuple[float, float] = (0.05, 0.95)


@dataclass
class AgentCfg:
    """Agent 配置"""
    type: str = "AgentRigid"
    collide_type: str = "particle"  # ['particle', 'grid', 'both']
    effectors: List[EffectorCfg] = field(default_factory=list)


@dataclass
class BodyCfg:
    """流体/物体配置"""
    type: str = "cylinder"  # cube, cylinder, ball
    material: str = "WATER"
    
    # Common parameters
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    # Cylinder parameters
    height: float = 0.2
    radius: float = 0.1
    
    # Cube parameters
    lower: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    upper: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    
    # Ball parameters (uses center and radius)


@dataclass
class StaticCfg:
    """静态物体配置"""
    file: str = ""
    file_vis: Optional[str] = None
    pos: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    material: str = "TABLE"
    has_dynamics: bool = True
    sdf_res: int = 128


@dataclass
class BoundaryCfg:
    """边界配置"""
    type: str = "cube"
    lower: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    upper: Tuple[float, float, float] = (0.95, 0.95, 0.95)
    
    # Cylinder boundary
    xz_radius: float = 0.42
    xz_center: Tuple[float, float] = (0.5, 0.5)
    y_range: Tuple[float, float] = (0.05, 0.95)


@dataclass
class RendererCfg:
    """GL 渲染器配置"""
    # camera_pos: Tuple[float, float, float] = (-0.15, 2.82, 2.5)
    # camera_lookat: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    fov: float = 30.0
    particle_radius: float = 0.0075
    
    # Light configuration
    # light_pos: Tuple[float, float, float] = (3.5, 15.0, 0.55)
    # light_lookat: Tuple[float, float, float] = (0.5, 0.5, 0.49)
    light_fov: float = 20.0


@dataclass
class RheoEnvCfg:
    """环境基础配置

    所有环境配置类应继承此类。
    """
    # 基础仿真参数
    dim: int = 3
    horizon: int = 500
    horizon_action: int = 500
    action_range: Tuple[float, float] = (-0.02, 0.02)
    
    # 粒子参数
    particle_density: float = 1e6
    n_obs_ptcls_per_body: int = 200
    
    # 物理参数
    gravity: Tuple[float, float, float] = (0.0, -9.8, 0.0)
    max_substeps_local: int = 50
    quality: int = 1
    
    # 运行时参数
    num_envs: int = 1
    enable_grad: bool = False
    seed: int = 0
    
    # 组件配置
    agent: Optional[AgentCfg] = None
    bodies: List[BodyCfg] = field(default_factory=list)
    statics: List[StaticCfg] = field(default_factory=list)
    boundary: Optional[BoundaryCfg] = None
    renderer: Optional[RendererCfg] = None
    
    # Loss 配置
    loss_type: str = "diff"  # diff, default
    loss_enabled: bool = True


def cfg_to_dict(cfg) -> dict:
    """将 dataclass 配置转换为字典"""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)

