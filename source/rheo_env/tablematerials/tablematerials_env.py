# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TableMaterials 环境 - 多材料展示任务"""

from typing import Optional
from yacs.config import CfgNode

from rheo.env_base import RheoEnvBase
from rheo.macros import *

from .tablematerials_env_cfg import TableMaterialsEnvCfg


class TableMaterialsEnv(RheoEnvBase):
    """TableMaterials 环境
    
    多材料展示任务：在桌面上展示多种 Herschel-Bulkley 流体材料。
    """
    
    cfg_class = TableMaterialsEnvCfg
    
    def __init__(
        self,
        cfg: Optional[TableMaterialsEnvCfg] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """初始化 TableMaterials 环境"""
        # 移除 renderer_type，只使用 GL 渲染器
        kwargs.pop('renderer_type', None)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
    
    def setup_agent(self):
        """设置 Agent - 直接使用 dataclass 配置"""
        agent_cfg = CfgNode(new_allowed=True)
        
        # 使用 dataclass 配置（TableMaterialsEnvCfg.agent）
        cfg_agent = self.cfg.agent
        agent_cfg.type = cfg_agent.type
        agent_cfg.params = CfgNode()
        agent_cfg.params.collide_type = cfg_agent.collide_type
        
        # 转换 effector 配置
        agent_cfg.effectors = []
        for eff_cfg in cfg_agent.effectors:
            effector = {
                'type': eff_cfg.type,
                'params': {
                    'init_pos': eff_cfg.init_pos,
                    'init_euler': eff_cfg.init_euler,
                    'action_dim': eff_cfg.action_dim,
                    'action_scale_p': eff_cfg.action_scale_p,
                    'action_scale_v': eff_cfg.action_scale_v,
                },
                'mesh': {
                    'file': eff_cfg.mesh_file,
                    'pos': (0.0, 0.0, 0.0),
                    'scale': eff_cfg.mesh_scale,
                    'material': eff_cfg.mesh_material,
                    'softness': eff_cfg.mesh_softness,
                },
                'boundary': {
                    'type': eff_cfg.boundary_type,
                    'xz_radius': eff_cfg.boundary_xz_radius,
                    'xz_center': eff_cfg.boundary_xz_center,
                    'y_range': eff_cfg.boundary_y_range,
                }
            }
            agent_cfg.effectors.append(effector)
        
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent
    
    def setup_statics(self):
        """设置静态物体 - 从配置文件读取"""
        for static_cfg in self.cfg.statics:
            # 解析材料常量
            material = static_cfg.material
            if isinstance(material, str):
                material = globals().get(material, RIGID)
            
            self.taichi_env.add_static(
                file=static_cfg.file,
                file_vis=getattr(static_cfg, 'file_vis', None),
                pos=static_cfg.pos,
                euler=static_cfg.euler,
                scale=static_cfg.scale,
                material=material,
                has_dynamics=static_cfg.has_dynamics,
                sdf_res=getattr(static_cfg, 'sdf_res', 128),
            )
    
    def setup_bodies(self):
        """设置流体 - 从配置文件读取"""
        for body_cfg in self.cfg.bodies:
            # 解析材料
            material = body_cfg.material
            if isinstance(material, str):
                material_const = globals().get(material, TOOTHPASTE)
            else:
                material_const = material
            
            body_type = body_cfg.type
            
            # 根据 body 类型传递不同参数
            if body_type == 'cylinder':
                self.taichi_env.add_body(
                    type='cylinder',
                    center=body_cfg.center,
                    height=body_cfg.height,
                    radius=body_cfg.radius,
                    material=material_const,
                )
            elif body_type == 'cube':
                self.taichi_env.add_body(
                    type='cube',
                    lower=body_cfg.lower,
                    upper=body_cfg.upper,
                    material=material_const,
                )
            elif body_type == 'ball':
                self.taichi_env.add_body(
                    type='ball',
                    center=body_cfg.center,
                    radius=body_cfg.radius,
                    material=material_const,
                )
            else:
                raise ValueError(f"Unknown body type: {body_type}")
    
    def setup_boundary(self):
        """设置边界 - 从配置文件读取"""
        boundary_cfg = self.cfg.boundary
        self.taichi_env.setup_boundary(
            type=boundary_cfg.type,
            lower=boundary_cfg.lower,
            upper=boundary_cfg.upper,
        )
    
    def setup_renderer(self):
        """设置 GL 渲染器 - 从配置文件读取"""
        renderer_cfg = self.cfg.renderer
        
        # 构建渲染器参数
        renderer_kwargs = {
            'fov': renderer_cfg.fov,
            'particle_radius': renderer_cfg.particle_radius,
            'light_fov': renderer_cfg.light_fov,
        }
        
        # 可选的相机/光照位置（如果配置了则使用，否则让渲染器自动计算）
        if renderer_cfg.camera_pos:
            renderer_kwargs['camera_pos'] = renderer_cfg.camera_pos
        if renderer_cfg.camera_lookat:
            renderer_kwargs['camera_lookat'] = renderer_cfg.camera_lookat
        if renderer_cfg.light_pos:
            renderer_kwargs['light_pos'] = renderer_cfg.light_pos
        if renderer_cfg.light_lookat:
            renderer_kwargs['light_lookat'] = renderer_cfg.light_lookat
        
        self.taichi_env.setup_renderer(**renderer_kwargs)
    
    def setup_loss(self):
        """设置损失函数"""
        pass
    
    def demo_policy(self):
        """返回演示策略"""
        return None
    
    def trainable_policy(self, optim_cfg, init_range):
        """返回可训练策略"""
        return None

