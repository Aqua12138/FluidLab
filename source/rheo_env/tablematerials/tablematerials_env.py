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
        """设置静态物体 - 桌子"""
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.5, 0.15, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=TABLE,
            has_dynamics=True,
            sdf_res=256,
        )
    
    def setup_bodies(self):
        """设置流体 - 从 dataclass 配置读取"""
        for body_cfg in self.cfg.bodies:
            # 解析材料
            material = body_cfg.material
            if isinstance(material, str):
                material_const = globals().get(material, TOOTHPASTE)
            else:
                material_const = material
            
            self.taichi_env.add_body(
                type=body_cfg.type,
                center=body_cfg.center,
                height=body_cfg.height,
                radius=body_cfg.radius,
                material=material_const,
            )
    
    def setup_boundary(self):
        """设置边界"""
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
        )
    
    def setup_renderer(self):
        """设置 GL 渲染器"""
        self.taichi_env.setup_renderer(
            camera_pos=(0.5, 0.8, 3.0),
            camera_lookat=(0.5, 0.3, 0.5),
            fov=30,
            particle_radius=0.004,
        )
    
    def setup_loss(self):
        """设置损失函数"""
        pass
    
    def demo_policy(self):
        """返回演示策略"""
        return None
    
    def trainable_policy(self, optim_cfg, init_range):
        """返回可训练策略"""
        return None

