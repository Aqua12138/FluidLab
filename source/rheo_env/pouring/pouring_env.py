# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pouring 环境 - 流体倾倒任务"""

from typing import Optional
from yacs.config import CfgNode

from rheo.env_base import RheoEnvBase
from rheo.macros import *

from .pouring_env_cfg import PouringEnvCfg


class PouringEnv(RheoEnvBase):
    """Pouring 环境
    
    流体倾倒任务：控制玻璃杯将流体倒入目标容器。
    """
    
    cfg_class = PouringEnvCfg
    
    def __init__(
        self,
        cfg: Optional[PouringEnvCfg] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """初始化 Pouring 环境
        
        Args:
            cfg: 环境配置
            render_mode: 渲染模式
            **kwargs: 额外参数，支持:
                - material: 流体材料类型 (默认 "WATER")
        """
        # 处理额外参数
        self.material = kwargs.pop('material', None)
        # 移除 renderer_type，只使用 GL 渲染器
        kwargs.pop('renderer_type', None)
        
        # 调用父类构造函数
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
    
    def setup_agent(self):
        """设置 Agent - 手持玻璃杯 (直接使用 dataclass 配置)"""
        agent_cfg = CfgNode(new_allowed=True)
        
        # 使用 dataclass 配置
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
                    'file_vis': eff_cfg.mesh_file_vis,
                    'pos': eff_cfg.mesh_pos,
                    'scale': eff_cfg.mesh_scale,
                    'material': eff_cfg.mesh_material,
                    'softness': eff_cfg.mesh_softness,
                } if eff_cfg.mesh_file else None,
                'boundary': {
                    'type': eff_cfg.boundary_type,
                    'lower': eff_cfg.boundary_lower,
                    'upper': eff_cfg.boundary_upper,
                }
            }
            agent_cfg.effectors.append(effector)
        
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent
    
    def setup_statics(self):
        """设置静态物体 - 目标杯子和桌子"""
        # 目标玻璃杯
        self.taichi_env.add_static(
            file='glass.obj',
            file_vis='glass_vis.obj',
            pos=(0.5, 0.15, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(0.3, 0.3, 0.3),
            material=BOTTLE,
            has_dynamics=True,
            sdf_res=256,
        )
        
        # 桌子
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.5, 0.0, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=TABLE,
            has_dynamics=True,
        )
    
    def setup_bodies(self):
        """设置流体"""
        # 确定材料类型
        material = self.material or self.cfg.material
        if isinstance(material, str):
            material_const = globals().get(material, WATER)
        else:
            material_const = material
        
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.55, 0.5),
            height=0.4,
            radius=0.07,
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
        """设置 GL 渲染器 - 使用自动计算的相机/光照位置"""
        # 只传递非相机/光照参数，让 GLRenderer 自动计算位置
        self.taichi_env.setup_renderer(
            fov=30,
            light_fov=20,
        )
    
    def setup_loss(self):
        """设置损失函数 - 可以为空，使用 MDP 中的奖励"""
        pass
    
    def demo_policy(self, user_input=False):
        """返回演示策略"""
        from rheo.utils.keyboard import KeyboardPolicy_6DOF
        
        if not user_input:
            raise NotImplementedError
        
        return KeyboardPolicy_6DOF(v_lin=1, v_ang=1)

