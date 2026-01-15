import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import *

class NewPouringEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI', material='WATER'):
        if seed is not None:
            self.seed(seed)

        self.horizon               = 3300
        self.horizon_action        = 3300
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.02, 0.02])
        self.renderer_type         = renderer_type
        self.material              = material  # 材料类型，默认为 WATER

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=20,
            gravity=(0.0, -9.8, 0.0),
            horizon=self.horizon,
            ckpt_dest="cpu"
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_newpouring.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='glass.obj',
            file_vis='glass_vis.obj',  # 使用与 agent 相同的 file_vis，确保颜色一致
            pos=(0.5, 0.15, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(0.3, 0.3, 0.3),  # 与 agent 的 scale 一致
            material=BOTTLE,
            has_dynamics=True,
            sdf_res=256,  # SDF分辨率：256 提高碰撞检测精度，减少穿模（默认128）
        )

        # 添加桌子（参考 flow_env.py）
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.5, 0.0, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=TABLE,  # 使用 TABLE 材料（高摩擦）
            has_dynamics=True,
        )

    def setup_bodies(self):
        # 添加流体体（参考 flow_env.py，但使用可配置的材料）
        # 将字符串转换为材料常量
        if isinstance(self.material, str):
            material_const = globals().get(self.material, WATER)  # 如果找不到，默认使用 WATER
        else:
            material_const = self.material
        
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.55, 0.5),
            height=0.4,
            radius=0.07,
            material=material_const,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        # 支持 GGUI 和 GL 两种渲染器（参考 flow_env.py）
        if self.renderer_type is None or self.renderer_type == 'None':
            return
        
        if self.renderer_type == 'GGUI':
            self.taichi_env.setup_renderer(
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )
        elif self.renderer_type == 'GL':
            self.taichi_env.setup_renderer(
                type='GL',
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(3.5, 15.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=20,
            )
        else:
            raise ValueError(f"Unknown renderer_type: {self.renderer_type}. Supported types: 'GGUI', 'GL', None")

    def setup_loss(self):
        # 可以设置损失函数，或者使用默认的
        pass

    def demo_policy(self, user_input=False):
        if not user_input:
            raise NotImplementedError

        init_p = np.array([0.5, 0.5, 0.5])
        return KeyboardPolicy_vxy_wz(init_p, v_lin=1, v_ang=1)

    def trainable_policy(self, optim_cfg, init_range):
        return PouringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[0, 1, 2, 3, 4])
