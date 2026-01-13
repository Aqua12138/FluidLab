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

class TableMaterialsEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI'):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 2000
        self.horizon_action        = 2000
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.01, 0.01])
        self.renderer_type         = renderer_type

        # create a taichi env
        # quality=2 提高网格分辨率（从64到128），减少穿模问题
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=20,
            gravity=(0.0, -9.8, 0.0),
            horizon=self.horizon,
            quality=2,  # 提高网格分辨率以减少穿模（默认1，2倍分辨率）
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_tablematerials.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        # 添加桌子（使用 table.obj）
        # 桌子位置：y=0.15（桌面高度），x和z方向居中
        # 
        # 碰撞检测说明：
        # 1. has_dynamics=True 是必需的，用于启用碰撞检测（会生成和加载SDF）
        # 2. sdf_res=256 提高SDF分辨率，减少薄物体穿模问题（默认128）
        # 3. 摩擦系数：TABLE 的摩擦系数为 2.0（在 macros.py 中定义，高摩擦）
        #    如需调整摩擦，修改 macros.py 中的 FRICTION[TABLE] 值
        #    0.0=无摩擦, 0.1-0.5=低-中摩擦, 1.0+=高摩擦
        # 4. 网格分辨率：通过 quality=2 提高（在 TaichiEnv 初始化中设置）
        #    这可以减少穿模，但会增加计算量
        self.taichi_env.add_static(
            file='table.obj',  # 文件路径，系统会自动查找 raw/table.obj
            pos=(0.5, 0.15, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),  # 桌子尺寸
            material=TABLE,  # 使用TABLE材料，具有高摩擦系数
            has_dynamics=True,  # 必需：启用碰撞检测（会生成和加载SDF）
            sdf_res=256,  # SDF分辨率：256 提高碰撞检测精度，减少穿模（默认128）
        )

    def setup_bodies(self):
        # 在桌面上每隔一段距离放置一些材料
        # 桌面高度 y = 0.15，材料放在 y = 0.25（桌面上方0.1）
        
        # 定义9种基于Herschel-Bulkley模型的常见材料
        # 1. WATER_NEWTON: 牛顿流体 (n=1, τ_y=0)
        # 2. HONEY: 蜂蜜，剪切变稀 (n=0.5, τ_y=0)
        # 3. KETCHUP: 番茄酱，剪切变稀+屈服应力 (n=0.3, τ_y=20)
        # 4. TOOTHPASTE: 牙膏，剪切变稀+屈服应力 (n=0.4, τ_y=25)
        # 5. BLOOD: 血液，剪切变稀 (n=0.7, τ_y=5)
        # 6. PAINT: 油漆，剪切变稀 (n=0.6, τ_y=10)
        # 7. CORNSTARCH: 玉米淀粉浆，剪切增稠 (n=1.5, τ_y=0)
        # 8. SILLY_PUTTY: 橡皮泥，剪切增稠 (n=1.3, τ_y=0)
        # 9. YOGURT: 酸奶，剪切变稀+屈服应力 (n=0.6, τ_y=12)
        materials = [
            WATER_NEWTON,  # 牛顿流体
            # HONEY,          # 剪切变稀
            # KETCHUP,        # 剪切变稀+屈服应力
            # TOOTHPASTE,     # 剪切变稀+屈服应力
            # BLOOD,          # 剪切变稀
            # PAINT,          # 剪切变稀
            # CORNSTARCH,     # 剪切增稠
            # SILLY_PUTTY,    # 剪切增稠
            # YOGURT,         # 剪切变稀+屈服应力
        ]
        
        # 直接在桌面上放置9种材料，使用固定的位置（参考mixing_env.py的方式）
        # 边界范围：(0.05, 0.05, 0.05) 到 (0.95, 0.95, 0.95)
        # 桌面高度 y = 0.15，材料放在 y = 0.25（桌面上方0.1）
        table_y = 0.25
        
        # 3x3网格布局，直接指定每个材料的位置
        positions = [
            # (0.25, table_y, 0.25),  # [0,0] 左上
            # (0.25, table_y, 0.5),   # [0,1] 左中
            # (0.25, table_y, 0.75),  # [0,2] 左下
            # (0.5, table_y, 0.25),  # [1,0] 中上
            (0.5, table_y, 0.5),    # [1,1] 中心
            # (0.5, table_y, 0.75),   # [1,2] 中下
            # (0.75, table_y, 0.25),  # [2,0] 右上
            # (0.75, table_y, 0.5),   # [2,1] 右中
            # (0.75, table_y, 0.75),  # [2,2] 右下
        ]
        
        # 为每个位置添加对应的材料
        for i, (x, y, z) in enumerate(positions):
            material = materials[i] if i < len(materials) else WATER_NEWTON
            self.taichi_env.add_body(
                type='cylinder',
                center=(x, y, z),
                height=0.1,
                radius=0.05,
                material=material,
            )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        self.taichi_env.setup_renderer(
            camera_pos=(0.5, 0.8, 2.0),
            camera_lookat=(0.5, 0.3, 0.5),
            fov=30,
            particle_radius=0.004,  # 减小粒子半径（默认0.0075），让粒子显示更小，与场景更成比例
            lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                    {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        )

    def setup_loss(self):
        # 可以设置一个简单的损失函数，或者使用默认的
        pass

    def demo_policy(self):
        # 返回一个演示策略（如果需要）
        return None

    def trainable_policy(self, optim_cfg, init_range):
        # 返回一个可训练的策略（如果需要）
        return None
