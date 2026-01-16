# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent 环境基类 (Environment Base Class)

所有流体仿真环境应继承此基类。
"""

import os
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from rheo.macros import *
from rheo.taichi_env import TaichiEnv, init_taichi
from rheo.utils.misc import set_random_seed, should_render
from rheo.env_cfg import RheoEnvCfg


class RheoEnvBase(gym.Env):
    """流体仿真环境基类
    
    支持 GPU 批并行 (num_envs > 1):
    - observations shape: (num_envs, obs_dim)
    - actions shape: (num_envs, action_dim)
    - rewards shape: (num_envs,)
    - dones shape: (num_envs,)
    
    子类需要实现:
    - setup_agent(): 设置 agent
    - setup_statics(): 设置静态物体
    - setup_bodies(): 设置流体/物体
    - setup_boundary(): 设置边界
    - setup_renderer(): 设置渲染器
    - setup_loss(): 设置损失函数 (可选)
    """
    
    # 子类应定义默认配置类
    cfg_class: type = RheoEnvCfg
    
    def __init__(
        self,
        cfg: Optional[RheoEnvCfg] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """初始化环境
        
        Args:
            cfg: 环境配置，如果为 None 则使用默认配置
            render_mode: 渲染模式 ('human', 'rgb_array', None)
            **kwargs: 额外参数（用于 gym.make 兼容性）
        """
        super().__init__()
        
        # 加载配置
        if cfg is None:
            cfg = self.cfg_class()
        self.cfg = cfg
        
        # 从 kwargs 或 cfg 获取运行时参数
        self.num_envs = kwargs.get('num_envs', cfg.num_envs)
        self.enable_grad = kwargs.get('enable_grad', cfg.enable_grad)
        seed = kwargs.get('seed', cfg.seed)
        
        # 设置随机种子
        if seed is not None:
            set_random_seed(seed)
        
        # 初始化 Taichi
        init_taichi(num_envs=self.num_envs, enable_grad=self.enable_grad)
        
        # 基础参数
        self.horizon = cfg.horizon
        self.horizon_action = cfg.horizon_action
        self.target_file = None
        self._n_obs_ptcls_per_body = cfg.n_obs_ptcls_per_body
        self.loss_enabled = cfg.loss_enabled
        self.loss_type = cfg.loss_type
        self.action_range = np.array(cfg.action_range)
        self.render_mode = render_mode
        
        # 创建 TaichiEnv
        self.taichi_env = TaichiEnv(
            dim=cfg.dim,
            quality=cfg.quality,
            particle_density=cfg.particle_density,
            max_substeps_local=cfg.max_substeps_local,
            gravity=cfg.gravity,
            horizon=self.horizon,
            num_envs=self.num_envs,
            enable_grad=self.enable_grad,
        )
        
        # 构建环境
        self.build_env()
        self._setup_gym_spaces()
    
    def build_env(self):
        """构建环境的所有组件"""
        self.setup_agent()
        self.setup_statics()
        self.setup_bodies()
        self.setup_smoke_field()
        self.setup_boundary()
        
        if should_render():
            self.setup_renderer()
        
        if self.loss_enabled:
            self.setup_loss()
        
        self.taichi_env.build()
        self._init_state = self.taichi_env.get_state()
        
        print(f'===>  {type(self).__name__} built successfully.')
    
    def setup_agent(self):
        """设置 agent - 子类应重写此方法"""
        pass
    
    def setup_statics(self):
        """设置静态物体 - 子类应重写此方法"""
        pass
    
    def setup_bodies(self):
        """设置流体/物体 - 子类应重写此方法"""
        pass
    
    def setup_smoke_field(self):
        """设置烟雾场 - 子类可重写此方法"""
        pass
    
    def setup_boundary(self):
        """设置边界 - 子类应重写此方法"""
        pass
    
    def setup_renderer(self):
        """设置 GL 渲染器 - 子类可重写此方法"""
        # 获取渲染器配置参数
        renderer_kwargs = {}
        if self.cfg.renderer is not None:
            from rheo.env_cfg import cfg_to_dict
            renderer_cfg = cfg_to_dict(self.cfg.renderer)
            renderer_kwargs.update(renderer_cfg)
        
        self.taichi_env.setup_renderer(**renderer_kwargs)
    
    def setup_loss(self):
        """设置损失函数 - 子类可重写此方法"""
        pass
    
    def _setup_gym_spaces(self):
        """设置 Gym 空间"""
        obs = self.reset()[0]  # gymnasium 返回 (obs, info)
        
        # Observation space
        if self.num_envs == 1:
            self.observation_space = Box(
                DTYPE_NP(-np.inf), DTYPE_NP(np.inf), 
                obs.shape, dtype=DTYPE_NP
            )
        else:
            single_obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
            self.observation_space = Box(
                DTYPE_NP(-np.inf), DTYPE_NP(np.inf), 
                (self.num_envs, single_obs_dim), dtype=DTYPE_NP
            )
        
        # Action space
        if self.taichi_env.agent is not None:
            if self.num_envs == 1:
                self.action_space = Box(
                    DTYPE_NP(self.action_range[0]), 
                    DTYPE_NP(self.action_range[1]),
                    (self.taichi_env.agent.action_dim,), 
                    dtype=DTYPE_NP
                )
            else:
                self.action_space = Box(
                    DTYPE_NP(self.action_range[0]), 
                    DTYPE_NP(self.action_range[1]),
                    (self.num_envs, self.taichi_env.agent.action_dim), 
                    dtype=DTYPE_NP
                )
        else:
            self.action_space = None
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        if seed is not None:
            set_random_seed(seed)
        
        self.taichi_env.set_state(**self._init_state)
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def _get_obs(self) -> np.ndarray:
        """获取观测
        
        Returns:
            obs: shape (num_envs, obs_dim) 或 (obs_dim,)
        """
        if self.num_envs == 1:
            return self._get_single_obs(0)
        else:
            obs_list = []
            for env_id in range(self.num_envs):
                obs_list.append(self._get_single_obs(env_id))
            return np.stack(obs_list, axis=0)
    
    def _get_single_obs(self, env_id: int) -> np.ndarray:
        """获取单个环境的观测"""
        state = self.taichi_env.simulator.get_state_RL(env_id=env_id)
        obs = []
        
        if 'x' in state:
            for body_id in range(self.taichi_env.particles['bodies']['n']):
                body_n_particles = self.taichi_env.particles['bodies']['n_particles'][body_id]
                body_particle_ids = self.taichi_env.particles['bodies']['particle_ids'][body_id]
                
                step_size = max(1, body_n_particles // self._n_obs_ptcls_per_body)
                body_x = state['x'][body_particle_ids][::step_size]
                body_v = state['v'][body_particle_ids][::step_size]
                body_used = state['used'][body_particle_ids][::step_size]
                
                obs.append(body_x.flatten())
                obs.append(body_v.flatten())
                obs.append(body_used.astype(DTYPE_NP).flatten())  # 确保 float32
        
        if 'agent' in state:
            obs += state['agent']
        
        if 'smoke_field' in state:
            obs.append(state['smoke_field']['v'][::10, 60:68, ::10].flatten())
            obs.append(state['smoke_field']['q'][::10, 60:68, ::10].flatten())
        
        obs = np.concatenate(obs).astype(DTYPE_NP)  # 确保返回 float32
        return obs
    
    def _get_reward(self) -> np.ndarray:
        """获取奖励"""
        if self.taichi_env.loss is None:
            if self.num_envs == 1:
                return 0.0
            else:
                return np.zeros(self.num_envs, dtype=DTYPE_NP)
        
        loss_info = self.taichi_env.get_step_loss()
        reward = loss_info['reward']
        
        if self.num_envs > 1 and np.isscalar(reward):
            reward = np.full(self.num_envs, reward, dtype=DTYPE_NP)
        
        return reward
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """执行一步
        
        Args:
            action: 动作，shape (num_envs, action_dim) 或 (action_dim,)
            
        Returns:
            observation: 观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        action = np.array(action).clip(self.action_range[0], self.action_range[1])
        
        # 处理多环境情况
        if self.num_envs > 1 and len(action.shape) == 1:
            action = np.tile(action, (self.num_envs, 1))
        
        self.taichi_env.step(action if self.num_envs == 1 else action[0])
        
        obs = self._get_obs()
        reward = self._get_reward()
        
        assert self.t <= self.horizon
        
        if self.num_envs == 1:
            terminated = self.t == self.horizon
            truncated = False
            if np.isnan(reward):
                reward = -1000
                terminated = True
        else:
            terminated = np.full(self.num_envs, self.t == self.horizon, dtype=bool)
            truncated = np.zeros(self.num_envs, dtype=bool)
            nan_mask = np.isnan(reward)
            if nan_mask.any():
                reward[nan_mask] = -1000
                terminated[nan_mask] = True
        
        info = {}
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """渲染环境
        
        Args:
            mode: 渲染模式，可选 'human', 'rgb_array'。
                  如果不指定，使用 self.render_mode 或默认 'human'
        """
        # headless 模式下跳过渲染
        if not should_render():
            return None
        
        # 确定渲染模式：参数 > self.render_mode > 默认 'human'
        render_mode = mode or self.render_mode or 'human'
        
        if render_mode == "rgb_array":
            return self.taichi_env.render('rgb_array')
        else:  # 'human' 或其他
            self.taichi_env.render('human')
            return None
    
    def close(self):
        """关闭环境"""
        pass
    
    @property
    def t(self) -> int:
        """当前时间步"""
        return self.taichi_env.t
    
    # ============ 兼容性方法 ============
    
    def demo_policy(self):
        """返回演示策略 - 子类可重写"""
        return None
    
    def trainable_policy(self, optim_cfg, init_range):
        """返回可训练策略 - 子类可重写"""
        return None
