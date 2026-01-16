# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent 环境基类 (Environment Base Class)

所有流体仿真环境应继承此基类。
支持 Isaac Lab 风格的 GPU-Only 数据流。
"""

import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union

from rheo.macros import *
from rheo.taichi_env import TaichiEnv, init_taichi
from rheo.utils.misc import set_random_seed, should_render, is_headless
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
        device: str = 'cuda',
        **kwargs
    ):
        """初始化环境
        
        Args:
            cfg: 环境配置，如果为 None 则使用默认配置
            render_mode: 渲染模式 ('human', 'rgb_array', None)
            device: 计算设备 ('cuda' for GPU-only flow, 'cpu' for legacy mode)
            **kwargs: 额外参数（用于 gym.make 兼容性）
        """
        super().__init__()
        
        # 加载配置
        if cfg is None:
            cfg = self.cfg_class()
        self.cfg = cfg
        
        # GPU-Only 模式设置
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._gpu_mode = (self.device.type == 'cuda')
        
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
        self.render_mode = render_mode
        
        # Action range: 存储为 PyTorch tensor (GPU) 用于快速 clamp
        self._action_range_np = np.array(cfg.action_range)
        self.action_range = torch.tensor(cfg.action_range, dtype=DTYPE_TC, device=self.device)
        
        # GPU buffer 占位符 (懒初始化)
        self._obs_buffer: Optional[torch.Tensor] = None
        self._reward_buffer: Optional[torch.Tensor] = None
        self._terminated_buffer: Optional[torch.Tensor] = None
        self._truncated_buffer: Optional[torch.Tensor] = None
        self._obs_dim: Optional[int] = None
        
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
            ckpt_dest=cfg.ckpt_dest,
        )
        
        # 构建环境
        self.build_env()
        self._setup_gym_spaces()
        
        # 预分配 GPU buffer
        self._allocate_gpu_buffers()
    
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
        # 临时使用 legacy 模式获取 obs 形状
        obs = self._get_obs_legacy()
        
        # 记录 obs 维度用于 buffer 分配
        if self.num_envs == 1:
            self._obs_dim = obs.shape[0]
        else:
            self._obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
        
        # Observation space
        if self.num_envs == 1:
            self.observation_space = Box(
                DTYPE_NP(-np.inf), DTYPE_NP(np.inf), 
                obs.shape, dtype=DTYPE_NP
            )
        else:
            self.observation_space = Box(
                DTYPE_NP(-np.inf), DTYPE_NP(np.inf), 
                (self.num_envs, self._obs_dim), dtype=DTYPE_NP
            )
        
        # Action space
        if self.taichi_env.agent is not None:
            action_range_np = self._action_range_np
            if self.num_envs == 1:
                self.action_space = Box(
                    DTYPE_NP(action_range_np[0]), 
                    DTYPE_NP(action_range_np[1]),
                    (self.taichi_env.agent.action_dim,), 
                    dtype=DTYPE_NP
                )
            else:
                self.action_space = Box(
                    DTYPE_NP(action_range_np[0]), 
                    DTYPE_NP(action_range_np[1]),
                    (self.num_envs, self.taichi_env.agent.action_dim), 
                    dtype=DTYPE_NP
                )
        else:
            self.action_space = None
    
    def _allocate_gpu_buffers(self):
        """预分配 GPU buffer 用于 GPU-only 数据流"""
        if not self._gpu_mode or self._obs_dim is None:
            return
        
        # Observation buffer
        if self.num_envs == 1:
            self._obs_buffer = torch.zeros(self._obs_dim, dtype=DTYPE_TC, device=self.device)
        else:
            self._obs_buffer = torch.zeros(
                (self.num_envs, self._obs_dim), dtype=DTYPE_TC, device=self.device
            )
        
        # Reward buffer
        if self.num_envs == 1:
            self._reward_buffer = torch.zeros(1, dtype=DTYPE_TC, device=self.device)
        else:
            self._reward_buffer = torch.zeros(self.num_envs, dtype=DTYPE_TC, device=self.device)
        
        # Terminated/Truncated buffers
        if self.num_envs == 1:
            self._terminated_buffer = torch.zeros(1, dtype=torch.bool, device=self.device)
            self._truncated_buffer = torch.zeros(1, dtype=torch.bool, device=self.device)
        else:
            self._terminated_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._truncated_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """重置环境 (支持 GPU-only 或 legacy 模式)
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 观测 (torch.Tensor if GPU mode, else np.ndarray)
            info: 额外信息
        """
        super().reset(seed=seed)
        
        if seed is not None:
            set_random_seed(seed)
        
        self.taichi_env.set_state(**self._init_state)
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def reset_gpu(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """重置环境 (GPU-only 模式，强制返回 CUDA tensor)
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: torch.Tensor on CUDA
            info: 额外信息
        """
        super().reset(seed=seed)
        
        if seed is not None:
            set_random_seed(seed)
        
        self.taichi_env.set_state(**self._init_state)
        
        obs = self._get_obs_gpu() if self._obs_buffer is not None else self._get_obs_legacy()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        
        info = {}
        return obs, info
    
    def _get_obs(self) -> Union[np.ndarray, torch.Tensor]:
        """获取观测 (自动选择 GPU 或 legacy 模式)
        
        Returns:
            obs: shape (num_envs, obs_dim) 或 (obs_dim,)
                 GPU 模式返回 torch.Tensor (CUDA), 否则返回 np.ndarray
        """
        if self._gpu_mode and self._obs_buffer is not None:
            return self._get_obs_gpu()
        return self._get_obs_legacy()
    
    def _get_obs_legacy(self) -> np.ndarray:
        """获取观测 (Legacy NumPy 模式)
        
        Returns:
            obs: shape (num_envs, obs_dim) 或 (obs_dim,)
        """
        if self.num_envs == 1:
            return self._get_single_obs_legacy(0)
        else:
            obs_list = []
            for env_id in range(self.num_envs):
                obs_list.append(self._get_single_obs_legacy(env_id))
            return np.stack(obs_list, axis=0)
    
    def _get_single_obs_legacy(self, env_id: int) -> np.ndarray:
        """获取单个环境的观测 (Legacy NumPy 模式)"""
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
    
    def _get_obs_gpu(self) -> torch.Tensor:
        """获取观测 (GPU-only 零拷贝模式)
        
        使用 Taichi->PyTorch 零拷贝获取粒子状态，
        在 GPU 上进行采样和拼接。
        
        Returns:
            obs: torch.Tensor on CUDA, shape (num_envs, obs_dim) 或 (obs_dim,)
        """
        if self.num_envs == 1:
            return self._get_single_obs_gpu(0)
        else:
            # 批量获取所有环境的观测
            obs_list = []
            for env_id in range(self.num_envs):
                obs_list.append(self._get_single_obs_gpu(env_id))
            
            # 在 GPU 上 stack
            stacked = torch.stack(obs_list, dim=0)
            self._obs_buffer.copy_(stacked)
            return self._obs_buffer
    
    def _get_single_obs_gpu(self, env_id: int) -> torch.Tensor:
        """获取单个环境的观测 (GPU-only 零拷贝模式)
        
        Args:
            env_id: 环境索引
            
        Returns:
            obs: torch.Tensor on CUDA, shape (obs_dim,)
        """
        # 使用零拷贝获取粒子状态
        state = self.taichi_env.simulator.get_state_RL_torch(env_id=env_id, device=self.device)
        obs_parts = []
        
        if 'x' in state:
            x = state['x']  # (n_particles, dim)
            v = state['v']  # (n_particles, dim)
            used = state['used']  # (n_particles,)
            
            for body_id in range(self.taichi_env.particles['bodies']['n']):
                body_n_particles = self.taichi_env.particles['bodies']['n_particles'][body_id]
                body_particle_ids = self.taichi_env.particles['bodies']['particle_ids'][body_id]
                
                # 转换 particle_ids 为 tensor (只需做一次，可以缓存)
                if not hasattr(self, '_body_particle_ids_gpu'):
                    self._body_particle_ids_gpu = {}
                
                if body_id not in self._body_particle_ids_gpu:
                    self._body_particle_ids_gpu[body_id] = torch.tensor(
                        body_particle_ids, dtype=torch.long, device=self.device
                    )
                
                ids = self._body_particle_ids_gpu[body_id]
                step_size = max(1, body_n_particles // self._n_obs_ptcls_per_body)
                
                # GPU 上索引和采样
                body_x = x[ids][::step_size]  # (sampled, dim)
                body_v = v[ids][::step_size]  # (sampled, dim)
                body_used = used[ids][::step_size].float()  # (sampled,)
                
                obs_parts.append(body_x.flatten())
                obs_parts.append(body_v.flatten())
                obs_parts.append(body_used.flatten())
        
        # Agent 状态
        if 'agent' in state:
            agent_state = state['agent']
            if isinstance(agent_state, list):
                for s in agent_state:
                    if isinstance(s, np.ndarray):
                        obs_parts.append(torch.from_numpy(s).to(dtype=DTYPE_TC, device=self.device))
                    elif isinstance(s, torch.Tensor):
                        obs_parts.append(s.to(dtype=DTYPE_TC, device=self.device).flatten())
                    else:
                        obs_parts.append(torch.tensor(s, dtype=DTYPE_TC, device=self.device).flatten())
            elif isinstance(agent_state, torch.Tensor):
                obs_parts.append(agent_state.flatten())
            elif isinstance(agent_state, np.ndarray):
                obs_parts.append(torch.from_numpy(agent_state).to(dtype=DTYPE_TC, device=self.device).flatten())
        
        # 烟雾场 (暂时使用 CPU fallback，因为复杂)
        if hasattr(self.taichi_env, 'smoke_field') and self.taichi_env.smoke_field is not None:
            # TODO: 实现烟雾场的 GPU-only 获取
            s = self.taichi_env.simulator.cur_step_local
            smoke_state = self.taichi_env.smoke_field.get_state(s)
            obs_parts.append(torch.from_numpy(
                smoke_state['v'][::10, 60:68, ::10].flatten()
            ).to(dtype=DTYPE_TC, device=self.device))
            obs_parts.append(torch.from_numpy(
                smoke_state['q'][::10, 60:68, ::10].flatten()
            ).to(dtype=DTYPE_TC, device=self.device))
        
        # 在 GPU 上拼接
        obs = torch.cat(obs_parts)
        return obs
    
    def _get_reward(self) -> Union[float, np.ndarray, torch.Tensor]:
        """获取奖励 (自动选择 GPU 或 legacy 模式)"""
        if self._gpu_mode and self._reward_buffer is not None:
            return self._get_reward_gpu()
        return self._get_reward_legacy()
    
    def _get_reward_legacy(self) -> Union[float, np.ndarray]:
        """获取奖励 (Legacy NumPy 模式)"""
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
    
    def _get_reward_gpu(self) -> torch.Tensor:
        """获取奖励 (GPU-only 模式)"""
        if self.taichi_env.loss is None:
            self._reward_buffer.zero_()
            return self._reward_buffer
        
        loss_info = self.taichi_env.get_step_loss()
        reward = loss_info['reward']
        
        if np.isscalar(reward):
            self._reward_buffer.fill_(reward)
        else:
            self._reward_buffer.copy_(torch.from_numpy(np.asarray(reward, dtype=DTYPE_NP)).to(self.device))
        
        return self._reward_buffer
    
    def step(
        self, 
        action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], 
               Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """执行一步 (支持 GPU-only 或 legacy 模式)
        
        Args:
            action: 动作，shape (num_envs, action_dim) 或 (action_dim,)
                    支持 np.ndarray 或 torch.Tensor
            
        Returns:
            observation: 观测 (torch.Tensor if GPU mode, else np.ndarray)
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        if self._gpu_mode:
            return self._step_gpu(action)
        return self._step_legacy(action)
    
    def _step_legacy(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """执行一步 (Legacy NumPy 模式)"""
        action = np.array(action).clip(self._action_range_np[0], self._action_range_np[1])
        
        # 处理多环境情况
        if self.num_envs > 1 and len(action.shape) == 1:
            action = np.tile(action, (self.num_envs, 1))
        
        self.taichi_env.step(action if self.num_envs == 1 else action[0])
        
        obs = self._get_obs_legacy()
        reward = self._get_reward_legacy()
        
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
    
    def _step_gpu(
        self, 
        action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """执行一步 (GPU-only 模式，零 CPU 往返)
        
        Args:
            action: 动作，支持 torch.Tensor (CUDA) 或 np.ndarray
            
        Returns:
            所有返回值均为 torch.Tensor on CUDA
        """
        # 处理 action: 确保在 GPU 上并 clamp
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action).to(dtype=DTYPE_TC, device=self.device)
        elif isinstance(action, torch.Tensor):
            action_t = action.to(dtype=DTYPE_TC, device=self.device)
        else:
            action_t = torch.tensor(action, dtype=DTYPE_TC, device=self.device)
        
        # GPU 上执行 clamp
        action_t = action_t.clamp(self.action_range[0], self.action_range[1])
        
        # 处理多环境情况
        if self.num_envs > 1 and action_t.dim() == 1:
            action_t = action_t.unsqueeze(0).expand(self.num_envs, -1)
        
        # 转换为 NumPy 传给 taichi_env (目前 taichi_env.step 仍需要 numpy)
        # TODO: 优化 taichi_env.step 接受 torch tensor
        action_np = action_t.cpu().numpy() if action_t.is_cuda else action_t.numpy()
        self.taichi_env.step(action_np if self.num_envs == 1 else action_np[0])
        
        # 获取观测 (GPU)
        obs = self._get_obs_gpu()
        
        # 获取奖励 (GPU)
        reward = self._get_reward_gpu()
        
        # 计算 terminated/truncated (GPU)
        is_done = (self.t == self.horizon)
        self._terminated_buffer.fill_(is_done)
        self._truncated_buffer.zero_()
        
        # 处理 NaN reward
        if self.num_envs == 1:
            if torch.isnan(reward).any():
                self._reward_buffer.fill_(-1000.0)
                self._terminated_buffer.fill_(True)
        else:
            nan_mask = torch.isnan(reward)
            if nan_mask.any():
                self._reward_buffer[nan_mask] = -1000.0
                self._terminated_buffer[nan_mask] = True
        
        info = {}
        
        # 返回适当形状
        if self.num_envs == 1:
            return (
                obs, 
                self._reward_buffer.squeeze(), 
                self._terminated_buffer.squeeze(), 
                self._truncated_buffer.squeeze(), 
                info
            )
        return obs, self._reward_buffer, self._terminated_buffer, self._truncated_buffer, info
    
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
