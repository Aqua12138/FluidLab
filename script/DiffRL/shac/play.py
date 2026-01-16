#!/usr/bin/env python3
# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent SHAC 推理脚本

加载训练好的 SHAC 模型进行推理。

用法:
    python script/DiffRL/shac/play.py --task Pouring-v0 --checkpoint logs/shac/best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'source'))

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from rheo import init_taichi
from rheo.utils.misc import set_headless_mode, set_random_seed


class PolicyNetwork(nn.Module):
    """策略网络 (与训练时相同)"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
        self.action_scale = 0.02
    
    def forward(self, obs):
        return self.net(obs) * self.action_scale


def parse_args():
    parser = argparse.ArgumentParser(description="RheoAgent SHAC 推理")
    
    parser.add_argument("--task", type=str, required=True, help="环境名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--num_episodes", type=int, default=10, help="运行回合数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--renderer_type", type=str, default="GGUI", help="渲染器类型")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_random_seed(args.seed)
    set_headless_mode(args.headless)
    
    # 推理不需要梯度
    enable_grad = False
    init_taichi(num_envs=args.num_envs, enable_grad=enable_grad)
    
    import rheo_env  # noqa: F401
    
    print(f"\n{'='*60}")
    print(f"RheoAgent SHAC 推理")
    print(f"{'='*60}")
    print(f"任务: {args.task}")
    print(f"检查点: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # 创建环境
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        enable_grad=enable_grad,
        seed=args.seed,
        renderer_type=args.renderer_type if not args.headless else None,
    )
    
    obs, _ = env.reset()
    obs_dim = obs.shape[-1] if len(obs.shape) > 1 else obs.shape[0]
    action_dim = env.action_space.shape[-1] if env.action_space is not None else 6
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建并加载策略
    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    checkpoint = torch.load(args.checkpoint)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    print(f"已加载检查点: {args.checkpoint}")
    if 'reward' in checkpoint:
        print(f"训练时奖励: {checkpoint['reward']:.4f}")
    print()
    
    # 运行推理
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        total_reward = 0
        step = 0
        
        while True:
            with torch.no_grad():
                action = policy(obs_tensor)
            action_np = action.cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            total_reward += reward.mean() if hasattr(reward, 'mean') else reward
            step += 1
            
            if not args.headless:
                env.render()
            
            done = terminated.any() if hasattr(terminated, 'any') else terminated
            if done:
                break
        
        print(f"回合 {episode + 1}: 步数={step}, 奖励={total_reward:.2f}")
    
    env.close()
    print("\n推理完成!")


if __name__ == "__main__":
    main()

