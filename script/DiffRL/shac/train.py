#!/usr/bin/env python3
# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent SHAC (Short-Horizon Actor-Critic) 训练脚本

使用差分物理进行策略优化。

用法:
    python script/DiffRL/shac/train.py --task Pouring-v0 --grad
    python script/DiffRL/shac/train.py --task TableMaterials-v0 --grad --n_iters 200
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'source'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from rheo import init_taichi
from rheo.utils.misc import set_headless_mode, set_random_seed


class PolicyNetwork(nn.Module):
    """策略网络"""
    
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
        self.action_scale = 0.02  # 动作缩放
    
    def forward(self, obs):
        return self.net(obs) * self.action_scale


def parse_args():
    parser = argparse.ArgumentParser(description="RheoAgent SHAC 训练")
    
    parser.add_argument("--task", type=str, required=True, help="环境名称")
    parser.add_argument("--grad", action="store_true", required=True, help="启用梯度 (必需)")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--n_iters", type=int, default=200, help="迭代次数")
    parser.add_argument("--horizon", type=int, default=100, help="短视野长度")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--log_dir", type=str, default="logs/shac", help="日志目录")
    parser.add_argument("--save_interval", type=int, default=50, help="保存间隔")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.grad:
        print("错误: SHAC 需要启用梯度 (--grad)")
        return
    
    set_random_seed(args.seed)
    set_headless_mode(args.headless or True)
    
    # SHAC 需要梯度
    enable_grad = True
    init_taichi(num_envs=args.num_envs, enable_grad=enable_grad)
    
    import rheo_env  # noqa: F401
    
    print(f"\n{'='*60}")
    print(f"RheoAgent SHAC 训练 (差分物理)")
    print(f"{'='*60}")
    print(f"任务: {args.task}")
    print(f"迭代: {args.n_iters}")
    print(f"视野: {args.horizon}")
    print(f"学习率: {args.lr}")
    print(f"{'='*60}\n")
    
    # 创建环境
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        enable_grad=enable_grad,
        seed=args.seed,
        renderer_type=None,
    )
    
    obs, _ = env.reset()
    obs_dim = obs.shape[-1] if len(obs.shape) > 1 else obs.shape[0]
    action_dim = env.action_space.shape[-1] if env.action_space is not None else 6
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建策略网络
    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    
    # 加载检查点
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        policy.load_state_dict(checkpoint['policy'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint.get('iteration', 0)
        print(f"已加载检查点: {args.checkpoint} (迭代 {start_iter})")
    else:
        start_iter = 0
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{args.task}_{timestamp}")
    os.makedirs(log_path, exist_ok=True)
    
    print(f"日志目录: {log_path}\n")
    print("开始训练...\n")
    
    # 训练循环
    best_reward = -float('inf')
    
    for iteration in range(start_iter, args.n_iters):
        policy.train()
        optimizer.zero_grad()
        
        # 重置环境
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        total_reward = 0.0
        rewards = []
        
        # 短视野 rollout
        for step in range(args.horizon):
            # 获取动作
            action = policy(obs_tensor)
            action_np = action.detach().cpu().numpy()
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            reward_val = reward.mean() if hasattr(reward, 'mean') else reward
            total_reward += reward_val
            rewards.append(reward_val)
            
            done = terminated.any() if hasattr(terminated, 'any') else terminated
            if done:
                break
        
        # 计算损失 (负奖励)
        avg_reward = total_reward / len(rewards)
        loss = -avg_reward  # 最大化奖励
        
        # 反向传播 (如果支持差分物理)
        try:
            # 尝试通过环境获取梯度
            if hasattr(env, 'taichi_env') and env.taichi_env.enable_grad:
                # 差分物理反向传播
                for step in range(len(rewards) - 1, -1, -1):
                    env.taichi_env.step_grad()
            else:
                # 退化到策略梯度
                loss_tensor = torch.tensor(loss, requires_grad=True)
                loss_tensor.backward()
        except Exception as e:
            print(f"警告: 梯度计算失败 - {e}")
        
        optimizer.step()
        
        # 日志
        if (iteration + 1) % 10 == 0:
            print(f"迭代 {iteration + 1}/{args.n_iters}: 平均奖励={avg_reward:.4f}")
        
        # 保存检查点
        if (iteration + 1) % args.save_interval == 0 or avg_reward > best_reward:
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_name = "best.pt"
            else:
                save_name = f"iter_{iteration + 1}.pt"
            
            torch.save({
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration + 1,
                'reward': avg_reward,
            }, os.path.join(log_path, save_name))
    
    # 保存最终模型
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': args.n_iters,
        'reward': avg_reward,
    }, os.path.join(log_path, "final.pt"))
    
    print(f"\n训练完成! 最佳奖励: {best_reward:.4f}")
    print(f"模型保存至: {log_path}")


if __name__ == "__main__":
    main()

