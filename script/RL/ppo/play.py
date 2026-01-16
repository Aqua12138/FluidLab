#!/usr/bin/env python3
# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent PPO 推理脚本

加载训练好的 PPO 模型进行推理。

用法:
    python script/RL/ppo/play.py --task Pouring-v0 --checkpoint logs/ppo/model.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'source'))

import numpy as np
import torch
import gymnasium as gym

from rheo import init_taichi
from rheo.utils.misc import set_headless_mode, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="RheoAgent PPO 推理")
    
    parser.add_argument("--task", type=str, required=True, help="环境名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--num_episodes", type=int, default=10, help="运行回合数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--renderer_type", type=str, default="GGUI", help="渲染器类型")
    parser.add_argument("--record", action="store_true", help="录制视频")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_random_seed(args.seed)
    set_headless_mode(args.headless)
    
    enable_grad = False
    init_taichi(num_envs=args.num_envs, enable_grad=enable_grad)
    
    import rheo_env  # noqa: F401
    
    print(f"\n{'='*60}")
    print(f"RheoAgent PPO 推理")
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
    
    try:
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    except ImportError:
        print("错误: 请安装 skrl 库")
        return
    
    env = wrap_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义模型 (与训练时相同)
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False,
                     clip_log_std=True, min_log_std=-20.0, max_log_std=2.0):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
            
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.num_observations, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, self.num_actions),
            )
            self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))
        
        def compute(self, inputs, role):
            return self.net(inputs["states"]), self.log_std_parameter, {}
    
    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)
            
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.num_observations, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 1),
            )
        
        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}
    
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device),
    }
    
    # 创建 Agent 并加载检查点
    cfg = PPO_DEFAULT_CONFIG.copy()
    agent = PPO(
        models=models,
        memory=None,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")
    print(f"已加载检查点: {args.checkpoint}")
    
    # 运行推理
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            with torch.no_grad():
                action = agent.act(obs, timestep=step, timesteps=0)[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward.mean().item() if hasattr(reward, 'mean') else reward
            step += 1
            
            if not args.headless:
                env.render()
            
            done = terminated.any() if hasattr(terminated, 'any') else terminated
            if done or truncated.any() if hasattr(truncated, 'any') else truncated:
                break
        
        print(f"回合 {episode + 1}: 步数={step}, 奖励={total_reward:.2f}")
    
    env.close()
    print("\n推理完成!")


if __name__ == "__main__":
    main()

