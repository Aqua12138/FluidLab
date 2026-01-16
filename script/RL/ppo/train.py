#!/usr/bin/env python3
# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent PPO 训练脚本

使用 skrl 库进行 PPO 训练。

用法:
    python script/RL/ppo/train.py --task Pouring-v0 --num_envs 16
    python script/RL/ppo/train.py --task TableMaterials-v0 --num_envs 32 --max_iterations 1000
"""

import argparse
import os
import sys
from datetime import datetime

# 添加 source 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'source'))

import numpy as np
import torch
import gymnasium as gym

from rheo import init_taichi
from rheo.utils.misc import set_headless_mode, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="RheoAgent PPO 训练")
    
    parser.add_argument("--task", type=str, required=True, help="环境名称")
    parser.add_argument("--num_envs", type=int, default=16, help="并行环境数量")
    parser.add_argument("--max_iterations", type=int, default=500, help="最大迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    parser.add_argument("--log_dir", type=str, default="logs/ppo", help="日志目录")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置
    set_random_seed(args.seed)
    set_headless_mode(args.headless or True)  # 训练时默认无头
    
    # PPO 不需要梯度
    enable_grad = False
    init_taichi(num_envs=args.num_envs, enable_grad=enable_grad)
    
    # 导入环境
    import rheo_env  # noqa: F401
    
    print(f"\n{'='*60}")
    print(f"RheoAgent PPO 训练")
    print(f"{'='*60}")
    print(f"任务: {args.task}")
    print(f"环境数: {args.num_envs}")
    print(f"最大迭代: {args.max_iterations}")
    print(f"{'='*60}\n")
    
    # 创建环境
    env = gym.make(
        args.task,
        num_envs=args.num_envs,
        enable_grad=enable_grad,
        seed=args.seed,
        renderer_type=None,  # 训练时不渲染
    )
    
    # 尝试导入 skrl
    try:
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.trainers.torch import SequentialTrainer
        from skrl.resources.preprocessors.torch import RunningStandardScaler
    except ImportError:
        print("错误: 请安装 skrl 库: pip install skrl")
        return
    
    # 包装环境
    env = wrap_env(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义模型
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
    
    # 创建模型
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device),
    }
    
    # 经验回放
    memory = RandomMemory(memory_size=32, num_envs=args.num_envs, device=device)
    
    # PPO 配置
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 32
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 8
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = torch.optim.lr_scheduler.LambdaLR
    cfg["learning_rate_scheduler_kwargs"] = {"lr_lambda": lambda step: 1.0}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"] = 2.0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # 实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{args.task}_{timestamp}"
    cfg["experiment"]["directory"] = args.log_dir
    cfg["experiment"]["experiment_name"] = experiment_name
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 1000
    
    # 创建 Agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    # 加载检查点
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"已加载检查点: {args.checkpoint}")
    
    # 创建训练器
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": args.max_iterations * args.num_envs * 32},
    )
    
    # 开始训练
    print("\n开始训练...\n")
    trainer.train()
    
    print(f"\n训练完成! 日志保存至: {args.log_dir}/{experiment_name}")


if __name__ == "__main__":
    main()

