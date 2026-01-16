#!/usr/bin/env python3
# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent 键盘控制脚本

使用 pynput 全局键盘监听实现流畅的键盘控制。
支持长按检测，控制更加流畅。

用法:
    python script/keyboard.py --task Pouring-v0 --num_envs 1
    python script/keyboard.py --task TableMaterials-v0 --material HONEY
    
控制说明:
    W/S: X轴移动 (前/后)
    A/D: Z轴移动 (左/右)
    Q/E: Y轴移动 (上/下)
    I/K: Roll 旋转
    J/L: Pitch 旋转
    U/O: Yaw 旋转
    R:   重置环境
    ESC: 退出
"""

import argparse
import sys
import os
import numpy as np
import gymnasium as gym
import cv2
import time

# 添加 source 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from rheo import init_taichi
from rheo.utils.misc import set_headless_mode, set_random_seed
from rheo.utils.keyboard import KeyboardPolicy_6DOF, PYNPUT_AVAILABLE


def parse_args():
    parser = argparse.ArgumentParser(description="RheoAgent 键盘控制")
    
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        help="环境名称 (如 Pouring-v0, TableMaterials-v0)"
    )
    parser.add_argument(
        "--num_envs", 
        type=int, 
        default=1,
        help="并行环境数量 (默认: 1)"
    )
    parser.add_argument(
        "--grad", 
        action="store_true",
        help="启用梯度计算 (默认: 无梯度)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式运行 (无渲染)"
    )
    parser.add_argument(
        "--material",
        type=str,
        default=None,
        help="流体材料类型 (如 WATER, HONEY, TOOTHPASTE)"
    )
    parser.add_argument(
        "--v_lin",
        type=float,
        default=0.01,
        help="线速度幅值 (默认: 0.01)"
    )
    parser.add_argument(
        "--v_ang",
        type=float,
        default=0.01,
        help="角速度幅值 (默认: 0.01)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查 pynput 是否可用
    if not PYNPUT_AVAILABLE:
        print("错误: pynput 库不可用。请安装: pip install pynput")
        return
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置无头模式
    set_headless_mode(args.headless)
    
    # 初始化 Taichi
    enable_grad = args.grad
    init_taichi(num_envs=args.num_envs, enable_grad=enable_grad)
    
    # 导入环境注册
    import rheo_env  # noqa: F401
    
    print(f"\n{'='*60}")
    print(f"RheoAgent 键盘控制 (pynput 全局监听)")
    print(f"{'='*60}")
    print(f"任务: {args.task}")
    print(f"环境数: {args.num_envs}")
    print(f"梯度: {'启用' if enable_grad else '禁用'}")
    print(f"{'='*60}\n")
    
    # 创建环境
    env_kwargs = {
        "num_envs": args.num_envs,
        "enable_grad": enable_grad,
        "seed": args.seed,
    }
    
    if args.material:
        env_kwargs["material"] = args.material
    
    env = gym.make(args.task, **env_kwargs)
    
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 初始化键盘控制器
    action_dim = env.action_space.shape[-1] if env.action_space is not None else 6
    policy = KeyboardPolicy_6DOF(v_lin=args.v_lin, v_ang=args.v_ang)
    
    # 重置环境
    obs, info = env.reset()
    
    print("\n" + "="*60)
    print("键盘控制说明 (全局监听，支持长按)")
    print("="*60)
    print("  W/S:  X轴移动 (前/后)")
    print("  A/D:  Z轴移动 (左/右)")
    print("  Q/E:  Y轴移动 (上/下)")
    print("  I/K:  Roll 旋转")
    print("  J/L:  Pitch 旋转")
    print("  U/O:  Yaw 旋转")
    print("  R:    重置环境")
    print("  ESC:  退出")
    print("="*60)
    print("\n提示: 使用 pynput 全局监听，按键响应更流畅！")
    print("      请确保焦点在终端或渲染窗口上。")
    print("-"*60 + "\n")
    
    # 主循环
    step = 0
    total_reward = 0
    
    # FPS 计时器
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 1.0  # 每秒更新一次 FPS 显示
    current_fps = 0.0
    
    try:
        while policy.is_running():
            loop_start = time.time()
            
            # 检查重置键 (R)
            if 'r' in policy.keys_activated:
                print("\n重置环境...")
                obs, info = env.reset()
                policy.keys_activated.discard('r')
                step = 0
                total_reward = 0
                continue
            
            # 获取动作
            action_v = policy.get_action_v()
            
            # 截断到实际动作维度
            if len(action_v) > action_dim:
                action = action_v[:action_dim]
            else:
                action = np.zeros(action_dim, dtype=np.float32)
                action[:len(action_v)] = action_v
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染（headless 模式跳过）
            if not args.headless:
                env.render()
                # cv2 窗口更新
                cv2.waitKey(1)
            
            # 统计
            step += 1
            total_reward += reward if np.isscalar(reward) else reward.mean()
            
            # FPS 计算
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= fps_display_interval:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()
                # 终端显示 FPS（覆盖同一行）
                print(f"\rStep: {step:6d} | FPS: {current_fps:6.1f} | Reward: {total_reward:8.2f}", end="", flush=True)
            
            # 检查终止
            done = terminated if np.isscalar(terminated) else terminated.any()
            if done:
                print(f"\n回合结束! 步数: {step}, 总奖励: {total_reward:.2f}, FPS: {current_fps:.1f}")
                obs, info = env.reset()
                step = 0
                total_reward = 0
    
    except KeyboardInterrupt:
        print("\n\n用户中断 (Ctrl+C)")
    
    finally:
        policy.stop()
        cv2.destroyAllWindows()
        env.close()
        print("\n环境已关闭")


if __name__ == "__main__":
    main()
