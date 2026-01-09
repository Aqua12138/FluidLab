import os
import gym
import torch
import random
import argparse
import numpy as np

import fluidlab.envs
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import solve_policy
from fluidlab.optimizer.recorder import record_target, replay_policy, replay_target
from fluidlab.utils.config import load_config
from fluidlab.optimizer.policies import KeyboardPolicy_vxy_wz, KeyboardPolicy_vxy, KeyboardPolicy_wz
from fluidlab.utils.misc import is_on_server

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--env_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--user_input", action='store_true')
    parser.add_argument("--replay_policy", action='store_true')
    parser.add_argument("--replay_target", action='store_true')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--renderer_type", type=str, default='GGUI')
    parser.add_argument("--keyboard", action='store_true', help='Manually control agent with keyboard, no recording')


    args = parser.parse_args()

    return args

def make_env_with_renderer(env_name, seed, loss, loss_type, renderer_type):
    """
    Create environment, only passing renderer_type if the environment supports it.
    """
    # Try with renderer_type first
    try:
        return gym.make(env_name, seed=seed, loss=loss, loss_type=loss_type, renderer_type=renderer_type)
    except TypeError:
        # If renderer_type is not supported, try without it
        return gym.make(env_name, seed=seed, loss=loss, loss_type=loss_type)

def keyboard_control(env, cfg=None):
    """
    Manually control agent with keyboard, no recording.
    """
    taichi_env = env.taichi_env
    
    # Get initial position from agent
    if taichi_env.agent is None:
        print("Error: Environment has no agent to control.")
        return
    
    # Try to use environment's demo_policy if it exists and returns a KeyboardPolicy
    # If no keyboard policy from demo_policy, create one

    # Try to get initial position from config file first
    init_p = None
    if cfg is not None and hasattr(cfg, 'SOLVER') and hasattr(cfg.SOLVER, 'init_range') and hasattr(cfg.SOLVER.init_range, 'p'):
        init_range_p = cfg.SOLVER.init_range.p
        if init_range_p and len(init_range_p) > 0:
            # init_range.p is a tuple of (min, max), we use the first one (min)
            # Usually min and max are the same, so we can use either
            init_p = np.array(init_range_p[0])
    
    # Fallback: try to get initial position from agent's effector
    if init_p is None:
        if hasattr(taichi_env.agent, 'effectors') and len(taichi_env.agent.effectors) > 0:
            effector = taichi_env.agent.effectors[0]
            if hasattr(effector, 'init_pos'):
                init_p = effector.init_pos

    # Fallback: use default position based on action dimension
    if init_p is None:
        action_dim = taichi_env.agent.action_dim
        if action_dim >= 6:
            init_p = np.array([0.5, 0.2, 0.5, 0.0, 0.0, 0.0])
        elif action_dim >= 3:
            init_p = np.array([0.5, 0.2, 0.5])
        else:
            init_p = np.zeros(action_dim)

    # Select keyboard policy based on action dimension
    if taichi_env.agent.action_dim >= 6:
        policy = KeyboardPolicy_vxy_wz(init_p, v_lin=0.003, v_ang=0.003)
        print("\n" + "="*60)
        print("智能体控制 (Agent Control):")
        print("  数字键 2/4/6/8: 控制 vxy 方向移动")
        print("  z/x 键: 控制 wz 旋转")
        print("  q 键: 退出程序")
        print("\n视角控制 (Camera Control):")
        print("  ↑/↓ 箭头键: 前后移动视角")
        print("  u/i 键: 上下移动视角")
        print("  鼠标右键 + 拖拽: 旋转视角")
        print("="*60)
    elif taichi_env.agent.action_dim >= 3:
        policy = KeyboardPolicy_vxy(init_p, v_lin=0.003)
        print("\n" + "="*60)
        print("智能体控制 (Agent Control):")
        print("  数字键 2/4/6/8: 控制 vxy 方向移动")
        print("  q 键: 退出程序")
        print("\n视角控制 (Camera Control):")
        print("  ↑/↓ 箭头键: 前后移动视角")
        print("  u/i 键: 上下移动视角")
        print("  鼠标右键 + 拖拽: 旋转视角")
        print("="*60)
    else:
        policy = KeyboardPolicy_wz(init_p, v_ang=0.003)
        print("\n" + "="*60)
        print("智能体控制 (Agent Control):")
        print("  z/x 键: 控制 wz 旋转")
        print("  q 键: 退出程序")
        print("\n视角控制 (Camera Control):")
        print("  ↑/↓ 箭头键: 前后移动视角")
        print("  u/i 键: 上下移动视角")
        print("  鼠标右键 + 拖拽: 旋转视角")
        print("="*60)
    
    print("\n提示: 详细控制说明请查看 KEYBOARD_CONTROLS.md")
    
    # Initialize environment
    taichi_env_state = taichi_env.get_state()
    taichi_env.set_state(**taichi_env_state)
    action_p = policy.get_actions_p()
    if action_p is not None:
        taichi_env.apply_agent_action_p(action_p)
    
    # Main control loop
    try:
        while True:
            # Get keyboard action
            action = policy.get_action_v(0)
            
            # Step environment
            taichi_env.step(action)
            
            # Render
            if not is_on_server():
                taichi_env.render('human')
            
            # Check for quit
            if hasattr(policy, 'keys_activated') and 'q' in policy.keys_activated:
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboard control stopped.")
    finally:
        # Clean up keyboard listener
        if hasattr(policy, 'listener'):
            policy.listener.stop()

def main():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    else:
        cfg = None

    if args.keyboard:
        if cfg is not None:
            env = make_env_with_renderer(cfg.EXP.env_name, cfg.EXP.seed, False, 'diff', args.renderer_type)
        else:
            if args.env_name:
                env = make_env_with_renderer(args.env_name, args.seed, False, 'diff', args.renderer_type)
            else:
                print("Error: --keyboard requires either --cfg_file or --env_name")
                return
        env.reset()
        keyboard_control(env, cfg)
        return

    if args.record:
        if cfg is not None:
            env = make_env_with_renderer(cfg.EXP.env_name, cfg.EXP.seed, False, 'diff', args.renderer_type)
        else:
            env = make_env_with_renderer(args.env_name, args.seed, False, 'diff', args.renderer_type)
        record_target(env, path=args.path, user_input=args.user_input)
    elif args.replay_target:
        if cfg is not None:
            env = make_env_with_renderer(cfg.EXP.env_name, cfg.EXP.seed, False, 'diff', args.renderer_type)
        else:
            env = make_env_with_renderer(args.env_name, args.seed, False, 'diff', args.renderer_type)
        replay_target(env)
    elif args.replay_policy:
        if cfg is not None:
            env = make_env_with_renderer(cfg.EXP.env_name, cfg.EXP.seed, False, 'diff', args.renderer_type)
        else:
            env = make_env_with_renderer(args.env_name, args.seed, False, 'diff', args.renderer_type)
        replay_policy(env, path=args.path)
    else:
        logger = Logger(args.exp_name)
        env = make_env_with_renderer(cfg.EXP.env_name, cfg.EXP.seed, True, 'diff', args.renderer_type)
        solve_policy(env, logger, cfg.SOLVER)

if __name__ == '__main__':
    main()
