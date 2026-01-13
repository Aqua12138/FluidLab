#!/usr/bin/env python3
"""
Shake test script: 
1. Stay still for a period
2. Move right (action_v[0] += linear_v_mag) for a period
3. Stop
4. Record particle count difference (left - right) around agent center
5. Plot time series
"""

import os
import sys
import gym
import argparse
import numpy as np
# Set matplotlib backend before importing pyplot to avoid Qt errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
from datetime import datetime

import fluidlab.envs
from fluidlab.utils.config import load_config
from fluidlab.run import make_env_with_renderer
from fluidlab.utils.misc import is_on_server

class ShakePolicy:
    """Simple policy for shake test"""
    def __init__(self, init_p, linear_v_mag=0.003):
        self.actions_p = init_p
        self.linear_v_mag = linear_v_mag
        self.current_action_v = None
    
    def get_actions_p(self):
        return self.actions_p
    
    def get_action_v(self, i):
        if self.current_action_v is None:
            return np.zeros(6) if len(self.actions_p) >= 6 else np.zeros(3)
        return self.current_action_v
    
    def set_action(self, action_v):
        """Set the action velocity"""
        self.current_action_v = action_v

def get_agent_center(taichi_env):
    """Get the center position of the agent"""
    if taichi_env.agent is None or len(taichi_env.agent.effectors) == 0:
        return None
    
    effector = taichi_env.agent.effectors[0]
    if hasattr(effector, 'latest_pos'):
        center = effector.latest_pos.to_numpy()[0]
        return center[:3]  # Only x, y, z
    elif hasattr(effector, 'init_pos'):
        return np.array(effector.init_pos[:3])
    return None

def count_particles_left_right(taichi_env, agent_center, threshold=0.05):
    """
    Count particles on left and right sides of agent center.
    Left: x < agent_x - threshold
    Right: x > agent_x + threshold
    """
    if not taichi_env.has_particles:
        return 0, 0
    
    # Get particle positions
    state = taichi_env.get_state_RL()
    if 'x' not in state or 'used' not in state:
        return 0, 0
    
    x = state['x']  # Shape: (n_particles, dim)
    used = state['used']  # Shape: (n_particles,)
    
    # Filter used particles
    used_mask = used.astype(bool)
    if not np.any(used_mask):
        return 0, 0
    
    x_used = x[used_mask]
    
    # Get agent x position
    agent_x = agent_center[0]
    
    # Count particles on left and right
    left_mask = x_used[:, 0] < (agent_x - threshold)
    right_mask = x_used[:, 0] > (agent_x + threshold)
    
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    
    return n_left, n_right

def shake_test(env, cfg=None, 
               rest_steps=100, 
               move_steps=50, 
               observe_steps=200,
               linear_v_mag=0.003,
               threshold=0.05,
               save_dir='shake_results'):
    """
    Perform shake test and record particle count difference.
    
    Args:
        env: Gym environment
        cfg: Config object (optional)
        rest_steps: Number of steps to rest initially
        move_steps: Number of steps to move right
        observe_steps: Number of steps to observe after stopping
        linear_v_mag: Linear velocity magnitude for movement
        threshold: Distance threshold for left/right particle counting
        save_dir: Directory to save results
    """
    taichi_env = env.taichi_env
    
    if taichi_env.agent is None:
        print("Error: Environment has no agent to control.")
        return
    
    # Get initial position from config or agent
    init_p = None
    if cfg is not None and hasattr(cfg, 'SOLVER') and hasattr(cfg.SOLVER, 'init_range') and hasattr(cfg.SOLVER.init_range, 'p'):
        init_range_p = cfg.SOLVER.init_range.p
        if init_range_p and len(init_range_p) > 0:
            init_p = np.array(init_range_p[0])
    
    if init_p is None:
        if hasattr(taichi_env.agent, 'effectors') and len(taichi_env.agent.effectors) > 0:
            effector = taichi_env.agent.effectors[0]
            if hasattr(effector, 'init_pos'):
                init_p = effector.init_pos
    
    if init_p is None:
        action_dim = taichi_env.agent.action_dim
        if action_dim >= 6:
            init_p = np.array([0.5, 0.2, 0.5, 0.0, 0.0, 0.0])
        elif action_dim >= 3:
            init_p = np.array([0.5, 0.2, 0.5])
        else:
            init_p = np.zeros(action_dim)
    
    # Create policy
    policy = ShakePolicy(init_p, linear_v_mag)
    
    # Initialize environment
    taichi_env_state = taichi_env.get_state()
    taichi_env.set_state(**taichi_env_state)
    action_p = policy.get_actions_p()
    if action_p is not None:
        taichi_env.apply_agent_action_p(action_p)
    
    # Data recording
    timesteps = []
    n_left_list = []
    n_right_list = []
    diff_list = []
    agent_x_list = []
    
    total_steps = rest_steps + move_steps + observe_steps
    current_step = 0
    
    print(f"\n{'='*60}")
    print("Shake Test Started")
    print(f"{'='*60}")
    print(f"Rest steps: {rest_steps}")
    print(f"Move steps: {move_steps}")
    print(f"Observe steps: {observe_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Linear velocity: {linear_v_mag}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    try:
        # Phase 1: Rest
        print("Phase 1: Resting...")
        policy.set_action(None)  # No movement
        for step in range(rest_steps):
            action = policy.get_action_v(0)
            taichi_env.step(action)
            
            # Record data
            agent_center = get_agent_center(taichi_env)
            if agent_center is not None:
                n_left, n_right = count_particles_left_right(taichi_env, agent_center, threshold)
                diff = n_left - n_right
                
                timesteps.append(current_step)
                n_left_list.append(n_left)
                n_right_list.append(n_right)
                diff_list.append(diff)
                agent_x_list.append(agent_center[0])
            
            current_step += 1
            
            if not is_on_server() and step % 10 == 0:
                taichi_env.render('human')
        
        # Phase 2: Move right
        print("Phase 2: Moving right...")
        action_dim = taichi_env.agent.action_dim
        if action_dim >= 6:
            action_v = np.array([linear_v_mag, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif action_dim >= 3:
            action_v = np.array([linear_v_mag, 0.0, 0.0])
        else:
            action_v = np.array([linear_v_mag])
        
        policy.set_action(action_v)
        
        for step in range(move_steps):
            action = policy.get_action_v(0)
            taichi_env.step(action)
            
            # Record data
            agent_center = get_agent_center(taichi_env)
            if agent_center is not None:
                n_left, n_right = count_particles_left_right(taichi_env, agent_center, threshold)
                diff = n_left - n_right
                
                timesteps.append(current_step)
                n_left_list.append(n_left)
                n_right_list.append(n_right)
                diff_list.append(diff)
                agent_x_list.append(agent_center[0])
            
            current_step += 1
            
            if not is_on_server() and step % 5 == 0:
                taichi_env.render('human')
        
        # Phase 3: Stop and observe
        print("Phase 3: Stopped, observing...")
        policy.set_action(None)  # Stop movement
        
        for step in range(observe_steps):
            action = policy.get_action_v(0)
            taichi_env.step(action)
            
            # Record data
            agent_center = get_agent_center(taichi_env)
            if agent_center is not None:
                n_left, n_right = count_particles_left_right(taichi_env, agent_center, threshold)
                diff = n_left - n_right
                
                timesteps.append(current_step)
                n_left_list.append(n_left)
                n_right_list.append(n_right)
                diff_list.append(diff)
                agent_x_list.append(agent_center[0])
            
            current_step += 1
            
            if not is_on_server() and step % 10 == 0:
                taichi_env.render('human')
        
        print("\nShake test completed!")
        
        # Save and plot results
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert timesteps to real time (seconds)
        # 1 simulation step = 2e-3 seconds = 0.002 seconds
        DT_PER_STEP = 2e-3  # seconds per step
        time_seconds = np.array(timesteps) * DT_PER_STEP
        
        # Convert phase boundaries to time
        rest_time = rest_steps * DT_PER_STEP
        move_end_time = (rest_steps + move_steps) * DT_PER_STEP
        
        # Save data
        data = {
            'timesteps': np.array(timesteps),
            'time_seconds': time_seconds,  # Add time in seconds
            'n_left': np.array(n_left_list),
            'n_right': np.array(n_right_list),
            'diff': np.array(diff_list),
            'agent_x': np.array(agent_x_list),
            'rest_steps': rest_steps,
            'move_steps': move_steps,
            'observe_steps': observe_steps,
            'rest_time': rest_time,  # Add time equivalents
            'move_time': move_steps * DT_PER_STEP,
            'observe_time': observe_steps * DT_PER_STEP,
            'linear_v_mag': linear_v_mag,
            'threshold': threshold,
        }
        
        data_file = os.path.join(save_dir, f'shake_data_{timestamp}.npz')
        np.savez(data_file, **data)
        print(f"Data saved to: {data_file}")
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Particle count difference (left - right)
        axes[0, 0].plot(time_seconds, diff_list, 'b-', linewidth=2, label='Left - Right')
        axes[0, 0].axvline(x=rest_time, color='g', linestyle='--', alpha=0.7, label='Start moving')
        axes[0, 0].axvline(x=move_end_time, color='r', linestyle='--', alpha=0.7, label='Stop moving')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Particle Count Difference')
        axes[0, 0].set_title('Particle Count Difference (Left - Right)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Left and right particle counts
        axes[0, 1].plot(time_seconds, n_left_list, 'r-', linewidth=2, label='Left particles')
        axes[0, 1].plot(time_seconds, n_right_list, 'b-', linewidth=2, label='Right particles')
        axes[0, 1].axvline(x=rest_time, color='g', linestyle='--', alpha=0.7, label='Start moving')
        axes[0, 1].axvline(x=move_end_time, color='r', linestyle='--', alpha=0.7, label='Stop moving')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Particle Count')
        axes[0, 1].set_title('Left vs Right Particle Counts')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Agent X position
        axes[1, 0].plot(time_seconds, agent_x_list, 'g-', linewidth=2, label='Agent X position')
        axes[1, 0].axvline(x=rest_time, color='g', linestyle='--', alpha=0.7, label='Start moving')
        axes[1, 0].axvline(x=move_end_time, color='r', linestyle='--', alpha=0.7, label='Stop moving')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('X Position')
        axes[1, 0].set_title('Agent X Position Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Combined view
        ax2 = axes[1, 1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(time_seconds, diff_list, 'b-', linewidth=2, label='Particle diff (L-R)')
        line2 = ax2_twin.plot(time_seconds, agent_x_list, 'g-', linewidth=2, label='Agent X')
        
        ax2.axvline(x=rest_time, color='g', linestyle='--', alpha=0.7)
        ax2.axvline(x=move_end_time, color='r', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Particle Count Difference', color='b')
        ax2_twin.set_ylabel('Agent X Position', color='g')
        ax2.set_title('Combined View: Particle Diff & Agent Position')
        ax2.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        plot_file = os.path.join(save_dir, f'shake_plot_{timestamp}.png')
        
        # Use non-interactive backend to avoid Qt errors
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except:
            pass
        
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        
        # Try to show plot, but handle errors gracefully
        try:
            if not is_on_server() and os.getenv('DISPLAY') is not None:
                # Only try to show if we have a display
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Note: Could not display plot (this is OK): {e}")
            plt.close()
        
        print(f"\nResults saved in: {save_dir}")
        
    except KeyboardInterrupt:
        print("\nShake test interrupted.")
    except Exception as e:
        print(f"\nError during shake test: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Shake test script')
    parser.add_argument("--cfg_file", type=str, default=None, help='Config file path')
    parser.add_argument("--env_name", type=str, default=None, help='Environment name')
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    parser.add_argument("--renderer_type", type=str, default='GGUI', help='Renderer type')
    parser.add_argument("--rest_steps", type=int, default=100, help='Steps to rest initially')
    parser.add_argument("--move_steps", type=int, default=50, help='Steps to move right')
    parser.add_argument("--observe_steps", type=int, default=200, help='Steps to observe after stopping')
    parser.add_argument("--linear_v_mag", type=float, default=0.003, help='Linear velocity magnitude')
    parser.add_argument("--threshold", type=float, default=0.05, help='Distance threshold for left/right counting')
    parser.add_argument("--save_dir", type=str, default='shake_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load config
    cfg = None
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
        env_name = cfg.EXP.env_name
        seed = cfg.EXP.seed
    elif args.env_name is not None:
        env_name = args.env_name
        seed = args.seed
    else:
        print("Error: Either --cfg_file or --env_name must be provided")
        return
    
    # Create environment
    env = make_env_with_renderer(env_name, seed, False, 'diff', args.renderer_type)
    env.reset()
    
    # Run shake test
    shake_test(
        env, 
        cfg=cfg,
        rest_steps=args.rest_steps,
        move_steps=args.move_steps,
        observe_steps=args.observe_steps,
        linear_v_mag=args.linear_v_mag,
        threshold=args.threshold,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()

