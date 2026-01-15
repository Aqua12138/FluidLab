import os
import gym
import numpy as np
from gym.spaces import Box, Dict
from fluidlab.configs.macros import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
import fluidlab.utils.misc as misc_utils

class BatchFluidEnv(gym.Env):
    '''
    Batch environment wrapper for FluidLab environments.
    Provides GPU batch parallelization similar to Isaac Lab.
    
    This wrapper creates a single TaichiEnv with batch_size > 1,
    allowing GPU-parallel execution of multiple environments.
    '''
    def __init__(self, base_env_class, batch_size, version=0, loss=True, loss_type='diff', seed=None, renderer_type='GGUI', **env_kwargs):
        """
        Args:
            base_env_class: The base environment class (e.g., NewPouringEnv)
            batch_size: Number of parallel environments
            version: Environment version
            loss: Whether to use loss function
            loss_type: Type of loss function
            seed: Random seed
            renderer_type: Renderer type ('GGUI', 'GL', or None)
            **env_kwargs: Additional keyword arguments for the base environment
        """
        if seed is not None:
            self.seed(seed)
        
        self.batch_size = batch_size
        self.base_env_class = base_env_class
        self.version = version
        self.loss = loss
        self.loss_type = loss_type
        self.renderer_type = renderer_type
        self.env_kwargs = env_kwargs
        
        # Create base environment instance to get configuration
        # We'll modify it to use batch_size
        self.base_env = base_env_class(
            version=version,
            loss=loss,
            loss_type=loss_type,
            seed=seed,
            renderer_type=None,  # Don't setup renderer yet
            **env_kwargs
        )
        
        # Replace TaichiEnv with batch-enabled version
        self._replace_taichi_env_with_batch()
        
        # Setup renderer if needed
        if not misc_utils.is_on_server() and self.renderer_type is not None:
            self.base_env.setup_renderer()
        
        # Copy references
        self.taichi_env = self.base_env.taichi_env
        self.horizon = self.base_env.horizon
        self.horizon_action = self.base_env.horizon_action
        self.target_file = self.base_env.target_file
        self._n_obs_ptcls_per_body = self.base_env._n_obs_ptcls_per_body
        self.action_range = self.base_env.action_range
        
        self.gym_misc()
        
        print(f'===>  BatchFluidEnv (batch_size={self.batch_size}) built successfully.')
    
    def _replace_taichi_env_with_batch(self):
        """Replace the base env's TaichiEnv with a batch-enabled version"""
        old_taichi_env = self.base_env.taichi_env
        
        # Create new TaichiEnv with batch_size
        new_taichi_env = TaichiEnv(
            dim=old_taichi_env.dim,
            quality=1,  # Default quality, can be extracted from old_taichi_env if needed
            particle_density=old_taichi_env.particle_density,
            max_substeps_local=old_taichi_env.max_substeps_local,
            max_substeps_global=old_taichi_env.max_substeps_global,
            horizon=old_taichi_env.horizon,
            ckpt_dest=old_taichi_env.ckpt_dest,
            gravity=old_taichi_env.simulator.gravity.to_numpy().tolist(),
            batch_size=self.batch_size,
        )
        
        # Copy all setup from old to new
        new_taichi_env.agent = old_taichi_env.agent
        new_taichi_env.statics = old_taichi_env.statics
        new_taichi_env.particle_bodies = old_taichi_env.particle_bodies
        new_taichi_env.smoke_field = old_taichi_env.smoke_field
        new_taichi_env.loss = old_taichi_env.loss
        
        # Update agent's batch_size if it exists
        if new_taichi_env.agent is not None:
            new_taichi_env.agent.batch_size = self.batch_size
            for effector in new_taichi_env.agent.effectors:
                effector.batch_size = self.batch_size
        
        # Rebuild with batch support
        new_taichi_env.build()
        
        # Replace
        self.base_env.taichi_env = new_taichi_env
    
    def seed(self, seed):
        misc_utils.set_random_seed(seed)
    
    def gym_misc(self):
        """Setup Gym spaces"""
        # Get observation and action spaces from a single batch
        single_state = self.taichi_env.get_state_RL(batch_id=0)
        single_obs = self._get_obs_from_state(single_state)
        
        # Observation space: Dict with batch dimension
        self.observation_space = Dict({
            'batch_obs': Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.batch_size, len(single_obs)),
                dtype=np.float32
            )
        })
        
        # Action space: batch dimension
        if self.taichi_env.agent is not None:
            action_dim = self.taichi_env.agent.action_dim
            self.action_space = Box(
                low=self.action_range[0],
                high=self.action_range[1],
                shape=(self.batch_size, action_dim),
                dtype=np.float32
            )
        else:
            self.action_space = Box(
                low=-1.0,
                high=1.0,
                shape=(self.batch_size, 1),
                dtype=np.float32
            )
    
    def _get_obs_from_state(self, state):
        """Convert state to observation vector (matching base env's _get_obs)"""
        obs = []
        
        if 'x' in state:
            # Get particles info from taichi_env
            particles = self.taichi_env.particles
            if particles is not None and 'bodies' in particles:
                for body_id in range(particles['bodies']['n']):
                    body_n_particles = particles['bodies']['n_particles'][body_id]
                    body_particle_ids = particles['bodies']['particle_ids'][body_id]
                    
                    step_size = max(1, body_n_particles // self._n_obs_ptcls_per_body)
                    if 'x' in state and len(state['x']) > 0:
                        body_x = state['x'][body_particle_ids][::step_size]
                        obs.append(body_x.flatten())
                    if 'v' in state and len(state['v']) > 0:
                        body_v = state['v'][body_particle_ids][::step_size]
                        obs.append(body_v.flatten())
                    if 'used' in state and len(state['used']) > 0:
                        body_used = state['used'][body_particle_ids][::step_size]
                        obs.append(body_used.flatten())
        
        if 'agent' in state:
            obs += state['agent']
        
        if 'smoke_field' in state:
            if 'v' in state['smoke_field']:
                obs.append(state['smoke_field']['v'][::10, 60:68, ::10].flatten())
            if 'q' in state['smoke_field']:
                obs.append(state['smoke_field']['q'][::10, 60:68, ::10].flatten())
        
        if len(obs) == 0:
            return np.array([0.0])  # Dummy observation
        
        return np.concatenate(obs)
    
    def reset(self):
        """Reset all environments in the batch"""
        init_state = self.base_env._init_state
        self.taichi_env.set_state(**init_state)
        
        obs_batch = []
        for b in range(self.batch_size):
            state = self.taichi_env.get_state_RL(batch_id=b)
            obs_batch.append(self._get_obs_from_state(state))
        
        return {
            'batch_obs': np.array(obs_batch, dtype=np.float32)
        }
    
    def step(self, action):
        """
        Step all environments in the batch.
        Args:
            action: (batch_size, action_dim) array
        Returns:
            obs: Dict with 'batch_obs' key
            reward: (batch_size,) array
            done: (batch_size,) array
            info: Dict with batch information
        """
        # Ensure action is correct shape
        action = np.asarray(action)
        assert action.shape[0] == self.batch_size, f"Action batch size {action.shape[0]} doesn't match env batch_size {self.batch_size}"
        
        # Clip actions
        action = action.clip(self.action_range[0], self.action_range[1])
        
        # Step all batches
        self.taichi_env.step(action)
        
        # Get observations and rewards for all batches
        obs_batch = []
        reward_batch = []
        done_batch = []
        
        for b in range(self.batch_size):
            state = self.taichi_env.get_state_RL(batch_id=b)
            obs_batch.append(self._get_obs_from_state(state))
            
            if self.loss:
                # Get reward for this batch
                # Note: This is a simplified version - loss function may need batch support
                reward = self._get_reward_for_batch(b)
            else:
                reward = 0.0
            
            reward_batch.append(reward)
            
            # Check if done
            done = (self.taichi_env.t >= self.horizon)
            done_batch.append(done)
        
        obs = {
            'batch_obs': np.array(obs_batch, dtype=np.float32)
        }
        reward = np.array(reward_batch, dtype=np.float32)
        done = np.array(done_batch, dtype=bool)
        info = {}
        
        return obs, reward, done, info
    
    def _get_reward_for_batch(self, batch_id):
        """Get reward for a specific batch"""
        # Simplified: return 0.0 for now
        # In practice, loss function would need batch support
        return 0.0
    
    def render(self, mode='human', batch_id=0):
        """
        Render a specific batch (default: batch 0).
        Note: Rendering all batches simultaneously is not supported.
        """
        assert mode in ['human', 'rgb_array']
        if self.taichi_env.renderer is not None:
            # Temporarily set renderer to use specific batch
            # This is a workaround - renderer may need batch support
            f = self.taichi_env.simulator.cur_substep_local
            state_render = self.taichi_env.simulator.get_state_render(f, batch_id=batch_id)
            return self.taichi_env.renderer.render_frame(mode, None, self.taichi_env.t)
        return None
    
    @property
    def t(self):
        return self.taichi_env.t
