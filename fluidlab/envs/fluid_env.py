import os
import gym
import numpy as np
from gym.spaces import Box
from fluidlab.configs.macros import *
from fluidlab.fluidengine.taichi_env import TaichiEnv, init_taichi
import fluidlab.utils.misc as misc_utils

class FluidEnv(gym.Env):
    '''
    Base env class with GPU batch parallelism support.
    
    When num_envs > 1:
    - observations have shape (num_envs, obs_dim)
    - actions have shape (num_envs, action_dim)
    - rewards have shape (num_envs,)
    - dones have shape (num_envs,)
    '''    
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI',
                 num_envs=1, enable_grad=True):
        """
        Initialize FluidEnv.
        
        Args:
            num_envs: Number of parallel environments (default 1 for backward compatibility)
            enable_grad: Whether to enable gradient computation (affects memory usage)
        """
        self.num_envs = num_envs
        self.enable_grad = enable_grad
        
        # Initialize Taichi with appropriate memory settings
        init_taichi(num_envs=num_envs, enable_grad=enable_grad)
        
        if seed is not None:
            self.seed(seed)

        self.horizon               = 500
        self.horizon_action        = 500
        self.target_file           = None
        self._n_obs_ptcls_per_body = 200
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-1.0, 1.0])
        self.renderer_type         = renderer_type

        # create a taichi env with num_envs support
        self.taichi_env = TaichiEnv(num_envs=num_envs, enable_grad=enable_grad)
        self.build_env()
        self.gym_misc()

    def seed(self, seed):
        # Note: gym.Env.seed() was removed in gym>=0.21, so we only set our own random seed
        misc_utils.set_random_seed(seed)

    def build_env(self):
        self.setup_agent()
        self.setup_statics()
        self.setup_bodies()
        self.setup_smoke_field()
        self.setup_boundary()
        if misc_utils.should_render():
            self.setup_renderer()
        if self.loss:
            self.setup_loss()
            
        self.taichi_env.build()
        self._init_state = self.taichi_env.get_state()
        
        print(f'===>  {type(self).__name__} built successfully.')

    def setup_agent(self):
        pass

    def setup_statics(self):
        # add static mesh-based objects in the scene
        pass

    def setup_bodies(self):
        # add fluid/object bodies
        self.taichi_env.add_body(
            type='cube',
            lower=(0.2, 0.2, 0.2),
            upper=(0.4, 0.4, 0.4),
            material=WATER,
        )
        self.taichi_env.add_body(
            type='ball',
            center=(0.6, 0.3, 0.6),
            radius=0.1,
            material=WATER,
        )

    def setup_smoke_field(self):
        pass

    def setup_boundary(self):
        pass

    def setup_renderer(self):
        self.taichi_env.setup_renderer()

    def setup_loss(self):
        pass

    def gym_misc(self):
        if self.loss_type == 'default':
            self.horizon = self.horizon_action
        obs = self.reset()
        
        # Observation space depends on num_envs
        if self.num_envs == 1:
            self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), obs.shape, dtype=DTYPE_NP)
        else:
            # For multi-env, obs has shape (num_envs, obs_dim)
            single_obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]
            self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), (self.num_envs, single_obs_dim), dtype=DTYPE_NP)
        
        if self.taichi_env.agent is not None:
            if self.num_envs == 1:
                self.action_space = Box(DTYPE_NP(self.action_range[0]), DTYPE_NP(self.action_range[1]), 
                                       (self.taichi_env.agent.action_dim,), dtype=DTYPE_NP)
            else:
                self.action_space = Box(DTYPE_NP(self.action_range[0]), DTYPE_NP(self.action_range[1]), 
                                       (self.num_envs, self.taichi_env.agent.action_dim), dtype=DTYPE_NP)
        else:
            self.action_space = None

    def reset(self, env_ids=None):
        """
        Reset environment(s).
        
        Args:
            env_ids: If provided, only reset these specific environments.
                     If None, reset all environments.
        
        Returns:
            obs: Observations for reset environments. Shape (num_envs, obs_dim) or (obs_dim,)
        """
        if env_ids is None:
            # Reset all environments
            self.taichi_env.set_state(**self._init_state)
        else:
            # Partial reset - only reset specified environments
            self.taichi_env.reset_envs(env_ids)
        
        return self._get_obs()

    def _get_obs(self, env_id=None):
        """
        Get observations for environment(s).
        
        Args:
            env_id: If provided, get obs for single env. If None, get for all envs.
        
        Returns:
            obs: Shape (num_envs, obs_dim) for multi-env, (obs_dim,) for single env
        """
        if self.num_envs == 1:
            # Original single-env behavior
            return self._get_single_obs(0)
        else:
            # Multi-env: get obs for all environments
            obs_list = []
            for env_id in range(self.num_envs):
                obs_list.append(self._get_single_obs(env_id))
            return np.stack(obs_list, axis=0)

    def _get_single_obs(self, env_id):
        """Get observation for a single environment."""
        state = self.taichi_env.simulator.get_state_RL(env_id=env_id)
        obs   = []

        if 'x' in state:
            for body_id in range(self.taichi_env.particles['bodies']['n']):
                body_n_particles  = self.taichi_env.particles['bodies']['n_particles'][body_id]
                body_particle_ids = self.taichi_env.particles['bodies']['particle_ids'][body_id]

                step_size = max(1, body_n_particles // self._n_obs_ptcls_per_body)
                body_x    = state['x'][body_particle_ids][::step_size]
                body_v    = state['v'][body_particle_ids][::step_size]
                body_used = state['used'][body_particle_ids][::step_size]

                obs.append(body_x.flatten())
                obs.append(body_v.flatten())
                obs.append(body_used.flatten())

        if 'agent' in state:
            obs += state['agent']

        if 'smoke_field' in state:
            obs.append(state['smoke_field']['v'][::10, 60:68, ::10].flatten())
            obs.append(state['smoke_field']['q'][::10, 60:68, ::10].flatten())

        obs = np.concatenate(obs)
        return obs

    def _get_reward(self):
        """Get reward(s). Returns shape (num_envs,) for multi-env, scalar for single."""
        if self.taichi_env.loss is None:
            if self.num_envs == 1:
                return 0.0
            else:
                return np.zeros(self.num_envs, dtype=DTYPE_NP)
        
        loss_info = self.taichi_env.get_step_loss()
        reward = loss_info['reward']
        
        # Handle multi-env case
        if self.num_envs > 1 and np.isscalar(reward):
            # If loss doesn't support multi-env yet, broadcast
            reward = np.full(self.num_envs, reward, dtype=DTYPE_NP)
        
        return reward

    def step(self, action):
        """
        Step all environments with given action(s).
        
        Args:
            action: Shape (num_envs, action_dim) for multi-env, (action_dim,) for single
        
        Returns:
            obs: Shape (num_envs, obs_dim) for multi-env, (obs_dim,) for single
            reward: Shape (num_envs,) for multi-env, scalar for single
            done: Shape (num_envs,) for multi-env, bool for single
            info: dict with additional info
        """
        action = np.array(action).clip(self.action_range[0], self.action_range[1])
        
        # For multi-env, action should be (num_envs, action_dim)
        # TODO: Currently simulator doesn't support batched actions directly
        # For now, we assume all envs receive the same action
        if self.num_envs > 1 and len(action.shape) == 1:
            # Broadcast single action to all envs
            action = np.tile(action, (self.num_envs, 1))
        
        self.taichi_env.step(action if self.num_envs == 1 else action[0])

        obs    = self._get_obs()
        reward = self._get_reward()

        assert self.t <= self.horizon
        
        if self.num_envs == 1:
            # Single env case
            done = self.t == self.horizon
            if np.isnan(reward):
                reward = -1000
                done = True
        else:
            # Multi-env case
            dones = np.full(self.num_envs, self.t == self.horizon, dtype=bool)
            nan_mask = np.isnan(reward)
            if nan_mask.any():
                reward[nan_mask] = -1000
                dones[nan_mask] = True
            done = dones

        info = dict()
        return obs, reward, done, info

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
        return self.taichi_env.render(mode)

    @property
    def t(self):
        return self.taichi_env.t
