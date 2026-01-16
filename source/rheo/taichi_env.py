import numpy as np
import taichi as ti
from yacs.config import CfgNode
from rheo.simulators import MPMSimulator, SmokeField
from rheo.agents import *
from rheo.meshes import Statics
from rheo.renderers import get_gl_renderer
from rheo.bodies import Bodies
from rheo.macros import *
from rheo.utils.misc import *

# Global flag to track if Taichi has been initialized
_TAICHI_INITIALIZED = False

def init_taichi(num_envs=1, enable_grad=True, device_memory_GB=None):
    """
    Initialize Taichi with appropriate memory settings.
    IMPORTANT: ti.init can only be called once, so this must be done before creating any fields.
    
    Args:
        num_envs: Number of parallel environments (affects memory allocation)
        enable_grad: Whether gradient computation is enabled (affects memory needs)
        device_memory_GB: Override default memory allocation (auto-calculated if None)
    """
    global _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED:
        return  # Already initialized
    
    if device_memory_GB is None:
        # Dynamic memory allocation based on num_envs and mode
        if enable_grad:
            # Differentiable mode needs more memory for gradients
            device_memory_GB = 20
        else:
            # Inference mode is much lighter
            device_memory_GB = min(8, 2 + num_envs * 0.2)
    
    # GL 渲染器不需要 Taichi 的 GGUI/Vulkan 功能
    # 禁用 offline_cache 避免 Vulkan 初始化问题
    ti.init(arch=ti.gpu, device_memory_GB=device_memory_GB, offline_cache=False)
    
    _TAICHI_INITIALIZED = True
    print(f'===>  Taichi initialized: num_envs={num_envs}, enable_grad={enable_grad}, device_memory_GB={device_memory_GB}')

# NOTE: Do NOT auto-initialize here - let scripts call init_taichi() explicitly
# This allows users to customize num_envs and enable_grad parameters

@ti.data_oriented
class TaichiEnv:
    '''
    TaichiEnv wraps all components in a simulation environment.
    Supports GPU batch parallelism with num_envs parameter.
    '''    
    def __init__(
            self,
            dim=3,
            quality=1,
            particle_density=1e6,
            max_substeps_local=50,
            max_substeps_global=100000,
            horizon=100,
            ckpt_dest='disk',
            gravity=(0.0, -10.0, 0.0),
            num_envs=1,
            enable_grad=True,
        ):
        """
        Initialize TaichiEnv.
        
        Args:
            num_envs: Number of parallel environments (default 1 for backward compatibility)
            enable_grad: Whether to enable gradient computation (affects memory layout)
                - True: Use full time frames for checkpointing (max_substeps_local+1)
                - False: Use minimal buffers for inference (2 frames particles, 1 frame grid)
        """
        self.num_envs            = num_envs
        self.enable_grad         = enable_grad
        self.particle_density    = particle_density
        self.dim                 = dim
        self.max_substeps_local  = max_substeps_local
        self.max_substeps_global = max_substeps_global
        self.horizon             = horizon
        self.ckpt_dest           = ckpt_dest
        self.t                   = 0

        # Per-env random seeds for sample diversity
        self.base_seed = 0
        self.env_seeds = [self.base_seed + env_id for env_id in range(num_envs)]

        # env components
        self.simulator = MPMSimulator(
            dim                 = self.dim,
            quality             = quality,
            horizon             = self.horizon,
            max_substeps_local  = self.max_substeps_local,
            max_substeps_global = self.max_substeps_global,
            gravity             = gravity,
            ckpt_dest           = ckpt_dest,
            num_envs            = num_envs,
            enable_grad         = enable_grad,
        )
        self.agent           = None
        self.statics         = Statics()
        self.particle_bodies = Bodies(dim=self.dim, particle_density=self.particle_density)
        self.renderer        = None
        self.loss            = None
        self.smoke_field     = None
        
        # Store initial state for reset
        self.init_x = None
        self.init_used = None

        print(f'===>  TaichiEnv created: num_envs={num_envs}, enable_grad={enable_grad}')

    def setup_agent(self, agent_cfg):
        self.agent = eval(agent_cfg.type)(
            max_substeps_local=self.max_substeps_local,
            max_substeps_global=self.max_substeps_global,
            max_action_steps_global=self.horizon,
            ckpt_dest=self.ckpt_dest,
            **agent_cfg.get('params', {}),
        )
        for effector_cfg_dict in agent_cfg.effectors:
            effector_cfg = CfgNode(effector_cfg_dict)
            self.agent.add_effector(
                type=effector_cfg.type,
                params=effector_cfg.params,
                mesh_cfg=effector_cfg.get('mesh', None),
                boundary_cfg=effector_cfg.boundary,
            )

    def setup_renderer(self, **kwargs):
        """设置 GL 渲染器
        
        Args:
            **kwargs: 渲染器参数
        """
        Renderer = get_gl_renderer()
        # Pass num_envs to renderer for multi-env visualization
        self.renderer = Renderer(num_envs=self.num_envs, **kwargs)

    def setup_boundary(self, **kwargs):
        self.simulator.setup_boundary(**kwargs)

    def add_static(self, **kwargs):
        self.statics.add_static(**kwargs)

    def add_body(self, **kwargs):
        self.particle_bodies.add_body(**kwargs)

    def setup_smoke_field(self, **kwargs):
        self.smoke_field = SmokeField(
            dim=self.dim,
            ckpt_dest=self.ckpt_dest,
             **kwargs
        )

    def setup_loss(self, loss_cls, **kwargs):
        self.loss = loss_cls(
            max_loss_steps=self.horizon,
            **kwargs
        )

    def build(self):
        # particles
        self.particles = self.particle_bodies.get()

        if self.particles is not None:
            self.n_particles = len(self.particles['x'])
            self.has_particles = True
            # Store initial state for reset
            self.init_x = self.particles['x'].copy().astype(DTYPE_NP)
            self.init_used = self.particles['used'].copy().astype(np.int32)
        else:
            self.n_particles = 0
            self.has_particles = False
            self.init_x = None
            self.init_used = None

        # build and initialize states of all environment components
        self.simulator.build(self.agent, self.smoke_field, self.statics, self.particles)

        if self.agent is not None:
            self.agent.build(self.simulator)

        if self.smoke_field is not None:
            self.smoke_field.build(self.simulator, self.agent)

        if self.renderer is not None:
            self.renderer.build(self.simulator, self.particles)

        if self.loss is not None:
            self.loss.build(self.simulator)

        self.t = 0

    def reset_grad(self):
        self.simulator.reset_grad()
        
        if self.agent is not None:
            self.agent.reset_grad()

        if self.smoke_field is not None:
            self.smoke_field.reset_grad()

        if self.loss is not None:
            self.loss.reset_grad()

    def enable_grad(self):
        self.simulator.enable_grad()

    def disable_grad(self):
        self.simulator.disable_grad()

    @property
    def grad_enabled(self):
        return self.simulator.grad_enabled

    def render(self, mode='human', tgt_particles=None):
        assert self.renderer is not None, 'No renderer available.'
        return self.renderer.render_frame(mode, tgt_particles, self.t)

    def get_state_RL(self):
        return self.simulator.get_state_RL()

    def step(self, action=None):
        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step(action=action)

        if self.loss:
            self.loss.step()

        self.t += 1

    def step_grad(self, action=None):
        if self.loss:
            self.loss.step_grad()

        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step_grad(action=action)

    def get_step_loss(self):
        assert self.loss is not None
        return self.loss.get_step_loss()

    def get_final_loss(self):
        assert self.loss is not None
        return self.loss.get_final_loss()

    def get_final_loss_grad(self):
        assert self.loss is not None
        self.loss.get_final_loss_grad()

    def get_state(self):
        return {
            'state': self.simulator.get_state(),
            'grad_enabled': self.grad_enabled
        }

    def set_state(self, state, grad_enabled=False):
        self.t = 0

        self.simulator.cur_substep_global = 0
        self.simulator.set_state(0, state)

        if grad_enabled:
            self.enable_grad()
        else:
            self.disable_grad()

        if self.loss:
            self.loss.reset()

    def apply_agent_action_p(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p(action_p)

    def apply_agent_action_p_grad(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p_grad(action_p)

    # --------------------------------- Multi-Env Support -----------------------------------
    def get_state_RL_all_envs(self):
        """Get RL state for all environments. Returns dict with arrays of shape (num_envs, ...)"""
        return self.simulator.get_state_RL_all_envs()

    def get_x_all_envs(self, f=None):
        """Get particle positions for all environments."""
        return self.simulator.get_x_all_envs(f)

    def reset_envs(self, env_ids=None):
        """
        Reset specific environments to initial state.
        
        Args:
            env_ids: array of environment indices to reset, or None to reset all
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        if self.has_particles:
            self.simulator.reset_envs(env_ids, self.init_x, self.init_used)
        
        # Reset agent state for specific envs if needed
        # TODO: Add agent reset support
        
    def reset_all(self):
        """Reset all environments to initial state."""
        self.reset_envs(None)
        self.t = 0
        
        if self.loss:
            self.loss.reset()

    def set_seed(self, seed):
        """Set base random seed for all environments."""
        self.base_seed = seed
        self.env_seeds = [self.base_seed + env_id for env_id in range(self.num_envs)]
    
    def get_env_seed(self, env_id):
        """Get random seed for a specific environment."""
        return self.env_seeds[env_id]