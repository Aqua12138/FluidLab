import taichi as ti
import numpy as np
import pickle as pkl
import uuid
import os
import torch
from time import time
from fluidlab.utils.misc import *
from fluidlab.utils.misc import is_headless
from fluidlab.configs.macros import *
from fluidlab.fluidengine.boundaries import create_boundary

@ti.data_oriented
class MPMSimulator:
    def __init__(self, dim, quality, gravity, horizon, max_substeps_local, max_substeps_global, ckpt_dest,
                 num_envs=1, enable_grad=True):
        """
        MPM Simulator with GPU parallel environment support.
        
        Args:
            num_envs: Number of parallel environments (default 1 for backward compatibility)
            enable_grad: Whether to enable gradient computation (affects memory layout)
                - True: Use full time frames for checkpointing (max_substeps_local+1)
                - False: Use minimal buffers for inference (2 frames for particles, 1 for grid)
        """
        self.num_envs = num_envs
        self.enable_grad = enable_grad
        
        self.dim       = dim
        self.ckpt_dest = ckpt_dest
        self.sim_id    = str(uuid.uuid4())
        self.gravity   = ti.Vector(gravity)

        self.n_grid              = int(64 * quality)
        self.dx                  = 1 / self.n_grid
        self.inv_dx              = float(self.n_grid)
        self.dt                  = 1e-4
        self.p_vol               = (self.dx * 0.5) ** 2
        self.res                 = (self.n_grid,) * self.dim
        self.max_substeps_local  = max_substeps_local
        self.max_substeps_global = max_substeps_global
        self.horizon             = horizon
        self.n_substeps          = int(2e-3 / self.dt)
        self.max_steps_local     = int(self.max_substeps_local / self.n_substeps)

        assert self.n_substeps * self.horizon < self.max_substeps_global
        assert self.max_substeps_local % self.n_substeps == 0

        # Determine time frames based on mode
        if self.enable_grad:
            self.particle_time_frames = self.max_substeps_local + 1
            self.grid_time_frames = self.max_substeps_local + 1
        else:
            # Inference mode: minimal buffers
            self.particle_time_frames = 2  # Double buffer for particles
            self.grid_time_frames = 1      # Single buffer for grid

        # Global maximum viscosity limit to prevent numerical instability
        # This is critical for n > 1 (shear-thickening) materials
        # Physical meaning: real materials have viscosity saturation at high shear rates
        # This prevents CFL condition from becoming too strict and causing simulation crash
        self.max_viscosity = 5e3  # Global viscosity ceiling (100,000)

        self.boundary      = None
        self.has_particles = False
        
        # Per-environment time tracking
        self.cur_substep_global_per_env = np.zeros(num_envs, dtype=np.int32)

    def setup_boundary(self, **kwargs):
        self.boundary = create_boundary(**kwargs)

    def build(self, agent, smoke_field, statics, particles):
        # default boundary
        if self.boundary is None:
            self.boundary = create_boundary()

        # statics
        self.n_statics = len(statics)
        self.statics = statics

        # particles and bodies
        if particles is not None:
            self.has_particles = True
            self.n_particles = len(particles['x'])
            self.setup_particle_fields()
            self.setup_grid_fields()
            self.setup_ckpt_vars()
            self.init_particles_and_bodies(particles)
        else:
            self.has_particles = False
            self.n_particles = 0

        # agent
        self.agent = agent

        # smoke
        self.smoke_field = smoke_field

        # misc
        self.cur_substep_global = 0
        self.disable_grad() # grad disabled by default

    def setup_particle_fields(self):
        # particle state
        particle_state = ti.types.struct(
            x     = ti.types.vector(self.dim, DTYPE_TI),  # position
            v     = ti.types.vector(self.dim, DTYPE_TI),  # velocity
            C     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # affine velocity field
            F     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # deformation gradient
            F_tmp = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # temp deformation gradient
            U     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            V     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            S     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
        )
        # particle state without gradient
        particle_state_ng = ti.types.struct(
            used = ti.i32,
        )
        # single frame particle state for rendering
        particle_state_render = ti.types.struct(
            x    = ti.types.vector(self.dim, ti.f32),
            used = ti.i32,
        )

        # particle info (shared across all envs - no num_envs dimension)
        particle_info = ti.types.struct(
            lam               = DTYPE_TI,
            mat               = ti.i32,
            mat_cls           = ti.i32,
            body_id           = ti.i32,
            mass              = DTYPE_TI,
            yield_stress      = DTYPE_TI,  # Herschel-Bulkley: τ_y
            consistency       = DTYPE_TI,   # Herschel-Bulkley: K (PRIMARY viscosity parameter)
            flow_index        = DTYPE_TI,   # Herschel-Bulkley: n
            use_herschel_bulkley = ti.i32,  # Flag to enable H-B model
        )

        # construct fields with num_envs dimension
        # Shape: (num_envs, time_frames, n_particles)
        self.particles    = particle_state.field(
            shape=(self.num_envs, self.particle_time_frames, self.n_particles), 
            needs_grad=self.enable_grad, layout=ti.Layout.SOA)
        self.particles_ng = particle_state_ng.field(
            shape=(self.num_envs, self.particle_time_frames, self.n_particles), 
            needs_grad=False, layout=ti.Layout.SOA)
        
        # Render buffer: (num_envs * n_particles) for multi-env rendering
        # Skip allocation in headless mode to save memory
        if not is_headless():
            self.particles_render = particle_state_render.field(
                shape=(self.num_envs * self.n_particles,), 
                needs_grad=False, layout=ti.Layout.SOA)
        else:
            self.particles_render = None
        
        # Material info shared across all envs (no num_envs dimension for memory efficiency)
        self.particles_i = particle_info.field(
            shape=(self.n_particles,), 
            needs_grad=False, layout=ti.Layout.SOA)

    def setup_grid_fields(self):
        grid_cell_state = ti.types.struct(
            v_in  = ti.types.vector(self.dim, DTYPE_TI), # input momentum/velocity
            mass  = DTYPE_TI,                            # mass
            v_out = ti.types.vector(self.dim, DTYPE_TI), # output momentum/velocity
        )
        # Grid shape: (num_envs, time_frames, *res)
        self.grid = grid_cell_state.field(
            shape=(self.num_envs, self.grid_time_frames, *self.res), 
            needs_grad=self.enable_grad, layout=ti.Layout.SOA)
    
    def get_particle_frame_idx(self, f):
        """Get particle buffer index based on mode."""
        if self.enable_grad:
            return f
        else:
            return f % 2  # Double buffer
    
    def get_grid_frame_idx(self, f):
        """Get grid buffer index based on mode."""
        if self.enable_grad:
            return f
        else:
            return 0  # Single buffer for inference

    def setup_ckpt_vars(self):
        if self.ckpt_dest == 'disk':
            # placeholder np array from checkpointing (with num_envs dimension)
            self.x_np    = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
            self.v_np    = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
            self.C_np    = np.zeros((self.num_envs, self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.F_np    = np.zeros((self.num_envs, self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.used_np = np.zeros((self.num_envs, self.n_particles,), dtype=np.int32)
        elif self.ckpt_dest == 'cpu' or 'gpu':
            self.ckpt_ram = dict()
        self.actions_buffer = []
        self.setup_ckpt_dir()

    def setup_ckpt_dir(self):
        self.ckpt_dir = os.path.join('/tmp', 'fluidlab', self.sim_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def init_particles_and_bodies(self, particles):
        x          = particles['x'].astype(DTYPE_NP)
        used       = particles['used'].astype(np.int32)
        mat        = particles['mat'].astype(np.int32)
        p_rho      = particles['rho'].astype(DTYPE_NP)
        body_id    = particles['body_id'].astype(np.int32)
        
        lam        = np.array([LAMDA[mat_i] for mat_i in mat]).astype(DTYPE_NP)
        mat_cls    = np.array([MAT_CLASS[mat_i] for mat_i in mat]).astype(np.int32)
        
        # 统一使用CONSISTENCY作为主要粘度参数（所有材料）
        # 如果CONSISTENCY未定义，使用0.0（不再使用MU）
        consistency  = np.array([CONSISTENCY.get(mat_i, 0.0) for mat_i in mat]).astype(DTYPE_NP)
        
        # Herschel-Bulkley parameters
        yield_stress = np.array([YIELD_STRESS.get(mat_i, 0.0) for mat_i in mat]).astype(DTYPE_NP)
        flow_index   = np.array([FLOW_INDEX.get(mat_i, 1.0) for mat_i in mat]).astype(DTYPE_NP)
        
        # Enable H-B for MAT_LIQUID materials if:
        # 1. yield_stress > 0 (non-Newtonian with yield stress)
        # 2. flow_index != 1.0 (non-Newtonian power-law)
        # 3. CONSISTENCY is explicitly set (ensures CONSISTENCY is used as primary parameter)
        # For non-liquid materials, H-B is disabled
        use_hb = np.array([
            1 if (MAT_CLASS.get(mat[i], -1) == MAT_LIQUID and 
                  (yield_stress[i] > 1e-6 or 
                   abs(flow_index[i] - 1.0) > 1e-6 or
                   mat[i] in CONSISTENCY))
            else 0
            for i in range(len(mat))
        ]).astype(np.int32)

        self.init_particles_kernel(x, mat, mat_cls, used, lam, p_rho, body_id, 
                                   yield_stress, consistency, flow_index, use_hb)
        self.init_bodies(mat_cls, body_id, particles['bodies'])

    @ti.kernel
    def init_particles_kernel(
            self,
            x            : ti.types.ndarray(),
            mat          : ti.types.ndarray(),
            mat_cls      : ti.types.ndarray(),
            used         : ti.types.ndarray(),
            lam          : ti.types.ndarray(),
            p_rho        : ti.types.ndarray(),
            body_id      : ti.types.ndarray(),
            yield_stress : ti.types.ndarray(),
            consistency  : ti.types.ndarray(),
            flow_index   : ti.types.ndarray(),
            use_hb       : ti.types.ndarray()
        ):
        # Initialize particles for all environments with same initial state
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[env_id, 0, i].x[j] = x[i, j]
            self.particles[env_id, 0, i].v       = ti.Vector.zero(DTYPE_TI, self.dim)
            self.particles[env_id, 0, i].F       = ti.Matrix.identity(DTYPE_TI, self.dim)
            self.particles[env_id, 0, i].C       = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles_ng[env_id, 0, i].used = used[i]

        # Material info is shared across all envs (only initialize once)
        for i in range(self.n_particles):
            self.particles_i[i].mat                = mat[i]
            self.particles_i[i].mat_cls            = mat_cls[i]
            self.particles_i[i].lam                = lam[i]
            self.particles_i[i].mass               = self.p_vol * p_rho[i]
            self.particles_i[i].body_id            = body_id[i]
            self.particles_i[i].yield_stress       = yield_stress[i]
            self.particles_i[i].consistency       = consistency[i]
            self.particles_i[i].flow_index         = flow_index[i]
            self.particles_i[i].use_herschel_bulkley = use_hb[i]

    def init_bodies(self, mat_cls, body_id, bodies):
        self.n_bodies = bodies['n']
        assert self.n_bodies == np.max(body_id) + 1

        # body state, for rigidity enforcement (with num_envs dimension)
        body_state = ti.types.struct(
            COM_t0 = ti.types.vector(self.dim, DTYPE_TI),
            COM_t1 = ti.types.vector(self.dim, DTYPE_TI),
            H      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            R      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            U      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            S      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            V      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
        )
        # body info (shared across envs)
        body_info = ti.types.struct(
            n_particles = ti.i32,
            mat_cls     = ti.i32,
        )
        # Bodies state needs num_envs dimension
        self.bodies   = body_state.field(shape=(self.num_envs, self.n_bodies,), needs_grad=self.enable_grad, layout=ti.Layout.SOA)
        # Body info is shared across envs
        self.bodies_i = body_info.field(shape=(self.n_bodies,), needs_grad=False, layout=ti.Layout.SOA)

        for i in range(self.n_bodies):
            self.bodies_i[i].n_particles = np.sum(body_id == i)
            self.bodies_i[i].mat_cls = mat_cls[body_id == i][0]

    def reset_grad(self):
        if self.enable_grad:
            self.particles.grad.fill(0)
            self.grid.grad.fill(0)

    def enable_grad(self):
        '''
        If grad_enable == True, we do checkpointing when gpu memory is not enough for storing the whole episode.
        '''
        self.grad_enabled       = True
        self.cur_substep_global = 0

    def disable_grad(self):
        self.grad_enabled       = False
        self.cur_substep_global = 0

    # --------------------------------- MPM part -----------------------------------
    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        grid_f = f if ti.static(self.enable_grad) else 0
        # Use ti.ndrange for parallelization across all dimensions
        if ti.static(self.dim == 3):
            for env_id, i, j, k in ti.ndrange(self.num_envs, self.res[0], self.res[1], self.res[2]):
                self.grid[env_id, grid_f, i, j, k].fill(0)
            if ti.static(self.enable_grad):
                for env_id, i, j, k in ti.ndrange(self.num_envs, self.res[0], self.res[1], self.res[2]):
                    self.grid.grad[env_id, grid_f, i, j, k].fill(0)
        else:  # dim == 2
            for env_id, i, j in ti.ndrange(self.num_envs, self.res[0], self.res[1]):
                self.grid[env_id, grid_f, i, j].fill(0)
            if ti.static(self.enable_grad):
                for env_id, i, j in ti.ndrange(self.num_envs, self.res[0], self.res[1]):
                    self.grid.grad[env_id, grid_f, i, j].fill(0)

    def f_global_to_f_local(self, f_global):
        f_local = f_global % self.max_substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self.n_substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self.n_substeps
        return s_global

    @property
    def cur_substep_local(self):
        return self.f_global_to_f_local(self.cur_substep_global)

    @property
    def cur_step_local(self):
        return self.f_global_to_s_local(self.cur_substep_global)

    @property
    def cur_step_global(self):
        return self.f_global_to_s_global(self.cur_substep_global)

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used:
                self.particles[env_id, pf, p].F_tmp = (ti.Matrix.identity(DTYPE_TI, self.dim) + self.dt * self.particles[env_id, pf, p].C) @ self.particles[env_id, pf, p].F

    @ti.kernel
    def svd(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used:
                self.particles[env_id, pf, p].U, self.particles[env_id, pf, p].S, self.particles[env_id, pf, p].V = ti.svd(self.particles[env_id, pf, p].F_tmp, DTYPE_TI)

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used:
                self.particles.grad[env_id, pf, p].F_tmp += self.backward_svd(
                    self.particles.grad[env_id, pf, p].U, self.particles.grad[env_id, pf, p].S, self.particles.grad[env_id, pf, p].V, 
                    self.particles[env_id, pf, p].U, self.particles[env_id, pf, p].S, self.particles[env_id, pf, p].V)

    @ti.func
    def backward_svd(self, grad_U, grad_S, grad_V, U, S, V):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = V.transpose()
        ut = U.transpose()
        S_term = U @ grad_S @ vt

        s = ti.Vector.zero(DTYPE_TI, self.dim)
        if ti.static(self.dim==2):
            s = ti.Vector([S[0, 0], S[1, 1]]) ** 2
        else:
            s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
        F = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
        v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
        return u_term + v_term + S_term

    @ti.func
    def clamp(self, a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a>=0:
            a = ti.max(a, 1e-8)
        else:
            a = ti.min(a, -1e-8)
        return a

    @ti.kernel
    def advect_used(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            self.particles_ng[env_id, pf_next, p].used = self.particles_ng[env_id, pf, p].used

    @ti.kernel
    def process_unused_particles(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used == 0:
                self.particles[env_id, pf_next, p].v = self.particles[env_id, pf, p].v
                self.particles[env_id, pf_next, p].x = self.particles[env_id, pf, p].x
                self.particles[env_id, pf_next, p].C = self.particles[env_id, pf, p].C
                self.particles[env_id, pf_next, p].F = self.particles[env_id, pf, p].F

    def agent_act(self, f, is_none_action):
        if not is_none_action:
            self.agent.act(f, self.cur_substep_global)

    def agent_act_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.act_grad(f, self.cur_substep_global)


    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.func
    def compute_shear_rate(self, C):
        """
        Compute shear rate from velocity gradient C.
        Returns: scalar shear rate γ̇
        """
        # Compute deformation rate tensor D = (C + C^T) / 2
        D = 0.5 * (C + C.transpose())
        
        # Compute shear rate: γ̇ = sqrt(2 * trace(D^T @ D))
        # Using Frobenius norm: ||D||_F = sqrt(sum(D_ij^2))
        D_squared = D @ D.transpose()
        trace_D_squared = 0.0
        for i in ti.static(range(self.dim)):
            trace_D_squared += D_squared[i, i]
        
        # Shear rate: γ̇ = sqrt(2 * ||D||_F^2) = sqrt(2 * trace(D^T @ D))
        shear_rate = ti.sqrt(2.0 * trace_D_squared)
        
        # Clamp to minimum value for numerical stability (1e-6 for gradient stability)
        return ti.max(shear_rate, 1e-6)

    @ti.func
    def compute_effective_viscosity(self, shear_rate, yield_stress, k, n, m=100.0):
        """
        Compute effective viscosity using Herschel-Bulkley model with Papanastasiou regularization.
        
        Note: CONSISTENCY (k) is the PRIMARY viscosity parameter.
        When n=1 and τ_y=0, μ_eff = k (constant viscosity, Newtonian fluid).
        
        Args:
            shear_rate: scalar shear rate γ̇
            yield_stress: yield stress τ_y
            k: consistency coefficient K (PRIMARY viscosity parameter)
            n: flow index n
            m: Papanastasiou regularization parameter (default 100.0)
        
        Returns:
            effective viscosity μ_eff (clamped by global max_viscosity)
        """
        # Ensure shear_rate is at least 1e-6 for numerical stability
        sr = ti.max(shear_rate, 1e-6)
        
        # Papanastasiou regularization for yield stress term
        # When sr -> 0: (1-exp(-m*sr))/sr -> m, avoiding division by zero
        # When sr -> ∞: (1-exp(-m*sr))/sr -> 1/sr, recovering standard H-B model
        exp_term = ti.exp(-m * sr)
        yield_term = yield_stress * (1.0 - exp_term) / sr
        
        # Power-law term: K * γ̇^(n-1)
        # Limit exponent to prevent gradient explosion when n < 1
        exponent = ti.max(n - 1.0, -0.9)
        
        # For n >= 1, limit shear rate to prevent power_term explosion
        # When n > 1, power_term grows with sr, so we need to clamp sr
        # This is critical for high-viscosity materials like HONEY and SILLY_PUTTY
        if n >= 1.0:
            # Limit sr to prevent power_term from becoming too large
            # For n=1: power_term = k (constant, no issue)
            # For n>1: power_term = k * sr^(n-1), so we limit sr based on k
            # Use a conservative limit to prevent numerical instability
            if n > 1.0:
                # More conservative: limit to prevent excessive growth
                # For n=1.3, limit sr to ~100 to keep power_term reasonable
                # This prevents power_term from growing too large and causing particle explosion
                sr_max = 100.0  # Conservative limit for n > 1
                sr = ti.min(sr, sr_max)
        
        power_term = k * (sr ** exponent)
        
        # Additional safety: limit power_term directly for n >= 1
        # This provides an extra layer of protection against numerical instability
        if n >= 1.0:
            # Limit power_term to prevent it from exceeding a reasonable multiple of k
            # For n=1: power_term = k, so limit is k * 10 (shouldn't trigger)
            # For n>1: limit power_term to k * 50 to prevent explosion
            power_term_max = k * 50.0 if n > 1.0 else k * 10.0
            power_term = ti.min(power_term, power_term_max)
        
        # Combine terms
        mu_eff = yield_term + power_term
        
        # Global viscosity ceiling: enforce maximum viscosity limit
        # This is critical for n > 1 (shear-thickening) materials to prevent numerical instability
        # Physical meaning: real materials have viscosity saturation at high shear rates
        # This prevents CFL condition from becoming too strict and causing simulation crash
        return ti.min(mu_eff, self.max_viscosity)

    @ti.kernel
    def p2g(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        grid_f = f if ti.static(self.enable_grad) else 0
        
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used:
                base = (self.particles[env_id, pf, p].x * self.inv_dx - 0.5).cast(int)
                fx   = self.particles[env_id, pf, p].x * self.inv_dx - base.cast(DTYPE_TI)
                w    = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

                J = self.particles[env_id, pf, p].S.determinant()

                r = self.particles[env_id, pf, p].U @ self.particles[env_id, pf, p].V.transpose()
                
                # Compute effective viscosity
                # 所有材料统一使用CONSISTENCY作为主要粘度参数
                mu_eff = self.particles_i[p].consistency
                
                if self.particles_i[p].use_herschel_bulkley != 0:
                    # Compute shear rate from velocity gradient C
                    shear_rate = self.compute_shear_rate(self.particles[env_id, pf, p].C)
                    
                    # Compute effective viscosity using Herschel-Bulkley model
                    # Global max_viscosity limit is applied inside compute_effective_viscosity
                    mu_eff = self.compute_effective_viscosity(
                        shear_rate,
                        self.particles_i[p].yield_stress,
                        self.particles_i[p].consistency,
                        self.particles_i[p].flow_index,
                        100.0  # Papanastasiou regularization parameter m
                    )
                
                # Apply global viscosity ceiling as final safety check
                # This ensures mu_eff never exceeds max_viscosity, even for non-H-B materials
                mu_eff = ti.min(mu_eff, self.max_viscosity)
                
                # Compute stress: deviatoric (viscous) + volumetric (pressure)
                # Deviatoric stress: uses dynamic effective viscosity
                deviatoric_stress = 2 * mu_eff * (self.particles[env_id, pf, p].F_tmp - r) @ self.particles[env_id, pf, p].F_tmp.transpose()
                
                # Volumetric stress: unchanged, uses LAMDA
                volumetric_stress = ti.Matrix.identity(DTYPE_TI, self.dim) * self.particles_i[p].lam * J * (J - 1)
                
                # Total stress
                stress = deviatoric_stress + volumetric_stress
                stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
                affine = stress + self.particles_i[p].mass * self.particles[env_id, pf, p].C

                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(DTYPE_TI) - fx) * self.dx
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]

                    self.grid[env_id, grid_f, base + offset].v_in += weight * (self.particles_i[p].mass * self.particles[env_id, pf, p].v + affine @ dpos)
                    self.grid[env_id, grid_f, base + offset].mass += weight * self.particles_i[p].mass

                # update deformation gradient based on material class
                F_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)

                if self.particles_i[p].mat_cls == MAT_LIQUID:
                    F_new = ti.Matrix.identity(DTYPE_TI, self.dim) * ti.pow(J, 1.0/self.dim)

                elif self.particles_i[p].mat_cls == MAT_ELASTIC:
                    F_new = self.particles[env_id, pf, p].F_tmp

                elif self.particles_i[p].mat_cls == MAT_RIGID:
                    F_new = self.particles[env_id, pf, p].F_tmp

                elif self.particles_i[p].mat_cls == MAT_PLASTO_ELASTIC:
                    S_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                    for d in ti.static(range(self.dim)):
                        S_new[d, d] = min(max(self.particles[env_id, pf, p].S[d, d], 1 - 2e-3), 1 + 3e-3)
                    F_new = self.particles[env_id, pf, p].U @ S_new @ self.particles[env_id, pf, p].V.transpose()
                elif self.particles_i[p].mat_cls == MAT_PLASTO_ELASTIC_DEMO:
                    S_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                    for d in ti.static(range(self.dim)):
                        S_new[d, d] = min(max(self.particles[env_id, pf, p].S[d, d], 1 - 2e-3), 1 + 3e-3)
                    F_new = self.particles[env_id, pf, p].U @ S_new @ self.particles[env_id, pf, p].V.transpose()

                self.particles[env_id, pf_next, p].F = F_new

    @ti.kernel
    def grid_op(self, f: ti.i32):
        grid_f = f if ti.static(self.enable_grad) else 0
        # Use ti.ndrange for parallelization, reconstruct I vector inside
        if ti.static(self.dim == 3):
            for env_id, gi, gj, gk in ti.ndrange(self.num_envs, self.res[0], self.res[1], self.res[2]):
                if self.grid[env_id, grid_f, gi, gj, gk].mass > EPS:
                    I = ti.Vector([gi, gj, gk])
                    v_out = (1 / self.grid[env_id, grid_f, gi, gj, gk].mass) * self.grid[env_id, grid_f, gi, gj, gk].v_in  # Momentum to velocity
                    v_out += self.dt * self.gravity # gravity

                    # collide with statics
                    if ti.static(self.n_statics>0):
                        for i in ti.static(range(self.n_statics)):
                            v_out = self.statics[i].collide(I*self.dx, v_out)

                    # collide with agent
                    # TODO: Agent collision needs env_id support
                    if ti.static(self.agent is not None):
                        if ti.static(self.agent.collide_type in ['grid', 'both']):
                            v_out = self.agent.collide(f, I*self.dx, v_out, self.dt)

                    # impose boundary
                    _, self.grid[env_id, grid_f, gi, gj, gk].v_out = self.boundary.impose_x_v(I*self.dx, v_out)
        else:  # dim == 2
            for env_id, gi, gj in ti.ndrange(self.num_envs, self.res[0], self.res[1]):
                if self.grid[env_id, grid_f, gi, gj].mass > EPS:
                    I = ti.Vector([gi, gj])
                    v_out = (1 / self.grid[env_id, grid_f, gi, gj].mass) * self.grid[env_id, grid_f, gi, gj].v_in  # Momentum to velocity
                    v_out += self.dt * self.gravity # gravity

                    # collide with statics
                    if ti.static(self.n_statics>0):
                        for i in ti.static(range(self.n_statics)):
                            v_out = self.statics[i].collide(I*self.dx, v_out)

                    # collide with agent
                    # TODO: Agent collision needs env_id support
                    if ti.static(self.agent is not None):
                        if ti.static(self.agent.collide_type in ['grid', 'both']):
                            v_out = self.agent.collide(f, I*self.dx, v_out, self.dt)

                    # impose boundary
                    _, self.grid[env_id, grid_f, gi, gj].v_out = self.boundary.impose_x_v(I*self.dx, v_out)

    @ti.kernel
    def g2p(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        grid_f = f if ti.static(self.enable_grad) else 0
        
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used:
                base = (self.particles[env_id, pf, p].x * self.inv_dx - 0.5).cast(int)
                fx = self.particles[env_id, pf, p].x * self.inv_dx - base.cast(DTYPE_TI)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(DTYPE_TI, self.dim)
                new_C = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(DTYPE_TI) - fx
                    g_v = self.grid[env_id, grid_f, base + offset].v_out
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

                # collide with agent
                # TODO: Agent collision needs env_id support
                if ti.static(self.agent is not None):
                    if ti.static(self.agent.collide_type in ['particle', 'both']):
                        new_x_tmp = self.particles[env_id, pf, p].x + self.dt * new_v
                        new_v = self.agent.collide(f, new_x_tmp, new_v, self.dt)

                # advect to next frame    
                self.particles[env_id, pf_next, p].v = new_v
                self.particles[env_id, pf_next, p].C = new_C

    def advect(self, f):
        self.reset_bodies_and_grad()
        self.compute_COM(f)
        self.compute_H(f)
        self.compute_H_svd(f)
        self.compute_R(f)
        self.advect_kernel(f)

    def advect_grad(self, f):
        self.reset_bodies_and_grad()
        self.compute_COM(f)
        self.compute_H(f)
        self.compute_H_svd(f)
        self.compute_R(f)

        self.advect_kernel.grad(f)
        self.compute_R.grad(f)
        self.compute_H_svd_grad(f)
        self.compute_H.grad(f)
        self.compute_COM.grad(f)

    @ti.kernel
    def reset_bodies_and_grad(self):
        for env_id, body_id in ti.ndrange(self.num_envs, self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[env_id, body_id].fill(0)
        if ti.static(self.enable_grad):
            for env_id, body_id in ti.ndrange(self.num_envs, self.n_bodies):
                if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                    self.bodies.grad[env_id, body_id].fill(0)

    @ti.kernel
    def compute_COM(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used and self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[env_id, body_id].COM_t0 += self.particles[env_id, pf, p].x / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)
                self.bodies[env_id, body_id].COM_t1 += (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v) / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)

    @ti.kernel
    def compute_H(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used and self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[env_id, body_id].H[0, 0] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[0] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[0]
                self.bodies[env_id, body_id].H[0, 1] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[0] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[1]
                self.bodies[env_id, body_id].H[0, 2] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[0] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[2]
                self.bodies[env_id, body_id].H[1, 0] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[1] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[0]
                self.bodies[env_id, body_id].H[1, 1] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[1] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[1]
                self.bodies[env_id, body_id].H[1, 2] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[1] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[2]
                self.bodies[env_id, body_id].H[2, 0] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[2] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[0]
                self.bodies[env_id, body_id].H[2, 1] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[2] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[1]
                self.bodies[env_id, body_id].H[2, 2] += (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0)[2] * (self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v - self.bodies[env_id, body_id].COM_t1)[2]

    @ti.kernel
    def compute_H_svd(self, f: ti.i32):
        for env_id, body_id in ti.ndrange(self.num_envs, self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[env_id, body_id].U, self.bodies[env_id, body_id].S, self.bodies[env_id, body_id].V = ti.svd(self.bodies[env_id, body_id].H, DTYPE_TI)

    @ti.kernel
    def compute_H_svd_grad(self, f: ti.i32):
        for env_id, body_id in ti.ndrange(self.num_envs, self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies.grad[env_id, body_id].H = self.backward_svd(
                    self.bodies.grad[env_id, body_id].U, self.bodies.grad[env_id, body_id].S, self.bodies.grad[env_id, body_id].V, 
                    self.bodies[env_id, body_id].U, self.bodies[env_id, body_id].S, self.bodies[env_id, body_id].V)

    @ti.kernel
    def compute_R(self, f: ti.i32):
        for env_id, body_id in ti.ndrange(self.num_envs, self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[env_id, body_id].R = self.bodies[env_id, body_id].V @ self.bodies[env_id, body_id].U.transpose()

    @ti.kernel
    def advect_kernel(self, f: ti.i32):
        pf = f if ti.static(self.enable_grad) else f % 2
        pf_next = (f + 1) if ti.static(self.enable_grad) else (f + 1) % 2
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            if self.particles_ng[env_id, pf, p].used:
                if self.particles_i[p].mat_cls == MAT_RIGID: # rigid objects
                    body_id = self.particles_i[p].body_id
                    self.particles[env_id, pf_next, p].x = self.bodies[env_id, body_id].R @ (self.particles[env_id, pf, p].x - self.bodies[env_id, body_id].COM_t0) + self.bodies[env_id, body_id].COM_t1
                else: # other particles
                    self.particles[env_id, pf_next, p].x = self.particles[env_id, pf, p].x + self.dt * self.particles[env_id, pf_next, p].v

    def agent_move(self, f, is_none_action):
        if not is_none_action:
            self.agent.move(f)

    def agent_move_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.move_grad(f)

    def substep(self, f, is_none_action):
        if self.has_particles:
            self.reset_grid_and_grad(f)
            self.advect_used(f)
            self.process_unused_particles(f)

        self.agent_act(f, is_none_action)

        if self.has_particles:
            self.compute_F_tmp(f)
            self.svd(f)
            self.p2g(f)

        self.agent_move(f, is_none_action)

        if self.has_particles:
            self.grid_op(f)
            self.g2p(f)
            self.advect(f)

    def substep_grad(self, f, is_none_action):
        if self.has_particles:
            self.advect_grad(f)
            self.g2p.grad(f)
            self.grid_op.grad(f)

        self.agent_move_grad(f, is_none_action)

        if self.has_particles:
            self.p2g.grad(f)
            self.svd_grad(f)
            self.compute_F_tmp.grad(f)

        self.agent_act_grad(f, is_none_action)

        if self.has_particles:
            self.process_unused_particles.grad(f)
            self.advect_used.grad(f)

    # ------------------------------------ io -------------------------------------#
    @ti.kernel
    def readframe(self, env_id: ti.i32, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), used: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[env_id, pf, i].x[j]
                v[i, j] = self.particles[env_id, pf, i].v[j]
                for k in ti.static(range(self.dim)):
                    C[i, j, k] = self.particles[env_id, pf, i].C[j, k]
                    F[i, j, k] = self.particles[env_id, pf, i].F[j, k]
            used[i] = self.particles_ng[env_id, pf, i].used

    @ti.kernel
    def readframe_all_envs(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), used: ti.types.ndarray()):
        """Read frame for all environments. Arrays shape: (num_envs, n_particles, ...)"""
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            for j in ti.static(range(self.dim)):
                x[env_id, i, j] = self.particles[env_id, pf, i].x[j]
                v[env_id, i, j] = self.particles[env_id, pf, i].v[j]
                for k in ti.static(range(self.dim)):
                    C[env_id, i, j, k] = self.particles[env_id, pf, i].C[j, k]
                    F[env_id, i, j, k] = self.particles[env_id, pf, i].F[j, k]
            used[env_id, i] = self.particles_ng[env_id, pf, i].used

    @ti.kernel
    def setframe(self, env_id: ti.i32, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), used: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[env_id, pf, i].x[j] = x[i, j]
                self.particles[env_id, pf, i].v[j] = v[i, j]
                for k in ti.static(range(self.dim)):
                    self.particles[env_id, pf, i].C[j, k] = C[i, j, k]
                    self.particles[env_id, pf, i].F[j, k] = F[i, j, k]
            self.particles_ng[env_id, pf, i].used = used[i]

    @ti.kernel
    def setframe_all_envs(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), used: ti.types.ndarray()):
        """Set frame for all environments. Arrays shape: (num_envs, n_particles, ...)"""
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[env_id, pf, i].x[j] = x[env_id, i, j]
                self.particles[env_id, pf, i].v[j] = v[env_id, i, j]
                for k in ti.static(range(self.dim)):
                    self.particles[env_id, pf, i].C[j, k] = C[env_id, i, j, k]
                    self.particles[env_id, pf, i].F[j, k] = F[env_id, i, j, k]
            self.particles_ng[env_id, pf, i].used = used[env_id, i]

    @ti.kernel
    def set_x(self, env_id: ti.i32, f:ti.i32, x: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[env_id, pf, i].x[j] = x[i, j]

    @ti.kernel
    def set_used(self, env_id: ti.i32, f:ti.i32, used: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            self.particles_ng[env_id, pf, i].used = used[i]

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        src_pf = source if ti.static(self.enable_grad) else source % 2
        tgt_pf = target if ti.static(self.enable_grad) else target % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            self.particles[env_id, tgt_pf, i].x = self.particles[env_id, src_pf, i].x
            self.particles[env_id, tgt_pf, i].v = self.particles[env_id, src_pf, i].v
            self.particles[env_id, tgt_pf, i].F = self.particles[env_id, src_pf, i].F
            self.particles[env_id, tgt_pf, i].C = self.particles[env_id, src_pf, i].C
            self.particles_ng[env_id, tgt_pf, i].used = self.particles_ng[env_id, src_pf, i].used

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        src_pf = source if ti.static(self.enable_grad) else source % 2
        tgt_pf = target if ti.static(self.enable_grad) else target % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            self.particles.grad[env_id, tgt_pf, i].x = self.particles.grad[env_id, src_pf, i].x
            self.particles.grad[env_id, tgt_pf, i].v = self.particles.grad[env_id, src_pf, i].v
            self.particles.grad[env_id, tgt_pf, i].F = self.particles.grad[env_id, src_pf, i].F
            self.particles.grad[env_id, tgt_pf, i].C = self.particles.grad[env_id, src_pf, i].C
            self.particles_ng[env_id, tgt_pf, i].used = self.particles_ng[env_id, src_pf, i].used

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for env_id, i, j in ti.ndrange(self.num_envs, f, self.n_particles):
            self.particles.grad[env_id, i, j].fill(0)

    def get_state(self, env_id=0):
        """Get state for a specific environment."""
        f = self.cur_substep_local
        s = self.cur_step_local

        state = {}

        if self.has_particles:
            state['x']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['C']    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['F']    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.n_particles,), dtype=np.int32)
            self.readframe(env_id, f, state['x'], state['v'], state['C'], state['F'], state['used'])

        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)

        if self.smoke_field is not None:
            state['smoke_field'] = self.smoke_field.get_state(s)

        return state

    def get_state_all_envs(self):
        """Get state for all environments."""
        f = self.cur_substep_local
        s = self.cur_step_local

        state = {}

        if self.has_particles:
            state['x']    = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
            state['C']    = np.zeros((self.num_envs, self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['F']    = np.zeros((self.num_envs, self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.num_envs, self.n_particles,), dtype=np.int32)
            self.readframe_all_envs(f, state['x'], state['v'], state['C'], state['F'], state['used'])

        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)

        if self.smoke_field is not None:
            state['smoke_field'] = self.smoke_field.get_state(s)

        return state

    def set_state(self, f_global, state, env_id=0):
        """Set state for a specific environment."""
        f = self.f_global_to_f_local(f_global)
        s = self.f_global_to_s_local(f_global)

        if self.has_particles:
            self.setframe(env_id, f, state['x'], state['v'], state['C'], state['F'], state['used'])

        if self.agent is not None:
            self.agent.set_state(f, state['agent'])

        if self.smoke_field is not None:
            self.smoke_field.set_state(s, state['smoke_field'])

    @ti.kernel
    def get_x_kernel(self, env_id: ti.i32, f: ti.i32, x: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[env_id, pf, i].x[j]

    @ti.kernel
    def get_x_all_envs_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        """Get x for all environments. Shape: (num_envs, n_particles, dim)"""
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            for j in ti.static(range(self.dim)):
                x[env_id, i, j] = self.particles[env_id, pf, i].x[j]

    @ti.kernel
    def get_used_kernel(self, env_id: ti.i32, f: ti.i32, used: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            used[i] = self.particles_ng[env_id, pf, i].used

    @ti.kernel
    def get_used_all_envs_kernel(self, f: ti.i32, used: ti.types.ndarray()):
        """Get used for all environments. Shape: (num_envs, n_particles)"""
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            used[env_id, i] = self.particles_ng[env_id, pf, i].used

    def get_x(self, f=None, env_id=0):
        if f is None:
            f = self.cur_substep_local

        x = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_x_kernel(env_id, f, x)
        return x

    def get_x_all_envs(self, f=None):
        """Get x for all environments."""
        if f is None:
            f = self.cur_substep_local

        x = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_x_all_envs_kernel(f, x)
        return x

    def get_used(self, f=None, env_id=0):
        if f is None:
            f = self.cur_substep_local

        used = np.zeros((self.n_particles), dtype=np.int32)
        if self.has_particles:
            self.get_used_kernel(env_id, f, used)
        return used

    def get_used_all_envs(self, f=None):
        """Get used for all environments."""
        if f is None:
            f = self.cur_substep_local

        used = np.zeros((self.num_envs, self.n_particles), dtype=np.int32)
        if self.has_particles:
            self.get_used_all_envs_kernel(f, used)
        return used

    @ti.kernel
    def get_state_RL_kernel(self, env_id: ti.i32, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), used: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[env_id, pf, i].x[j]
                v[i, j] = self.particles[env_id, pf, i].v[j]
            used[i] = self.particles_ng[env_id, pf, i].used

    @ti.kernel
    def get_state_RL_all_envs_kernel(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), used: ti.types.ndarray()):
        """Get RL state for all environments. Shape: (num_envs, n_particles, ...)"""
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            for j in ti.static(range(self.dim)):
                x[env_id, i, j] = self.particles[env_id, pf, i].x[j]
                v[env_id, i, j] = self.particles[env_id, pf, i].v[j]
            used[env_id, i] = self.particles_ng[env_id, pf, i].used

    def get_state_RL(self, env_id=0):
        f = self.cur_substep_local
        s = self.cur_step_local
        state = {}
        if self.has_particles:
            state['x']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.n_particles,), dtype=np.int32)
            self.get_state_RL_kernel(env_id, f, state['x'], state['v'], state['used'])
        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)
        if self.smoke_field is not None:
            state['smoke_field'] = self.smoke_field.get_state(s)
        return state

    def get_state_RL_all_envs(self):
        """Get RL state for all environments."""
        f = self.cur_substep_local
        s = self.cur_step_local
        state = {}
        if self.has_particles:
            state['x']    = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.num_envs, self.n_particles,), dtype=np.int32)
            self.get_state_RL_all_envs_kernel(f, state['x'], state['v'], state['used'])
        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)
        if self.smoke_field is not None:
            state['smoke_field'] = self.smoke_field.get_state(s)
        return state

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        """Get render state for all environments - merged into single buffer for rendering."""
        pf = f if ti.static(self.enable_grad) else f % 2
        for env_id, i in ti.ndrange(self.num_envs, self.n_particles):
            global_idx = env_id * self.n_particles + i
            for j in ti.static(range(self.dim)):
                self.particles_render[global_idx].x[j] = ti.cast(self.particles[env_id, pf, i].x[j], ti.f32)
            self.particles_render[global_idx].used = ti.cast(self.particles_ng[env_id, pf, i].used, ti.i32)

    def get_state_render(self, f):
        if self.particles_render is None:
            return None  # Headless mode - no render buffer
        self.get_state_render_kernel(f)
        return self.particles_render

    @ti.kernel
    def get_v_kernel(self, env_id: ti.i32, f: ti.i32, v: ti.types.ndarray()):
        pf = f if ti.static(self.enable_grad) else f % 2
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v[i, j] = self.particles[env_id, pf, i].v[j]

    def get_v(self, f, env_id=0):
        v = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_v_kernel(env_id, f, v)
        return v

    def step(self, action=None):
        if self.grad_enabled:
            if self.cur_substep_local == 0:
                self.actions_buffer = []

        self.step_(action)

        if self.grad_enabled:
            self.actions_buffer.append(action)

        if self.cur_substep_local == 0:
            self.memory_to_cache()


    def step_(self, action=None):
        is_none_action = action is None
        if not is_none_action:
            self.agent.set_action(
                s=self.cur_step_local,
                s_global=self.cur_step_global,
                n_substeps=self.n_substeps,
                action=action
            )

        # smoke simulates at step level, not substep
        if self.smoke_field is not None:
            self.smoke_field.step(s=self.cur_step_local, f=self.cur_substep_local)

        for i in range(0, self.n_substeps):
            self.substep(self.cur_substep_local, is_none_action)
            self.cur_substep_global += 1

        assert self.cur_substep_global <= self.max_substeps_global

    def step_grad(self, action=None):     
        if self.cur_substep_local == 0:
            self.memory_from_cache()

        is_none_action = action is None

        for i in range(self.n_substeps-1, -1, -1):
            self.cur_substep_global -= 1
            self.substep_grad(self.cur_substep_local, is_none_action)

        # smoke simulates at step level, not substep
        if self.smoke_field is not None:
            self.smoke_field.step_grad(s=self.cur_step_local, f=self.cur_substep_local)

        if not is_none_action:
            self.agent.set_action_grad(
                s=self.cur_substep_local//self.n_substeps,
                s_global=self.cur_substep_global//self.n_substeps, 
                n_substeps=self.n_substeps,
                action=action
            )

    def memory_to_cache(self):
        if self.grad_enabled:
            ckpt_start_step = self.cur_substep_global - self.max_substeps_local
            ckpt_end_step = self.cur_substep_global - 1
            ckpt_name = f'{ckpt_start_step:06d}'

            if self.ckpt_dest == 'disk':
                ckpt = {}
                if self.has_particles:
                    self.readframe_all_envs(0, self.x_np, self.v_np, self.C_np, self.F_np, self.used_np)
                    ckpt['x']       = self.x_np.copy()
                    ckpt['v']       = self.v_np.copy()
                    ckpt['C']       = self.C_np.copy()
                    ckpt['F']       = self.F_np.copy()
                    ckpt['used']    = self.used_np.copy()
                    ckpt['actions'] = self.actions_buffer

                if self.smoke_field is not None:
                    ckpt['smoke_field'] = self.smoke_field.get_ckpt()

                if self.agent is not None:
                    ckpt['agent'] = self.agent.get_ckpt()

                # save to /tmp
                ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_name}.pkl')
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
                pkl.dump(ckpt, open(ckpt_file, 'wb'))

            elif self.ckpt_dest in ['cpu', 'gpu']:
                if not ckpt_name in self.ckpt_ram:
                    self.ckpt_ram[ckpt_name] = {}
                    
                    if self.ckpt_dest == 'cpu':
                        device = 'cpu'
                    elif self.ckpt_dest == 'gpu':
                        device = 'cuda'
                    if self.has_particles:
                        # With num_envs dimension
                        self.ckpt_ram[ckpt_name]['x']    = torch.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['v']    = torch.zeros((self.num_envs, self.n_particles, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['C']    = torch.zeros((self.num_envs, self.n_particles, self.dim, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['F']    = torch.zeros((self.num_envs, self.n_particles, self.dim, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['used'] = torch.zeros((self.num_envs, self.n_particles,), dtype=torch.int32, device=device)

                if self.has_particles:
                    self.readframe_all_envs(
                        0,
                        self.ckpt_ram[ckpt_name]['x'],
                        self.ckpt_ram[ckpt_name]['v'],
                        self.ckpt_ram[ckpt_name]['C'],
                        self.ckpt_ram[ckpt_name]['F'],
                        self.ckpt_ram[ckpt_name]['used'],
                    )

                self.ckpt_ram[ckpt_name]['actions'] = list(self.actions_buffer)

                if self.smoke_field is not None:
                    self.smoke_field.get_ckpt(ckpt_name)

                if self.agent is not None:
                    self.agent.get_ckpt(ckpt_name)

            else:
                assert False

            # print(f'[Forward] Cached step {ckpt_start_step} to {ckpt_end_step}. {t2-t1:.2f}s {t3-t2:.2f}s')

        # restart from frame 0 in memory
        if self.has_particles:
            self.copy_frame(self.max_substeps_local, 0)

        if self.smoke_field is not None:
            self.smoke_field.copy_frame(self.max_steps_local, 0)

        if self.agent is not None:
            self.agent.copy_frame(self.max_substeps_local, 0)

        # print(f'[Forward] Memory refreshed. Now starts from global step {self.cur_substep_global}.')

    def memory_from_cache(self):
        assert self.grad_enabled
        if self.has_particles:
            self.copy_frame(0, self.max_substeps_local)
            self.copy_grad(0, self.max_substeps_local)
            self.reset_grad_till_frame(self.max_substeps_local)
            
        if self.smoke_field is not None:
            self.smoke_field.copy_frame(0, self.max_steps_local)
            self.smoke_field.copy_grad(0, self.max_steps_local)
            self.smoke_field.reset_grad_till_frame(self.max_steps_local)

        if self.agent is not None:
            self.agent.copy_frame(0, self.max_substeps_local)
            self.agent.copy_grad(0, self.max_substeps_local)
            self.agent.reset_grad_till_frame(self.max_substeps_local)

        ckpt_start_step = self.cur_substep_global - self.max_substeps_local
        ckpt_end_step = self.cur_substep_global - 1
        ckpt_name = f'{ckpt_start_step:06d}'

        if self.ckpt_dest == 'disk':
            ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_start_step:06d}.pkl')
            assert os.path.exists(ckpt_file)
            ckpt = pkl.load(open(ckpt_file, 'rb'))

            if self.has_particles:
                self.setframe_all_envs(0, ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'], ckpt['used'])


            if self.smoke_field is not None:
                self.smoke_field.set_ckpt(ckpt=ckpt['smoke_field'])

            if self.agent is not None:
                self.agent.set_ckpt(ckpt=ckpt['agent'])

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if self.has_particles:
                ckpt = self.ckpt_ram[ckpt_name]
                self.setframe_all_envs(0, ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'], ckpt['used'])

            if self.smoke_field is not None:
                self.smoke_field.set_ckpt(ckpt_name=ckpt_name)

            if self.agent is not None:
                self.agent.set_ckpt(ckpt_name=ckpt_name)

        else:
            assert False

        # now that we loaded the first frame, we do a forward pass to fill up the rest 
        self.cur_substep_global = ckpt_start_step
        for action in ckpt['actions']:
            self.step_(action)

        # print(f'[Backward] Loading step {ckpt_start_step} to {ckpt_end_step} from cache. {t2-t1:.2f}s {t3-t2:.2f}s {t4-t3:.2f}s')
        # print(f'[Backward] Memory reloaded. Now starts from global step {ckpt_start_step}.')

    # --------------------------------- Partial Reset for Vectorized Envs -----------------------------------
    @ti.kernel
    def reset_env_particles_kernel(self, env_id: ti.i32, f: ti.i32, 
                                    init_x: ti.types.ndarray(), init_used: ti.types.ndarray()):
        """
        Reset particles for a specific environment.
        CRITICAL: Must reset x, v, F, C to avoid explosion from accumulated deformation.
        """
        pf = f if ti.static(self.enable_grad) else f % 2
        for p in range(self.n_particles):
            # 1. Reset position to initial
            for d in ti.static(range(self.dim)):
                self.particles[env_id, pf, p].x[d] = init_x[p, d]
            # 2. Reset velocity to zero
            self.particles[env_id, pf, p].v = ti.Vector.zero(DTYPE_TI, self.dim)
            # 3. Reset deformation gradient F to identity (CRITICAL - prevents explosion)
            self.particles[env_id, pf, p].F = ti.Matrix.identity(DTYPE_TI, self.dim)
            # 4. Reset affine velocity field C to zero
            self.particles[env_id, pf, p].C = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            # 5. Reset F_tmp, U, S, V
            self.particles[env_id, pf, p].F_tmp = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles[env_id, pf, p].U = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles[env_id, pf, p].S = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles[env_id, pf, p].V = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            # 6. Reset used flag
            self.particles_ng[env_id, pf, p].used = init_used[p]

    @ti.kernel
    def reset_env_grid_kernel(self, env_id: ti.i32, f: ti.i32):
        """Reset grid for a specific environment to prevent residual momentum."""
        grid_f = f if ti.static(self.enable_grad) else 0
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.grid[env_id, grid_f, I].fill(0)

    @ti.kernel
    def reset_env_bodies_kernel(self, env_id: ti.i32):
        """Reset bodies for a specific environment."""
        for body_id in range(self.n_bodies):
            self.bodies[env_id, body_id].fill(0)

    def reset_envs(self, env_ids, init_x, init_used):
        """
        Reset specific environments to initial state.
        
        Args:
            env_ids: array of environment indices to reset
            init_x: initial particle positions, shape (n_particles, dim)
            init_used: initial used flags, shape (n_particles,)
        """
        f = self.cur_substep_local
        
        for env_id in env_ids:
            self.reset_env_particles_kernel(env_id, f, init_x, init_used)
            self.reset_env_grid_kernel(env_id, f)
            if self.n_bodies > 0:
                self.reset_env_bodies_kernel(env_id)

    def reset_all_envs(self, init_x, init_used):
        """Reset all environments to initial state."""
        self.reset_envs(np.arange(self.num_envs), init_x, init_used)
