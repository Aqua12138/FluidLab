import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.configs.macros import *
from fluidlab.fluidengine.effectors import *

@ti.data_oriented
class AgentPouring(Agent):
    # Agent with one Rigid and a collector
    
    def __init__(self, collector_boundary, **kwargs):
        super(AgentPouring, self).__init__(collide_type='both', **kwargs)
        self.collector_boundary = create_boundary(**collector_boundary)
        self.nowhere = ti.Vector(NOWHERE, dt=DTYPE_TI)

    def build(self, sim):
        super(AgentPouring, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Rigid)
        self.rigid = self.effectors[0]

    def act(self, batch_id, f, f_global):
        self.collector_act_kernel(batch_id, f, f_global)

    def act_grad(self, batch_id, f, f_global):
        self.collector_act_kernel.grad(batch_id, f, f_global)

    @ti.kernel
    def collector_act_kernel(self, batch_id: ti.i32, f: ti.i32, f_global: ti.i32):
        # collect out-of-boundary particles
        for p in range(self.sim.n_particles):
            # if self.sim.particles_ng[batch_id, f, p].used and self.sim.particles_i[p].mat == WATER:
            if self.sim.particles_ng[batch_id, f, p].used:
                if self.collector_boundary.is_out(self.sim.particles[batch_id, f, p].x):
                    self.sim.particles_ng[batch_id, f+1, p].used = 0
                    self.sim.particles[batch_id, f+1, p].x = self.nowhere
                    
                    # set current step used to false to avoid later options in simulator's substep()
                    self.sim.particles_ng[batch_id, f, p].used = 0
                    
    @ti.func
    def collide(self, batch_id, f, pos_world, mat_v, dt):
        return self.rigid.collide(batch_id, f, pos_world, mat_v, dt)
