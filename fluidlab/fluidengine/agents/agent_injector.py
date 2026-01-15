import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *

@ti.data_oriented
class AgentInjector(Agent):
    # Agent with one Injector

    def __init__(self, **kwargs):
        super(AgentInjector, self).__init__(**kwargs)
        
    def build(self, sim):
        super(AgentInjector, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Injector)
        self.injector = self.effectors[0]

        self.injector.set_act_range(self.sim.particles_ng.used.to_numpy()[0])

    def act(self, batch_id, f, f_global):
        self.act_kernel(batch_id, f, f_global)
        self.check_act_range(batch_id, f)

    def act_grad(self, batch_id, f, f_global):
        self.act_kernel.grad(batch_id, f, f_global)

    @ti.kernel
    def act_kernel(self, batch_id: ti.i32, f: ti.i32, f_global: ti.i32):
        self.injector.act(batch_id, f, f_global, self.sim.particles_ng, self.sim.particles)

    @ti.func
    def collide(self, batch_id, f, pos_world, mat_v, dt):
        return mat_v

    def check_act_range(self, batch_id, f):
        assert self.injector.act_id[batch_id, f+1] <= self.injector.act_range.shape[0], 'too many particles added'
