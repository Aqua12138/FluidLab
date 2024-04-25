import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *
from fluidlab.fluidengine.sensor import *


@ti.data_oriented
class AgentSensor(Agent):
    # Agent with one Rigid
    def __init__(self, **kwargs):
        super(AgentSensor, self).__init__(**kwargs)
    def build(self, sim):
        super(AgentSensor, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Rigid)
        self.rigid = self.effectors[0]
        self.sensors = []
    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        return self.rigid.collide(f, pos_world, mat_v, dt)
    def reset_grad(self):
        for i in range(self.n_effectors):
            self.effectors[i].reset_grad()
        self.sensors[0].reset_grad()

    def get_obs(self):
        sensor_obs = []
        for sensor in self.sensors:
            sensor_obs.append(sensor.get_obs())
        return sensor_obs

    def add_sensor(self, sensor_handle, sensor_cfg=None):
        sensor = sensor_handle(**sensor_cfg, AgentGameObject=self, sim=self.sim)
        self.sensors.append(sensor)

    def set_next_vector_grad(self, grad):
        self.effectors[0].set_next_state_grad(self.sim.cur_substep_local, grad)

    def set_next_grid3d_grad(self, grad):
        self.sensors[0].set_next_state_grad(self.sim.cur_substep_local, grad)

    def set_state(self, f, state):
        for i in range(self.n_effectors):
            self.effectors[i].set_state(f, state[i])
        for sensor in self.sensors :
            if isinstance(sensor, GridSensor3DGrad):
                sensor.reset()
                sensor.set_obs(f)
    def get_state(self, f):
        out = []
        for i in range(self.n_effectors):
            out.append(self.effectors[i].get_state(f))
        return out

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            out = []
            for effector in self.effectors:
                out.append(effector.get_ckpt())
            for sensor in self.sensors:
                if isinstance(sensor, GridSensor3DGrad):
                    out.append(sensor.get_ckpt())
            return out

        elif self.ckpt_dest in ['cpu', 'gpu']:
            for effector in self.effectors:
                effector.get_ckpt(ckpt_name)
            for sensor in self.sensors:
                if isinstance(sensor, GridSensor3DGrad):
                    sensor.get_ckpt(ckpt_name)

    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            for effector, ckpt_effector in zip(self.effectors, ckpt):
                effector.set_ckpt(ckpt=ckpt_effector)
            self.sensors[0].set_ckpt(ckpt=ckpt[1])

        elif self.ckpt_dest in ['cpu', 'gpu']:
            for effector in self.effectors:
                effector.set_ckpt(ckpt_name=ckpt_name)
            self.sensors[0].set_ckpt(ckpt_name=ckpt_name)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i in ti.static(range(self.n_effectors)):
            self.effectors[i].copy_frame(source, target)
        self.sensors[0].copy_frame(source, target)

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i in ti.static(range(self.n_effectors)):
            self.effectors[i].copy_grad(source, target)
        self.sensors[0].copy_grad(source, target)

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for i in ti.static(range(self.n_effectors)):
            self.effectors[i].reset_grad_till_frame(f)
        self.sensors[0].reset_grad_till_frame(f)