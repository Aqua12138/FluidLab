import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion

@ti.data_oriented
class VectorSensor:
    def __init__(self, AgentGameObject, device):
        self.AgentGameObject = AgentGameObject
        self.device = device
    def get_obs(self):
        agent_state = self.AgentGameObject.effectors[0].get_state(self.AgentGameObject.sim.cur_substep_local)
        body_id = self.AgentGameObject.sim.get_state()['body_id']
        body_x = self.AgentGameObject.sim.get_state()['x']
        indices = np.where(body_id == 1)[0]
        rigid_x = body_x[indices].reshape(-1)
        # print([state[0], state[1], 1 - state[2], -state[4], -state[5], state[6], state[3]])
        return torch.tensor([agent_state[0], agent_state[1], agent_state[2], agent_state[3], agent_state[4], agent_state[5], agent_state[6]], dtype=torch.float32, device=self.device)
