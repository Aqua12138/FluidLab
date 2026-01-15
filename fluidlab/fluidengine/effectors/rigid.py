import taichi as ti
import numpy as np
import yaml
import torch
from .effector import Effector
from fluidlab.fluidengine.meshes import Dynamic
from fluidlab.configs.macros import *

@ti.data_oriented
class Rigid(Effector):
    # Rigid end-effector. Can be a stirrer, ice-cream cone, laddle, whatever...
    def __init__(self, **kwargs):
        super(Rigid, self).__init__(**kwargs)
        self.mesh = None

        # magic, don't touch. (I finally got a chance to write something like this lol, thanks to stupid taichi)
        self.magic = ti.field(dtype=DTYPE_TI, shape=())
        self.magic.fill(0)

    def setup_mesh(self, **kwargs):
        self.mesh = Dynamic(
            container=self,
            has_dynamics=True,
            **kwargs
        )

    def move(self, batch_id, f):
        self.move_kernel(batch_id, f)
        self.update_latest_pos(batch_id, f)
        self.update_mesh_pose(batch_id, f)
        
    def update_mesh_pose(self, batch_id, f):
        # For visualization only. No need to compute grad.
        self.mesh.update_vertices(batch_id, f)

    @ti.func
    def collide(self, batch_id, f, pos_world, mat_v, dt):
        return self.mesh.collide(batch_id, f, pos_world, mat_v, dt)
