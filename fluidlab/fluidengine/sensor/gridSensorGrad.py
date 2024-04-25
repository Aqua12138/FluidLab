import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridSensor import GridSensor
from fluidlab.utils.geom import quaternion_to_rotation_matrix

@ti.data_oriented
class GridSensor3DGrad(GridSensor):
    def __init__(self, SensorName, AgentGameObject, ObservationStacks, CellArc, LatAngleNorth, LatAngleSouth, LonAngle,
                 MaxDistance, MinDistance, DistanceNormalization, n_particles, device, sim, ckpt_dest,):
        super(GridSensor3DGrad, self).__init__()
        '''
        SensorName: 传感器名字
        CellScale: 网格尺寸
        GridSize: 网格检测范围（cellArc latAngleSouth latAngleNorth LonAngle maxDistance minDistance DistanceNormalization）
        RotateWithAgent: 是否随Agent旋转
        AgentGameObject: Agent
        AgentID: effetor ID
        DetectableTags: 检测物体body tuple
        MaxColliderBufferSize: 最大检测物数量
        DebugColors: 颜色显示，用于debug
        GizmoZOffset: 沿着Z偏移的尺寸
        ObservationStacks: 时间维度堆叠数量
        DataType: 数据类型 目前支持one-hot

        '''
        # Geometry
        self.SensorName = SensorName
        self.m_AgentGameObject = AgentGameObject
        self.m_ObservationStacks = ObservationStacks
        self.m_CellArc = CellArc
        self.m_LatAngleNorth = LatAngleNorth
        self.m_LatAngleSouth = LatAngleSouth
        self.m_LonAngle = LonAngle
        self.m_MaxDistance = MaxDistance
        self.m_MinDistance = MinDistance
        self.m_DistanceNormalization = DistanceNormalization
        self.dim = 3
        self.n_particles = n_particles
        self.n_bodies = self.m_AgentGameObject.sim.n_bodies
        self.device = device
        # self.agent_groups = self.m_AgentGameObject.sim.agent_groups
        self.dynamics = []
        self.sim = sim
        self.ckpt_dest = ckpt_dest
        particle_state = ti.types.struct(
            relative_x=ti.types.vector(self.dim, DTYPE_TI),
            rotated_x=ti.types.vector(self.dim, DTYPE_TI),
            latitudes=DTYPE_TI,
            longitudes=DTYPE_TI,
            distance=DTYPE_TI
        )

        grid_sensor = ti.types.struct(
            distance=DTYPE_TI
        )


        self.particle_state = particle_state.field(shape=(self.sim.max_substeps_local+1, self.n_particles,), needs_grad=True,
                                                         layout=ti.Layout.SOA)
        self.grid_sensor = grid_sensor.field(shape=(self.sim.max_substeps_local+1, (self.m_LonAngle // self.m_CellArc) * 2,
                     (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc, self.n_bodies), needs_grad=True, layout=ti.Layout.SOA)
        self.init_ckpt()

    def init_ckpt(self):
        if self.ckpt_dest == 'disk':
            self.grid_sensor_np = np.zeros(((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies), dtype=DTYPE_NP)
        elif self.ckpt_dest in ['cpu', 'gpu']:
            self.ckpt_ram = dict()
    def setup_dynamic_mesh(self):
        # 把所有非自身的agent.rigid对象提取出来
        for group in self.agent_groups:
            for agent in group.agents:
                if agent != self.m_AgentGameObject:
                    self.dynamics.append(agent.rigid.mesh)
    @ti.kernel
    def transform_point_particle(self, f: ti.i32):
        # 计算point相对agent位置
        for i in range(self.n_particles):
            self.particle_state[f, i].relative_x[0] = self.sim.particles[f, i].x[0] - self.m_AgentGameObject.effectors[0].pos[f][0]
            self.particle_state[f, i].relative_x[1] = self.sim.particles[f, i].x[1] - self.m_AgentGameObject.effectors[0].pos[f][1]
            self.particle_state[f, i].relative_x[2] = self.sim.particles[f, i].x[2] - self.m_AgentGameObject.effectors[0].pos[f][2]

            # 获取四元数数据
            a = self.m_AgentGameObject.effectors[0].quat[f][0]
            b = -self.m_AgentGameObject.effectors[0].quat[f][1]
            c = -self.m_AgentGameObject.effectors[0].quat[f][2]
            d = -self.m_AgentGameObject.effectors[0].quat[f][3]
            rotation_matrix = quaternion_to_rotation_matrix(a, b, c, d)
            self.particle_state[f, i].rotated_x = rotation_matrix @ self.particle_state[f, i].relative_x

    @ti.kernel
    def compute_lat_lon_particle(self, f: ti.i32):
        for i in range(self.n_particles):
            # 提取局部坐标系中的坐标
            x = self.particle_state[f, i].rotated_x[0]
            y = self.particle_state[f, i].rotated_x[1]
            z = self.particle_state[f, i].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.particle_state[f, i].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = y / self.particle_state[f, i].distance
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)

            self.particle_state[f, i].latitudes = lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.particle_state[f, i].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def normal_distance_particle(self, f: ti.i32):
        # 清空
        # 1. 判断距离是否在球体内
        for p in range(self.n_particles):
            if self.particle_state[f, p].distance < self.m_MaxDistance and self.particle_state[f, p].distance > self.m_MinDistance:
                # 2. 判断经度范围和纬度范围
                if (90 - self.particle_state[f, p].latitudes < self.m_LatAngleNorth and 90 - self.particle_state[f, p].latitudes >= 0) or \
                        (ti.abs(self.particle_state[f, p].latitudes - 90) < self.m_LatAngleSouth and ti.abs(self.particle_state[f, p].latitudes - 90) >= 0):
                    if ti.abs(self.particle_state[f, p].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.particle_state[f, p].distance - self.m_MinDistance) / (self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                        self.m_DistanceNormalization + 1)
                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.particle_state[f, p].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                 (self.particle_state[f, p].latitudes - (90 - self.m_LatAngleNorth)) / self.m_CellArc),
                            ti.i32)

                        # 使用 atomic_max 更新 normal_distance 的值
                        ti.atomic_max(self.grid_sensor[f, longitude_index, latitude_index, self.sim.particles_i[p].body_id].distance, normal_d)
    @ti.kernel
    def clear_gird_sensor(self, f: ti.i32):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor[f, i, j, k].distance = 0
    @ti.kernel
    def get_sensor_data_kernel(self, f: ti.i32, grid_sensor: ti.types.ndarray()):
        # 这里假设 output 已经是一个正确维度和类型的 Taichi field
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            grid_sensor[i, j, k] = self.grid_sensor[f, i, j, k].distance
    def step(self, f):
        # particle
        self.transform_point_particle(f)
        self.compute_lat_lon_particle(f)
        self.clear_gird_sensor(f)
        self.normal_distance_particle(f)

    def step_grad(self, f):
        self.normal_distance_particle.grad(f)
        self.clear_gird_sensor.grad(f)
        self.compute_lat_lon_particle.grad(f)
        self.transform_point_particle.grad(f)

        self.debug(f)
    @ti.kernel
    def debug(self, f: ti.i32):
        print(self.sim.particles.grad[f, 0].x[0], self.m_AgentGameObject.effectors[0].pos.grad[f][0], self.particle_state.grad[f, 0].relative_x[0])
        print(self.sim.particles[f, 0].x[0], self.m_AgentGameObject.effectors[0].pos[f][0], self.particle_state[f, 0].relative_x[0])
    def get_obs(self):
        grid_sensor = torch.zeros(((self.m_LonAngle // self.m_CellArc) * 2,
                                   (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc, self.n_bodies), dtype=torch.float32, device=self.device)
        self.get_sensor_data_kernel(self.sim.cur_substep_local, grid_sensor)
        return grid_sensor

    @ti.kernel
    def set_next_state_grad(self, f: ti.i32, grad: ti.types.ndarray()):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor.grad[f, i, j, k].distance = grad[i, j, k]

    @ti.kernel
    def get_ckpt_kernel(self, grid_sensor_np: ti.types.ndarray()):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            grid_sensor_np[i, j, k] = self.grid_sensor[0, i, j, k].distance
    @ti.kernel
    def set_ckpt_kernel(self, grid_sensor_np: ti.types.ndarray()):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor[0, i, j, k].distance = grid_sensor_np[i, j, k]

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            ckpt = {
                'grid_sensor3d'  : self.grid_sensor_np,
            }
            self.get_ckpt_kernel(self.grid_sensor_np)
            return ckpt

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if not ckpt_name in self.ckpt_ram:
                if self.ckpt_dest == 'cpu':
                    device = 'cpu'
                elif self.ckpt_dest == 'gpu':
                    device = 'cuda'
                self.ckpt_ram[ckpt_name] = {
                    'grid_sensor3d': torch.zeros(((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies), dtype=DTYPE_TC, device=device),
                }
            self.get_ckpt_kernel(
                self.ckpt_ram[ckpt_name]['grid_sensor3d'],
            )
    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            assert ckpt is not None
        elif self.ckpt_dest in ['cpu', 'gpu']:
            ckpt = self.ckpt_ram[ckpt_name]
        self.set_ckpt_kernel(ckpt['grid_sensor3d'])

    @ti.func
    def copy_frame(self, source, target):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor[target, i, j, k] = self.grid_sensor[source, i, j, k]

    @ti.func
    def copy_grad(self, source, target):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor.grad[target, i, j, k] = self.grid_sensor.grad[source, i, j, k]

    @ti.func
    def reset_grad_till_frame(self, f):
        for t, i, j, k in ti.ndrange(f, (self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor.grad[t, i, j, k].fill(0)

    def reset_grad(self):
        self.particle_state.grad.fill(0)
        self.grid_sensor.grad.fill(0)

    def reset(self):
        self.particle_state.fill(0)
        self.particle_state.grad.fill(0)
        self.grid_sensor.fill(0)
        self.grid_sensor.grad.fill(0)




