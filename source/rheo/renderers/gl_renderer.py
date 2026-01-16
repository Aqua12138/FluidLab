import os
import cv2
import time
import skimage
import numpy as np
import taichi as ti
from time import time
from scipy import ndimage
from rheo.macros import *
from .gl_renderer_src import flex_renderer
import rheo.utils.geom as geom_utils
import rheo.utils.misc as misc_utils
from rheo.utils.misc import is_headless

class GLRenderer:
    # Default parameter values for detecting if user provided custom values
    _DEFAULT_CAMERA_POS = (0.5, 2.5, 3.5)
    _DEFAULT_CAMERA_LOOKAT = (0.5, 0.5, 0.5)
    _DEFAULT_LIGHT_POS = (0.5, 5.0, 0.5)
    _DEFAULT_LIGHT_LOOKAT = (0.5, 0.5, 0.49)
    _DEFAULT_FLOOR_HEIGHT = 0.0
    _DEFAULT_SCENE_RADIUS = 10.0
    _DEFAULT_CAMERA_FAR = 10.0
    
    def __init__(self, 
        res             = (960, 540),  # 与 RheoAgent 保持一致，提高默认分辨率
        camera_pos      = None,  # None means auto-calculate for multi-env
        camera_lookat   = None,
        camera_near     = 0.1,
        camera_far      = None,
        fov             = 30,
        particle_radius = 0.01,  # 默认值，不要设置太小（如 0.001）会导致颗粒感
        smoke_radius    = 0.01,
        render_particle = False,  # 默认使用流体渲染模式，减少颗粒感
        light_pos       = None,
        light_lookat    = None,
        light_fov       = 50,
        floor_height    = None,
        scene_radius    = None,
        cam_rotate_v    = 0.0,
        _smoothing      = 0.5,  # 默认平滑度
        num_envs        = 1,
        env_spacing     = 1.2,
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """
        Initialize GLRenderer with multi-environment support.
        
        Args:
            num_envs: Number of parallel environments to render
            env_spacing: World-space distance between environment centers
            camera_pos: Camera position. If None, auto-calculated for multi-env
            camera_lookat: Camera look-at point. If None, auto-calculated for multi-env
            light_pos: Light position. If None, auto-calculated for multi-env
            floor_height: Floor height. If None, auto-calculated for multi-env
            scene_radius: Scene radius. If None, auto-calculated for multi-env
        """
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        
        # Calculate grid layout (as close to square as possible)
        self.grid_cols = int(np.ceil(np.sqrt(num_envs)))
        self.grid_rows = int(np.ceil(num_envs / self.grid_cols))
        
        # Camera mode: 'overview' for all envs, 'follow' for single env
        self.camera_mode = 'overview' if num_envs > 1 else 'follow'
        self.follow_env_id = 0
        
        # Compute environment offsets
        self.env_offsets = self._compute_env_offsets()
        
        # Compute multi-env parameters first
        self._compute_multi_env_params()
        
        self.res = res
        
        # Apply camera settings: use provided values or auto-calculated
        if camera_pos is not None:
            self.camera_pos = np.array(camera_pos)
        # else: use self.camera_pos from _compute_multi_env_params
        
        if camera_lookat is not None:
            self.camera_lookat = np.array(camera_lookat)
        # else: use self.camera_lookat from _compute_multi_env_params
        
        self.camera_vec         = self.camera_pos - self.camera_lookat
        self.camera_init_xz_rad = np.arctan2(self.camera_vec[0], self.camera_vec[2])
        self.camera_angle       = geom_utils.compute_camera_angle(self.camera_pos, self.camera_lookat)
        self.camera_near        = camera_near
        
        # Camera far: use provided or auto-calculated
        if camera_far is not None:
            self.camera_far = camera_far
        # else: use self.camera_far from _compute_multi_env_params
        
        self.lights             = []
        self.render_particle    = render_particle
        self.fov                = fov / 180.0 * np.pi
        self.particle_radius    = particle_radius
        self.smoke_radius       = smoke_radius
        
        # Light settings: use provided or auto-calculated
        if light_pos is not None:
            self.light_pos = np.array(light_pos)
        # else: use self.light_pos from _compute_multi_env_params
        
        if light_lookat is not None:
            self.light_lookat = np.array(light_lookat)
        # else: use self.light_lookat from _compute_multi_env_params
        
        self.light_fov = light_fov
        
        # Floor and scene settings: use provided or auto-calculated
        if floor_height is not None:
            self.floor_height = floor_height
        # else: use self.floor_height from _compute_multi_env_params
        
        if scene_radius is not None:
            self.scene_radius = scene_radius
        # else: use self.scene_radius from _compute_multi_env_params
        
        self.cam_rotate_v = cam_rotate_v

        self._msaaSamples = 8
        self._anisotropy_scale = 1.0
        self._smoothing = _smoothing
        self._rendering_scale = 2.0
        self._fluid_rest_distance = 0.0125
        self._gl_color_gamma = 3.5

        self.aa = []

        flex_renderer.init(self.res[0], self.res[1], self._msaaSamples, self.fov)
    
    def _compute_multi_env_params(self):
        """Compute optimal camera, lights, floor, etc. for multi-environment rendering."""
        # Single environment center offset (default env center is at 0.5, 0.5, 0.5)
        env_center_offset = 0.5
        
        # Calculate grid center
        center_x = (self.grid_cols - 1) * self.env_spacing / 2 + env_center_offset
        center_z = (self.grid_rows - 1) * self.env_spacing / 2 + env_center_offset
        center_y = env_center_offset
        
        self._grid_center = np.array([center_x, center_y, center_z])
        
        # Calculate grid extent
        grid_width = (self.grid_cols - 1) * self.env_spacing
        grid_depth = (self.grid_rows - 1) * self.env_spacing
        grid_extent = np.sqrt(grid_width**2 + grid_depth**2)
        
        if self.num_envs > 1:
            # Multi-env: calculate optimal parameters
            # Camera height scales with grid size for good overview
            camera_height = 3.0 + grid_extent * 1.0
            camera_distance = 2.0 + grid_extent * 0.8
            
            self.camera_pos = np.array([center_x, camera_height, center_z + camera_distance])
            self.camera_lookat = np.array([center_x, center_y, center_z])
            
            # Light position: much higher above grid center for even illumination
            # Light should be high enough to cover entire grid without falloff
            light_height = camera_height*1.2
            self.light_pos = np.array([center_x, light_height, center_z])
            self.light_lookat = np.array([center_x+0.01, 0.0, center_z])
            
            # Floor height: below all environments
            self.floor_height = -0.1
            
            # Scene radius: cover entire grid with large margin for light coverage
            self.scene_radius = max(grid_extent * 3, 20.0)
            
            # Camera far: ensure all envs are visible
            self.camera_far = (camera_height + camera_distance) * 2
        else:
            # Single env: use default values
            self.camera_pos = np.array(self._DEFAULT_CAMERA_POS)
            self.camera_lookat = np.array(self._DEFAULT_CAMERA_LOOKAT)
            self.light_pos = np.array(self._DEFAULT_LIGHT_POS)
            self.light_lookat = np.array(self._DEFAULT_LIGHT_LOOKAT)
            self.floor_height = self._DEFAULT_FLOOR_HEIGHT
            self.scene_radius = self._DEFAULT_SCENE_RADIUS
            self.camera_far = self._DEFAULT_CAMERA_FAR
    
    def _compute_env_offsets(self):
        """Compute world-space offsets for each environment in grid layout."""
        offsets = np.zeros((self.num_envs, 3), dtype=np.float32)
        for env_id in range(self.num_envs):
            row = env_id // self.grid_cols
            col = env_id % self.grid_cols
            offsets[env_id] = [
                col * self.env_spacing,  # X direction
                0,                        # Y stays at ground
                row * self.env_spacing   # Z direction
            ]
        return offsets
    
    def set_camera_mode(self, mode='overview', follow_env_id=0):
        """
        Set camera mode for multi-environment rendering.
        
        Args:
            mode: 'overview' - bird's eye view of all envs, 'follow' - track single env
            follow_env_id: environment to follow in 'follow' mode
        """
        self.camera_mode = mode
        self.follow_env_id = follow_env_id
    
    def setup_overview_camera(self):
        """Position camera to see all environments in grid layout.
        
        Uses pre-computed camera_pos and camera_lookat from _compute_multi_env_params.
        """
        camera_angle = geom_utils.compute_camera_angle(self.camera_pos, self.camera_lookat)
        
        flex_renderer.set_camera_params(
            self.scaled(self.camera_pos),
            camera_angle,
            self.scaled(self.camera_near),
            self.scaled(self.camera_far),
        )


    def build(self, sim, particles):
        self.sim = sim
        self.n_particles_per_env = sim.n_particles
        self.n_particles = sim.n_particles * self.num_envs  # Total particles for all envs

        # compute bodies info - replicate for all environments
        bodies_color = []
        bodies_needs_smoothing = []
        bodies_particle_offset = []
        
        for env_id in range(self.num_envs):
            env_particle_base = env_id * self.n_particles_per_env
            for body_particle_ids in particles['bodies']['particle_ids']:
                body_particle_offset = body_particle_ids[0] + env_particle_base
                bodies_particle_offset.append(body_particle_offset)
                bodies_color.append(misc_utils.alpha_to_transparency(particles['color'][body_particle_ids[0]]))
                bodies_needs_smoothing.append(MAT_CLASS[particles['mat'][body_particle_ids[0]]] == MAT_LIQUID)

        # Replicate n_particles for each env
        bodies_n_particles_single = np.array(particles['bodies']['n_particles'])
        bodies_n_particles = np.tile(bodies_n_particles_single, self.num_envs)
        bodies_color           = np.array(bodies_color).flatten()
        bodies_needs_smoothing = np.array(bodies_needs_smoothing)
        bodies_particle_offset = np.array(bodies_particle_offset)

        n_bodies_per_env = particles['bodies']['n']
        n_bodies = n_bodies_per_env * self.num_envs
        assert np.sum(bodies_n_particles) == self.n_particles

        # create scene (light_pos, light_lookat, floor_height, scene_radius already
        # computed in _compute_multi_env_params for multi-env mode)
        flex_renderer.create_scene(
            self.render_particle,
            self.particle_radius,
            self.smoke_radius,
            self._anisotropy_scale,
            self._smoothing,
            self._gl_color_gamma,
            self.scaled(self._fluid_rest_distance),
            self.scaled(self.light_pos),
            self.scaled(self.light_lookat),
            self.light_fov,
            self.scaled(self.floor_height),
            self.scaled(self.scene_radius),
            bodies_n_particles,
            bodies_particle_offset,
            bodies_color,
            bodies_needs_smoothing,
            n_bodies,
        )

        # add statics for each environment
        self.static_mesh_ids = []  # Store mesh IDs for each env
        if len(self.sim.statics) != 0:
            for env_id in range(self.num_envs):
                env_mesh_ids = []
                for static in self.sim.statics:
                    colors = static.colors_np.flatten()
                    faces = static.faces_np
                    mesh_id = flex_renderer.add_mesh(static.n_vertices, static.n_faces, colors, faces)
                    env_mesh_ids.append(mesh_id)
                self.static_mesh_ids.append(env_mesh_ids)
            # Set gl_renderer_id for env 0 (backward compatibility)
            for i, static in enumerate(self.sim.statics):
                static.gl_renderer_id = self.static_mesh_ids[0][i]

        # add effectors for each environment
        self.effector_mesh_ids = []
        if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
            for env_id in range(self.num_envs):
                env_mesh_ids = []
                for effector in self.sim.agent.effectors:
                    if effector.mesh is not None:
                        colors = effector.mesh.colors_np.flatten()
                        faces = effector.mesh.faces_np
                        mesh_id = flex_renderer.add_mesh(effector.mesh.n_vertices, effector.mesh.n_faces, colors, faces)
                        env_mesh_ids.append(mesh_id)
                    else:
                        env_mesh_ids.append(None)
                self.effector_mesh_ids.append(env_mesh_ids)
            # Set gl_renderer_id for env 0 (backward compatibility)
            for i, effector in enumerate(self.sim.agent.effectors):
                if effector.mesh is not None:
                    effector.mesh.gl_renderer_id = self.effector_mesh_ids[0][i]

        # add smoke particles (skip in headless mode)
        if self.sim.smoke_field is not None and not is_headless():
            num_smoke_particles = self.sim.smoke_field.vis_particles.shape[0]
            positions           = self.sim.smoke_field.vis_particles.to_numpy().flatten()
            colors              = self.sim.smoke_field.vis_particles_c.to_numpy().flatten()
            positions_scaled    = self.scaled(positions)
            flex_renderer.add_smoke_particles(num_smoke_particles, positions_scaled, colors)

        # camera - use overview for multi-env
        if self.num_envs > 1 and self.camera_mode == 'overview':
            self.setup_overview_camera()
        else:
            flex_renderer.set_camera_params(
                self.scaled(self.camera_pos),
                self.camera_angle,
                self.scaled(self.camera_near),
                self.scaled(self.camera_far),
            )

        # timer
        self.t = time()

    def rotate_camera(self, t=None):
        if t is not None:
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = self.cam_rotate_v * np.pi * t + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                    x + self.camera_lookat[0],
                    self.camera_pos[1],
                    z + self.camera_lookat[2]]) 
            new_camera_angle = geom_utils.compute_camera_angle(new_camera_pos, self.camera_lookat)

            flex_renderer.set_camera_params(
                self.scaled(new_camera_pos),
                new_camera_angle,
                self.scaled(self.camera_near),
                self.scaled(self.camera_far),
            )

    def scaled(self, x):
        return x * self._rendering_scale

    def update_fps(self):
        self.fps = 1.0 / max(time() - self.t, 1e-6)
        self.t = time()
        # FPS 显示已移到 keyboard.py 终端输出，这里不再打印

    def render_frame(self, mode='human', x_target=None, t=0):
        # Headless 模式检查: 跳过所有渲染和 NumPy 转换
        if is_headless():
            return None
        
        self.rotate_camera(t)

        # particles - merge all environments with spatial offsets
        if self.sim.has_particles:
            state_render = self.sim.get_state_render(self.sim.cur_substep_local)
            if state_render is None:
                # Headless mode: no render buffer allocated
                return None
            # state_render contains merged data for all envs
            # GPU->CPU transfer only happens during actual rendering
            positions = state_render.x.to_numpy()  # Shape: (num_envs * n_particles, 3)
            used = state_render.used.to_numpy()    # Shape: (num_envs * n_particles,)
            
            # Apply env offsets to particle positions
            positions_with_offset = np.zeros_like(positions)
            for env_id in range(self.num_envs):
                start_idx = env_id * self.n_particles_per_env
                end_idx = (env_id + 1) * self.n_particles_per_env
                positions_with_offset[start_idx:end_idx] = positions[start_idx:end_idx] + self.env_offsets[env_id]
            
            flex_renderer.set_particles_state(self.scaled(positions_with_offset.flatten()), used)

        # statics - update for each environment with offset
        if len(self.sim.statics) != 0:
            for env_id in range(self.num_envs):
                offset = self.env_offsets[env_id]
                for i, static in enumerate(self.sim.statics):
                    mesh_render_id = self.static_mesh_ids[env_id][i]
                    # Apply offset to vertices
                    vertices = static.init_vertices_np.copy()
                    vertices += offset
                    vertices_flattened = vertices.flatten()
                    vertex_normals = static.init_vertex_normals_np_flattened
                    flex_renderer.update_mesh(
                        mesh_render_id,
                        self.scaled(vertices_flattened),
                        vertex_normals,
                    )

        # effectors - update for each environment with offset
        if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
            for env_id in range(self.num_envs):
                offset = self.env_offsets[env_id]
                for i, effector in enumerate(self.sim.agent.effectors):
                    if effector.mesh is not None:
                        mesh_render_id = self.effector_mesh_ids[env_id][i]
                        # 使用预分配缓冲区获取顶点数据（避免每帧分配新数组）
                        vertices = effector.mesh.get_vertices_np()
                        vertices_with_offset = vertices + offset  # 广播加法
                        vertex_normals = effector.mesh.get_vertex_normals_np().flatten()
                        flex_renderer.update_mesh(
                            mesh_render_id,
                            self.scaled(vertices_with_offset.flatten()),
                            vertex_normals,
                        )

        # smoke field (skip GPU->CPU transfer in headless mode)
        if self.sim.smoke_field is not None and not is_headless():
            colors = self.sim.smoke_field.vis_particles_c.to_numpy()
            flex_renderer.update_smoke_particles(colors)

        # render
        img = flex_renderer.render()
        img = np.flip(img.reshape([self.res[1], self.res[0], 4]), 0)[:, :, :3]

        self.update_fps()
        
        # Window title with multi-env info
        title = f'FluidLab [FPS: {self.fps:.2f}]'
        if self.num_envs > 1:
            title += f' | {self.num_envs} envs ({self.grid_rows}x{self.grid_cols})'

        if mode == 'human':
            cv2.imshow('FluidLab', img[..., ::-1])
            cv2.setWindowTitle('FluidLab', title)
            cv2.waitKey(1)
            return None

        elif mode == 'rgb_array':
            cv2.imshow('FluidLab', img[..., ::-1])
            cv2.setWindowTitle('FluidLab', title)
            cv2.waitKey(1)
            return img






