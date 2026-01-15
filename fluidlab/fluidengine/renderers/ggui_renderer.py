import os
import trimesh
import skimage
import numpy as np
import taichi as ti
from time import time
from scipy import ndimage
from fluidlab.configs.macros import *

@ti.data_oriented
class GGUIRenderer:
    def __init__(self, 
        res=(640, 640),
        camera_pos=(0.5, 2.5, 3.5),
        camera_lookat=(0.5, 0.5, 0.5),
        fov=30,
        mode='human',
        particle_radius=0.0075,
        lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        num_envs=1,
        env_spacing=1.2,
    ):
        """
        Initialize GGUIRenderer with multi-environment support.
        
        Args:
            num_envs: Number of parallel environments to render
            env_spacing: World-space distance between environment centers
        """
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        
        # Calculate grid layout (as close to square as possible)
        self.grid_cols = int(np.ceil(np.sqrt(num_envs)))
        self.grid_rows = int(np.ceil(num_envs / self.grid_cols))
        
        # Camera mode: 'overview' for all envs, 'follow' for single env
        self.camera_mode = 'overview' if num_envs > 1 else 'follow'
        self.follow_env_id = 0
        
        self.res = res
        self.camera_pos = np.array(camera_pos)
        self.camera_lookat = np.array(camera_lookat)
        self.camera_vec = self.camera_pos - self.camera_lookat
        self.camera_init_xz_rad = np.arctan2(self.camera_vec[0], self.camera_vec[2])
        self.lights = []
        self.uninit = True

        for light in lights:
            self.add_light(light['pos'], light['color'])

        self.fov = fov
        self.particle_radius = particle_radius
        self.frame = ti.Vector.field(3, dtype=ti.f32, shape=(9,))
        self.frames = [ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,)),
                        ti.Vector.field(3, dtype=ti.f32, shape=(200,))]
        self.color_target = None
        
        # Environment offset positions for Isaac Lab-style layout
        self.env_offsets_np = self._compute_env_offsets()
        self.env_offsets = ti.Vector.field(3, ti.f32, shape=(num_envs,))
        
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

    def add_light(self, pos, color=(0.5, 0.5, 0.5)):
        light = {
            'pos': pos,
            'color': color
        }
        self.lights.append(light)

    def build(self, sim, particles):
        self.sim = sim
        self.n_particles = len(particles['color'])
        
        # Setup env offsets field
        self.env_offsets.from_numpy(self.env_offsets_np)
        
        # Colors for all environments (replicated)
        self.particles_color = ti.Vector.field(4, ti.f32, shape=(self.num_envs * self.n_particles,))
        # Replicate colors for each environment
        colors_replicated = np.tile(particles['color'].astype(np.float32), (self.num_envs, 1))
        self.particles_color.from_numpy(colors_replicated)
        
        # Render positions field for merged particles
        self.render_positions = ti.Vector.field(3, ti.f32, shape=(self.num_envs * self.n_particles,))

        for i in range(200):
            self.frames[0][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[1][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([0., 1., 0.])
            self.frames[2][i] = ti.Vector([0., 0., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[3][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([-1., 0., 0.])
            self.frames[4][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([0., -1., 0.])
            self.frames[5][i] = ti.Vector([1., 1., 1.]) + i/200 * ti.Vector([0., 0., -1.])
            self.frames[6][i] = ti.Vector([0., 1., 0.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[7][i] = ti.Vector([0., 1., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[8][i] = ti.Vector([1., 0., 0.]) + i/200 * ti.Vector([0., 1., 0.])
            self.frames[9][i] = ti.Vector([1., 0., 0.]) + i/200 * ti.Vector([0., 0., 1.])
            self.frames[10][i] = ti.Vector([0., 0., 1.]) + i/200 * ti.Vector([1., 0., 0.])
            self.frames[11][i] = ti.Vector([0., 0., 1.]) + i/200 * ti.Vector([0., 1., 0.])

        # timer
        self.t = time()
    
    @ti.kernel
    def prepare_render_positions(self):
        """Merge all environments' particles into single render buffer with spatial offsets."""
        for env_id, p in ti.ndrange(self.num_envs, self.n_particles):
            global_idx = env_id * self.n_particles + p
            # Apply environment offset to particle position
            self.render_positions[global_idx] = (
                self.sim.particles_render[global_idx].x + self.env_offsets[env_id]
            )
    
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
        """Position camera to see all environments in grid layout."""
        # Calculate center of the grid
        center_x = (self.grid_cols - 1) * self.env_spacing / 2
        center_z = (self.grid_rows - 1) * self.env_spacing / 2
        
        # Calculate camera height based on grid size
        grid_extent = max(self.grid_rows, self.grid_cols) * self.env_spacing
        camera_height = grid_extent * 1.5
        camera_distance = grid_extent * 1.2
        
        # Position camera for overview
        self.camera.position(center_x, camera_height, center_z + camera_distance)
        self.camera.lookat(center_x, 0.3, center_z)

    def update_fps(self):
        self.fps = 1.0 / (time() - self.t)
        self.t = time()
        # Print FPS to terminal (real-time update, overwrite same line)
        import sys
        sys.stdout.write(f'\r[FPS: {self.fps:.2f}]')
        sys.stdout.flush()

    def update_camera(self, t=None, rotate=False):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        speed = 1e-2
        if self.window.is_pressed(ti.ui.UP):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.DOWN):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) - camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) - camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('u'):
            camera_dir = np.array([0, 1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed('i'):
            camera_dir = np.array([0, -1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)

        # rotate
        if rotate and t is not None:
            speed = 7.5e-4
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = speed * np.pi * t + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                    x + self.camera_lookat[0],
                    self.camera_pos[1],
                    z + self.camera_lookat[2]]) 
            self.camera.position(*new_camera_pos)

        self.scene.set_camera(self.camera)

    def init_window(self, mode):
        show_window = mode == 'human'
        self.window = ti.ui.Window('FluidLab', self.res, vsync=True, show_window=show_window)

        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1,1,1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(*self.camera_pos)
        self.camera.lookat(*self.camera_lookat)
        self.camera.fov(self.fov)


    def render_frame(self, mode='human', x_target=None, t=0):
        if self.uninit:
            self.uninit = False
            self.init_window(mode)
            # Setup overview camera for multi-env mode
            if self.num_envs > 1 and self.camera_mode == 'overview':
                self.setup_overview_camera()

        self.update_camera(t)

        # # reference frame
        # for i in range(12):
        #     self.scene.particles(self.frames[i], color=COLOR[FRAME], radius=self.particle_radius*0.5)
            
        # particles - render all environments merged
        if self.sim.has_particles:
            # Get render state (already merged with offsets applied in get_state_render)
            state = self.sim.get_state_render(self.sim.cur_substep_local)
            
            # Prepare render positions with env offsets
            self.prepare_render_positions()
            
            # Render all particles from all environments
            self.scene.particles(self.render_positions, per_vertex_color=self.particles_color, radius=self.particle_radius)

            # target particles
            if x_target is not None:
                if self.color_target is None:
                    self.color_target = ti.Vector.field(4, ti.f32, x_target.shape)
                    self.color_target.from_numpy(np.repeat(np.array([COLOR[TARGET]]).astype(np.float32), x_target.shape[0], axis=0))
                self.scene.particles(x_target, per_vertex_color=self.color_target, radius=self.particle_radius)

        # statics - render for each environment with offset
        if len(self.sim.statics) != 0:
            for env_id in range(self.num_envs):
                offset = self.env_offsets_np[env_id]
                for static in self.sim.statics:
                    # Create offset vertices for this env
                    offset_vertices = static.vertices.to_numpy() + offset[np.newaxis, :]
                    offset_field = ti.Vector.field(3, dtype=ti.f32, shape=offset_vertices.shape[0])
                    offset_field.from_numpy(offset_vertices.astype(np.float32))
                    self.scene.mesh(offset_field, static.faces, per_vertex_color=static.colors)

        # effectors - render for each environment with offset  
        if self.sim.agent is not None and self.sim.agent.n_effectors != 0:
            for env_id in range(self.num_envs):
                offset = self.env_offsets_np[env_id]
                for effector in self.sim.agent.effectors:
                    if effector.mesh is not None:
                        # Create offset vertices for this env
                        offset_vertices = effector.mesh.vertices.to_numpy() + offset[np.newaxis, :]
                        offset_field = ti.Vector.field(3, dtype=ti.f32, shape=offset_vertices.shape[0])
                        offset_field.from_numpy(offset_vertices.astype(np.float32))
                        self.scene.mesh(offset_field, effector.mesh.faces, per_vertex_color=effector.mesh.colors)

        # smoke
        if self.sim.smoke_field is not None:
            self.scene.particles(self.sim.smoke_field.vis_particles, per_vertex_color=self.sim.smoke_field.vis_particles_c, radius=1.0/128)

        # Lights - adjust for overview mode
        for light in self.lights:
            if self.num_envs > 1 and self.camera_mode == 'overview':
                # Offset light to center of grid
                center_x = (self.grid_cols - 1) * self.env_spacing / 2
                center_z = (self.grid_rows - 1) * self.env_spacing / 2
                light_pos = (light['pos'][0] + center_x, light['pos'][1], light['pos'][2] + center_z)
                self.scene.point_light(pos=light_pos, color=light['color'])
            else:
                self.scene.point_light(pos=light['pos'], color=light['color'])

        self.canvas.scene(self.scene)

        # camera gui with multi-env info
        # if self.num_envs > 1:
        #     self.window.GUI.begin("Multi-Env Info", 0.05, 0.1, 0.2, 0.2)
        #     self.window.GUI.text(f'Environments: {self.num_envs}')
        #     self.window.GUI.text(f'Layout: {self.grid_rows}x{self.grid_cols}')
        #     self.window.GUI.text(f'Mode: {self.camera_mode}')
        #     self.window.GUI.end()

        if mode == 'human':
            self.window.show()
            return None

        elif mode == 'rgb_array':
            img = np.rot90(self.window.get_image_buffer_as_numpy())[:, :, :3]
            img = (img * 255).astype(np.uint8)
            self.update_fps()
            # print(f'===> GGUIRenderer: {self.fps:.2f} FPS')
            return img





