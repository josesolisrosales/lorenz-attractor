"""OpenGL rendering components."""

import numpy as np
import moderngl
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Optional, Tuple, List, Dict, Any
import math

from ..core.simulator import SimulationResult


class OpenGLRenderer:
    """OpenGL renderer for high-performance visualization."""
    
    def __init__(self, window_size: Tuple[int, int] = (1200, 800)):
        """
        Initialize renderer.
        
        Args:
            window_size: Window size (width, height)
        """
        self.window_size = window_size
        self.ctx = None
        self.program = None
        self.camera_distance = 80.0
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.mouse_sensitivity = 0.5
        self.zoom_sensitivity = 5.0
        
    def initialize_context(self):
        """Initialize OpenGL context."""
        # Initialize Pygame
        pygame.init()
        pygame.display.set_mode(self.window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("High-Performance Lorenz Attractor")
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        
        # Enable depth testing and blending
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Create shader program
        self._create_shader_program()
        
    def _create_shader_program(self):
        """Create shader program for trajectory rendering."""
        vertex_shader = '''
        #version 330 core
        
        in vec3 in_position;
        in float in_alpha;
        in vec3 in_color;
        
        uniform mat4 mvp;
        uniform vec3 camera_pos;
        uniform float point_size;
        
        out float alpha;
        out vec3 color;
        out float distance_from_camera;
        
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            gl_PointSize = point_size;
            alpha = in_alpha;
            color = in_color;
            distance_from_camera = length(camera_pos - in_position);
        }
        '''
        
        fragment_shader = '''
        #version 330 core
        
        in float alpha;
        in vec3 color;
        in float distance_from_camera;
        
        out vec4 fragColor;
        
        void main() {
            // Distance-based alpha falloff
            float distance_alpha = 1.0 / (1.0 + distance_from_camera * 0.01);
            fragColor = vec4(color, alpha * distance_alpha);
        }
        '''
        
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
    def render_trajectory(self, result: SimulationResult, 
                         render_mode: str = 'line',
                         color_mode: str = 'time',
                         point_size: float = 2.0):
        """
        Render trajectory using OpenGL.
        
        Args:
            result: Simulation result
            render_mode: 'line', 'points', or 'both'
            color_mode: 'time', 'velocity', or 'position'
            point_size: Size of points when rendering points
        """
        if not self.ctx:
            raise RuntimeError("OpenGL context not initialized")
            
        # Prepare vertex data
        vertices = result.trajectory.astype(np.float32)
        
        # Generate colors based on mode
        colors = self._generate_colors(result, color_mode)
        
        # Generate alpha values (fade effect)
        alphas = np.linspace(0.1, 1.0, len(vertices), dtype=np.float32)
        
        # Create vertex buffer objects
        vbo_vertices = self.ctx.buffer(vertices.tobytes())
        vbo_colors = self.ctx.buffer(colors.tobytes())
        vbo_alphas = self.ctx.buffer(alphas.tobytes())
        
        # Create vertex array object
        vao = self.ctx.vertex_array(self.program, [
            (vbo_vertices, '3f', 'in_position'),
            (vbo_colors, '3f', 'in_color'),
            (vbo_alphas, '1f', 'in_alpha')
        ])
        
        # Set uniforms
        mvp_matrix = self._calculate_mvp_matrix()
        camera_pos = self._get_camera_position()
        
        self.program['mvp'].write(mvp_matrix.tobytes())
        self.program['camera_pos'].write(camera_pos.tobytes())
        self.program['point_size'].value = point_size
        
        # Clear screen
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render based on mode
        if render_mode == 'line':
            vao.render(moderngl.LINE_STRIP)
        elif render_mode == 'points':
            vao.render(moderngl.POINTS)
        elif render_mode == 'both':
            # Render line first
            vao.render(moderngl.LINE_STRIP)
            # Then render points with larger size
            self.program['point_size'].value = point_size * 2
            vao.render(moderngl.POINTS)
        
        # Clean up
        vao.release()
        vbo_vertices.release()
        vbo_colors.release()
        vbo_alphas.release()
        
    def _generate_colors(self, result: SimulationResult, color_mode: str) -> np.ndarray:
        """Generate colors for trajectory points."""
        n_points = len(result.trajectory)
        colors = np.zeros((n_points, 3), dtype=np.float32)
        
        if color_mode == 'time':
            # Color by time progression
            t_norm = np.linspace(0, 1, n_points)
            colors[:, 0] = np.sin(2 * np.pi * t_norm)  # Red
            colors[:, 1] = np.cos(2 * np.pi * t_norm)  # Green
            colors[:, 2] = 0.5 + 0.5 * np.sin(4 * np.pi * t_norm)  # Blue
            
        elif color_mode == 'velocity':
            # Color by velocity magnitude
            velocities = np.diff(result.trajectory, axis=0)
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            vel_magnitudes = np.append(vel_magnitudes, vel_magnitudes[-1])  # Add last point
            
            vel_norm = (vel_magnitudes - vel_magnitudes.min()) / (vel_magnitudes.max() - vel_magnitudes.min())
            colors[:, 0] = vel_norm  # Red increases with velocity
            colors[:, 1] = 1.0 - vel_norm  # Green decreases with velocity
            colors[:, 2] = 0.5  # Constant blue
            
        elif color_mode == 'position':
            # Color by Z coordinate
            z_norm = (result.z - result.z.min()) / (result.z.max() - result.z.min())
            colors[:, 0] = 0.2  # Low red
            colors[:, 1] = 0.5 + 0.5 * z_norm  # Green varies with Z
            colors[:, 2] = 1.0 - z_norm  # Blue inversely varies with Z
            
        # Ensure colors are in valid range
        colors = np.clip(colors, 0.0, 1.0)
        
        return colors
        
    def _calculate_mvp_matrix(self) -> np.ndarray:
        """Calculate Model-View-Projection matrix."""
        # Model matrix (identity)
        model = np.eye(4, dtype=np.float32)
        
        # View matrix
        view = np.eye(4, dtype=np.float32)
        
        # Apply camera transformations
        # Translate back
        view[2, 3] = -self.camera_distance
        
        # Rotation matrices
        cos_x, sin_x = math.cos(math.radians(self.camera_angle_x)), math.sin(math.radians(self.camera_angle_x))
        cos_y, sin_y = math.cos(math.radians(self.camera_angle_y)), math.sin(math.radians(self.camera_angle_y))
        
        # X rotation
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, cos_x, -sin_x, 0],
            [0, sin_x, cos_x, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Y rotation
        rot_y = np.array([
            [cos_y, 0, sin_y, 0],
            [0, 1, 0, 0],
            [-sin_y, 0, cos_y, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        view = view @ rot_x @ rot_y
        
        # Projection matrix
        fov = 45.0
        aspect = self.window_size[0] / self.window_size[1]
        near = 0.1
        far = 200.0
        
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        # Combine matrices
        mvp = projection @ view @ model
        return mvp
        
    def _get_camera_position(self) -> np.ndarray:
        """Get camera position for shader calculations."""
        cos_x, sin_x = math.cos(math.radians(self.camera_angle_x)), math.sin(math.radians(self.camera_angle_x))
        cos_y, sin_y = math.cos(math.radians(self.camera_angle_y)), math.sin(math.radians(self.camera_angle_y))
        
        x = self.camera_distance * sin_y * cos_x
        y = self.camera_distance * sin_x
        z = self.camera_distance * cos_y * cos_x
        
        return np.array([x, y, z], dtype=np.float32)
        
    def handle_mouse_input(self, event):
        """Handle mouse input for camera control."""
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:  # Left mouse button
                self.camera_angle_x += event.rel[1] * self.mouse_sensitivity
                self.camera_angle_y += event.rel[0] * self.mouse_sensitivity
                
                # Clamp vertical angle
                self.camera_angle_x = max(-90, min(90, self.camera_angle_x))
                
        elif event.type == pygame.MOUSEWHEEL:
            self.camera_distance -= event.y * self.zoom_sensitivity
            self.camera_distance = max(10, min(200, self.camera_distance))
            
    def render_axes(self, length: float = 30.0):
        """Render coordinate axes."""
        # Axes vertices
        axes_vertices = np.array([
            # X axis
            [0, 0, 0], [length, 0, 0],
            # Y axis  
            [0, 0, 0], [0, length, 0],
            # Z axis
            [0, 0, 0], [0, 0, length]
        ], dtype=np.float32)
        
        # Axes colors
        axes_colors = np.array([
            # X axis (red)
            [1, 0, 0], [1, 0, 0],
            # Y axis (green)
            [0, 1, 0], [0, 1, 0],
            # Z axis (blue)
            [0, 0, 1], [0, 0, 1]
        ], dtype=np.float32)
        
        # Full alpha
        axes_alphas = np.ones(6, dtype=np.float32)
        
        # Create buffers
        vbo_vertices = self.ctx.buffer(axes_vertices.tobytes())
        vbo_colors = self.ctx.buffer(axes_colors.tobytes())
        vbo_alphas = self.ctx.buffer(axes_alphas.tobytes())
        
        # Create VAO
        vao = self.ctx.vertex_array(self.program, [
            (vbo_vertices, '3f', 'in_position'),
            (vbo_colors, '3f', 'in_color'),
            (vbo_alphas, '1f', 'in_alpha')
        ])
        
        # Set uniforms
        mvp_matrix = self._calculate_mvp_matrix()
        camera_pos = self._get_camera_position()
        
        self.program['mvp'].write(mvp_matrix.tobytes())
        self.program['camera_pos'].write(camera_pos.tobytes())
        self.program['point_size'].value = 1.0
        
        # Render axes as lines
        vao.render(moderngl.LINES)
        
        # Clean up
        vao.release()
        vbo_vertices.release()
        vbo_colors.release()
        vbo_alphas.release()
        
    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.program:
            self.program.release()
        if self.ctx:
            self.ctx.release()
        pygame.quit()