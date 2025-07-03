"""Real-time visualization for Lorenz attractor simulations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import time
from typing import Optional, Callable, Tuple, List
import threading
from queue import Queue
import moderngl

from ..core.simulator import Simulator
from ..core.parameters import LorenzParameters, InitialConditions, SimulationConfig


class RealtimeVisualizer:
    """Real-time visualization of Lorenz attractor evolution."""
    
    def __init__(self, simulator: Simulator, trail_length: int = 2000):
        """
        Initialize real-time visualizer.
        
        Args:
            simulator: Simulator instance
            trail_length: Number of points to keep in trail
        """
        self.simulator = simulator
        self.trail_length = trail_length
        self.is_running = False
        self.current_state = None
        self.trajectory_buffer = []
        self.time_buffer = []
        self.current_time = 0
        
    def start_matplotlib_animation(self, initial_conditions: InitialConditions,
                                  config: SimulationConfig,
                                  update_interval: int = 50) -> FuncAnimation:
        """
        Start real-time animation using matplotlib.
        
        Args:
            initial_conditions: Initial conditions
            config: Simulation configuration
            update_interval: Update interval in milliseconds
            
        Returns:
            Animation object
        """
        # Setup figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize trajectory
        self.current_state = initial_conditions.to_array()
        self.trajectory_buffer = [self.current_state.copy()]
        self.time_buffer = [0]
        
        # Setup plot elements
        line, = ax.plot([], [], [], 'b-', linewidth=0.5, alpha=0.7)
        point, = ax.plot([], [], [], 'ro', markersize=8)
        
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([0, 60])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Real-time Lorenz Attractor')
        
        def update_frame(frame):
            # Simulate next step
            dt = config.dt
            derivative = self.simulator.system.derivative(self.current_state)
            self.current_state += derivative * dt
            self.current_time += dt
            
            # Update buffers
            self.trajectory_buffer.append(self.current_state.copy())
            self.time_buffer.append(self.current_time)
            
            # Keep only recent points
            if len(self.trajectory_buffer) > self.trail_length:
                self.trajectory_buffer.pop(0)
                self.time_buffer.pop(0)
            
            # Update plot
            if len(self.trajectory_buffer) > 1:
                trajectory = np.array(self.trajectory_buffer)
                line.set_data(trajectory[:, 0], trajectory[:, 1])
                line.set_3d_properties(trajectory[:, 2])
                
                # Current point
                point.set_data([self.current_state[0]], [self.current_state[1]])
                point.set_3d_properties([self.current_state[2]])
            
            return line, point
        
        # Create animation
        anim = FuncAnimation(fig, update_frame, interval=update_interval, 
                           blit=True, cache_frame_data=False)
        
        return anim
    
    def start_pygame_visualization(self, initial_conditions: InitialConditions,
                                  config: SimulationConfig,
                                  window_size: Tuple[int, int] = (1200, 800)):
        """
        Start real-time visualization using Pygame and OpenGL.
        
        Args:
            initial_conditions: Initial conditions
            config: Simulation configuration
            window_size: Window size (width, height)
        """
        # Initialize Pygame
        pygame.init()
        display = pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Real-time Lorenz Attractor")
        
        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Set up projection
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, window_size[0] / window_size[1], 0.1, 200.0)
        
        glMatrixMode(GL_MODELVIEW)
        
        # Initialize simulation
        self.current_state = initial_conditions.to_array()
        self.trajectory_buffer = []
        self.is_running = True
        
        # Camera parameters
        camera_distance = 80
        camera_angle_x = 0
        camera_angle_y = 0
        
        clock = pygame.time.Clock()
        
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
                    elif event.key == pygame.K_SPACE:
                        # Reset simulation
                        self.current_state = initial_conditions.to_array()
                        self.trajectory_buffer = []
                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:  # Left mouse button
                        camera_angle_x += event.rel[1] * 0.5
                        camera_angle_y += event.rel[0] * 0.5
                elif event.type == pygame.MOUSEWHEEL:
                    camera_distance -= event.y * 5
                    camera_distance = max(20, min(150, camera_distance))
            
            # Simulate next step
            dt = config.dt
            derivative = self.simulator.system.derivative(self.current_state)
            self.current_state += derivative * dt
            
            # Update trajectory buffer
            self.trajectory_buffer.append(self.current_state.copy())
            if len(self.trajectory_buffer) > self.trail_length:
                self.trajectory_buffer.pop(0)
            
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up camera
            glLoadIdentity()
            glTranslatef(0, 0, -camera_distance)
            glRotatef(camera_angle_x, 1, 0, 0)
            glRotatef(camera_angle_y, 0, 1, 0)
            
            # Draw coordinate axes
            self._draw_axes()
            
            # Draw trajectory
            if len(self.trajectory_buffer) > 1:
                self._draw_trajectory_opengl()
            
            # Draw current point
            self._draw_current_point_opengl()
            
            # Update display
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        pygame.quit()
    
    def _draw_axes(self):
        """Draw coordinate axes."""
        glLineWidth(2)
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(30, 0, 0)
        
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 30, 0)
        
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 30)
        
        glEnd()
    
    def _draw_trajectory_opengl(self):
        """Draw trajectory using OpenGL."""
        glLineWidth(1)
        glBegin(GL_LINE_STRIP)
        
        # Color gradient based on position in buffer
        for i, point in enumerate(self.trajectory_buffer):
            alpha = i / len(self.trajectory_buffer)
            glColor4f(0, 0.5, 1, alpha)  # Blue with varying alpha
            glVertex3f(point[0], point[1], point[2])
        
        glEnd()
    
    def _draw_current_point_opengl(self):
        """Draw current point using OpenGL."""
        glPointSize(8)
        glColor3f(1, 0, 0)  # Red
        glBegin(GL_POINTS)
        glVertex3f(self.current_state[0], self.current_state[1], self.current_state[2])
        glEnd()
    
    def start_moderngl_visualization(self, initial_conditions: InitialConditions,
                                   config: SimulationConfig,
                                   window_size: Tuple[int, int] = (1200, 800)):
        """
        Start high-performance visualization using ModernGL.
        
        Args:
            initial_conditions: Initial conditions
            config: Simulation configuration
            window_size: Window size (width, height)
        """
        # Initialize Pygame for window management
        pygame.init()
        display = pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("High-Performance Lorenz Attractor")
        
        # Create ModernGL context
        ctx = moderngl.create_context()
        
        # Vertex shader
        vertex_shader = '''
        #version 330 core
        
        in vec3 in_position;
        in float in_alpha;
        
        uniform mat4 mvp;
        
        out float alpha;
        
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            alpha = in_alpha;
        }
        '''
        
        # Fragment shader
        fragment_shader = '''
        #version 330 core
        
        in float alpha;
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(0.0, 0.5, 1.0, alpha);
        }
        '''
        
        # Create shader program
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Initialize simulation
        self.current_state = initial_conditions.to_array()
        self.trajectory_buffer = []
        self.is_running = True
        
        clock = pygame.time.Clock()
        
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
            
            # Simulate next step
            dt = config.dt
            derivative = self.simulator.system.derivative(self.current_state)
            self.current_state += derivative * dt
            
            # Update trajectory buffer
            self.trajectory_buffer.append(self.current_state.copy())
            if len(self.trajectory_buffer) > self.trail_length:
                self.trajectory_buffer.pop(0)
            
            if len(self.trajectory_buffer) > 1:
                # Prepare vertex data
                vertices = np.array(self.trajectory_buffer, dtype=np.float32)
                alphas = np.linspace(0.1, 1.0, len(vertices), dtype=np.float32)
                
                # Create vertex buffer
                vbo = ctx.buffer(vertices.tobytes())
                alpha_buffer = ctx.buffer(alphas.tobytes())
                
                # Create vertex array
                vao = ctx.vertex_array(program, [
                    (vbo, '3f', 'in_position'),
                    (alpha_buffer, '1f', 'in_alpha')
                ])
                
                # Set up MVP matrix (simplified)
                mvp = np.eye(4, dtype=np.float32)
                mvp[0, 0] = 0.02  # Scale X
                mvp[1, 1] = 0.02  # Scale Y
                mvp[2, 2] = 0.02  # Scale Z
                mvp[2, 3] = -0.5  # Translate Z
                
                program['mvp'].write(mvp.tobytes())
                
                # Clear and render
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                
                vao.render(moderngl.LINE_STRIP)
                
                # Clean up
                vao.release()
                vbo.release()
                alpha_buffer.release()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    def stop(self):
        """Stop the real-time visualization."""
        self.is_running = False


class StreamingVisualizer:
    """Stream simulation data for real-time visualization."""
    
    def __init__(self, simulator: Simulator, buffer_size: int = 10000):
        """
        Initialize streaming visualizer.
        
        Args:
            simulator: Simulator instance
            buffer_size: Size of the streaming buffer
        """
        self.simulator = simulator
        self.buffer_size = buffer_size
        self.data_queue = Queue(maxsize=buffer_size)
        self.is_streaming = False
        self.stream_thread = None
    
    def start_streaming(self, initial_conditions: InitialConditions,
                       config: SimulationConfig):
        """
        Start streaming simulation data.
        
        Args:
            initial_conditions: Initial conditions
            config: Simulation configuration
        """
        self.is_streaming = True
        self.stream_thread = threading.Thread(
            target=self._stream_worker,
            args=(initial_conditions, config)
        )
        self.stream_thread.start()
    
    def _stream_worker(self, initial_conditions: InitialConditions,
                      config: SimulationConfig):
        """Worker thread for streaming simulation data."""
        current_state = initial_conditions.to_array()
        current_time = 0
        
        while self.is_streaming:
            # Simulate next step
            dt = config.dt
            derivative = self.simulator.system.derivative(current_state)
            current_state += derivative * dt
            current_time += dt
            
            # Add to queue (non-blocking)
            try:
                self.data_queue.put_nowait({
                    'time': current_time,
                    'state': current_state.copy(),
                    'x': current_state[0],
                    'y': current_state[1],
                    'z': current_state[2]
                })
            except:
                # Queue is full, skip this point
                pass
            
            # Control simulation speed
            time.sleep(config.dt / 10)  # 10x real-time
    
    def get_latest_data(self, num_points: int = 100) -> List[dict]:
        """
        Get latest simulation data points.
        
        Args:
            num_points: Number of points to retrieve
            
        Returns:
            List of data points
        """
        data_points = []
        
        # Get all available data
        while not self.data_queue.empty() and len(data_points) < num_points:
            try:
                data_points.append(self.data_queue.get_nowait())
            except:
                break
        
        return data_points
    
    def stop_streaming(self):
        """Stop streaming simulation data."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_streaming()