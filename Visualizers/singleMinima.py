import numpy as np
import math
import plotly.graph_objects as go

class GradientDescentSimulator:
    def __init__(self, 
                 learning_rate=10, 
                 title="3D Surface Plot: Stochastic Gradient Descent",
                 ball_radius=0.15, 
                 max_iter=300, 
                 tolerance=1e-4, 
                 initial_pos=(6, 6),
                 grid_range=(-10, 10),
                 grid_points=100,
                 camera_distance=1.0,
                 noise_level=0,
                 random_seed = None):
        """
        :param learning_rate: SGD step size.
        :param title: Title for the 3D plot.
        :param ball_radius: The ball marker offset above the surface.
        :param max_iter: Max number of iterations for SGD.
        :param tolerance: Threshold for stopping SGD.
        :param initial_pos: (x0, y0) starting position of the ball.
        :param grid_range: (min, max) range for both x and y axes.
        :param grid_points: Number of points in each dimension for the surface grid.
        :param camera_distance: Distance of the camera from the origin (smaller => more zoom).
        :param noise_level: Standard deviation of the noise added to the gradient.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.learning_rate = learning_rate
        self.title = title
        self.ball_radius = ball_radius
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_pos = initial_pos
        self.grid_range = grid_range
        self.grid_points = grid_points
        self.camera_distance = camera_distance
        self.noise_level = noise_level
        
        # Build the simulation immediately and store the figure
        self.fig = self._build_simulation()
    
    def f_single_func(self, x, y):
        return -0.05 * (
            (10/np.sqrt(2*np.pi) * np.exp(- (x**2)/16)) *
            (10/np.sqrt(2*np.pi) * np.exp(- (y**2)/16))
        )
    
    def numerical_grad(self, f, x, y, h=1e-5):
        df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return df_dx, df_dy
    
    def _build_simulation(self):
        # ------------------------------------------------------------
        # 1. Plot the singleMinima surface over the chosen grid range
        # ------------------------------------------------------------
        x_range = self.grid_range
        x_vals = np.linspace(x_range[0], x_range[1], self.grid_points)
        y_vals = np.linspace(x_range[0], x_range[1], self.grid_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.f_single_func(X, Y)
        
        surface_single = go.Surface(
            x=X, y=Y, z=Z, name="singleMinima", opacity=0.8
        )
        
        default_z_range = [-3.5, 1]
        fig = go.Figure(data=[surface_single])
        fig.update_layout(
            title=self.title,
            title_font_color="white",
            paper_bgcolor="grey",
            plot_bgcolor="grey",
            width=800,
            height=600,
            scene=dict(
                xaxis=dict(visible=False, range=[x_range[0], x_range[1]]),
                yaxis=dict(visible=False, range=[x_range[0], x_range[1]]),
                zaxis=dict(visible=False, range=default_z_range),
                bgcolor="grey"
            )
        )
        
        # ------------------------------------------------------------
        # 2. Run Stochastic Gradient Descent and collect the path
        # ------------------------------------------------------------
        x0, y0 = self.initial_pos
        traj_x = [x0]
        traj_y = [y0]
        traj_z = [self.f_single_func(x0, y0) + self.ball_radius]
        
        for i in range(self.max_iter):
            current_x = traj_x[-1]
            current_y = traj_y[-1]
            grad_x, grad_y = self.numerical_grad(self.f_single_func, current_x, current_y)
            
            # Add noise to simulate stochastic gradient estimates.
            grad_x += np.random.normal(0, self.noise_level)
            grad_y += np.random.normal(0, self.noise_level)
            
            step_x = self.learning_rate * grad_x
            step_y = self.learning_rate * grad_y
            if np.sqrt(step_x**2 + step_y**2) < self.tolerance:
                break
            new_x = current_x - step_x
            new_y = current_y - step_y
            traj_x.append(new_x)
            traj_y.append(new_y)
            traj_z.append(self.f_single_func(new_x, new_y) + self.ball_radius)
            
        num_iterations = len(traj_x)
        
        # ------------------------------------------------------------
        # 3. Adjust Scene Ranges to ensure the ball is visible
        # ------------------------------------------------------------
        x_min, x_max = min(x_range[0], np.min(traj_x))-1, max(x_range[1], np.max(traj_x))+1
        y_min, y_max = min(x_range[0], np.min(traj_y))-1, max(x_range[1], np.max(traj_y))+1
        z_min_sim = min(default_z_range[0], np.min(traj_z))-0.5
        z_max_sim = max(default_z_range[1], np.max(traj_z))+0.5
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[x_min, x_max]),
                yaxis=dict(visible=False, range=[y_min, y_max]),
                zaxis=dict(visible=False, range=[z_min_sim, z_max_sim])
            )
        )
        
        # ------------------------------------------------------------
        # 4. Build Animation Frames
        # ------------------------------------------------------------
        frames = []
        for i in range(num_iterations):
            ball_trace = go.Scatter3d(
                x=[traj_x[i]],
                y=[traj_y[i]],
                z=[traj_z[i]],
                mode='markers',
                marker=dict(color='red', size=10),
                name="Ball"
            )
            path_trace = go.Scatter3d(
                x=traj_x[:i+1],
                y=traj_y[:i+1],
                z=traj_z[:i+1],
                mode='markers',
                marker=dict(color='red', size=5),
                name="Path"
            )
            frame_data = [surface_single, ball_trace, path_trace]
            frames.append(go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    annotations=[dict(
                        text=f"Iterations: {i}",
                        x=0.95, y=0.95, xref="paper", yref="paper",
                        font=dict(color="red", size=16),
                        showarrow=False
                    )]
                )
            ))
        
        # ------------------------------------------------------------
        # 5. Add Traces & Animation Button
        # ------------------------------------------------------------
        initial_ball = go.Scatter3d(
            x=[traj_x[0]],
            y=[traj_y[0]],
            z=[traj_z[0]],
            mode='markers',
            marker=dict(color='red', size=10),
            name="Ball"
        )
        initial_path = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(color='red', size=5),
            name="Path"
        )
        fig.add_trace(initial_ball)
        fig.add_trace(initial_path)
        
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [{
                    "label": "simulate SGD",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]
                }],
                "pad": {"r": 10, "t": 10},
                "x": 0.5, "xanchor": "center",
                "y": 0.0, "yanchor": "bottom"
            }],
            annotations=[dict(
                text="Iterations: 0",
                x=0.95, y=0.95, xref="paper", yref="paper",
                font=dict(color="red", size=16),
                showarrow=False
            )],
            uirevision='constant'
        )
        
        fig.frames = frames
        
        # ------------------------------------------------------------
        # 6. Set the Camera Position (zoomed in by camera_distance)
        # ------------------------------------------------------------
        r_val = self.camera_distance  # The smaller this is, the more "zoomed in" it appears
        azim = math.radians(135)      # 135° around the xy-plane
        elev = math.radians(20)       # ~20° above the xy-plane
        x_eye = r_val * math.cos(elev) * math.cos(azim)
        y_eye = r_val * math.cos(elev) * math.sin(azim)
        z_eye = r_val * math.sin(elev)
        
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=x_eye, y=y_eye, z=z_eye)
                )
            ),
            width=800,
            height=600
        )
        
        return fig

# Usage Examples:
# 1) Default settings:
# sim = StochasticGradientDescentSimulator()
#
# 2) Zoom in more (camera_distance=0.5):
# sim = StochasticGradientDescentSimulator(camera_distance=0.5)
#
# 3) Use a smaller domain (grid_range=(-5, 5)) with a closer camera:
# sim = StochasticGradientDescentSimulator(grid_range=(-5, 5), camera_distance=0.5)
#
# Finally, to display the figure:
# sim.fig.show()
