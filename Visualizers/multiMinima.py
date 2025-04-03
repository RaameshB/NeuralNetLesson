import numpy as np
import math
import plotly.graph_objects as go

class MultiMinimaSimulator:
    def __init__(self, 
                 learning_rate=0.7, 
                 title="3D Surface Plot: multiMinima",
                 ball_radius=0.15, 
                 max_iter=100, 
                 tolerance=1e-2, 
                 initial_pos=(7.5, 4.2),
                 grid_range=(-5, 10),  # Domain for surface grid (for both x and y)
                 grid_points=100,
                 azim = 90,
                 elev = 15,
                 camera_distance=2):  # Controls the zoom level for the camera
        """
        :param learning_rate: Step size for gradient descent.
        :param title: Title of the plot.
        :param ball_radius: Offset to draw the ball above the surface.
        :param max_iter: Maximum iterations for the simulation.
        :param tolerance: Stopping threshold for the gradient descent steps.
        :param initial_pos: Starting position (x0, y0) for the descent.
        :param grid_range: Tuple (min, max) used to generate the x and y grid for the surface.
        :param grid_points: Number of grid points in each dimension.
        :param camera_distance: Distance of the camera from the origin (smaller values zoom in).
        """
        self.learning_rate = learning_rate
        self.title = title
        self.ball_radius = ball_radius
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_pos = initial_pos
        self.grid_range = grid_range
        self.grid_points = grid_points
        self.camera_distance = camera_distance
        self.azim = azim
        self.elev = elev
        
        # Build the simulation immediately and store the resulting figure.
        self.fig = self._build_simulation()
    
    def f_multi_func(self, x, y):
        # The multiMinima objective function.
        return -0.25 * (
            (10/np.sqrt(2*np.pi) * np.exp(- (x**2)/4) + 4/np.sqrt(2*np.pi) * np.exp(-((x-4)**2)/4)) *
            (10/np.sqrt(2*np.pi) * np.exp(- (y**2)/4) + 4/np.sqrt(2*np.pi) * np.exp(-((y-2)**2)/4))
        )
    
    def numerical_grad(self, f, x, y, h=1e-5):
        # Central difference numerical gradient.
        df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return df_dx, df_dy

    def _build_simulation(self):
        # ------------------------------------------------------------
        # 1. Build the multiMinima surface.
        # ------------------------------------------------------------
        x_vals = np.linspace(self.grid_range[0], self.grid_range[1], self.grid_points)
        y_vals = np.linspace(self.grid_range[0], self.grid_range[1], self.grid_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.f_multi_func(X, Y)
        
        # Compute a default z-axis range that covers the entire surface.
        default_z_range = [np.min(Z) - 1, np.max(Z) + 1]
        
        surface_multi = go.Surface(
            x=X, y=Y, z=Z, name="multiMinima", opacity=0.8
        )
        
        # Create the base figure with the surface.
        # (Initial scene ranges for x and y are set to [-10,10] as in your original code.)
        fig = go.Figure(data=[surface_multi])
        fig.update_layout(
            title=self.title,
            title_font_color="white",
            paper_bgcolor="grey",
            plot_bgcolor="grey",
            scene=dict(
                xaxis=dict(visible=False, range=[-10, 10]),
                yaxis=dict(visible=False, range=[-10, 10]),
                zaxis=dict(visible=False, range=default_z_range),
                bgcolor="grey"
            )
        )
        
        # ------------------------------------------------------------
        # 2. Precompute the Gradient Descent Trajectory.
        # ------------------------------------------------------------
        x0, y0 = self.initial_pos
        traj_x = [x0]
        traj_y = [y0]
        traj_z = [self.f_multi_func(x0, y0) + self.ball_radius]
        
        for i in range(self.max_iter):
            current_x = traj_x[-1]
            current_y = traj_y[-1]
            grad_x, grad_y = self.numerical_grad(self.f_multi_func, current_x, current_y)
            step_x = self.learning_rate * grad_x
            step_y = self.learning_rate * grad_y
            if np.sqrt(step_x**2 + step_y**2) < self.tolerance:
                break
            new_x = current_x - step_x
            new_y = current_y - step_y
            traj_x.append(new_x)
            traj_y.append(new_y)
            traj_z.append(self.f_multi_func(new_x, new_y) + self.ball_radius)
            
        num_iterations = len(traj_x)
        
        # Update scene ranges based on the trajectory to ensure the ball stays visible.
        x_min = min(-10, np.min(traj_x)) - 1
        x_max = max(10, np.max(traj_x)) + 1
        y_min = min(-10, np.min(traj_y)) - 1
        y_max = max(10, np.max(traj_y)) + 1
        z_min_sim = min(default_z_range[0], np.min(traj_z)) - 0.5
        z_max_sim = max(default_z_range[1], np.max(traj_z)) + 0.5
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[x_min, x_max]),
                yaxis=dict(visible=False, range=[y_min, y_max]),
                zaxis=dict(visible=False, range=[z_min_sim, z_max_sim])
            )
        )
        
        # ------------------------------------------------------------
        # 3. Build Animation Frames.
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
            frame_data = [surface_multi, ball_trace, path_trace]
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
        # 4. Add Initial Traces and Animation Button.
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
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(
                        label="simulate gradient descent",
                        method="animate",
                        args=[None,
                              {"frame": {"duration": 100, "redraw": True},
                               "mode": "immediate",
                               "transition": {"duration": 0}}]
                    )],
                    pad={"r": 10, "t": 10},
                    x=0.5, xanchor="center",
                    y=0.0, yanchor="bottom"
                )
            ],
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
        # 5. Set the Initial Camera Position.
        # ------------------------------------------------------------
        # Use a camera position similar to the original code:
        # r = camera_distance, azimuth = 90°, elevation = 15°
        r_val = self.camera_distance
        azim = math.radians(self.azim)
        elev = math.radians(self.elev)
        x_eye = r_val * math.cos(elev) * math.cos(azim)
        y_eye = r_val * math.cos(elev) * math.sin(azim)
        z_eye = r_val * math.sin(elev)
        
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=x_eye, y=y_eye, z=z_eye)
                )
            )
        )
        
        return fig

# ------------------------------------------------------------------------------
# Usage Example:
