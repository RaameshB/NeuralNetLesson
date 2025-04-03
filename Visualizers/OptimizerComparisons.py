import numpy as np
import math
import plotly.graph_objects as go

class MultiMinimaSimulatorAdaptive:
    def __init__(self, 
                 learning_rate=0.7,
                 title="3D Surface Plot: multiMinima",
                 ball_radius=0.15,
                 max_iter=100,
                 tolerance=1e-2,
                 initial_pos=(7.5, 4.2),
                 grid_range=(-5, 10),
                 grid_points=100,
                 # Parameters for adaptive methods:
                 beta=0.9,         # for RMSprop
                 epsilon=1e-8,     # for RMSprop, AdaGrad, and Adam
                 beta1=0.9,        # for Adam
                 beta2=0.999,      # for Adam
                 # Camera parameters:
                 azim=135,         # in degrees
                 elev=25,          # in degrees
                 r_cam=2):         # camera distance
        """
        :param learning_rate: Common step size for all methods.
        :param title: Plot title.
        :param ball_radius: Vertical offset for the markers (so they appear above the surface).
        :param max_iter: Maximum number of iterations.
        :param tolerance: Convergence tolerance based on step size.
        :param initial_pos: Starting position (x0, y0) for all methods.
        :param grid_range: Tuple (min, max) for the x and y domain of the surface.
        :param grid_points: Number of points along each axis for the surface grid.
        :param beta: Decay parameter for RMSprop.
        :param epsilon: Small constant to avoid division by zero.
        :param beta1: Exponential decay rate for Adam first moment.
        :param beta2: Exponential decay rate for Adam second moment.
        :param azim: Camera azimuth angle (degrees).
        :param elev: Camera elevation angle (degrees).
        :param r_cam: Camera distance from the origin.
        """
        self.learning_rate = learning_rate
        self.title = title
        self.ball_radius = ball_radius
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_pos = initial_pos
        self.grid_range = grid_range
        self.grid_points = grid_points
        self.beta = beta
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.azim = azim
        self.elev = elev
        self.r_cam = r_cam
        
        # Build simulation immediately.
        self.fig = self._build_simulation()
    
    def f_multi_func(self, x, y):
        """MultiMinima objective function."""
        return -0.25 * (
            (10/np.sqrt(2*np.pi) * np.exp(- (x**2)/4) + 4/np.sqrt(2*np.pi) * np.exp(-((x-4)**2)/4)) *
            (10/np.sqrt(2*np.pi) * np.exp(- (y**2)/4) + 4/np.sqrt(2*np.pi) * np.exp(-((y-2)**2)/4))
        )
    
    def numerical_grad(self, f, x, y, h=1e-5):
        """Compute the numerical gradient using central differences."""
        df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return df_dx, df_dy
    
    def _build_simulation(self):
        # ------------------------------------------------------------
        # 1. Create the multiMinima surface.
        # ------------------------------------------------------------
        x_vals = np.linspace(self.grid_range[0], self.grid_range[1], self.grid_points)
        y_vals = np.linspace(self.grid_range[0], self.grid_range[1], self.grid_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.f_multi_func(X, Y)
        
        # Create the surface; hide it from the legend.
        surface_multi = go.Surface(
            x=X, y=Y, z=Z, name="multiMinima", opacity=0.8, showlegend=False
        )
        
        # Default z-range (will update after simulation).
        default_z_range = [-3.5, 1]
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
        # 2. Precompute the Gradient Descent Simulations for each method.
        # ------------------------------------------------------------
        # Unpack the starting point.
        x0, y0 = self.initial_pos
        
        # Initialize trajectories for each method.
        # RMSprop (green)
        traj_rms_x = [x0]
        traj_rms_y = [y0]
        traj_rms_z = [self.f_multi_func(x0, y0) + self.ball_radius]
        v_rms_x, v_rms_y = 0.0, 0.0
        rms_converged = False
        rms_iter = None
        
        # AdaGrad (purple)
        traj_ada_x = [x0]
        traj_ada_y = [y0]
        traj_ada_z = [self.f_multi_func(x0, y0) + self.ball_radius]
        g_ada_x, g_ada_y = 0.0, 0.0
        ada_converged = False
        ada_iter = None
        
        # Normal Gradient Descent (red)
        traj_gd_x = [x0]
        traj_gd_y = [y0]
        traj_gd_z = [self.f_multi_func(x0, y0) + self.ball_radius]
        gd_converged = False
        gd_iter = None
        
        # Adam (light blue)
        traj_adam_x = [x0]
        traj_adam_y = [y0]
        traj_adam_z = [self.f_multi_func(x0, y0) + self.ball_radius]
        m_adam_x, m_adam_y = 0.0, 0.0
        v_adam_x, v_adam_y = 0.0, 0.0
        adam_converged = False
        adam_iter = None
        t_adam = 0
        
        # Run the simulation for max_iter iterations.
        for i in range(self.max_iter):
            # --- RMSprop update (green) ---
            current_rms_x = traj_rms_x[-1]
            current_rms_y = traj_rms_y[-1]
            if not rms_converged:
                grad_rms_x, grad_rms_y = self.numerical_grad(self.f_multi_func, current_rms_x, current_rms_y)
                v_rms_x = self.beta * v_rms_x + (1 - self.beta) * (grad_rms_x**2)
                v_rms_y = self.beta * v_rms_y + (1 - self.beta) * (grad_rms_y**2)
                step_rms_x = (self.learning_rate / (math.sqrt(v_rms_x) + self.epsilon)) * grad_rms_x
                step_rms_y = (self.learning_rate / (math.sqrt(v_rms_y) + self.epsilon)) * grad_rms_y
                step_rms = math.sqrt(step_rms_x**2 + step_rms_y**2)
                if step_rms < self.tolerance:
                    rms_converged = True
                    rms_iter = i
                    new_rms_x, new_rms_y = current_rms_x, current_rms_y
                else:
                    new_rms_x = current_rms_x - step_rms_x
                    new_rms_y = current_rms_y - step_rms_y
            else:
                new_rms_x, new_rms_y = current_rms_x, current_rms_y
            traj_rms_x.append(new_rms_x)
            traj_rms_y.append(new_rms_y)
            traj_rms_z.append(self.f_multi_func(new_rms_x, new_rms_y) + self.ball_radius)
            
            # --- AdaGrad update (purple) ---
            current_ada_x = traj_ada_x[-1]
            current_ada_y = traj_ada_y[-1]
            if not ada_converged:
                grad_ada_x, grad_ada_y = self.numerical_grad(self.f_multi_func, current_ada_x, current_ada_y)
                g_ada_x += grad_ada_x**2
                g_ada_y += grad_ada_y**2
                step_ada_x = (self.learning_rate / (math.sqrt(g_ada_x) + self.epsilon)) * grad_ada_x
                step_ada_y = (self.learning_rate / (math.sqrt(g_ada_y) + self.epsilon)) * grad_ada_y
                step_ada = math.sqrt(step_ada_x**2 + step_ada_y**2)
                if step_ada < self.tolerance:
                    ada_converged = True
                    ada_iter = i
                    new_ada_x, new_ada_y = current_ada_x, current_ada_y
                else:
                    new_ada_x = current_ada_x - step_ada_x
                    new_ada_y = current_ada_y - step_ada_y
            else:
                new_ada_x, new_ada_y = current_ada_x, current_ada_y
            traj_ada_x.append(new_ada_x)
            traj_ada_y.append(new_ada_y)
            traj_ada_z.append(self.f_multi_func(new_ada_x, new_ada_y) + self.ball_radius)
            
            # --- Normal Gradient Descent update (red) ---
            current_gd_x = traj_gd_x[-1]
            current_gd_y = traj_gd_y[-1]
            if not gd_converged:
                grad_gd_x, grad_gd_y = self.numerical_grad(self.f_multi_func, current_gd_x, current_gd_y)
                step_gd_x = self.learning_rate * grad_gd_x
                step_gd_y = self.learning_rate * grad_gd_y
                step_gd = math.sqrt(step_gd_x**2 + step_gd_y**2)
                if step_gd < self.tolerance:
                    gd_converged = True
                    gd_iter = i
                    new_gd_x, new_gd_y = current_gd_x, current_gd_y
                else:
                    new_gd_x = current_gd_x - step_gd_x
                    new_gd_y = current_gd_y - step_gd_y
            else:
                new_gd_x, new_gd_y = current_gd_x, current_gd_y
            traj_gd_x.append(new_gd_x)
            traj_gd_y.append(new_gd_y)
            traj_gd_z.append(self.f_multi_func(new_gd_x, new_gd_y) + self.ball_radius)
            
            # --- Adam update (light blue) ---
            current_adam_x = traj_adam_x[-1]
            current_adam_y = traj_adam_y[-1]
            if not adam_converged:
                t_adam += 1
                grad_adam_x, grad_adam_y = self.numerical_grad(self.f_multi_func, current_adam_x, current_adam_y)
                m_adam_x = self.beta1 * m_adam_x + (1 - self.beta1) * grad_adam_x
                m_adam_y = self.beta1 * m_adam_y + (1 - self.beta1) * grad_adam_y
                v_adam_x = self.beta2 * v_adam_x + (1 - self.beta2) * (grad_adam_x**2)
                v_adam_y = self.beta2 * v_adam_y + (1 - self.beta2) * (grad_adam_y**2)
                m_hat_x = m_adam_x / (1 - self.beta1**t_adam)
                m_hat_y = m_adam_y / (1 - self.beta1**t_adam)
                v_hat_x = v_adam_x / (1 - self.beta2**t_adam)
                v_hat_y = v_adam_y / (1 - self.beta2**t_adam)
                step_adam_x = self.learning_rate * m_hat_x / (math.sqrt(v_hat_x) + self.epsilon)
                step_adam_y = self.learning_rate * m_hat_y / (math.sqrt(v_hat_y) + self.epsilon)
                step_adam = math.sqrt(step_adam_x**2 + step_adam_y**2)
                if step_adam < self.tolerance:
                    adam_converged = True
                    adam_iter = i
                    new_adam_x, new_adam_y = current_adam_x, current_adam_y
                else:
                    new_adam_x = current_adam_x - step_adam_x
                    new_adam_y = current_adam_y - step_adam_y
            else:
                new_adam_x, new_adam_y = current_adam_x, current_adam_y
            traj_adam_x.append(new_adam_x)
            traj_adam_y.append(new_adam_y)
            traj_adam_z.append(self.f_multi_func(new_adam_x, new_adam_y) + self.ball_radius)
        
        # Use the length from one of the methods (all are updated equally).
        num_iterations = len(traj_rms_x)
        
        # Update scene ranges so all trajectories remain visible.
        all_x = np.array(traj_rms_x + traj_ada_x + traj_gd_x + traj_adam_x)
        all_y = np.array(traj_rms_y + traj_ada_y + traj_gd_y + traj_adam_y)
        all_z = np.array(traj_rms_z + traj_ada_z + traj_gd_z + traj_adam_z)
        x_min = min(-10, np.min(all_x)) - 1
        x_max = max(10, np.max(all_x)) + 1
        y_min = min(-10, np.min(all_y)) - 1
        y_max = max(10, np.max(all_y)) + 1
        z_min_sim = min(default_z_range[0], np.min(all_z)) - 0.5
        z_max_sim = max(default_z_range[1], np.max(all_z)) + 0.5
        
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
            # For each method, create the ball and path traces at iteration i.
            rms_ball = go.Scatter3d(
                x=[traj_rms_x[i]],
                y=[traj_rms_y[i]],
                z=[traj_rms_z[i]],
                mode='markers',
                marker=dict(color='green', size=10),
                name="RMSprop Ball"
            )
            rms_path = go.Scatter3d(
                x=traj_rms_x[:i+1],
                y=traj_rms_y[:i+1],
                z=traj_rms_z[:i+1],
                mode='markers',
                marker=dict(color='green', size=5),
                name="RMSprop Path"
            )
            adagrad_ball = go.Scatter3d(
                x=[traj_ada_x[i]],
                y=[traj_ada_y[i]],
                z=[traj_ada_z[i]],
                mode='markers',
                marker=dict(color='purple', size=10),
                name="AdaGrad Ball"
            )
            adagrad_path = go.Scatter3d(
                x=traj_ada_x[:i+1],
                y=traj_ada_y[:i+1],
                z=traj_ada_z[:i+1],
                mode='markers',
                marker=dict(color='purple', size=5),
                name="AdaGrad Path"
            )
            gd_ball = go.Scatter3d(
                x=[traj_gd_x[i]],
                y=[traj_gd_y[i]],
                z=[traj_gd_z[i]],
                mode='markers',
                marker=dict(color='red', size=10),
                name="GD Ball"
            )
            gd_path = go.Scatter3d(
                x=traj_gd_x[:i+1],
                y=traj_gd_y[:i+1],
                z=traj_gd_z[:i+1],
                mode='markers',
                marker=dict(color='red', size=5),
                name="GD Path"
            )
            adam_ball = go.Scatter3d(
                x=[traj_adam_x[i]],
                y=[traj_adam_y[i]],
                z=[traj_adam_z[i]],
                mode='markers',
                marker=dict(color='lightblue', size=10),
                name="Adam Ball"
            )
            adam_path = go.Scatter3d(
                x=traj_adam_x[:i+1],
                y=traj_adam_y[:i+1],
                z=traj_adam_z[:i+1],
                mode='markers',
                marker=dict(color='lightblue', size=5),
                name="Adam Path"
            )
            
            # For display, if a method has converged, show its converged iteration.
            rms_disp = rms_iter if rms_converged and i >= rms_iter else i
            ada_disp = ada_iter if ada_converged and i >= ada_iter else i
            gd_disp  = gd_iter  if gd_converged  and i >= gd_iter  else i
            adam_disp= adam_iter if adam_converged and i >= adam_iter else i
            
            annotation_text = (
                f"RMSprop: {rms_disp} | AdaGrad: {ada_disp} | GD: {gd_disp} | Adam: {adam_disp}"
            )
            
            frame_data = [surface_multi, rms_ball, rms_path,
                          adagrad_ball, adagrad_path,
                          gd_ball, gd_path,
                          adam_ball, adam_path]
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    annotations=[dict(
                        text=annotation_text,
                        x=0.95, y=0.95, xref="paper", yref="paper",
                        font=dict(color="red", size=16),
                        showarrow=False
                    )]
                )
            ))
        
        # ------------------------------------------------------------
        # 4. Add Initial Traces and Animation Button.
        # ------------------------------------------------------------
        # Initial traces for each method.
        initial_rms_ball = go.Scatter3d(
            x=[traj_rms_x[0]],
            y=[traj_rms_y[0]],
            z=[traj_rms_z[0]],
            mode='markers',
            marker=dict(color='green', size=10),
            name="RMSprop Ball"
        )
        initial_rms_path = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(color='green', size=5),
            name="RMSprop Path"
        )
        initial_adagrad_ball = go.Scatter3d(
            x=[traj_ada_x[0]],
            y=[traj_ada_y[0]],
            z=[traj_ada_z[0]],
            mode='markers',
            marker=dict(color='purple', size=10),
            name="AdaGrad Ball"
        )
        initial_adagrad_path = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(color='purple', size=5),
            name="AdaGrad Path"
        )
        initial_gd_ball = go.Scatter3d(
            x=[traj_gd_x[0]],
            y=[traj_gd_y[0]],
            z=[traj_gd_z[0]],
            mode='markers',
            marker=dict(color='red', size=10),
            name="GD Ball"
        )
        initial_gd_path = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(color='red', size=5),
            name="GD Path"
        )
        initial_adam_ball = go.Scatter3d(
            x=[traj_adam_x[0]],
            y=[traj_adam_y[0]],
            z=[traj_adam_z[0]],
            mode='markers',
            marker=dict(color='lightblue', size=10),
            name="Adam Ball"
        )
        initial_adam_path = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(color='lightblue', size=5),
            name="Adam Path"
        )
        
        # Add these initial traces to the figure.
        fig.add_trace(initial_rms_ball)
        fig.add_trace(initial_rms_path)
        fig.add_trace(initial_adagrad_ball)
        fig.add_trace(initial_adagrad_path)
        fig.add_trace(initial_gd_ball)
        fig.add_trace(initial_gd_path)
        fig.add_trace(initial_adam_ball)
        fig.add_trace(initial_adam_path)
        
        # Add an animation button.
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
                text="RMSprop: 0 | AdaGrad: 0 | GD: 0 | Adam: 0",
                x=0.95, y=0.95, xref="paper", yref="paper",
                font=dict(color="red", size=16),
                showarrow=False
            )],
            # Position the legend just below the annotations.
            legend=dict(
                x=0.95,
                y=0.87,
                xanchor="right",
                yanchor="top",
                font=dict(color="white", size=12),
                bgcolor='rgba(0,0,0,0)'
            ),
            uirevision='constant'
        )
        
        fig.frames = frames
        
        # ------------------------------------------------------------
        # 5. Set the Initial Camera Position.
        # ------------------------------------------------------------
        # Convert azimuth and elevation to radians.
        azim_rad = math.radians(self.azim)
        elev_rad = math.radians(self.elev)
        x_eye = self.r_cam * math.cos(elev_rad) * math.cos(azim_rad)
        y_eye = self.r_cam * math.cos(elev_rad) * math.sin(azim_rad)
        z_eye = self.r_cam * math.sin(elev_rad)
        
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
# Create an instance of the simulator with your desired settings.
