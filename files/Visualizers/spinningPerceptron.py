import numpy as np
import plotly.graph_objects as go

class PerceptronPlotlyDemo:
    def __init__(
        self, 
        title="Perceptron Simulation (Plotly)", 
        n_points=100, 
        max_updates=200, 
        seed=42, 
        linearlySeperable=0,
        blueName="Rich",
        redName="Not Rich"
    ):
        """
        Initialize the simulation parameters.
        
        Parameters:
          - title: Title of the figure.
          - n_points: Number of points per cluster (for XOR, per class).
          - max_updates: Maximum number of perceptron updates.
          - seed: Random seed for reproducibility.
          - linearlySeperable: An integer flag for the dataset type:
                0 -> Linearly separable clusters.
                1 -> Overlapping clusters with slight overlap.
                2 -> XOR problem in all four quadrants.
          - blueName: Name for the blue (positive) class.
          - redName: Name for the red (negative) class.
        """
        self.title = title
        self.n_points = n_points
        self.max_updates = max_updates
        self.seed = seed
        self.linearlySeperable = linearlySeperable
        self.blueName = blueName
        self.redName = redName
        
        # Set random seed
        np.random.seed(seed)
        
        if self.linearlySeperable == 0:
            # Linearly separable clusters:
            self.blue = np.random.randn(n_points, 2) * 1 + np.array([2, 2])
            self.red = np.random.randn(n_points, 2) * 1 + np.array([-2, -2])
        elif self.linearlySeperable == 1:
            # Overlapping clusters with slight overlap:
            self.blue = np.random.randn(n_points, 2) + np.array([1, 1])
            self.red = np.random.randn(n_points, 2) + np.array([-1, -1])
        elif self.linearlySeperable == 2:
            # XOR problem in all four quadrants with clipping:
            # For class +1 (blue): two clusters
            n_half = self.n_points // 2
            # Cluster in Quadrant I (centered at (1,1)): clip so x,y >= 0
            blue1 = np.random.randn(n_half, 2) * 0.5 + np.array([1, 1])
            blue1 = np.clip(blue1, 0, None)
            # Cluster in Quadrant III (centered at (-1,-1)): clip so x,y <= 0
            blue2 = np.random.randn(self.n_points - n_half, 2) * 0.5 + np.array([-1, -1])
            blue2 = np.clip(blue2, None, 0)
            self.blue = np.vstack([blue1, blue2])
            
            # For class -1 (red): two clusters
            # Cluster in Quadrant II (centered at (-1,1)): clip so x <= 0 and y >= 0
            red1 = np.random.randn(n_half, 2) * 0.5 + np.array([-1, 1])
            red1_x = np.clip(red1[:, 0], None, 0)[:, np.newaxis]
            red1_y = np.clip(red1[:, 1], 0, None)[:, np.newaxis]
            red1 = np.hstack([red1_x, red1_y])
            # Cluster in Quadrant IV (centered at (1,-1)): clip so x >= 0 and y <= 0
            red2 = np.random.randn(self.n_points - n_half, 2) * 0.5 + np.array([1, -1])
            red2_x = np.clip(red2[:, 0], 0, None)[:, np.newaxis]
            red2_y = np.clip(red2[:, 1], None, 0)[:, np.newaxis]
            red2 = np.hstack([red2_x, red2_y])
            self.red = np.vstack([red1, red2])
        else:
            raise ValueError("linearlySeperable must be 0, 1, or 2.")
        
        # Combine data and labels
        self.X = np.vstack([self.blue, self.red])
        # Blue: label +1, Red: label -1
        self.y = np.hstack([np.ones(self.blue.shape[0]), -np.ones(self.red.shape[0])])
        
        # Add bias term to each datapoint (homogeneous coordinates)
        ones = np.ones((self.X.shape[0], 1))
        self.X_aug = np.hstack([self.X, ones])
        
        # Initialize hyperplane: 3 parameters (w1, w2, bias)
        self.w = np.random.randn(3)
    
    def hyperplane_line(self, w, x_range):
        """Compute y-values for the hyperplane given x_range and weight vector w."""
        if abs(w[1]) > 1e-5:
            return -(w[0] / w[1]) * x_range - (w[2] / w[1])
        else:
            return None  # handle vertical line separately
    
    def create_animation(self):
        """
        Precompute frames for the simulation and return:
          - x_line: array of x values for the hyperplane
          - frames: a list of go.Frame objects for animation
          - initial_line: the initial hyperplane line trace
          - maxReached: boolean flag indicating if max_updates was reached
        """
        # Prepare x values for drawing the hyperplane
        x_line = np.linspace(np.min(self.X[:, 0]) - 1, np.max(self.X[:, 0]) + 1, 200)
        
        # Make a copy of w so as not to overwrite the original
        current_w = self.w.copy()
        
        frames = []
        updates = 0
        frame_count = 0
        
        while updates < self.max_updates:
            updated = False
            # Try to find a misclassified point
            for i in range(self.X_aug.shape[0]):
                x_i = self.X_aug[i]
                y_i = self.y[i]
                prediction = np.sign(np.dot(current_w, x_i))
                if prediction == 0:
                    prediction = -1
                if prediction != y_i:
                    current_w += y_i * x_i
                    updates += 1
                    updated = True
                    break
            
            if not updated and self.linearlySeperable == 0:
                # For linearly separable data, we stop when no misclassified point is found.
                break
            
            # Compute new line trace for this frame
            y_line = self.hyperplane_line(current_w, x_line)
            if y_line is None:
                line_trace = go.Scatter(
                    x=[-current_w[2] / current_w[0]] * 2,
                    y=[np.min(self.X[:, 1]) - 1, np.max(self.X[:, 1]) + 1],
                    mode="lines",
                    name="Hyperplane",
                )
            else:
                line_trace = go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Hyperplane",
                )
            frames.append(
                go.Frame(
                    data=[line_trace],
                    name=f"frame{frame_count}",
                    traces=[2]
                )
            )
            frame_count += 1
        
        maxReached = (updates == self.max_updates)
        
        # Create the initial line trace from the original w
        init_y_line = self.hyperplane_line(self.w, x_line)
        if init_y_line is None:
            initial_line = go.Scatter(
                x=[-self.w[2] / self.w[0]] * 2,
                y=[np.min(self.X[:, 1]) - 1, np.max(self.X[:, 1]) + 1],
                mode="lines",
                name="Hyperplane",
            )
        else:
            initial_line = go.Scatter(
                x=x_line,
                y=init_y_line,
                mode="lines",
                name="Hyperplane",
            )
        
        return x_line, frames, initial_line, maxReached

    def __call__(self):
        # Prepare the data for the figure
        x_line, frames, initial_line, maxReached = self.create_animation()
        
        # Scatter traces for blue & red clusters (constant, indices 0 and 1)
        trace_blue = go.Scatter(
            x=self.blue[:, 0],
            y=self.blue[:, 1],
            mode="markers",
            marker=dict(color="blue"),
            name=self.blueName + " (+1)"
        )
        trace_red = go.Scatter(
            x=self.red[:, 0],
            y=self.red[:, 1],
            mode="markers",
            marker=dict(color="red"),
            name=self.redName + " (0)"
        )
        
        data = [trace_blue, trace_red, initial_line]
        
        layout = go.Layout(
            title=self.title,
            width=1000,
            height=600,
            xaxis=dict(range=[np.min(self.X[:, 0]) - 1, np.max(self.X[:, 0]) + 1]),
            yaxis=dict(range=[np.min(self.X[:, 1]) - 1, np.max(self.X[:, 1]) + 1]),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    showactive=False,
                    x=0.5,
                    y=0.95,
                    xanchor="center",
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}
                                },
                            ],
                        ),
                        dict(
                            label="Restart",
                            method="animate",
                            args=[
                                ["frame0"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                },
                            ],
                        ),
                    ],
                )
            ],
        )
        
        # If max iterations were reached, add an annotation in red text
        if maxReached:
            layout.annotations = [
                dict(
                    text="<b>FORCE STOP: Perceptron could not classify the data.</b>",
                    font=dict(color="red", size=16),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.1,
                    showarrow=False,
                )
            ]
        
        fig = go.Figure(
            data=data,
            layout=layout,
            frames=frames
        )
        
        fig.show()
