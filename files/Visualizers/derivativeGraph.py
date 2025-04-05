import numpy as np
import plotly.graph_objects as go

class PlotlySurfaceWithGradient:
    def __init__(self, flip=False, title=""):
        self.flip = flip
        self.title = title
        # Generate polar grid data so the top of the surface appears circular,
        # and restrict the domain so that f(x,y)=x^2+y^2 is between 0 and 1.
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.linspace(0, 1, 100)  # r in [0,1] ensures z = x^2+y^2 is in [0,1]
        r, theta = np.meshgrid(r, theta)
        self.X = r * np.cos(theta)
        self.Y = r * np.sin(theta)
        self.Z = self.X**2 + self.Y**2

    def __call__(self, gradient_point=None):
        # Create the surface trace with a Viridis colorscale and 80% opacity.
        surface = go.Surface(
            x=self.X,
            y=self.Y,
            z=self.Z,
            colorscale='Viridis',
            showscale=False,
            opacity=0.8
        )
        
        data = [surface]
        
        if gradient_point is not None:
            x0, y0 = gradient_point
            z0 = x0**2 + y0**2  # f(x0,y0)
            
            # For f(x,y)=x^2+y^2, the domain gradient is (2x, 2y)
            grad_domain = np.array([2 * x0, 2 * y0])
            if self.flip:
                grad_domain = -grad_domain
            # Desired arrow length in 3D will equal the magnitude of the domain gradient:
            L = np.linalg.norm(grad_domain)
            
            # A displacement (dx, dy) in the domain causes a change in z given by:
            # dz = f_x * dx + f_y * dy, where f_x = 2x and f_y = 2y.
            dx = 2 * x0
            dy = 2 * y0
            dz = 2 * x0 * dx + 2 * y0 * dy  # equals 4*x0^2 + 4*y0^2
            v_natural = np.array([dx, dy, dz])
            norm_v = np.linalg.norm(v_natural)
            # Scale v_natural to have length L (the original gradient magnitude)
            v_arrow = (v_natural / norm_v) * L
            
            # If flip is True, flip the 3D arrow direction as well.
            if self.flip:
                v_arrow = -v_arrow
            
            # Split the arrow into:
            # 1. Shaft: 80% of the arrow as a red line.
            # 2. Arrowhead: 20% of the arrow as a red cone.
            shaft_fraction = 0.8
            arrow_head_fraction = 1 - shaft_fraction
            
            start_point = np.array([x0, y0, z0])
            shaft_end = start_point + shaft_fraction * v_arrow
            arrow_head_vector = arrow_head_fraction * v_arrow
            
            # Add the shaft as a red line.
            arrow_line = go.Scatter3d(
                x=[start_point[0], shaft_end[0]],
                y=[start_point[1], shaft_end[1]],
                z=[start_point[2], shaft_end[2]],
                mode='lines',
                line=dict(color='red', width=5),
                showlegend=False
            )
            data.append(arrow_line)
            
            # Add the arrowhead as a cone with a smaller head (sizeref=0.5).
            arrow_cone = go.Cone(
                x=[shaft_end[0]],
                y=[shaft_end[1]],
                z=[shaft_end[2]],
                u=[arrow_head_vector[0]],
                v=[arrow_head_vector[1]],
                w=[arrow_head_vector[2]],
                anchor="tail",
                showscale=False,
                colorscale=[[0, 'red'], [1, 'red']],
                sizemode="absolute",
                sizeref=0.5
            )
            data.append(arrow_cone)
        
        # Create the figure with no visible axes, grey background, and set size to 800x600.
        fig = go.Figure(data=data)
        fig.update_layout(
            title=dict(
                text=self.title,
                x=0.5,
                xanchor='center',
                font=dict(color='white', size=20)
            ),
            width=800,   # width in pixels
            height=600,  # height in pixels
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='grey'
            ),
            paper_bgcolor='grey',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        fig.show()
