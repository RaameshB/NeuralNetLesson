# neural_network_visualization.py

import plotly.graph_objects as go
import numpy as np
import math

class NeuralNetworkVisualization:
    def __init__(self, batchSize=3, nodes_per_layer=None, input_offset=-0.5, arrow_offset=0.2,
                 bias=False, node_diameter=0.5):
        """
        Initializes and displays a 3D neural network visualization.
        
        Parameters:
          batchSize (int): Number of batch slices along the z-axis.
          nodes_per_layer (list): List of number of nodes per layer.
            Default: [3, 5, 4, 7, 3, 5, 5]
          input_offset (float): Horizontal offset for the input layer.
          arrow_offset (float): Offset for arrow lines from node centers.
          bias (bool): If True, add a bias node to the input and eligible hidden layers.
          node_diameter (float): Used for spacing the bias node above the real nodes.
        """
        if nodes_per_layer is None:
            nodes_per_layer = [3, 5, 4, 7, 3, 5, 5]
        self.batchSize = batchSize
        self.nodes_per_layer = nodes_per_layer
        self.n_layers = len(nodes_per_layer)
        self.input_offset = input_offset
        self.arrow_offset = arrow_offset
        self.bias = bias
        self.node_diameter = node_diameter

        # Compute z-offsets for batch replication.
        if self.batchSize % 2 == 1:
            self.z_offsets = np.linspace(-((self.batchSize-1)/2), ((self.batchSize-1)/2), self.batchSize)
        else:
            self.z_offsets = np.linspace(-((self.batchSize)/2 - 0.5), ((self.batchSize)/2 - 0.5), self.batchSize)

        # Build 2D node coordinates for each layer.
        # (If bias is enabled, we add an extra node on top of real nodes for eligible layers.)
        self.layers_coords_2d = []
        for layer in range(self.n_layers):
            n_nodes = self.nodes_per_layer[layer]
            coords = []
            if n_nodes > 1:
                ys = np.linspace(-(n_nodes-1)/2, (n_nodes-1)/2, n_nodes)
            else:
                ys = [0]
            # For input layer, adjust x by input_offset.
            for y in ys:
                x = layer if layer != 0 else layer + self.input_offset
                coords.append((x, y))
            # Determine if we add a bias node:
            # For input layer and hidden layers except the second-to-last.
            if self.bias and (layer == 0 or (layer > 0 and layer < self.n_layers - 1 and layer != self.n_layers - 2)):
                if layer == 0:
                    # Place bias well above the top real input node so it clears the rectangle label.
                    top_y = max(ys)
                    bias_y = top_y + (self.node_diameter * 4)
                else:
                    spacing = ys[1] - ys[0] if len(ys) > 1 else 1
                    bias_y = max(ys) + spacing
                x = layer if layer != 0 else layer + self.input_offset
                coords.append((x, bias_y))
            self.layers_coords_2d.append(coords)

        # Replicate network for each batch (apply z-offset).
        self.all_layers_coords = []
        for z_off in self.z_offsets:
            batch_layers = []
            for layer in range(self.n_layers):
                coords = []
                for (x, y) in self.layers_coords_2d[layer]:
                    coords.append((x, y, z_off))
                batch_layers.append(coords)
            self.all_layers_coords.append(batch_layers)

        # Build arrow lines (separately for bias and non-bias).
        self.arrow_lines_traces = self.build_arrow_lines()

        # Build node traces (one trace per layer, combining batches) with bias nodes colored red.
        self.all_node_traces = []
        for layer in range(self.n_layers):
            xs, ys, zs, colors = [], [], [], []
            qualifies_bias = self.bias and (layer == 0 or (layer > 0 and layer < self.n_layers - 1 and layer != self.n_layers - 2))
            for batch in self.all_layers_coords:
                for idx, (x, y, z) in enumerate(batch[layer]):
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    if qualifies_bias and idx == len(batch[layer]) - 1:
                        colors.append("red")
                    else:
                        colors.append("lightgreen")
            trace = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=10, color=colors, symbol='circle'),
                hoverinfo='none'
            )
            self.all_node_traces.append(trace)

        # Build filled rectangular prisms for input and output.
        self.input_box_mesh = self.build_input_box()
        self.output_box_mesh = self.build_output_box()

        # Build input and output labels.
        self.input_label_trace = go.Scatter3d(
            x=[(self.x0_in_box + self.x1_in_box) / 2],
            y=[self.y_max_in_box + 0.2],
            z=[self.z_max_in_box],
            mode='text',
            text=["Input"],
            textfont=dict(family="Arial, sans-serif", color="black", size=14),
            hoverinfo='none'
        )
        self.output_label_trace = go.Scatter3d(
            x=[(self.x0_out_box + self.x1_out_box) / 2],
            y=[self.y_max_out_box + 0.2],
            z=[self.z_max_out_box],
            mode='text',
            text=["Output"],
            textfont=dict(family="Arial, sans-serif", color="black", size=14),
            hoverinfo='none'
        )

        # Build and show the figure.
        self.build_figure().show()

    def is_bias_node(self, layer, i, batch):
        """
        Returns True if the node at index i in the given batch and layer is a bias node.
        """
        qualifies_bias = self.bias and (layer == 0 or (layer > 0 and layer < self.n_layers - 1 and layer != self.n_layers - 2))
        return qualifies_bias and i == len(batch[layer]) - 1

    def build_arrow_lines(self):
        # We'll collect arrow segments separately for bias-origin arrows and regular arrows.
        arrow_line_x_bias, arrow_line_y_bias, arrow_line_z_bias = [], [], []
        arrow_line_x_regular, arrow_line_y_regular, arrow_line_z_regular = [], [], []
        for batch in self.all_layers_coords:
            for layer in range(self.n_layers - 1):
                # Determine if the target layer qualifies for a bias node.
                target_has_bias = self.bias and (layer+1 < self.n_layers - 1 and (layer+1) != self.n_layers - 2)
                if layer != self.n_layers - 2:
                    for i, (x0, y0, z0) in enumerate(batch[layer]):
                        for j, (x1, y1, z1) in enumerate(batch[layer+1]):
                            # Skip arrow if target node is bias.
                            if target_has_bias and j == len(batch[layer+1]) - 1:
                                continue
                            dx = x1 - x0
                            dy = y1 - y0
                            dz = z1 - z0
                            r = math.sqrt(dx**2 + dy**2 + dz**2)
                            if r > 0:
                                x_tail = x0 + self.arrow_offset * (dx / r)
                                y_tail = y0 + self.arrow_offset * (dy / r)
                                z_tail = z0 + self.arrow_offset * (dz / r)
                                x_tip = x1 - self.arrow_offset * (dx / r)
                                y_tip = y1 - self.arrow_offset * (dy / r)
                                z_tip = z1 - self.arrow_offset * (dz / r)
                            else:
                                x_tail, y_tail, z_tail = x0, y0, z0
                                x_tip, y_tip, z_tip = x1, y1, z1
                            if self.is_bias_node(layer, i, batch):
                                arrow_line_x_bias += [x_tail, x_tip, None]
                                arrow_line_y_bias += [y_tail, y_tip, None]
                                arrow_line_z_bias += [z_tail, z_tip, None]
                            else:
                                arrow_line_x_regular += [x_tail, x_tip, None]
                                arrow_line_y_regular += [y_tail, y_tip, None]
                                arrow_line_z_regular += [z_tail, z_tip, None]
                else:
                    # For connection from last hidden layer to output layer (assumed one-to-one mapping).
                    for (x0, y0, z0), (x1, y1, z1) in zip(batch[layer], batch[layer+1]):
                        x_tail = x0 + self.arrow_offset
                        x_tip = x1 - self.arrow_offset
                        y_tail = y0
                        y_tip = y1
                        z_tail = z0
                        z_tip = z1
                        arrow_line_x_regular += [x_tail, x_tip, None]
                        arrow_line_y_regular += [y_tail, y_tip, None]
                        arrow_line_z_regular += [z_tail, z_tip, None]
        # Create two separate traces.
        trace_bias = go.Scatter3d(
            x=arrow_line_x_bias,
            y=arrow_line_y_bias,
            z=arrow_line_z_bias,
            mode='lines',
            line=dict(color="red", width=2),
            hoverinfo='none'
        )
        trace_regular = go.Scatter3d(
            x=arrow_line_x_regular,
            y=arrow_line_y_regular,
            z=arrow_line_z_regular,
            mode='lines',
            line=dict(color="white", width=2),
            hoverinfo='none'
        )
        # Return the list of arrow traces.
        return [trace_regular, trace_bias]

    def build_input_box(self):
        # Gather all input layer nodes across batches, excluding the bias node.
        input_all = []
        for batch in self.all_layers_coords:
            if self.bias:
                input_all += batch[0][:self.nodes_per_layer[0]]
            else:
                input_all += batch[0]
        input_xs = [c[0] for c in input_all]
        input_ys = [c[1] for c in input_all]
        input_zs = [c[2] for c in input_all]
        x_center_in = np.mean(input_xs)
        x_box_margin = 0.3 + 0.1  # add x_buffer=0.1
        self.x0_in_box = x_center_in - x_box_margin
        self.x1_in_box = x_center_in + x_box_margin
        self.y_min_in_box = min(input_ys) - 0.3 - 0.1  # y_buffer=0.1
        self.y_max_in_box = max(input_ys) + 0.3 + 0.1
        self.z_min_in_box = min(input_zs) - 0.1
        self.z_max_in_box = max(input_zs) + 0.1
        return self.build_box_mesh(self.x0_in_box, self.x1_in_box,
                                   self.y_min_in_box, self.y_max_in_box,
                                   self.z_min_in_box, self.z_max_in_box,
                                   "RoyalBlue")

    def build_output_box(self):
        # Gather all output layer nodes across batches.
        output_all = []
        for batch in self.all_layers_coords:
            output_all += batch[-1]
        output_xs = [c[0] for c in output_all]
        output_ys = [c[1] for c in output_all]
        output_zs = [c[2] for c in output_all]
        x_center_out = np.mean(output_xs)
        x_box_margin = 0.3 + 0.1  # x_buffer=0.1
        self.x0_out_box = x_center_out - x_box_margin
        self.x1_out_box = x_center_out + x_box_margin
        self.y_min_out_box = min(output_ys) - 0.3 - 0.1  # y_buffer=0.1
        self.y_max_out_box = max(output_ys) + 0.3 + 0.1
        self.z_min_out_box = min(output_zs) - 0.1
        self.z_max_out_box = max(output_zs) + 0.1
        return self.build_box_mesh(self.x0_out_box, self.x1_out_box,
                                   self.y_min_out_box, self.y_max_out_box,
                                   self.z_min_out_box, self.z_max_out_box,
                                   "MistyRose")

    def build_box_mesh(self, x0, x1, y0, y1, z0, z1, color):
        vertices = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])
        faces = [
            (0,1,2), (0,2,3),  # bottom
            (4,5,6), (4,6,7),  # top
            (0,1,5), (0,5,4),  # front
            (3,2,6), (3,6,7),  # back
            (0,3,7), (0,7,4),  # left
            (1,2,6), (1,6,5)   # right
        ]
        I, J, K = [], [], []
        for (a, b, c) in faces:
            I.append(a)
            J.append(b)
            K.append(c)
        return go.Mesh3d(
            x=vertices[:,0],
            y=vertices[:,1],
            z=vertices[:,2],
            i=I,
            j=J,
            k=K,
            color=color,
            opacity=0.5,
            flatshading=True,
            hoverinfo='none'
        )

    def build_figure(self):
        # Build a legend (key) annotation in the bottom-right.
        legend_text = "<span style='color:lightgreen;'>●</span> Node<br>"
        if self.bias:
            legend_text += "<span style='color:red;'>●</span> Bias<br>"
        legend_text += "<span style='color:RoyalBlue;'>Input</span> Box<br><span style='color:MistyRose;'>Output</span> Box"
        legend_annotation = dict(
            x=0.95,
            y=0.05,
            xref="paper",
            yref="paper",
            text=legend_text,
            showarrow=False,
            align="right",
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            opacity=0.8
        )

        # Combine all data traces.
        data = self.arrow_lines_traces + [self.input_box_mesh, self.output_box_mesh,
                self.input_label_trace, self.output_label_trace] + self.all_node_traces

        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=dict(
                    text="Neural Network visualization for the entire Dataset",
                    font=dict(color="white")
                ),
                scene=dict(
                    xaxis=dict(title="Layer", range=[-1, self.n_layers + 0.5], visible=False),
                    yaxis=dict(title="Node Position", visible=False),
                    zaxis=dict(title="Z", range=[min(self.z_offsets)-1, max(self.z_offsets)+1], visible=False),
                    camera=dict(
                        projection=dict(type="orthographic"),
                        eye=dict(x=0, y=0, z=2)
                    ),
                    bgcolor="grey"
                ),
                width=1000,
                height=800,
                showlegend=False,
                paper_bgcolor="grey",
                annotations=[legend_annotation]
            )
        )
        return fig
