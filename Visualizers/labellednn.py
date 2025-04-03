import plotly.graph_objects as go
import numpy as np
import math

class NeuralNetworkPlot:
    def __init__(self,
                 title="Neural Network Diagram",
                 nodes_per_layer=[3, 5, 1, 2, 1, 1],
                 bias=False,
                 label_arrows=False,
                 arrow_label_overrides=None,
                 special_inter_layer_arrow_target_layer=None,
                 special_inter_layer_arrow_color="red",
                 special_inter_layer_arrow_offset = "layer",
                 node_diameter=0.5,
                 arrow_offset=0.2,
                 label_fraction=0.5,
                 # Colors for rectangles (you can add more customization as needed)
                 input_rect_line="RoyalBlue",
                 input_rect_fill="LightSkyBlue",
                 output_rect_line="DarkMagenta",
                 output_rect_fill="MistyRose",
                 classType="",
                 # Figure layout options:
                 fig_width=800,
                 fig_height=600,
                 bg_color='grey'):
        if arrow_label_overrides is None:
            arrow_label_overrides = {}
        self.classType = classType
        self.title = title
        self.nodes_per_layer = nodes_per_layer
        self.n_layers = len(nodes_per_layer)
        self.bias = bias  # New bias flag
        self.label_arrows = label_arrows
        self.arrow_label_overrides = arrow_label_overrides
        self.special_inter_layer_arrow_target_layer = special_inter_layer_arrow_target_layer
        self.special_inter_layer_arrow_color = special_inter_layer_arrow_color
        self.special_inter_layer_arrow_offset = special_inter_layer_arrow_offset
        self.node_diameter = node_diameter
        self.arrow_offset = arrow_offset
        self.label_fraction = label_fraction

        self.input_rect_line = input_rect_line
        self.input_rect_fill = input_rect_fill
        self.output_rect_line = output_rect_line
        self.output_rect_fill = output_rect_fill

        self.fig_width = fig_width
        self.fig_height = fig_height
        self.bg_color = bg_color

        # Build the neural network diagram.
        self._compute_node_coordinates()
        self._compute_rectangles()
        self._build_arrow_annotations()
        self._build_node_traces()
        self._build_figure()

    def _compute_node_coordinates(self):
        """
        Compute the coordinates for each layer’s nodes.
        For the input layer, if bias is enabled, we place the bias node significantly above
        the top real input node so that it won't overlap the 'Inputs:' label.
        """
        self.layers_coords = []

        for layer in range(self.n_layers):
            n_nodes = self.nodes_per_layer[layer]
            # The real nodes in this layer:
            if n_nodes > 1:
                ys = np.linspace(-(n_nodes - 1) / 2, (n_nodes - 1) / 2, n_nodes)
            else:
                ys = [0]

            layer_coords = []
            for y in ys:
                x = layer
                layer_coords.append((x, y))

            # Decide if we add a bias node for this layer.
            # 1) Always for the input layer if bias=True
            # 2) For hidden layers, except the second-to-last layer
            add_bias = False
            if self.bias:
                if layer == 0:  # input layer
                    add_bias = True
                elif (layer > 0 and layer < self.n_layers - 1 and layer != self.n_layers - 2):
                    add_bias = True

            if add_bias:
                # For input layer, place the bias node further above the top real node
                # to avoid overlapping with the rectangle label.
                if layer == 0:
                    top_node_y = max(ys)
                    bias_y = top_node_y + (self.node_diameter * 4)
                else:
                    if len(ys) > 1:
                        spacing = ys[1] - ys[0]
                    else:
                        spacing = 1
                    bias_y = max(ys) + spacing

                layer_coords.append((layer, bias_y))

            self.layers_coords.append(layer_coords)

    def _compute_rectangles(self):
        """Compute rectangle coordinates for the input and output layers.
           We exclude the bias node from the input rectangle area to keep it above the rectangle."""
        # For the input rectangle, use only the original input nodes (exclude bias node).
        if self.bias:
            input_non_bias = self.layers_coords[0][:self.nodes_per_layer[0]]
        else:
            input_non_bias = self.layers_coords[0]

        input_ys = [y for (_, y) in input_non_bias]
        input_y_min = min(input_ys)
        input_y_max = max(input_ys)
        margin_y_input = 0.5
        self.rect_y0_input = input_y_min - margin_y_input
        self.rect_y1_input = input_y_max + margin_y_input

        self.rect_width = 1.5 * self.node_diameter
        self.half_width = self.rect_width / 2
        # For input layer (layer 0), center the rectangle around x=0.
        self.rect_x0_input = 0 - self.half_width
        self.rect_x1_input = 0 + self.half_width

        # Output rectangle for the last layer.
        output_ys = [y for (_, y) in self.layers_coords[-1]]
        output_y_min = min(output_ys)
        output_y_max = max(output_ys)
        margin_y_output = 0.5
        self.rect_y0_output = output_y_min - margin_y_output
        self.rect_y1_output = output_y_max + margin_y_output
        # For output layer (at x = n_layers - 1), center the rectangle around that x.
        self.rect_x0_output = (self.n_layers - 1) - self.half_width
        self.rect_x1_output = (self.n_layers - 1) + self.half_width

    def _build_arrow_annotations(self):
        """Build arrow annotations for connections between nodes (and the special inter-layer arrow)."""
        self.annotations = []
        self.arrow_label_annotations = []

        # For layers 0 to n_layers-2.
        for layer in range(self.n_layers - 1):
            if layer != self.n_layers - 2:
                for i, (x0, y0) in enumerate(self.layers_coords[layer]):
                    for j, (x1, y1) in enumerate(self.layers_coords[layer + 1]):
                        # Skip drawing an arrow if the target node is a bias node in the next layer.
                        if (
                            self.bias
                            and (layer+1 > 0 and layer+1 < self.n_layers - 1 and layer+1 != self.n_layers - 2)
                            and j == len(self.layers_coords[layer + 1]) - 1
                        ):
                            continue

                        dx = x1 - x0
                        dy = y1 - y0
                        r = math.hypot(dx, dy)
                        if r > 0:
                            x_tail = x0 + self.arrow_offset * (dx / r)
                            y_tail = y0 + self.arrow_offset * (dy / r)
                            x_tip = x1 - self.arrow_offset * (dx / r)
                            y_tip = y1 - self.arrow_offset * (dy / r)
                        else:
                            x_tail, y_tail = x0, y0
                            x_tip, y_tip = x1, y1

                        # Determine if source node is actually a bias node
                        # The source layer has self.nodes_per_layer[layer] real nodes;
                        # if i == self.nodes_per_layer[layer], it's the bias node.
                        if (self.bias and i == self.nodes_per_layer[layer]):
                            arrow_color = "red"
                        else:
                            arrow_color = "white"

                        default_label = f"L{layer}:{self.nodes_per_layer[layer] - i} → L{layer+1}:{self.nodes_per_layer[layer+1] - j}"
                        label_text = self.arrow_label_overrides.get((layer, i, j), default_label) if self.label_arrows else ""

                        arrow_ann = dict(
                            x=x_tip,
                            y=y_tip,
                            ax=x_tail,
                            ay=y_tail,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor=arrow_color,
                            xanchor="center",
                            yanchor="middle"
                        )
                        self.annotations.append(arrow_ann)

                        if self.label_arrows:
                            label_x = x_tail + self.label_fraction * (x_tip - x_tail)
                            label_y = y_tail + self.label_fraction * (y_tip - y_tail)
                            label_ann = dict(
                                x=label_x,
                                y=label_y,
                                text=label_text,
                                xref="x",
                                yref="y",
                                showarrow=False,
                                font=dict(family="Arial, sans-serif", color="black", size=10),
                                xanchor="center",
                                yanchor="middle"
                            )
                            self.arrow_label_annotations.append(label_ann)

            else:
                # This is the second-to-last layer connecting to the last layer
                for i in range(len(self.layers_coords[layer])):
                    x0, y0 = self.layers_coords[layer][i]
                    if i < len(self.layers_coords[layer + 1]):
                        x1, y1 = self.layers_coords[layer + 1][i]
                    else:
                        x1, y1 = self.layers_coords[layer + 1][-1]

                    y_arrow = y0
                    x_tail = x0 + self.arrow_offset
                    x_tip = x1 - self.arrow_offset

                    # Again, check if the source node is truly a bias node
                    if (self.bias and i == self.nodes_per_layer[layer]):
                        arrow_color = "red"
                    else:
                        arrow_color = "white"

                    default_label = f"L{layer}:{self.nodes_per_layer[layer] - i}→L{self.n_layers-1}:{i}"
                    label_text = self.arrow_label_overrides.get((layer, i, i), default_label) if self.label_arrows else ""

                    arrow_ann = dict(
                        x=x_tip,
                        y=y_arrow,
                        ax=x_tail,
                        ay=y_arrow,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor=arrow_color,
                        xanchor="center",
                        yanchor="middle"
                    )
                    self.annotations.append(arrow_ann)

                    if self.label_arrows:
                        label_x = x_tail + self.label_fraction * (x_tip - x_tail)
                        label_ann = dict(
                            x=label_x,
                            y=y_arrow,
                            text=label_text,
                            xref="x",
                            yref="y",
                            showarrow=False,
                            font=dict(family="Arial, sans-serif", color="black", size=10),
                            xanchor="center",
                            yanchor="middle"
                        )
                        self.arrow_label_annotations.append(label_ann)

        # Special arrow if requested
        if self.special_inter_layer_arrow_target_layer is not None:
            target_layer = self.special_inter_layer_arrow_target_layer
            if self.special_inter_layer_arrow_offset == "layer":
                target_x = target_layer - 0.5
            else:
                target_x = target_layer
            target_ys = [y for (_, y) in self.layers_coords[target_layer]]
            target_bottom = min(target_ys) - 0.5
            overall_bottom = -max(self.nodes_per_layer) / 2 - 1
            extra_arrow = dict(
                x=target_x,
                y=target_bottom,
                ax=target_x,
                ay=overall_bottom,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=self.special_inter_layer_arrow_color,
                xanchor="center",
                yanchor="middle"
            )
            self.annotations.append(extra_arrow)
        
        self.all_annotations = self.annotations + self.arrow_label_annotations

    def _build_node_traces(self):
        """
        Create scatter traces for input, hidden, and output nodes.
        Only the input nodes (and bias in input) and the output nodes will show labels.
        Hidden nodes will have no text labels except bias nodes.
        All hover popups are disabled.
        """
        # --- Input layer ---
        self.input_node_x = []
        self.input_node_y = []
        self.input_text = []
        input_colors = []
        real_input_count = self.nodes_per_layer[0]  # real input nodes count
        for i, (x, y) in enumerate(self.layers_coords[0]):
            self.input_node_x.append(x)
            self.input_node_y.append(y)
            if i < real_input_count:
                # Label real input nodes in reversed order: topmost gets x1.
                self.input_text.append(f"x{real_input_count - i}")
                input_colors.append("lightgreen")
            else:
                # Bias node
                self.input_text.append("bias")
                input_colors.append("red")

        self.input_node_trace = go.Scatter(
            x=self.input_node_x,
            y=self.input_node_y,
            mode='markers+text',
            marker=dict(size=40, color=input_colors, line=dict(width=1, color='black')),
            text=self.input_text,
            textposition="middle center",
            hoverinfo='none'
        )

        # --- Hidden layers ---
        self.hidden_node_x = []
        self.hidden_node_y = []
        self.hidden_text = []
        hidden_colors = []

        for layer in range(1, self.n_layers - 1):
            for node_idx, (x, y) in enumerate(self.layers_coords[layer]):
                self.hidden_node_x.append(x)
                self.hidden_node_y.append(y)
                # Only label the bias node if present; otherwise leave blank.
                # We can detect the bias node if node_idx == self.nodes_per_layer[layer].
                if self.bias and node_idx == self.nodes_per_layer[layer]:
                    self.hidden_text.append("bias")
                    hidden_colors.append("red")
                else:
                    self.hidden_text.append("")
                    hidden_colors.append("lightgreen")

        self.hidden_node_trace = go.Scatter(
            x=self.hidden_node_x,
            y=self.hidden_node_y,
            mode='markers+text',
            marker=dict(size=40, color=hidden_colors, line=dict(width=1, color='black')),
            text=self.hidden_text,
            textposition="middle center",
            hoverinfo='none'
        )

        # --- Output layer ---
        self.output_node_x = []
        self.output_node_y = []
        self.output_text = []
        n_output = len(self.layers_coords[-1])
        for i, (x, y) in enumerate(self.layers_coords[-1]):
            self.output_node_x.append(x)
            self.output_node_y.append(y)
            # Label output nodes in reversed order so that the topmost node gets y1.
            self.output_text.append(f"y{n_output - i}")

        self.output_node_trace = go.Scatter(
            x=self.output_node_x,
            y=self.output_node_y,
            mode='markers+text',
            marker=dict(size=40, color='lightgreen', line=dict(width=1, color='black')),
            text=self.output_text,
            textposition="middle center",
            hoverinfo='none'
        )

    def _build_figure(self):
        """Create the full Plotly figure with nodes, arrows, rectangle annotations, etc."""
        input_rectangle_annotation = dict(
            x=(self.rect_x0_input + self.rect_x1_input) / 2,
            y=self.rect_y1_input + 0.2,
            xref="x",
            yref="y",
            text="Inputs:",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(family="Arial, sans-serif", color="black", size=14)
        )

        output_rectangle_annotation = dict(
            x=(self.rect_x0_output + self.rect_x1_output) / 2,
            y=self.rect_y1_output + 0.2,
            xref="x",
            yref="y",
            text="Outputs: " + self.classType,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(family="Arial, sans-serif", color="black", size=14)
        )

        self.fig = go.Figure(
            data=[
                self.input_node_trace,
                self.hidden_node_trace,
                self.output_node_trace
            ],
            layout=go.Layout(
                title=dict(
                    text=self.title,
                    font=dict(color="white", size=16)
                ),
                showlegend=False,
                xaxis=dict(
                    visible=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    visible=False,
                    showgrid=False,
                    zeroline=False
                ),
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
                width=self.fig_width,
                height=self.fig_height,
                shapes=[
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=self.rect_x0_input,
                        y0=self.rect_y0_input,
                        x1=self.rect_x1_input,
                        y1=self.rect_y1_input,
                        line=dict(color=self.input_rect_line),
                        fillcolor=self.input_rect_fill,
                        layer="below"
                    ),
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=self.rect_x0_output,
                        y0=self.rect_y0_output,
                        x1=self.rect_x1_output,
                        y1=self.rect_y1_output,
                        line=dict(color=self.output_rect_line),
                        fillcolor=self.output_rect_fill,
                        layer="below"
                    )
                ],
                annotations=[
                    input_rectangle_annotation,
                    output_rectangle_annotation
                ] + self.all_annotations
            )
        )

    def show(self):
        """Display the figure."""
        self.fig.show()


# -------------------------
# Example usage:
# -------------------------
