import plotly.graph_objects as go
import numpy as np

# Define colormap
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    unique_labels = np.unique(obj_labels)
    colors = np.random.rand(len(unique_labels), 3)  # Random colors for simplicity
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    return np.array([label_to_color[label] for label in obj_labels])


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    """
    Visualize point cloud and bounding boxes in Plotly.
    """
    fig = go.Figure()

    # Add points to the plot
    if point_colors is None:
        point_colors = np.ones((points.shape[0], 3))  # Default white points

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=2, color=point_colors, opacity=0.4),
            name="Points",
        )
    )

    # Add the origin coordinate frame
    if draw_origin:
        fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0], mode="lines", name="X-axis", line=dict(color="red", width=5)))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 1], z=[0, 0], mode="lines", name="Y-axis", line=dict(color="green", width=5)))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1], mode="lines", name="Z-axis", line=dict(color="blue", width=5)))

    # Add GT boxes
    if gt_boxes is not None:
        fig = add_boxes(fig, gt_boxes, color="blue", name="GT Boxes")

    # Add reference boxes
    if ref_boxes is not None:
        fig = add_boxes(fig, ref_boxes, color="green", labels=ref_labels, scores=ref_scores, name="Ref Boxes")

    fig.update_layout(scene=dict(aspectmode="data"), title="3D Visualization")
    fig.show()


def add_boxes(fig, boxes, color="green", labels=None, scores=None, name="Bounding Boxes"):
    """
    Add 3D bounding boxes to the Plotly figure.
    """
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label_color = box_colormap[labels[i]] if labels is not None else color

        # Get box vertices and edges
        vertices, edges = get_box_vertices_and_edges(box)

        # Add edges as lines
        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[vertices[edge[0], 0], vertices[edge[1], 0]],
                    y=[vertices[edge[0], 1], vertices[edge[1], 1]],
                    z=[vertices[edge[0], 2], vertices[edge[1], 2]],
                    mode="lines",
                    line=dict(color=label_color, width=2),
                    name=name,
                )
            )

        # Optionally add a score label
        if scores is not None:
            center = np.mean(vertices, axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode="text",
                    text=[f"{scores[i]:.2f}"],
                    textposition="top center",
                    name=f"Score: {scores[i]:.2f}",
                )
            )
    return fig


def get_box_vertices_and_edges(box):
    """
    Generate 3D bounding box vertices and edges.
    Args:
        box: [x, y, z, length, width, height, heading]
    Returns:
        vertices: [8, 3] array of vertices
        edges: [12, 2] list of edges connecting vertices
    """
    x, y, z, l, w, h, heading = box
    rot_matrix = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1],
    ])
    dx, dy, dz = l / 2, w / 2, h / 2

    # Vertices in local box coordinates
    vertices_local = np.array([
        [-dx, -dy, -dz], [-dx, dy, -dz], [dx, dy, -dz], [dx, -dy, -dz],
        [-dx, -dy, dz], [-dx, dy, dz], [dx, dy, dz], [dx, -dy, dz],
    ])

    # Rotate and translate to global coordinates
    vertices = vertices_local @ rot_matrix.T + np.array([x, y, z])

    # Edges connecting vertices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical lines
    ]

    return vertices, edges
