"""Plotly 3D Interactive Orbit Viewer."""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

def create_interactive_viewer(
    trajectory_positions: np.ndarray,
    labels: list[str],
    output_path: str = "orbit.html",
) -> None:
    """Create a 3D interactive Plotly visualization for the simulated trajectory."""
    T, N, _ = trajectory_positions.shape
    
    fig = go.Figure()
    
    # Premium colors: Planet, A, B, C, D
    # Let's map Planet to Gold, A/B/C/D to vibrant modern colors
    colors = ['#FFD700', '#FF3B30', '#34C759', '#007AFF', '#AF52DE', '#FF9500']
    
    # Add trace for each body
    # Trace indices:
    # 2*i: Path line (fixed)
    # 2*i+1: Current marker (animated)
    for i in range(N):
        color = colors[i % len(colors)]
        name = labels[i]
        
        # Path line
        fig.add_trace(go.Scatter3d(
            x=trajectory_positions[:, i, 0],
            y=trajectory_positions[:, i, 1],
            z=trajectory_positions[:, i, 2],
            mode='lines',
            line=dict(color=color, width=3, dash='solid'),
            name=f"{name} Path",
            legendgroup=name,
            showlegend=False,
        ))
        
        # Current marker
        fig.add_trace(go.Scatter3d(
            x=[trajectory_positions[0, i, 0]],
            y=[trajectory_positions[0, i, 1]],
            z=[trajectory_positions[0, i, 2]],
            mode='markers+text',
            marker=dict(size=8 if name == "Planet" else 6, color=color),
            text=[name],
            textposition="top center",
            name=name,
            legendgroup=name,
            showlegend=True,
        ))
        
    # Frames for animation
    frames = []
    # Limit number of frames for performance and file size
    max_frames = 300
    step_size = max(1, T // max_frames)
    frame_indices = list(range(0, T, step_size))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)
        
    for t_idx in frame_indices:
        frame_data = []
        for i in range(N):
            frame_data.append(go.Scatter3d(
                x=[trajectory_positions[t_idx, i, 0]],
                y=[trajectory_positions[t_idx, i, 1]],
                z=[trajectory_positions[t_idx, i, 2]],
            ))
        frames.append(go.Frame(
            data=frame_data,
            name=f"frame_{t_idx}",
            traces=[2 * i + 1 for i in range(N)],
        ))
        
    fig.frames = frames
    
    # Play / Pause buttons
    updatemenus = [
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, transition=dict(duration=0))],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))],
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )
    ]
    
    # Time slider
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Time Step: ", "font": {"size": 12, "color": "#FFFFFF"}},
            pad={"b": 10, "t": 50},
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[[f"frame_{t_idx}"], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
                    label=str(t_idx),
                    method="animate",
                ) for t_idx in frame_indices
            ],
        )
    ]
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="Fairy Orbit Interactive 3D Viewer",
            x=0.5,
            y=0.95,
            font=dict(size=20, color="#FFFFFF"),
        ),
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="rgb(20, 20, 20)", gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            yaxis=dict(title="Y", backgroundcolor="rgb(20, 20, 20)", gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            zaxis=dict(title="Z", backgroundcolor="rgb(20, 20, 20)", gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            aspectmode='data',
        ),
        updatemenus=updatemenus,
        sliders=sliders,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=0.85,
            y=0.9,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(color="#FFFFFF"),
        ),
    )
    
    fig.write_html(output_path, include_plotlyjs="cdn")
