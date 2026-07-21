"""Visualization for event timeline and sequences."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.analysis.periodicity_evaluator import PeriodicityScore
from fairy_orbit.simulation.adaptive_simulator import Event, EventType


def plot_event_timeline(
    events: list[Event],
    output_path: Path,
    title: str = "Event Timeline",
) -> None:
    """
    Plot event timeline showing event types and participants over time.
    
    Args:
        events: List of events from adaptive simulator
        output_path: Path to save the plot
        title: Plot title
    """
    if not events:
        print("No events to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color map for event types
    colors = {
        EventType.PAIR: 'blue',
        EventType.TRIPLE: 'orange',
        EventType.FULL: 'red',
    }
    
    # Plot events as horizontal bars
    for i, event in enumerate(events):
        color = colors.get(event.event_type, 'gray')
        label = f"{event.event_type.name}"
        
        # Draw bar
        ax.barh(
            i,
            event.end_time - event.start_time,
            left=event.start_time,
            height=0.8,
            color=color,
            alpha=0.7,
            label=label if i == 0 or events[i-1].event_type != event.event_type else "",
        )
        
        # Add participant labels
        participants_str = ",".join(map(str, event.participants))
        ax.text(
            (event.start_time + event.end_time) / 2,
            i,
            participants_str,
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            fontweight='bold',
        )
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Event Index")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right')
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved event timeline to {output_path}")


def plot_event_sequence(
    events: list[Event],
    output_path: Path,
    title: str = "Event Sequence",
) -> None:
    """
    Plot event sequence as a time series of event types.
    
    Args:
        events: List of events from adaptive simulator
        output_path: Path to save the plot
        title: Plot title
    """
    if not events:
        print("No events to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Extract event types and start times
    times = [event.start_time for event in events]
    types = [event.event_type.value for event in events]
    
    # Plot as step plot
    ax.step(times, types, where='post', linewidth=2)
    
    # Mark points
    ax.scatter(times, types, s=50, zorder=5)
    
    # Add labels for event types
    type_labels = {1: 'PAIR', 2: 'TRIPLE', 3: 'FULL'}
    for t, typ in zip(times, types):
        ax.text(t, typ + 0.1, type_labels.get(typ, ''), 
                ha='center', fontsize=8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Event Type")
    ax.set_title(title)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['PAIR', 'TRIPLE', 'FULL'])
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved event sequence to {output_path}")


def plot_score_breakdown(
    score: PeriodicityScore,
    output_path: Path,
    title: str = "Score Breakdown",
) -> None:
    """
    Plot score components as a bar chart.
    
    Args:
        score: PeriodicityScore from evaluator
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = [
        'Time Variance',
        'Event Sequence Error',
        'Event Count Difference',
        'Center Motion Error',
        'Energy Drift',
    ]
    
    values = [
        score.time_variance,
        score.event_sequence_error,
        score.event_count_difference,
        score.center_motion_error,
        score.energy_drift,
    ]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    bars = ax.bar(components, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel("Score Component Value")
    ax.set_title(f"{title}\nTotal Score: {score.total_score:.4f}")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved score breakdown to {output_path}")


def plot_normalized_intervals(
    intervals: list[float],
    output_path: Path,
    title: str = "Normalized Time Intervals",
) -> None:
    """
    Plot normalized time intervals between events.
    
    Args:
        intervals: Normalized time intervals
        output_path: Path to save the plot
        title: Plot title
    """
    if not intervals:
        print("No intervals to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Time series plot
    ax = axes[0]
    ax.plot(range(len(intervals)), intervals, 'o-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ideal (1.0)')
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Normalized Interval")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram
    ax = axes[1]
    ax.hist(intervals, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Ideal (1.0)')
    ax.set_xlabel("Normalized Interval")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Normalized Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved normalized intervals to {output_path}")


def plot_cycle_comparison(
    cycles: list[list[int]],
    output_path: Path,
    title: str = "Cycle Comparison",
) -> None:
    """
    Compare event sequences across cycles.
    
    Args:
        cycles: List of encoded event sequences for each cycle
        output_path: Path to save the plot
        title: Plot title
    """
    if not cycles:
        print("No cycles to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each cycle as a horizontal line
    for i, cycle in enumerate(cycles):
        if cycle:
            ax.plot(range(len(cycle)), cycle, 'o-', label=f'Cycle {i+1}', linewidth=2)
    
    ax.set_xlabel("Event Index in Cycle")
    ax.set_ylabel("Encoded Event")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved cycle comparison to {output_path}")


def plot_all_diagnostics(
    events: list[Event],
    score: PeriodicityScore,
    output_dir: Path,
) -> None:
    """
    Generate all diagnostic plots.
    
    Args:
        events: List of events from adaptive simulator
        score: PeriodicityScore from evaluator
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating diagnostic plots...")
    
    plot_event_timeline(events, output_dir / "event_timeline.png")
    plot_event_sequence(events, output_dir / "event_sequence.png")
    plot_score_breakdown(score, output_dir / "score_breakdown.png")
    plot_normalized_intervals(score.normalized_intervals, output_dir / "normalized_intervals.png")
    
    if score.event_sequences:
        plot_cycle_comparison(score.event_sequences, output_dir / "cycle_comparison.png")
    
    print(f"All diagnostic plots saved to {output_dir}")
