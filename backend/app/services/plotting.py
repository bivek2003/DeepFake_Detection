"""
Chart and plot generation for analysis results.
"""

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend

from app.logging_config import get_logger

logger = get_logger(__name__)


def generate_timeline_chart(
    timestamps: list[float],
    scores: list[float],
    title: str = "Fake Score Timeline",
) -> bytes:
    """
    Generate timeline chart showing fake scores over video duration.

    Args:
        timestamps: List of timestamps in seconds
        scores: List of fake probability scores (0-1)
        title: Chart title

    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot scores
    ax.plot(timestamps, scores, "b-", linewidth=1.5, alpha=0.8)
    ax.fill_between(timestamps, scores, alpha=0.3)

    # Threshold line
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Threshold")

    # Styling
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Fake Probability", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(timestamps) if timestamps else 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Color coding background
    ax.axhspan(0, 0.3, alpha=0.1, color="green")  # Real zone
    ax.axhspan(0.7, 1, alpha=0.1, color="red")  # Fake zone

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()


def generate_distribution_chart(
    scores: list[float],
    title: str = "Score Distribution",
) -> bytes:
    """
    Generate histogram showing score distribution.

    Args:
        scores: List of fake probability scores
        title: Chart title

    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create histogram
    bins = np.linspace(0, 1, 21)
    counts, _, patches = ax.hist(scores, bins=bins, edgecolor="black", alpha=0.7)

    # Color bars by score range
    for i, patch in enumerate(patches):
        bin_center = bins[i] + (bins[i + 1] - bins[i]) / 2
        if bin_center < 0.3:
            patch.set_facecolor("green")
        elif bin_center > 0.7:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("orange")

    # Add mean line
    mean_score = np.mean(scores) if scores else 0
    ax.axvline(
        x=mean_score, color="blue", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.2f}"
    )

    # Styling
    ax.set_xlabel("Fake Probability", fontsize=10)
    ax.set_ylabel("Frame Count", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()


def generate_suspicious_frames_montage(
    frames: list[np.ndarray],
    scores: list[float],
    timestamps: list[float],
    title: str = "Suspicious Frames",
) -> bytes:
    """
    Generate montage of suspicious frames with captions.

    Args:
        frames: List of frame images as numpy arrays
        scores: Corresponding fake scores
        timestamps: Corresponding timestamps
        title: Montage title

    Returns:
        PNG image as bytes
    """
    n_frames = len(frames)
    if n_frames == 0:
        # Create empty placeholder
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No suspicious frames detected", ha="center", va="center")
        ax.axis("off")
    else:
        # Create grid of frames
        cols = min(5, n_frames)
        rows = (n_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.5))

        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for idx, (frame, score, ts) in enumerate(zip(frames, scores, timestamps, strict=True)):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            ax.imshow(frame)
            ax.set_title(f"t={ts:.1f}s | Score: {score:.2f}", fontsize=9)
            ax.axis("off")

            # Border color based on score
            border_color = "red" if score > 0.7 else "orange" if score > 0.5 else "green"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

        # Hide empty subplots
        for idx in range(n_frames, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()
