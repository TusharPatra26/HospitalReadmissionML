"""Visualization styling and helper functions."""

import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure matplotlib uses a writable temp cache directory to avoid permissions warnings
cache_dir = Path(__file__).resolve().parent.parent.parent / "logs" / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

# Set non-interactive backend for headless plotting
matplotlib.use("Agg")

# Modern healthcare palette: navy, teal, coral, slate
HEALTHCARE_PALETTE = ["#1E3A8A", "#0D9488", "#F59E0B", "#EF4444", "#6B7280", "#3B82F6"]


def set_plot_theme() -> None:
    """Sets a clean, modern publication-grade theme for all plots."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.titlesize": 14,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def save_plot(fig: plt.Figure, file_path: str | Path, dpi: int = 300) -> None:
    """Saves a matplotlib figure cleanly."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

