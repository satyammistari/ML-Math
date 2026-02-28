"""viz/terminal_plots.py — plotext-based terminal plots (no window needed)."""

from typing import Sequence, List, Optional, Tuple
import numpy as np


def _get_plt():
    try:
        import plotext as plt
        return plt
    except ImportError:
        return None


def _no_plotext():
    print("  ℹ  plotext not installed. Run:  pip install plotext")
    print("  Showing ASCII fallback instead.")


# ── Loss curve ────────────────────────────────────────────────────────────────
def loss_curve(losses: Sequence[float], title: str = "Training Loss",
               xlabel: str = "Step", ylabel: str = "Loss") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        from ui.widgets import print_sparkline
        print_sparkline(losses, label="Loss")
        return
    plt.clf()
    plt.plot(list(range(len(losses))), list(losses), label="loss")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.show()


# ── Distribution shape ────────────────────────────────────────────────────────
def distribution_plot(xs: Sequence[float], ys: Sequence[float],
                       title: str = "Distribution",
                       fill: bool = True) -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        from viz.ascii_plots import line_plot
        line_plot(xs, ys, title=title)
        return
    plt.clf()
    if fill:
        plt.plot(list(xs), list(ys), fillx=True)
    else:
        plt.plot(list(xs), list(ys))
    plt.title(title)
    plt.show()


# ── Multiple distributions ─────────────────────────────────────────────────────
def multi_distribution(series: List[Tuple[Sequence, Sequence, str]],
                        title: str = "Distributions") -> None:
    """series: list of (xs, ys, label)"""
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        return
    plt.clf()
    for xs, ys, lbl in series:
        plt.plot(list(xs), list(ys), label=lbl)
    plt.title(title)
    plt.show()


# ── Scatter plot ───────────────────────────────────────────────────────────────
def scatter_plot(xs: Sequence[float], ys: Sequence[float],
                  title: str = "Scatter",
                  xlabel: str = "x", ylabel: str = "y") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        from viz.ascii_plots import scatter
        scatter(xs, ys, title=title)
        return
    plt.clf()
    plt.scatter(list(xs), list(ys))
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.show()


# ── Multi-series loss curves ───────────────────────────────────────────────────
def multi_loss(series: List[Tuple[Sequence, str]],
               title: str = "Optimiser Comparison") -> None:
    """series: list of (losses, label)"""
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        for losses, lbl in series:
            from ui.widgets import print_sparkline
            print_sparkline(losses, label=lbl)
        return
    plt.clf()
    for losses, lbl in series:
        plt.plot(list(range(len(losses))), list(losses), label=lbl)
    plt.title(title)
    plt.show()


# ── Bar chart ─────────────────────────────────────────────────────────────────
def bar(labels: Sequence[str], values: Sequence[float],
        title: str = "") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        from ui.widgets import bar_chart
        bar_chart(list(labels), list(values), title=title)
        return
    plt.clf()
    plt.bar(list(labels), list(values))
    if title:
        plt.title(title)
    plt.show()


# ── Histogram ─────────────────────────────────────────────────────────────────
def histogram(data: Sequence[float], bins: int = 30,
              title: str = "Histogram", xlabel: str = "value") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        # Simple ASCII histogram
        data = np.asarray(data)
        counts, edges = np.histogram(data, bins=min(bins, 20))
        from ui.widgets import bar_chart
        bar_chart([f"{e:.2g}" for e in edges[:-1]], counts.tolist(), title=title)
        return
    plt.clf()
    plt.hist(list(data), bins=bins)
    plt.title(title); plt.xlabel(xlabel)
    plt.show()


# ── ROC curve ────────────────────────────────────────────────────────────────
def roc_curve_plot(fpr: Sequence[float], tpr: Sequence[float],
                    auc: float = 0.0) -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        return
    plt.clf()
    plt.plot(list(fpr), list(tpr), label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], label="Random")
    plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.show()


# ── Eigenvalue spectrum ───────────────────────────────────────────────────────
def eigenvalue_spectrum(eigenvalues: Sequence[float],
                         title: str = "Eigenvalue Spectrum") -> None:
    plt = _get_plt()
    evals = sorted(np.abs(eigenvalues), reverse=True)
    if plt is None:
        _no_plotext()
        from ui.widgets import bar_chart
        bar_chart([f"λ{i+1}" for i in range(len(evals))], evals, title=title)
        return
    plt.clf()
    plt.bar(list(range(1, len(evals)+1)), evals)
    plt.title(title); plt.xlabel("Component"); plt.ylabel("|λ|")
    plt.show()


# ── Scree plot ────────────────────────────────────────────────────────────────
def scree_plot(explained_var: Sequence[float],
               cumulative: bool = True) -> None:
    plt = _get_plt()
    evs = list(explained_var)
    cum = list(np.cumsum(evs))
    if plt is None:
        _no_plotext()
        from ui.widgets import bar_chart
        bar_chart([f"PC{i+1}" for i in range(len(evs))], evs, title="Scree Plot")
        return
    plt.clf()
    plt.bar(list(range(1, len(evs)+1)), evs, label="Var %")
    if cumulative:
        plt.plot(list(range(1, len(cum)+1)), cum, label="Cumulative")
    plt.title("Scree Plot"); plt.xlabel("Component"); plt.ylabel("Explained Variance")
    plt.show()


# ── Activation function plot ──────────────────────────────────────────────────
def activation_plot(x: Sequence[float],
                    fns: List[Tuple[Sequence[float], str]],
                    title: str = "Activation Functions") -> None:
    """fns: list of (y_values, name)"""
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        return
    plt.clf()
    for ys, name in fns:
        plt.plot(list(x), list(ys), label=name)
    plt.title(title)
    plt.show()


# ── Learning rate schedule ────────────────────────────────────────────────────
def lr_schedule_plot(schedules: List[Tuple[Sequence[float], str]],
                      title: str = "LR Schedules") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        return
    plt.clf()
    for lrs, name in schedules:
        plt.plot(list(range(len(lrs))), list(lrs), label=name)
    plt.title(title); plt.xlabel("Step"); plt.ylabel("LR")
    plt.show()


# ── MCMC trace ────────────────────────────────────────────────────────────────
def mcmc_trace(samples: Sequence[float],
               title: str = "MCMC Trace") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        from ui.widgets import print_sparkline
        print_sparkline(samples, label=title)
        return
    plt.clf()
    plt.plot(list(range(len(samples))), list(samples))
    plt.title(title); plt.xlabel("Iteration"); plt.ylabel("Value")
    plt.show()


# ── Convergence plot ──────────────────────────────────────────────────────────
def convergence_plot(values: Sequence[float],
                      title: str = "Convergence",
                      ylabel: str = "Value") -> None:
    plt = _get_plt()
    if plt is None:
        _no_plotext()
        from ui.widgets import print_sparkline
        print_sparkline(values, label=ylabel)
        return
    plt.clf()
    plt.plot(list(range(len(values))), list(values))
    plt.title(title); plt.xlabel("Iteration"); plt.ylabel(ylabel)
    plt.show()


# Alias — blocks import loss_curve_plot; the implementation lives in loss_curve
loss_curve_plot = loss_curve
