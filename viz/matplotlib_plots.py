"""viz/matplotlib_plots.py — matplotlib figures (opens window). Graceful fallback if absent."""

from typing import Sequence, List, Tuple, Optional
import numpy as np


def _mpl():
    """Return (plt, np) or (None, np) with a friendly message."""
    try:
        import matplotlib
        matplotlib.use("TkAgg") if True else None   # try, ignore errors
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  ℹ  Install matplotlib for rich plots:  pip install matplotlib")
        print("  Showing ASCII/terminal fallback instead.")
        return None


# ── Heatmap ──────────────────────────────────────────────────────────────────
def heatmap(M, title: str = "", row_labels=None, col_labels=None,
            cmap: str = "viridis", annot: bool = True) -> None:
    plt = _mpl()
    if plt is None:
        from viz.ascii_plots import heatmap as ah
        ah(M, title=title, row_labels=row_labels, col_labels=col_labels)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)
    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2g}", ha="center", va="center",
                        color="white" if M[i,j] < (M.max()+M.min())/2 else "black",
                        fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ── Scatter with classes ──────────────────────────────────────────────────────
def scatter_classes(X: np.ndarray, y: np.ndarray,
                     title: str = "Scatter",
                     xlabel: str = "x₁", ylabel: str = "x₂") -> None:
    plt = _mpl()
    if plt is None:
        from viz.ascii_plots import scatter
        scatter(X[:, 0], X[:, 1], title=title)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = np.unique(y)
    cmap    = plt.cm.get_cmap("tab10")
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], label=str(cls),
                   color=cmap(i), alpha=0.7, s=40)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout(); plt.show()


# ── Decision boundary ─────────────────────────────────────────────────────────
def decision_boundary(model, X: np.ndarray, y: np.ndarray,
                        title: str = "Decision Boundary",
                        resolution: float = 0.02) -> None:
    plt = _mpl()
    if plt is None:
        return
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap="RdYlBu")
    scatter_classes(X, y, title=title)


# ── 3-D loss surface ─────────────────────────────────────────────────────────
def loss_surface_3d(f, x_range=(-3, 3), y_range=(-3, 3),
                     title: str = "Loss Surface",
                     trajectory: Optional[np.ndarray] = None) -> None:
    plt = _mpl()
    if plt is None:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    xs = np.linspace(*x_range, 80)
    ys = np.linspace(*y_range, 80)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
    if trajectory is not None:
        tz = np.array([f(p) for p in trajectory])
        ax.plot(trajectory[:, 0], trajectory[:, 1], tz,
                "r-o", ms=3, label="trajectory")
    ax.set_title(title)
    plt.tight_layout(); plt.show()


# ── Contour plot + gradient arrows ───────────────────────────────────────────
def contour_gradient(f, x_range=(-3, 3), y_range=(-3, 3),
                      title: str = "Contour",
                      trajectory: Optional[np.ndarray] = None) -> None:
    plt = _mpl()
    if plt is None:
        return
    xs  = np.linspace(*x_range, 200)
    ys  = np.linspace(*y_range, 200)
    X, Y = np.meshgrid(xs, ys)
    Z   = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)
    fig, ax = plt.subplots(figsize=(7, 5))
    cs  = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    plt.colorbar(cs, ax=ax)
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "r-o", ms=4,
                label="Trajectory")
        ax.legend()
    ax.set_title(title)
    plt.tight_layout(); plt.show()


# ── Bias-variance curves ──────────────────────────────────────────────────────
def bias_variance_curves(complexities: Sequence,
                          bias2: Sequence, variance: Sequence,
                          total: Sequence) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(complexities, bias2, "b-", label="Bias²")
    ax.plot(complexities, variance, "r-", label="Variance")
    ax.plot(complexities, total, "g--", lw=2, label="Total Error")
    ax.set_xlabel("Model Complexity")
    ax.set_ylabel("Error")
    ax.set_title("Bias-Variance Trade-off")
    ax.legend(); plt.tight_layout(); plt.show()


# ── ROC curve ─────────────────────────────────────────────────────────────────
def roc_curve_mpl(fpr: Sequence, tpr: Sequence, auc: float) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout(); plt.show()


# ── Confusion matrix ──────────────────────────────────────────────────────────
def confusion_matrix_plot(cm: np.ndarray,
                            class_names: Optional[List[str]] = None) -> None:
    plt = _mpl()
    if plt is None:
        from viz.ascii_plots import heatmap as ah
        ah(cm, title="Confusion Matrix", row_labels=class_names,
           col_labels=class_names)
        return
    heatmap(cm, title="Confusion Matrix",
            row_labels=class_names, col_labels=class_names)


# ── Calibration plot ──────────────────────────────────────────────────────────
def calibration_plot(mean_pred: Sequence, frac_pos: Sequence) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, "s-", label="Calibration")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean Predicted"); ax.set_ylabel("Fraction Positive")
    ax.set_title("Calibration Curve"); ax.legend()
    plt.tight_layout(); plt.show()


# ── Gaussian ellipses (GMM) ───────────────────────────────────────────────────
def gmm_ellipses(X: np.ndarray, means: np.ndarray,
                  covs: np.ndarray, labels: np.ndarray) -> None:
    plt = _mpl()
    if plt is None:
        return
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.tab10
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(X[mask, 0], X[mask, 1], color=cmap(cls), alpha=0.5, s=20)
    for k, (mean, cov) in enumerate(zip(means, covs)):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        for nsig in (1, 2):
            w, h = 2 * nsig * np.sqrt(vals)
            ell  = Ellipse((mean[0], mean[1]), w, h, angle=theta,
                           color=cmap(k), alpha=0.3 if nsig == 2 else 0.6,
                           fill=False, lw=2)
            ax.add_patch(ell)
    ax.set_title("GMM Components")
    plt.tight_layout(); plt.show()


# ── Positional encoding heatmap ───────────────────────────────────────────────
def positional_encoding_heatmap(PE: np.ndarray) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(PE, cmap="RdBu", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Embedding Dimension"); ax.set_ylabel("Position")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.tight_layout(); plt.show()


# ── Attention heatmap ─────────────────────────────────────────────────────────
def attention_heatmap(attn: np.ndarray,
                       row_labels: Optional[List[str]] = None,
                       col_labels: Optional[List[str]] = None) -> None:
    heatmap(attn, title="Attention Weights",
            row_labels=row_labels, col_labels=col_labels, cmap="Blues")


# ── Radar chart ───────────────────────────────────────────────────────────────
def radar_chart(metrics: List[str],
                models: List[Tuple[str, List[float]]]) -> None:
    plt = _mpl()
    if plt is None:
        return
    N    = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name, vals in models:
        vals_ = vals + vals[:1]
        ax.plot(angles, vals_, label=name)
        ax.fill(angles, vals_, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Model Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout(); plt.show()


# ── PCA scatter ───────────────────────────────────────────────────────────────
def pca_scatter(Z: np.ndarray, labels: Optional[np.ndarray] = None,
                evr: Optional[np.ndarray] = None) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    if labels is not None:
        scatter_classes(Z, labels, title="PCA Projection",
                        xlabel=f"PC1 ({evr[0]*100:.1f}%)" if evr is not None else "PC1",
                        ylabel=f"PC2 ({evr[1]*100:.1f}%)" if evr is not None else "PC2")
    else:
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.6)
        ax.set_title("PCA Projection")
        plt.tight_layout(); plt.show()


# ── GP posterior ──────────────────────────────────────────────────────────────
def gp_posterior(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, mu: np.ndarray,
                  std: np.ndarray) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_test, mu, "b-", label="Posterior mean")
    ax.fill_between(x_test, mu - 2*std, mu + 2*std, alpha=0.3, label="95% CI")
    ax.scatter(x_train, y_train, c="red", zorder=5, label="Observations")
    ax.set_title("Gaussian Process Posterior")
    ax.legend(); plt.tight_layout(); plt.show()


# ── Learning curves (sklearn style) ──────────────────────────────────────────
def learning_curves(train_sizes: Sequence, train_scores: Sequence,
                     val_scores: Sequence, title: str = "Learning Curves") -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores, "o-", color="blue", label="Train")
    ax.plot(train_sizes, val_scores,   "o-", color="green", label="Validation")
    ax.set_xlabel("Training Size"); ax.set_ylabel("Score")
    ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()


# ── Polynomial fit demo ───────────────────────────────────────────────────────
def poly_fit_demo(x: np.ndarray, y_true: np.ndarray,
                   degrees: List[int]) -> None:
    plt = _mpl()
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(degrees), figsize=(4*len(degrees), 4), sharey=True)
    x_plot = np.linspace(x.min(), x.max(), 200)
    for ax, deg in zip(axes, degrees):
        coeffs  = np.polyfit(x, y_true, deg)
        y_fit   = np.polyval(coeffs, x_plot)
        ax.scatter(x, y_true, c="red", s=20, alpha=0.7)
        ax.plot(x_plot, y_fit, "b-")
        ax.set_title(f"Degree {deg}")
    plt.suptitle("Polynomial Degree Comparison")
    plt.tight_layout(); plt.show()


# Aliases — blocks reference these names; implementations use shorter names above
plot_decision_boundary = decision_boundary
show_heatmap = heatmap
