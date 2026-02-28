"""
Block 19: Kernel Methods
Covers: Mercer's theorem, kernel functions, Gram matrix, SVM dual, GP kernels, Kernel PCA
"""
import numpy as np
import math


def run():
    topics = [
        ("Mercer's Theorem",               mercers_theorem),
        ("Kernel Functions",               kernel_functions),
        ("Gram Matrix",                    gram_matrix),
        ("Kernel Trick",                   kernel_trick),
        ("Kernel SVM",                     kernel_svm),
        ("Gaussian Processes (kernel)",    gp_kernel),
        ("Kernel PCA",                     kernel_pca),
        ("Combining Kernels",              combining_kernels),
    ]
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║         BLOCK 19 — KERNEL METHODS                ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m")
        print("\033[90mmmlmath > Block 19 > Kernel Methods\033[0m\n")
        for i, (name, _) in enumerate(topics, 1):
            print(f"  \033[93m{i:2d}.\033[0m {name}")
        print("\n  \033[90m[0] Back to main menu\033[0m")
        choice = input("\n\033[96mSelect topic: \033[0m").strip()
        if choice == "0":
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(topics):
                topics[idx][1]()
        except (ValueError, IndexError):
            print("\033[91mInvalid choice.\033[0m")


def mercers_theorem():
    print("\n\033[95m━━━ Mercer's Theorem ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Mercer's theorem tells us WHEN a function k(x, x') is a valid kernel —
i.e., when it corresponds to an inner product in some feature space Φ.

A kernel k is valid iff the Gram matrix K is positive semi-definite (PSD)
for all datasets: Kᵢⱼ = k(xᵢ, xⱼ), and K ⪰ 0 (all eigenvalues ≥ 0).

This is crucial: we can define distance/similarity in a (possibly
infinite-dimensional, abstract) feature space without ever computing
or storing the features explicitly — only the kernel function needed.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Mercer's Theorem: k(x,x') is a valid kernel ⟺
    1. k is symmetric: k(x,x') = k(x',x)
    2. For any finite set {x₁,...,xₙ}, the Gram matrix K ⪰ 0

  Feature map: ∃ φ : X → H (Hilbert space) such that
    k(x, x') = ⟨φ(x), φ(x')⟩_H

  PSD check: ∀ c ∈ ℝⁿ, Σᵢ Σⱼ cᵢ cⱼ k(xᵢ, xⱼ) ≥ 0
  ⟺ all eigenvalues of K are non-negative
""")

    print("\033[93mNUMERICAL PSD CHECK\033[0m")
    X = np.array([[1, 0], [0, 1], [1, 1]])

    def rbf(x, y, sigma=1.0):
        return math.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))

    n = len(X)
    K = np.array([[rbf(X[i], X[j]) for j in range(n)] for i in range(n)])
    eigs = np.linalg.eigvalsh(K)
    print(f"\n  Gram matrix (RBF kernel, σ=1):\n{np.round(K, 4)}")
    print(f"\n  Eigenvalues: {np.round(eigs, 4)}")
    print(f"  All eigenvalues ≥ 0: {all(e >= -1e-10 for e in eigs)} → Valid kernel ✓")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • Linear kernel: k(x,x') = xᵀx' → φ(x) = x (identity)
  • RBF kernel: k(x,x') = exp(−||x−x'||²/2σ²) → infinite-dim φ
  • Mercer's gives freedom: choose any PSD function as kernel
  • Negative definite kernels can still be useful (e.g. Laplacian)
  • Numerical PSD: add ε·I for stability (called jitter / nugget)
""")
    input("\033[90m[Enter to continue]\033[0m")


def kernel_functions():
    print("\n\033[95m━━━ Kernel Functions ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Different kernels encode different notions of similarity and inductive
biases. The choice of kernel is the most important hyperparameter in
kernel methods — it encodes our prior beliefs about the function space.
""")
    print("\033[93mFORMULAS\033[0m")
    kernels_info = [
        ("Linear",         "k(x,x') = xᵀx'",                                  "Linear boundaries"),
        ("Polynomial",     "k(x,x') = (xᵀx' + c)^d",                          "Polynomial features"),
        ("RBF / Gaussian", "k(x,x') = exp(−||x−x'||²/2σ²)",                   "Universal approximator"),
        ("Laplacian",      "k(x,x') = exp(−||x−x'||₁/σ)",                     "L1 distance, sharp"),
        ("Matérn (ν=3/2)", "k(r) = (1+√3r/l)exp(−√3r/l)  r=||x−x'||",        "Rough functions"),
        ("Sigmoid/Tanh",   "k(x,x') = tanh(αxᵀx'+c)  (conditionally valid)", "Neural-net-like"),
        ("Periodic",       "k(x,x') = exp(−2sin²(π||x−x'||/p)/l²)",           "Periodic patterns"),
    ]
    print("  " + "─" * 78)
    print(f"  {'Kernel':<18} {'Formula':<44} {'Use case'}")
    print("  " + "─" * 78)
    for row in kernels_info:
        print(f"  {row[0]:<18} {row[1]:<44} {row[2]}")
    print("  " + "─" * 78)

    # plotext: kernel similarity vs distance
    try:
        import plotext as plt
        dists = list(np.linspace(0, 4, 80))
        rbf_k  = [math.exp(-d**2 / 2) for d in dists]
        lap_k  = [math.exp(-d) for d in dists]
        poly_k = [max(0, (1 - d**2/4)**3) for d in dists]
        plt.clf()
        plt.plot(dists, rbf_k, label="RBF (σ=1)")
        plt.plot(dists, lap_k, label="Laplacian (σ=1)")
        plt.plot(dists, poly_k, label="Polynomial d=3")
        plt.title("Kernel Similarity vs Distance")
        plt.xlabel("||x - x'||"); plt.ylabel("k(x, x')")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for kernel plot]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • RBF is the default choice — smooth, universal, one hyperparameter (σ)
  • Length scale σ: small→wiggly fit, large→smooth fit
  • Polynomial degree d: controls interaction order between features
  • Matérn kernels: control differentiability of the resulting function
  • Stationary kernels: k(x,x') = k(||x−x'||) — shift invariant
""")
    input("\033[90m[Enter to continue]\033[0m")


def gram_matrix():
    print("\n\033[95m━━━ Gram Matrix ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
The Gram matrix K is the n×n matrix of all pairwise kernel evaluations:
  Kᵢⱼ = k(xᵢ, xⱼ)

It captures all pairwise similarities in the training set and is the
fundamental object in kernel methods. All predictions and learning
are expressed through K.

Computing K costs O(n²d) time and O(n²) memory — the main bottleneck
for large datasets (n > 100k).
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  K ∈ ℝⁿˣⁿ,  Kᵢⱼ = k(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)

  K = Φ·Φᵀ  where rows of Φ are φ(x₁),...,φ(xₙ)

  Properties:
    • Symmetric: K = Kᵀ
    • PSD if k is valid: vᵀKv ≥ 0 ∀v
    • Diagonal: Kᵢᵢ = k(xᵢ,xᵢ) = ||φ(xᵢ)||²

  Kernel Ridge Regression:
    α = (K + λI)⁻¹ y     [dual form, n×n system]
    ŷ(x*) = k(x*, X)ᵀ α  [prediction using kernel evaluations]

  vs Primal Ridge: w = (XᵀX + λI)⁻¹ Xᵀy  [d×d system]
  Use dual if n < d, primal if d < n.
""")

    np.random.seed(42)
    X = np.random.randn(5, 2)
    def rbf_matrix(X, sigma=1.0):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = math.exp(-np.sum((X[i] - X[j])**2) / (2*sigma**2))
        return K

    K = rbf_matrix(X)
    print(f"\n  Data X (5×2):\n{np.round(X, 3)}")
    print(f"\n  RBF Gram Matrix K (5×5):\n{np.round(K, 4)}")
    eigs = np.linalg.eigvalsh(K)
    print(f"\n  Eigenvalues of K: {np.round(eigs, 4)}")
    print(f"  PSD check (all ≥ 0): {all(e >= -1e-8 for e in eigs)} ✓")

    input("\033[90m[Enter to continue]\033[0m")


def kernel_trick():
    print("\n\033[95m━━━ The Kernel Trick ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
The kernel trick lets us implicitly compute inner products in a
high/infinite dimensional feature space without ever constructing
the feature vectors.

Classic example: polynomial kernel k(x,x') = (xᵀx'+1)² corresponds
to feature map φ(x) = [x₁², x₂², √2x₁x₂, √2x₁, √2x₂, 1].
Computing k(x,x') directly costs O(d), vs O(d²) for explicit φ.

RBF kernel: feature space is infinite-dimensional (Taylor expansion of
exp), yet k(x,x') = exp(−||x−x'||²/2σ²) takes O(d) to compute.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Polynomial kernel k(x,x') = (xᵀx' + 1)²:
  For x,x' ∈ ℝ²:

  Explicit: φ(x) = [x₁², x₂², √2x₁x₂, √2x₁, √2x₂, 1]
  φ(x)ᵀφ(x') = x₁²x'₁² + x₂²x'₂² + 2x₁x₂x'₁x'₂
               + 2x₁x'₁ + 2x₂x'₂ + 1
             = (x₁x'₁ + x₂x'₂ + 1)²
             = (xᵀx' + 1)²  ✓

  Cost comparison:
    Explicit φ: O(d^p) to compute feature vector (exponential in degree p)
    Kernel trick: O(d) to compute k(x,x') — independent of p!
""")

    x = np.array([2.0, 3.0])
    xp = np.array([1.0, 4.0])

    # Explicit feature map
    phi = lambda v: np.array([v[0]**2, v[1]**2, math.sqrt(2)*v[0]*v[1],
                               math.sqrt(2)*v[0], math.sqrt(2)*v[1], 1.0])
    explicit = phi(x) @ phi(xp)
    kernel   = (x @ xp + 1) ** 2

    print(f"\n  x = {x},  x' = {xp}")
    print(f"  φ(x)  = {np.round(phi(x), 3)}")
    print(f"  φ(x') = {np.round(phi(xp), 3)}")
    print(f"  φ(x)ᵀφ(x') = {explicit:.4f}")
    print(f"  k(x,x') = (xᵀx'+1)² = ({x@xp:.0f}+1)² = {kernel:.4f}")
    print(f"  Equal: {abs(explicit - kernel) < 1e-8} ✓")

    input("\033[90m[Enter to continue]\033[0m")


def kernel_svm():
    print("\n\033[95m━━━ Kernel SVM ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Support Vector Machines in the kernel (dual) form can classify
non-linearly separable data by implicitly mapping to higher dimensions.

The dual SVM formulation only requires kernel evaluations k(xᵢ,xⱼ),
never the explicit feature vector φ(xᵢ).
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Primal (may be infinite-dim):   min_{w,b} ½||w||² + C·Σ max(0,1−yᵢ(wᵀφ(xᵢ)+b))

  Dual (always finite n×n system):
    max_{α} Σᵢαᵢ − ½ Σᵢⱼ αᵢαⱼyᵢyⱼk(xᵢ,xⱼ)
    s.t. 0≤αᵢ≤C, Σᵢαᵢyᵢ=0

  Prediction: f(x*) = sign(Σᵢ αᵢyᵢk(xᵢ,x*) + b)

  Support vectors: xᵢ with αᵢ > 0 (only these matter for prediction)
  Sparsity: most αᵢ = 0 → efficient prediction

  KKT conditions:  αᵢ = 0    ⟺  yᵢf(xᵢ) ≥ 1  (correct, outside margin)
                   αᵢ = C    ⟺  yᵢf(xᵢ) ≤ 1  (on or inside margin)
                   0<αᵢ<C   ⟺  yᵢf(xᵢ) = 1  (on the margin, support vectors)
""")

    try:
        from sklearn.svm import SVC
        from sklearn.datasets import make_circles
        import matplotlib.pyplot as plt2
        X_c, y_c = make_circles(n_samples=100, noise=0.1, factor=0.4, random_state=0)
        y_c = 2 * y_c - 1  # {-1, +1}
        clf = SVC(kernel='rbf', C=1.0, gamma=1.0)
        clf.fit(X_c, y_c)
        acc = clf.score(X_c, y_c)
        print(f"\n  make_circles dataset (non-linearly separable)")
        print(f"  RBF-SVM training accuracy: {acc:.4f}")
        print(f"  Number of support vectors: {len(clf.support_vectors_)}")

        fig, axes = plt2.subplots(1, 2, figsize=(12, 5))
        xx, yy = np.meshgrid(np.linspace(-1.8, 1.8, 200), np.linspace(-1.8, 1.8, 200))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        axes[0].contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        axes[0].contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
        axes[0].scatter(X_c[:, 0], X_c[:, 1], c=y_c, cmap='RdBu', edgecolors='k', s=40)
        sv = clf.support_vectors_
        axes[0].scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='yellow', linewidths=2)
        axes[0].set_title(f'RBF-SVM Decision Boundary\n{len(sv)} support vectors (yellow)')
        axes[1].bar(['Total', 'Support Vectors'], [len(X_c), len(sv)], color=['steelblue', 'tomato'])
        axes[1].set_title('Data vs Support Vectors')
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install sklearn + matplotlib for SVM demo]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • Kernel SVM is a maximum-margin classifier in feature space
  • C controls margin-vs-violation tradeoff: large C = hard margin
  • For RBF: γ=1/2σ² controls smoothness; too large → overfitting
  • SVMs scale as O(n²-n³) due to QP solving → kernel SVM hard for n>100k
  • Modern alternative: random features approximate kernel, then linear SVM
""")
    input("\033[90m[Enter to continue]\033[0m")


def gp_kernel():
    print("\n\033[95m━━━ Gaussian Processes & Kernels ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
A Gaussian Process (GP) is a probability distribution over functions.
A sample from a GP is an entire function f(·).
The kernel k(x,x') defines the covariance between function values at x and x'.

GP prior:  f ~ GP(μ(x), k(x,x'))
GP posterior (given data): Gaussian with closed-form mean and variance.

The kernel choice determines:
• Smoothness (RBF → infinitely differentiable)
• Periodicity (periodic kernel)
• Length scale (how fast the function varies)
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Prior:  f(x) ~ GP(0, k(x,x'))
  Observations: y = f(X) + ε,  ε ~ N(0, σ²I)

  Posterior at test x*:
    μ*(x*)  = k(x*, X)(K + σ²I)⁻¹y
    σ²*(x*) = k(x*,x*) − k(x*,X)(K+σ²I)⁻¹k(X,x*)

  Marginal likelihood (for hyperparameter optimization):
    log P(y|X,θ) = −½yᵀ(K+σ²I)⁻¹y − ½log|K+σ²I| − n/2 log 2π
""")

    np.random.seed(5)
    n_obs = 6
    X_obs = np.sort(np.random.uniform(0, 5, n_obs))
    y_obs = np.sin(X_obs) + 0.1 * np.random.randn(n_obs)
    X_test = np.linspace(0, 5, 100)
    sigma_noise = 0.1

    def rbf_cov(a, b, l=1.0):
        return np.array([[math.exp(-0.5*((ai-bi)/l)**2) for bi in b] for ai in a])

    K = rbf_cov(X_obs, X_obs) + sigma_noise**2 * np.eye(n_obs)
    K_star = rbf_cov(X_test, X_obs)
    K_ss = rbf_cov(X_test, X_test)
    K_inv_y = np.linalg.solve(K, y_obs)
    mu_post = K_star @ K_inv_y
    var_post = np.diag(K_ss) - np.einsum('ij,jk,ki->i', K_star, np.linalg.inv(K), K_star.T)
    std_post = np.sqrt(np.maximum(var_post, 0))

    # ASCII display
    print("\n\033[93mASCII GP Posterior (6 observations)\033[0m")
    w, h = 60, 12
    x_min, x_max = 0, 5
    y_min, y_max = -2, 2
    grid = [[' '] * w for _ in range(h)]
    for i, (m, s) in enumerate(zip(mu_post, std_post)):
        xi = int((X_test[i] - x_min) / (x_max - x_min) * (w - 1))
        yi = int((m - y_min) / (y_max - y_min) * (h - 1))
        yi = h - 1 - yi
        yi_up = int(((m + s) - y_min) / (y_max - y_min) * (h - 1))
        yi_lo = int(((m - s) - y_min) / (y_max - y_min) * (h - 1))
        yi_up = h - 1 - yi_up; yi_lo = h - 1 - yi_lo
        if 0 <= xi < w:
            if 0 <= yi < h:
                grid[yi][xi] = '\033[96m─\033[0m'
            for yy in range(min(yi_up, yi_lo), max(yi_up, yi_lo)+1):
                if 0 <= yy < h and grid[yy][xi] == ' ':
                    grid[yy][xi] = '\033[90m·\033[0m'
    for xo, yo in zip(X_obs, y_obs):
        xi = int((xo - x_min) / (x_max - x_min) * (w - 1))
        yi = int((yo - y_min) / (y_max - y_min) * (h - 1))
        yi = h - 1 - yi
        if 0 <= xi < w and 0 <= yi < h:
            grid[yi][xi] = '\033[93m●\033[0m'
    for row in grid:
        print("  " + "".join(row))
    print("  \033[96m─ posterior mean   · 1σ band   \033[93m● observations\033[0m\n")

    # matplotlib
    try:
        import matplotlib.pyplot as plt2
        fig, ax = plt2.subplots(figsize=(10, 5))
        ax.plot(X_test, np.sin(X_test), 'g--', label='True sin(x)', lw=2, alpha=0.5)
        ax.plot(X_test, mu_post, 'b-', label='GP mean', lw=2)
        ax.fill_between(X_test, mu_post - 2*std_post, mu_post + 2*std_post,
                         alpha=0.2, color='blue', label='95% CI')
        ax.scatter(X_obs, y_obs, c='red', zorder=5, s=80, label='Observations')
        ax.legend(); ax.set_title('GP Posterior Regression')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for GP posterior plot]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


def kernel_pca():
    print("\n\033[95m━━━ Kernel PCA ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Kernel PCA extends PCA to nonlinear dimensionality reduction by
performing standard PCA in the kernel feature space.

Since we never need explicit φ(x), we work entirely with the n×n
Gram matrix K. The principal components in feature space become
eigenvectors of the centered Gram matrix.

Result: can "unfold" nonlinear manifolds that standard PCA cannot.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  [1] Compute Gram matrix: Kᵢⱼ = k(xᵢ, xⱼ)
  [2] Center the kernel matrix:
    K̃ = K − 1ₙK − K1ₙ + 1ₙK1ₙ   where 1ₙ = (1/n)·11ᵀ
  [3] Eigendecompose: K̃ = VΛVᵀ
  [4] Normalize eigenvectors: αₖ = vₖ / √λₖ
  [5] Project new point: (zₖ)ᵢ = Σⱼ αₖ(j) k(xⱼ, xᵢ)

  vs Linear PCA: PCA eigendecomp on d×d covariance (d << n)
  Kernel PCA: eigendecomp on n×n Gram (n << ∞-dim feature space)
""")

    try:
        from sklearn.datasets import make_circles
        from sklearn.decomposition import KernelPCA, PCA as SkPCA
        import matplotlib.pyplot as plt2
        X_c, y_c = make_circles(n_samples=200, noise=0.05, factor=0.3, random_state=0)
        pca_lin = SkPCA(n_components=2).fit_transform(X_c)
        pca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=5).fit_transform(X_c)
        fig, axes = plt2.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(X_c[:, 0], X_c[:, 1], c=y_c, cmap='RdBu', s=20)
        axes[0].set_title("Original 2D Data (circles)")
        axes[1].scatter(pca_lin[:, 0], pca_lin[:, 1], c=y_c, cmap='RdBu', s=20)
        axes[1].set_title("Linear PCA (not separable)")
        axes[2].scatter(pca_rbf[:, 0], pca_rbf[:, 1], c=y_c, cmap='RdBu', s=20)
        axes[2].set_title("Kernel PCA RBF (separable!)")
        plt2.suptitle("Kernel PCA vs Linear PCA on Circles")
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install sklearn + matplotlib for Kernel PCA demo]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


def combining_kernels():
    print("\n\033[95m━━━ Combining Kernels ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
New valid kernels can be built from existing ones using closure rules.
This allows composing kernels that encode rich prior knowledge.
""")
    print("\033[93mCLOSURE RULES\033[0m")
    print("""
  If k₁, k₂ are valid kernels and c > 0:

  Sum:         k(x,x') = k₁(x,x') + k₂(x,x')       ← valid
  Product:     k(x,x') = k₁(x,x') · k₂(x,x')        ← valid
  Scaled:      k(x,x') = c · k₁(x,x')                ← valid
  Exp:         k(x,x') = exp(k₁(x,x'))                ← valid
  Power:       k(x,x') = k₁(x,x')^p, p∈ℤ⁺            ← valid
  Composition: k(x,x') = k₁(φ(x), φ(x'))             ← valid

  Example — sum for multi-scale:
    k(x,x') = exp(−||x−x'||²/σ₁²) + exp(−||x−x'||²/σ₂²)
    Captures both local and global structure simultaneously.

  Example — product for independence:
    k((x₁,x₂),(x'₁,x'₂)) = k_text(x₁,x'₁) · k_numeric(x₂,x'₂)
    For mixed feature types.
""")

    np.random.seed(42)
    X = np.random.randn(5, 2)

    def rbf(a, b, s=1.0): return math.exp(-np.sum((a-b)**2)/(2*s**2))
    def poly(a, b, d=2):  return (a @ b + 1) ** d

    n = len(X)
    K_rbf  = np.array([[rbf(X[i], X[j]) for j in range(n)] for i in range(n)])
    K_poly = np.array([[poly(X[i], X[j]) for j in range(n)] for i in range(n)])
    K_sum  = K_rbf + K_poly
    K_prod = K_rbf * K_poly

    for name, K in [("RBF", K_rbf), ("Poly-2", K_poly),
                    ("Sum (RBF+Poly)", K_sum), ("Product (RBF×Poly)", K_prod)]:
        eigs = np.linalg.eigvalsh(K)
        psd = all(e >= -1e-8 for e in eigs)
        print(f"  {name:<22} PSD={psd}  min_eig={min(eigs):.4f}")

    input("\033[90m[Enter to continue]\033[0m")
