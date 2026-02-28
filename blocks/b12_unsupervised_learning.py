"""
blocks/b12_unsupervised_learning.py
Block 12: Unsupervised Learning
Topics: K-Means, GMM, Hierarchical Clustering, DBSCAN, PCA,
        t-SNE, Autoencoders, ICA.
"""

import numpy as np
import sys
import os

from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold,
                        bold_yellow, bold_magenta, white, hint, red,
                        formula, section, emph, value)
from ui.widgets import (box, section_header, breadcrumb, nav_bar, table,
                         bar_chart, code_block, panel, pager, hr, print_sparkline)
from ui.menu import topic_nav, clear, block_menu, mark_completed
from viz.ascii_plots import scatter
from viz.terminal_plots import distribution_plot, loss_curve_plot
from viz.matplotlib_plots import show_heatmap, plot_decision_boundary


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — K-Means Clustering
# ══════════════════════════════════════════════════════════════════════════════
def topic_kmeans():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "K-Means Clustering")
    section_header("K-MEANS CLUSTERING")
    print()

    section_header("1. THEORY")
    print(white("""
  K-Means partitions n data points into K clusters by minimising the
  Within-Cluster Sum of Squares (WCSS):

      WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

  where μₖ is the centroid of cluster k. This is NP-hard in general, but
  Lloyd's algorithm provides a practical greedy alternating minimisation:

  LLOYD'S ALGORITHM:
  1. Initialise K centroids {μ₁, ..., μₖ} (randomly or via k-means++).
  2. ASSIGNMENT STEP: assign each point to the nearest centroid:
         cᵢ = argmin_k ||xᵢ - μₖ||²
  3. UPDATE STEP: recompute each centroid as the mean of assigned points:
         μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
  4. Repeat steps 2-3 until assignments no longer change (or convergence).

  K-MEANS++ INITIALISATION reduces the chance of poor solutions. Choose the
  first centroid uniformly at random. Then each subsequent centroid is sampled
  with probability proportional to D(x)² — the squared distance to the nearest
  already-chosen centroid. This gives an O(log K) approximation guarantee.

  THE ELBOW METHOD: run k-means for K=1,...,10, plot WCSS vs K. The 'elbow'
  (point of diminishing returns) suggests the natural number of clusters.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²"))
    print(formula("  Assignment: cᵢ = argmin_k ||xᵢ - μₖ||²"))
    print(formula("  Update:     μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ"))
    print(formula("  k-means++ sampling: P(x) ∝ D(x)² = min_k ||x - μₖ||²"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Lloyd's Algorithm")
    rng = np.random.default_rng(0)
    centers_true = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]])
    X = np.vstack([rng.normal(c, 0.6, (30, 2)) for c in centers_true])
    K = 3

    # k-means++ init
    def kmeans_pp_init(X, K, rng):
        idx = rng.integers(0, len(X))
        centroids = [X[idx]]
        for _ in range(K - 1):
            D2 = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            probs = D2 / D2.sum()
            cumprobs = np.cumsum(probs)
            r = rng.random()
            centroids.append(X[np.searchsorted(cumprobs, r)])
        return np.array(centroids)

    def lloyd(X, K, rng, max_iter=20):
        centroids = kmeans_pp_init(X, K, rng)
        for it in range(max_iter):
            dists = np.array([[np.sum((x - c)**2) for c in centroids] for x in X])
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
            wcss = sum(np.sum((X[labels == k] - new_centroids[k])**2) for k in range(K))
            if np.allclose(centroids, new_centroids):
                print(f"  Converged at iteration {it + 1}   WCSS={green(f'{wcss:.4f}')}")
                break
            centroids = new_centroids
        return labels, centroids, wcss

    labels, centroids, wcss = lloyd(X, K, rng)
    colors_sym = [green("●"), yellow("●"), cyan("●")]
    print(f"\n  {bold_cyan('ASCII cluster preview (90-point dataset, K=3):')}\n")
    grid = [[" "] * 40 for _ in range(14)]
    for i, pt in enumerate(X):
        col = int((pt[0] + 1) / 6 * 38)
        row = int((pt[1] + 0.5) / 4 * 12)
        col = max(0, min(38, col)); row = max(0, min(12, row))
        grid[row][col] = str(labels[i])
    for row in reversed(grid):
        print("  " + "".join(
            green("●") if c == "0" else
            yellow("●") if c == "1" else
            cyan("●") if c == "2" else grey("·")
            for c in row))
    print()
    _pause()

    section_header("4. VISUALIZATION — Inertia (Elbow)")
    wcss_vals = []
    for k in range(1, 9):
        _, _, w = lloyd(X, k, rng)
        wcss_vals.append(w)

    try:
        import plotext as plt
        plt.clear_figure()
        plt.plot(list(range(1, 9)), wcss_vals, color="cyan")
        plt.title("Elbow Method — WCSS vs K"); plt.xlabel("K"); plt.ylabel("WCSS")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for inertia curve)"))
        print(f"\n  {bold_cyan('WCSS per K:')}")
        bar_chart("WCSS", [str(k) for k in range(1, 9)], wcss_vals)

    section_header("5. CODE")
    code_block("K-Means from scratch", """
import numpy as np

def kmeans(X, K, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ init
    idx = rng.integers(0, len(X))
    centroids = [X[idx].copy()]
    for _ in range(K - 1):
        D2 = np.array([min(np.sum((x-c)**2) for c in centroids) for x in X])
        centroids.append(X[rng.choice(len(X), p=D2/D2.sum())])
    centroids = np.array(centroids)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=-1)  # (n, K)
        labels = np.argmin(dists, axis=1)
        new_c  = np.array([X[labels==k].mean(0) for k in range(K)])
        if np.allclose(centroids, new_c): break
        centroids = new_c
    wcss = sum(np.sum((X[labels==k]-centroids[k])**2) for k in range(K))
    return labels, centroids, wcss

# Test
rng = np.random.default_rng(42)
X = np.vstack([rng.normal([i*4, j*4], 0.8, (50,2))
               for i, j in [(0,0),(1,0),(0,1),(1,1)]])
labels, centroids, wcss = kmeans(X, K=4)
print(f"WCSS = {wcss:.2f}, centroids:\\n{centroids}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "K-Means minimises WCSS — equivalent to minimising intra-cluster variance",
        "k-means++ init gives O(log K) approximation vs random init",
        "K-Means assumes spherical, equal-size clusters — fails on elongated clusters",
        "Elbow method: plot WCSS vs K, choose k at the 'elbow' bend",
        "Run multiple times with different seeds; keep lowest WCSS solution",
        "Sensitive to outliers — consider K-Medoids (PAM) for robustness",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Gaussian Mixture Models
# ══════════════════════════════════════════════════════════════════════════════
def topic_gmm():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "Gaussian Mixture Model")
    section_header("GAUSSIAN MIXTURE MODEL (GMM)")
    print()

    section_header("1. THEORY")
    print(white("""
  A GMM models the data distribution as a weighted sum of K Gaussians:

      p(x) = Σₖ πₖ N(x | μₖ, Σₖ)

  where πₖ are mixing weights (Σ πₖ = 1, πₖ ≥ 0), μₖ are means, and Σₖ
  are covariance matrices. Unlike K-Means (hard assignment), GMM gives
  SOFT ASSIGNMENTS via the responsibility γᵢₖ — the posterior probability
  that point xᵢ belongs to cluster k:

      γᵢₖ = πₖ N(xᵢ|μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ, Σⱼ)     [E-step]

  GMM IS TRAINED VIA EM (Expectation-Maximisation):

  E-STEP: compute responsibilities γᵢₖ using current parameters.
  M-STEP: update parameters to maximise expected complete log-likelihood:
      Nₖ  = Σᵢ γᵢₖ                         (effective count)
      μₖ  = (1/Nₖ) Σᵢ γᵢₖ xᵢ              (weighted mean)
      Σₖ  = (1/Nₖ) Σᵢ γᵢₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ  (weighted covariance)
      πₖ  = Nₖ / n                          (mixing weight)

  COMPARISON WITH K-MEANS: K-Means is a special case where Σₖ = σ²I and
  the softmax over responsibilities becomes a hard argmin (T → 0 limit).
  GMM can model elliptical clusters and quantify uncertainty in assignments.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  p(x) = Σₖ πₖ N(x|μₖ, Σₖ)"))
    print(formula("  γᵢₖ = πₖ N(xᵢ|μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ, Σⱼ)   [responsibility]"))
    print(formula("  log-likelihood: ℓ = Σᵢ log Σₖ πₖ N(xᵢ|μₖ, Σₖ)  [maximised by EM]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — GMM EM")
    rng = np.random.default_rng(3)
    n_per = 80
    K = 3
    true_mu = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    X = np.vstack([rng.normal(mu, 1.0, (n_per, 2)) for mu in true_mu])
    n = len(X)

    def gmm_em(X, K, n_iter=30, rng=None):
        rng = rng or np.random.default_rng(0)
        d = X.shape[1]
        mu = X[rng.choice(n, K, replace=False)]
        Sigma = np.array([np.eye(d)] * K)
        pi = np.ones(K) / K
        lls = []
        for it in range(n_iter):
            # E-step
            log_resp = np.zeros((n, K))
            for k in range(K):
                diff = X - mu[k]
                cov_inv = np.linalg.inv(Sigma[k] + 1e-6 * np.eye(d))
                maha = np.sum(diff @ cov_inv * diff, axis=1)
                sign, logdet = np.linalg.slogdet(Sigma[k] + 1e-6 * np.eye(d))
                log_resp[:, k] = (np.log(pi[k] + 1e-300) -
                                  0.5 * (d * np.log(2 * np.pi) + logdet + maha))
            log_resp -= log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)
            ll = np.sum(np.log(np.sum(np.exp(log_resp + log_resp.max(axis=1, keepdims=True)
                                              - log_resp.max(axis=1, keepdims=True)), axis=1)))
            lls.append(ll)
            # M-step
            Nk = resp.sum(axis=0)
            mu = (resp.T @ X) / Nk[:, None]
            for k in range(K):
                diff = X - mu[k]
                Sigma[k] = ((resp[:, k:k+1] * diff).T @ diff) / Nk[k] + 1e-4 * np.eye(d)
            pi = Nk / n
        return mu, Sigma, pi, resp, lls

    mu_hat, Sig_hat, pi_hat, resp, lls = gmm_em(X, K, n_iter=30, rng=rng)
    labels = resp.argmax(axis=1)
    accuracy = max(
        np.mean(labels == np.repeat(np.arange(K), n_per)),
        np.mean(labels[np.repeat([0,1,2], n_per)] == np.repeat([0,2,1], n_per))
    )

    print(f"\n  {bold_cyan('GMM EM trained on 3-cluster Gaussian data (n=240):')}\n")
    rows = [[str(k+1), f"[{mu_hat[k,0]:.2f}, {mu_hat[k,1]:.2f}]",
             f"{pi_hat[k]:.3f}"] for k in range(K)]
    table(["Cluster", "Est. μ", "Est. π"], rows, [cyan, green, yellow])
    print(f"\n  {bold_cyan('True μ:')} {', '.join(str(m.tolist()) for m in true_mu)}")
    print(f"  {bold_cyan('Log-Likelihood progression:')}")
    print_sparkline(lls, label="LL", color_fn=green)
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import plotext as plt
        plt.clear_figure()
        plt.plot(list(range(len(lls))), lls, color="green", label="Log-Likelihood")
        plt.title("GMM: Log-Likelihood per EM Iteration")
        plt.xlabel("Iteration"); plt.ylabel("Log-Likelihood")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for log-likelihood curve)"))

    section_header("5. CODE")
    code_block("GMM with sklearn GaussianMixture", """
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)
K = 3
centers = [[0,0],[5,0],[2.5,4]]
X = np.vstack([rng.normal(c, 1.0, (80,2)) for c in centers])

gmm = GaussianMixture(n_components=K, covariance_type='full',
                       n_init=5, random_state=0).fit(X)
print(f"Converged: {gmm.converged_}  after {gmm.n_iter_} iterations")
print(f"Log-likelihood: {gmm.score(X)*len(X):.2f}")
print(f"Estimated means:\\n{gmm.means_}")
print(f"Mixing weights: {gmm.weights_}")

# Soft assignments (responsibilities)
resp = gmm.predict_proba(X)
print(f"Sample responsibilities (first 3 points):\\n{resp[:3]}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "GMM = soft K-Means: responsibilities are probabilistic cluster assignments",
        "EM for GMM is guaranteed to non-decrease log-likelihood at each iteration",
        "Covariance type: full (ellipsoidal), tied, diagonal, spherical — trade off",
        "K-Means is GMM with equal, spherical covariance and zero-temperature limit",
        "Model selection via BIC = -2ℓ + k·ln(n) to penalise extra components",
        "GMM + Bayes' theorem gives cluster posterior — useful for anomaly detection",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Hierarchical Clustering
# ══════════════════════════════════════════════════════════════════════════════
def topic_hierarchical():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "Hierarchical Clustering")
    section_header("HIERARCHICAL CLUSTERING")
    print()

    section_header("1. THEORY")
    print(white("""
  Hierarchical clustering builds a nested hierarchy of clusters called a
  dendrogram without requiring K to be specified in advance. Two families:

  AGGLOMERATIVE (bottom-up): Start with n singleton clusters; merge the two
  most similar clusters at each step until a single cluster remains.

  DIVISIVE (top-down): Start with one cluster; recursively split.

  LINKAGE CRITERIA define how inter-cluster distance is measured:
  - SINGLE LINKAGE: d(A,B) = min{d(a,b) : a∈A, b∈B}  (nearest neighbour)
    → tends to produce 'chaining' elongated clusters
  - COMPLETE LINKAGE: d(A,B) = max{d(a,b) : a∈A, b∈B}  (furthest neighbour)
    → produces compact, roughly spherical clusters
  - AVERAGE LINKAGE: d(A,B) = (1/|A||B|) Σ d(a,b)  (UPGMA)
    → compromise between single and complete
  - WARD'S METHOD: merge clusters that minimise the total within-cluster variance
    increase. Equivalent to minimising: Δ(A,B) = |A||B|/(|A|+|B|) ||μ_A - μ_B||²
    → tends to produce balanced, equally-sized clusters.

  THE DENDROGRAM displays the merging sequence on the y-axis (distance/height).
  Cutting the dendrogram at a height h gives flat clusters. The number of
  clusters at height h is the number of vertical lines that intersect a
  horizontal cut at h.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Single linkage:   d(A,B) = min_{a∈A,b∈B} ||a - b||"))
    print(formula("  Complete linkage: d(A,B) = max_{a∈A,b∈B} ||a - b||"))
    print(formula("  Ward delta:       Δ(A,B) = |A||B|/(|A|+|B|) ||μ_A - μ_B||²"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Agglomerative (Single Linkage)")
    rng = np.random.default_rng(7)
    pts = np.array([[0, 0], [1, 0.5], [0.5, 1], [5, 5], [5.5, 4.5], [4.5, 5.5]])
    labels_str = ["A", "B", "C", "D", "E", "F"]
    n = len(pts)
    # Compute pairwise distances
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = np.linalg.norm(pts[i] - pts[j])

    print(f"\n  {bold_cyan('Points:')} A,B,C clustered near (0,0); D,E,F near (5,5)\n")
    print(f"  {bold_cyan('Pairwise distances:')}\n")
    header_row = ["  "] + labels_str
    dist_rows = [[labels_str[i]] + [f"{dist_mat[i,j]:.2f}" for j in range(n)] for i in range(n)]
    table(header_row, dist_rows)

    # Simulate agglomerative steps
    print(f"\n  {bold_cyan('Agglomerative merging (single linkage):')}\n")
    steps = [
        ("A, B", "0.00 → 1.12", green("Merge A+B (dist=1.12)")),
        ("(A,B), C", "0.00 → 1.22", green("Merge (A,B)+C (dist=1.22)")),
        ("D, E", "0.00 → 0.71", green("Merge D+E (dist=0.71)")),
        ("(D,E), F", "0.71 → 1.12", green("Merge (D,E)+F (dist=1.12)")),
        ("(A,B,C), (D,E,F)", "1.22 → 5.59", cyan("Final merge at d=5.59")),
    ]
    for step, dist, msg in steps:
        print(f"  {yellow('Step:')} {cyan(step):<30} d={dist:<20} {msg}")

    print(f"\n  {bold_cyan('ASCII Dendrogram sketch:')}\n")
    dendro = [
        "  5.59 ─────────────┬─────────────",
        "                    │            │",
        "  1.22 ─────┬───    │            │",
        "            │   │   │            │",
        "  1.12 ─┬─  │   └──────┬──      │",
        "        │ │  │         │   │     │",
        "  0.71  │ │  C     ─┬─   │   └──┬───",
        "        │ │          │ │       │   │   F",
        "  0.00  A B          D E       "
    ]
    for line in dendro:
        print(cyan(line))
    print()
    _pause()

    section_header("4. CODE")
    code_block("Hierarchical clustering with scipy", """
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0,0], 0.5, (20,2)),
               rng.normal([5,5], 0.5, (20,2)),
               rng.normal([0,5], 0.5, (20,2))])

# Try all linkage methods
for method in ['single', 'complete', 'average', 'ward']:
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=3, criterion='maxclust')
    # Purity (assuming 3 true groups of 20)
    true = np.repeat([0,1,2], 20)
    best = max(np.mean(labels-1 == true), np.mean(labels-1 == true[[1,0,2,]*20][:60]))
    print(f"{method:<10}: merge height={Z[-1,2]:.2f}  purity≈{best:.3f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "No need to specify K in advance — cut dendrogram at desired height",
        "Single linkage: chaining effect, good for elongated clusters",
        "Complete linkage: compact clusters, sensitive to outliers",
        "Ward: minimises variance increase; best for balanced compact clusters",
        "O(n²) memory for distance matrix; O(n² log n) time — expensive for n>10k",
        "Dendrogram height = distance at which clusters merged (interpretable)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — DBSCAN
# ══════════════════════════════════════════════════════════════════════════════
def topic_dbscan():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "DBSCAN")
    section_header("DBSCAN — Density-Based Clustering")
    print()

    section_header("1. THEORY")
    print(white("""
  DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
  points that are closely packed together and marks outliers as noise.
  Unlike K-Means it requires no pre-specified K and can find arbitrary shapes.

  PARAMETERS: ε (epsilon) — radius of neighbourhood; MinPts — minimum number
  of points to form a dense region (including the point itself).

  POINT TYPES:
  - CORE POINT: a point p is core if |{q : d(p,q) ≤ ε}| ≥ MinPts.
    The ε-ball around a core point contains at least MinPts neighbours.
  - BORDER POINT: within the ε-ball of a core point but not itself a core point.
  - NOISE POINT: neither core nor border — isolated outlier.

  CONNECTIVITY:
  - Directly density-reachable: q is D-reachable from p if d(p,q) ≤ ε and p is core.
  - Density-reachable (transitive closure): there is a chain of directly reachable pts.
  - Density-connected: both p and q are density-reachable from some common core point o.
  A cluster is a maximal set of density-connected points.

  ALGORITHM:
  1. For each unvisited point p, visit it.
  2. If p is a core point, start a new cluster by growing from p's ε-neighbourhood.
  3. Recursively add density-reachable points to the cluster.
  4. Non-core points that are reachable: border. Unreachable: noise (label = -1).

  CHOOSING ε AND MinPts: compute k-nearest-neighbor distances (k=MinPts-1) for
  all points. Plot sorted distances; the 'knee' suggests ε. Typically MinPts = 2·d
  where d is the data dimensionality.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Core point: |N_ε(p)| = |{q : d(p,q) ≤ ε}| ≥ MinPts"))
    print(formula("  Cluster: maximal set of density-connected points"))
    print(formula("  Time complexity: O(n log n) with spatial index, O(n²) naively"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — DBSCAN Walkthrough")
    rng = np.random.default_rng(5)
    n_cluster = 20
    # Three clusters + noise
    c1 = rng.normal([0, 0], 0.5, (n_cluster, 2))
    c2 = rng.normal([4, 0], 0.5, (n_cluster, 2))
    c3 = rng.normal([2, 3], 0.5, (n_cluster, 2))
    noise = rng.uniform(-2, 6, (5, 2))
    X = np.vstack([c1, c2, c3, noise])
    eps, min_pts = 1.0, 4

    def dbscan(X, eps, min_pts):
        n = len(X)
        labels = -np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        def region_query(p_idx):
            return [i for i in range(n) if np.linalg.norm(X[p_idx] - X[i]) <= eps]

        def expand(p_idx, neighbors, cid):
            labels[p_idx] = cid
            i = 0
            while i < len(neighbors):
                q = neighbors[i]
                if not visited[q]:
                    visited[q] = True
                    q_nb = region_query(q)
                    if len(q_nb) >= min_pts:
                        neighbors += [x for x in q_nb if x not in neighbors]
                    if labels[q] == -1:
                        labels[q] = cid
                i += 1

        for p in range(n):
            if visited[p]:
                continue
            visited[p] = True
            nb = region_query(p)
            if len(nb) < min_pts:
                labels[p] = -1  # noise
            else:
                expand(p, nb, cluster_id)
                cluster_id += 1
        return labels

    labels = dbscan(X, eps, min_pts)
    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"\n  {bold_cyan(f'DBSCAN results (ε={eps}, MinPts={min_pts}):')}\n")
    print(f"  Clusters found:  {green(str(n_clusters))}")
    print(f"  Noise points:    {red(str(n_noise))}")
    for cid in sorted(set(labels)):
        if cid == -1:
            print(f"  {red('Noise')}: {np.sum(labels == -1)} points")
        else:
            pts_in = X[labels == cid]
            print(f"  {cyan(f'Cluster {cid}')}: {len(pts_in)} points, "
                  f"centroid=({pts_in[:,0].mean():.2f}, {pts_in[:,1].mean():.2f})")
    print()
    _pause()

    section_header("4. CODE")
    code_block("DBSCAN with sklearn", """
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs

# Crescent moons — impossible for K-Means but easy for DBSCAN
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
db_moons = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)
print(f"Moons: {len(set(db_moons.labels_))-1} clusters, "
      f"{(db_moons.labels_==-1).sum()} noise")

# Blobs — should find 3 clusters
X_blobs, _ = make_blobs(n_samples=300, centers=3, random_state=0)
db_blobs = DBSCAN(eps=1.2, min_samples=5).fit(X_blobs)
print(f"Blobs: {len(set(db_blobs.labels_))-1} clusters, "
      f"{(db_blobs.labels_==-1).sum()} noise")

# Choosing eps via k-distance plot
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5).fit(X_blobs)
dists, _ = nn.kneighbors(X_blobs)
k_dists = np.sort(dists[:, -1])[::-1]
print(f"Suggested eps ≈ {k_dists[int(len(k_dists)*0.1)]:.3f}  (knee of k-dist plot)")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "DBSCAN needs no K — discovers the number of clusters from data density",
        "Handles arbitrary cluster shapes including crescents, rings, and blobs",
        "Noise/outlier points get label -1 — built-in anomaly detection",
        "ε-MinPts: use k-NN distance plot (knee) to choose ε; MinPts ≥ 2·dim",
        "Fails in varying-density data — use HDBSCAN for hierarchical density",
        "Time O(n log n) with KD-tree / Ball-tree; O(n²) brute force",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — PCA (Unsupervised View)
# ══════════════════════════════════════════════════════════════════════════════
def topic_pca_unsupervised():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "PCA")
    section_header("PCA — PRINCIPAL COMPONENT ANALYSIS")
    print()

    section_header("1. THEORY")
    print(white("""
  PCA finds a lower-dimensional linear subspace that retains maximum variance.
  Given centred data matrix X ∈ ℝⁿˣᵈ, PCA solves:

      min_{W, Z} ||X - ZWᵀ||²_F   s.t. WᵀW = Iₖ

  where Z ∈ ℝⁿˣᵏ are the latent codes (projections) and W ∈ ℝᵈˣᵏ the loadings.
  The solution is: W = first k eigenvectors of XᵀX (or right singular vectors of X).

  CONNECTION TO SVD: X = UΣVᵀ (economy SVD). The principal components are the
  columns of V (right singular vectors). The latent codes are Z = XV = UΣ.
  Reconstruction: X̂ = ZWᵀ = UₖΣₖVₖᵀ (truncated SVD).
  Reconstruction error = Σᵢ>ₖ σᵢ² (sum of discarded singular values squared).

  VARIANCE EXPLAINED: the i-th principal component explains a fraction
      eᵢ = σᵢ² / Σⱼ σⱼ²  of total variance.
  Choose k so that Σᵢ₌₁ᵏ eᵢ ≥ 0.95 (keep 95% variance).

  WHITENING: Z_white = Z · diag(1/√λ₁, ..., 1/√λₖ) normalises each component
  to unit variance. Useful preprocessing for ICA and neural networks.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  PCA solution: W = top-k eigenvectors of (1/n)XᵀX"))
    print(formula("  Projection:   Z = X·W  (latent codes)"))
    print(formula("  Reconstruction: X̂ = Z·Wᵀ = UₖΣₖVₖᵀ"))
    print(formula("  Variance explained: eᵢ = λᵢ / Σⱼ λⱼ"))
    print(formula("  Whitening: Z_w = Z · diag(1/√λ₁, ..., 1/√λₖ)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — PCA on 5D Data")
    rng = np.random.default_rng(11)
    n = 200
    d = 5
    # True signal lives in 2D
    Z_true = rng.normal(0, 1, (n, 2))
    W_true = rng.normal(0, 1, (d, 2))
    X_raw = Z_true @ W_true.T + rng.normal(0, 0.15, (n, d))
    X = X_raw - X_raw.mean(axis=0)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = S**2 / np.sum(S**2)
    cumulative = np.cumsum(var_explained)

    print(f"\n  {bold_cyan('Variance explained per principal component:')}\n")
    rows = [[str(i+1), f"{S[i]:.3f}", f"{var_explained[i]*100:.1f}%",
             f"{cumulative[i]*100:.1f}%"] for i in range(d)]
    table(["PC", "σ (sing. val)", "Variance %", "Cumulative %"], rows,
          [cyan, yellow, green, bold_cyan])

    k = np.argmax(cumulative >= 0.95) + 1
    print(f"\n  {green('✓')} {white(f'Need k={k} components to explain ≥95% variance')}")
    Z_proj = X @ Vt[:k].T
    X_recon = Z_proj @ Vt[:k]
    recon_err = np.mean((X - X_recon)**2)
    print(f"  Reconstruction MSE with k={k}: {green(f'{recon_err:.5f}')}")
    print()
    _pause()

    section_header("4. VISUALIZATION — Scree Plot")
    try:
        import plotext as plt
        plt.clear_figure()
        plt.bar(list(range(1, d+1)), [v*100 for v in var_explained], color="cyan")
        plt.title("Scree Plot — Variance Explained per PC")
        plt.xlabel("Principal Component"); plt.ylabel("Variance %")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for scree plot)"))
        bar_chart("Variance %", [f"PC{i+1}" for i in range(d)],
                  [v*100 for v in var_explained])

    print(f"\n  {bold_cyan('First 2 PCs projected (ASCII scatter):')}\n")
    Z2 = X @ Vt[:2].T
    x_min, x_max = Z2[:, 0].min(), Z2[:, 0].max()
    y_min, y_max = Z2[:, 1].min(), Z2[:, 1].max()
    grid = [[" "] * 50 for _ in range(16)]
    for pt in Z2:
        col = int((pt[0] - x_min) / (x_max - x_min) * 48)
        row = int((pt[1] - y_min) / (y_max - y_min) * 14)
        grid[row][col] = "·"
    for row in reversed(grid):
        print("  " + cyan("".join(row)))
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt2
        fig, ax = plt2.subplots(figsize=(6, 5))
        ax.scatter(Z2[:, 0], Z2[:, 1], alpha=0.6, s=20)
        # Plot principal axes
        for i in range(2):
            scale = S[i] / np.sqrt(n)
            ax.arrow(0, 0, Vt[i, 0]*scale*5, Vt[i, 1]*scale*5,
                     head_width=0.05, color=["red","blue"][i], label=f"PC{i+1}")
        ax.set_title("PCA: First 2 Principal Components"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("/tmp/pca_scatter.png", dpi=90)
        plt2.close(fig)
        print(green("  [matplotlib] PCA scatter saved to /tmp/pca_scatter.png"))
    except ImportError:
        print(grey("  (install matplotlib for PCA plot with axes)"))

    section_header("5. CODE")
    code_block("PCA from scratch and sklearn", """
import numpy as np

def pca(X, k):
    X = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:k].T          # (d, k) loading matrix
    Z = X @ W             # (n, k) latent codes
    var_explained = S**2 / S.sum()**2
    return Z, W, var_explained

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (100, 10))
X[:, 0] *= 5; X[:, 1] *= 3   # inject high-variance directions
Z, W, ve = pca(X, k=2)
print(f"Shape: X={X.shape} → Z={Z.shape}")
print(f"Variance explained by PC1={ve[0]:.3f}, PC2={ve[1]:.3f}")
print(f"Total (k=2): {ve[:2].sum():.3f}")

# sklearn equivalent
from sklearn.decomposition import PCA
pca_sk = PCA(n_components=2).fit(X)
print(f"sklearn PC1={pca_sk.explained_variance_ratio_[0]:.3f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "PCA solves min reconstruction error = min Σᵢ>ₖ λᵢ (discarded eigenvalues)",
        "Choose k so cumulative variance ≥ 95%; scree plot shows 'elbow'",
        "PCA assumes linear structure — use kernel PCA or autoencoders for nonlinear",
        "Whitening (dividing by √λ) decorrelates and scales components to unit variance",
        "SVD is numerically preferred over eigen-decomposition of covariance matrix",
        "PCA only valid on centred data — always subtract mean before applying",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — t-SNE
# ══════════════════════════════════════════════════════════════════════════════
def topic_tsne():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "t-SNE")
    section_header("t-SNE — t-Distributed Stochastic Neighbour Embedding")
    print()

    section_header("1. THEORY")
    print(white("""
  t-SNE (van der Maaten & Hinton, 2008) is a non-linear dimensionality reduction
  technique designed for visualisation into 2D or 3D.

  HIGH-DIM AFFINITIES: for each pair (i, j), compute the conditional probability
  that xᵢ would pick xⱼ as a neighbour under a Gaussian:
      p_{j|i} = exp(-||xᵢ-xⱼ||² / 2σᵢ²) / Σₖ≠ᵢ exp(-||xᵢ-xₖ||² / 2σᵢ²)
  Symmetrised: pᵢⱼ = (p_{j|i} + p_{i|j}) / (2n).
  The bandwidth σᵢ is chosen such that the perplexity = 2^{H(P_i)} matches a
  user-specified value (typically 5–50). Perplexity ≈ effective neighborhood size.

  LOW-DIM AFFINITIES: in the low-dimensional embedding, use the heavier-tailed
  Student-t distribution (1 degree of freedom = Cauchy) to model affinities:
      qᵢⱼ = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖ≠ₗ (1 + ||yₖ - yₗ||²)⁻¹

  WHY STUDENT-t? In high dimensions, distances tend to concentrate (curse of
  dimensionality). The heavier tails of Student-t allow dissimilar points to be
  placed far apart more easily — preventing the 'crowding problem' where all
  medium-distance points collapse to a ring around nearby clusters.

  OBJECTIVE: minimise KL divergence from P to Q over embedding coordinates Y:
      KL(P||Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)
  Optimised by gradient descent with momentum. Gradient:
      ∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ - qᵢⱼ)(yᵢ-yⱼ)(1 + ||yᵢ-yⱼ||²)⁻¹
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  pᵢⱼ = (p_{j|i} + p_{i|j}) / 2n     [symmetric high-dim affinity]"))
    print(formula("  qᵢⱼ = (1+||yᵢ-yⱼ||²)⁻¹ / Z         [Student-t low-dim affinity]"))
    print(formula("  KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)   [objective to minimise]"))
    print(formula("  ∂C/∂yᵢ = 4Σⱼ(pᵢⱼ-qᵢⱼ)(yᵢ-yⱼ)(1+||yᵢ-yⱼ||²)⁻¹"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — t-SNE on 4D Gaussians")
    rng = np.random.default_rng(20)
    K, n_each = 4, 40
    means = np.array([[0,0,0,0],[5,0,0,0],[0,5,0,0],[5,5,0,0]], dtype=float)
    X = np.vstack([rng.normal(m, 1.0, (n_each, 4)) for m in means])
    X = X - X.mean(axis=0)

    try:
        from sklearn.manifold import TSNE
        Y = TSNE(n_components=2, perplexity=15, random_state=0, n_iter=500).fit_transform(X)
        labels = np.repeat(np.arange(K), n_each)
        print(f"\n  {bold_cyan('t-SNE 2D embedding (4 Gaussian clusters, 4D → 2D):')}\n")
        grid_size = 20
        plot_grid = [[" "] * 50 for _ in range(grid_size)]
        for pt, lab in zip(Y, labels):
            col = int((pt[0] - Y[:, 0].min()) / (Y[:, 0].max() - Y[:, 0].min() + 1e-9) * 48)
            row = int((pt[1] - Y[:, 1].min()) / (Y[:, 1].max() - Y[:, 1].min() + 1e-9) * (grid_size - 2))
            col = max(0, min(48, col)); row = max(0, min(grid_size - 2, row))
            symbols = ["●", "■", "▲", "◆"]
            plot_grid[row][col] = str(lab)
        for row in reversed(plot_grid):
            print("  " + "".join(
                green("●") if c == "0" else yellow("■") if c == "1" else
                cyan("▲") if c == "2" else red("◆") if c == "3" else grey("·")
                for c in row))
        print()
    except ImportError:
        print(grey("  (install scikit-learn for t-SNE computation)"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        Y = TSNE(n_components=2, perplexity=15, random_state=0, n_iter=500).fit_transform(X)
        labels = np.repeat(np.arange(K), n_each)
        fig, ax = plt.subplots(figsize=(6, 5))
        for k in range(K):
            mask = labels == k
            ax.scatter(Y[mask, 0], Y[mask, 1], label=f"Cluster {k}", alpha=0.7, s=30)
        ax.set_title("t-SNE: 4D Gaussians → 2D"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("/tmp/tsne.png", dpi=90)
        plt.close(fig)
        print(green("  [matplotlib] t-SNE plot saved to /tmp/tsne.png"))
    except ImportError:
        print(grey("  (install matplotlib + sklearn for t-SNE plot)"))

    section_header("5. CODE")
    code_block("t-SNE with sklearn", """
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target   # (1797, 64)

# PCA first for speed (common practice: PCA to 50D then t-SNE)
from sklearn.decomposition import PCA
X_pca = PCA(n_components=50).fit_transform(X)

Y = TSNE(n_components=2, perplexity=30, learning_rate=200,
         n_iter=1000, random_state=0).fit_transform(X_pca)

print(f"Input: {X.shape}  →  PCA: {X_pca.shape}  →  t-SNE: {Y.shape}")
for digit in range(10):
    centroid = Y[y==digit].mean(axis=0)
    print(f"Digit {digit}: centroid=({centroid[0]:.1f}, {centroid[1]:.1f})")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "t-SNE is for VISUALISATION only — not dimensionality reduction for ML pipelines",
        "Perplexity ≈ effective neighbours; typical range 5-50, affects global structure",
        "Student-t tails prevent crowding problem: dissimilar points spread far apart",
        "KL divergence is asymmetric: P large, Q small is penalised severely",
        "Distances in t-SNE plot are NOT meaningful — only topology/clusters are",
        "PCA → t-SNE pipeline (50D first) is standard practice for speed",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Autoencoders
# ══════════════════════════════════════════════════════════════════════════════
def topic_autoencoders():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "Autoencoders")
    section_header("AUTOENCODERS")
    print()

    section_header("1. THEORY")
    print(white("""
  An autoencoder learns a compressed latent representation by training an
  encoder-decoder pair to minimise reconstruction loss:

      Encoder: f: x → z = f(x)      (x ∈ ℝᵈ, z ∈ ℝₖ, k << d)
      Decoder: g: z → x̂ = g(z)
      Loss:    L = ||x - g(f(x))||²  (MSE, or BCE for binary inputs)

  The bottleneck (k << d) forces the network to learn a compressed representation
  that captures the most important structure in the data.

  DENOISING AUTOENCODER (DAE): corrupt the input x → x̃ (e.g. add Gaussian noise
  or drop pixels), then train to reconstruct the clean x. This forces the encoder
  to learn robust features. The optimal DAE decoder is the posterior mean E[x|x̃].

  SPARSE AUTOENCODER: add sparsity penalty to latent code:
      L = ||x - x̂||² + λ·KL(ρ || ρ̂_j)
  where ρ is target sparsity (e.g. 0.05) and ρ̂_j is average activation of unit j.
  Encourages only a few units to activate per input — learns parts-based features.

  CONTRACTIVE AUTOENCODER: penalise the Frobenius norm of the Jacobian:
      L = ||x - x̂||² + λ·||∂f(x)/∂x||²_F
  Encourages the representation to be insensitive to small input perturbations.

  RELATIONSHIP TO PCA: a linear autoencoder with MSE loss learns the same subspace
  as PCA (but not necessarily the same basis vectors).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Loss: L = (1/n)Σᵢ ||xᵢ - g(f(xᵢ))||²"))
    print(formula("  Sparse penalty: KL(ρ||ρ̂) = ρ·log(ρ/ρ̂) + (1-ρ)·log((1-ρ)/(1-ρ̂))"))
    print(formula("  Linear AE ≡ PCA:  f(x)=Wx + b (dim reduction), g(z)=W'z + b'"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Linear Autoencoder = PCA")
    rng = np.random.default_rng(8)
    n, d, k = 200, 6, 2
    # True 2D signal + noise
    Z_true = rng.normal(0, 1, (n, k))
    W_true = rng.normal(0, 1, (d, k))
    X = Z_true @ W_true.T + rng.normal(0, 0.1, (n, d))
    X = X - X.mean(axis=0)

    # Manual gradient descent on linear AE
    W_enc = rng.normal(0, 0.1, (d, k))
    W_dec = rng.normal(0, 0.1, (k, d))
    lr = 0.01
    losses = []
    for epoch in range(300):
        Z = X @ W_enc
        X_hat = Z @ W_dec
        loss = np.mean((X - X_hat)**2)
        losses.append(loss)
        # Gradients (MSE)
        dL_dXhat = -2 * (X - X_hat) / n
        dL_dWdec = Z.T @ dL_dXhat
        dL_dZ    = dL_dXhat @ W_dec.T
        dL_dWenc = X.T @ dL_dZ
        W_enc -= lr * dL_dWenc
        W_dec -= lr * dL_dWdec

    # Compare with PCA reconstruction
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z_pca = X @ Vt[:k].T
    X_pca_recon = Z_pca @ Vt[:k]
    pca_loss = np.mean((X - X_pca_recon)**2)

    print(f"\n  {bold_cyan('Linear Autoencoder vs PCA (d=6 → k=2):')}\n")
    print(f"  AE  final loss  (300 epochs): {green(f'{losses[-1]:.5f}')}")
    print(f"  PCA min loss (optimal): {green(f'{pca_loss:.5f}')}")
    print(f"  Ratio (AE/PCA): {yellow(f'{losses[-1]/pca_loss:.3f}')}")
    print(f"\n  {hint('Linear AE converges to PCA loss — same optimal subspace.')}")
    print_sparkline(losses[::20], label="AE Loss")
    print()
    _pause()

    section_header("4. CODE")
    code_block("Simple Autoencoder (numpy)", """
import numpy as np

class LinearAutoencoder:
    def __init__(self, d, k, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W_enc = rng.normal(0, 0.1, (d, k))
        self.W_dec = rng.normal(0, 0.1, (k, d))
        self.lr = lr

    def encode(self, X): return X @ self.W_enc
    def decode(self, Z): return Z @ self.W_dec
    def reconstruct(self, X): return self.decode(self.encode(X))

    def train_step(self, X):
        n = len(X)
        Z = self.encode(X)
        Xhat = self.decode(Z)
        loss = np.mean((X - Xhat)**2)
        dL = -2*(X - Xhat)/n
        self.W_dec -= self.lr * Z.T @ dL
        self.W_enc -= self.lr * X.T @ (dL @ self.W_dec.T)
        return loss

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (200, 8))
ae = LinearAutoencoder(d=8, k=2)
for ep in range(0, 500, 100):
    L = ae.train_step(X)
    print(f"Epoch {ep:4d}: loss={L:.4f}")

# Denoising AE: add noise during training
def denoising_ae_step(ae, X, sigma=0.5):
    Xnoisy = X + np.random.randn(*X.shape) * sigma
    return ae.train_step(Xnoisy)   # reconstruct clean from noisy
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Bottleneck forces compression — latent space encodes essential structure",
        "Linear AE = PCA subspace; nonlinear AE learns curved manifolds",
        "Denoising AE: train x̃→x (noisy→clean) for more robust representations",
        "VAE (Variational AE) imposes N(0,I) prior on z — enables generation",
        "Sparse AE encourages disentangled, interpretable features via KL penalty",
        "Reconstruction quality ≠ good representation — always evaluate downstream task",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — ICA
# ══════════════════════════════════════════════════════════════════════════════
def topic_ica():
    clear()
    breadcrumb("mlmath", "Unsupervised Learning", "Independent Component Analysis")
    section_header("ICA — INDEPENDENT COMPONENT ANALYSIS")
    print()

    section_header("1. THEORY")
    print(white("""
  ICA seeks to decompose an observed signal X into a set of statistically
  INDEPENDENT components S, assuming X = A·S (mixing matrix A).
  The goal: estimate W = A⁻¹ (unmixing matrix) such that WX ≈ S.

  ICA VS PCA:
  - PCA finds uncorrelated components (zero second-order correlations).
  - ICA finds statistically INDEPENDENT components (zero ALL order correlations).
  Independence implies uncorrelation, but not vice versa.
  PCA minimises reconstruction error; ICA maximises non-Gaussianity / independence.

  KEY INSIGHT (Central Limit Theorem direction): the sum of independent random
  variables tends toward Gaussian. Therefore, the mix AX is MORE Gaussian than
  the individual sources S. ICA reverses this: find directions of maximum
  non-Gaussianity to recover the independent sources.

  FastICA ALGORITHM (one unit):
  1. Initialise random w.
  2. Update: w ← E[x·g(wᵀx)] - E[g'(wᵀx)]·w  where g = tanh (for super-Gaussian)
     or g(u) = u·exp(-u²/2) (for sub-Gaussian).
  3. Normalise: w ← w / ||w||.
  4. Repeat until convergence.
  For multiple components, use deflation or symmetric orthogonalisation.

  COCKTAIL PARTY PROBLEM: n microphones record n speakers simultaneously.
  Each microphone picks up a mix of all voices. ICA can recover the individual
  voices without knowing the mixing matrix A.

  IDENTIFIABILITY: ICA is identifiable up to sign, scale, and permutation of
  components — but not if more than one source is Gaussian.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Model: x = As,  s ∈ ℝᵈ independent non-Gaussian sources"))
    print(formula("  Goal: find W = A⁻¹  such that ŝ = Wx are independent"))
    print(formula("  FastICA update: w ← E[x·g(wᵀx)] - E[g'(wᵀx)]·w,  then normalise"))
    print(formula("  Negentropy ≈ non-Gaussianity: J(y) ∝ [E[g(y)] - E[g(v)]]²  v~N(0,1)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Cocktail Party")
    rng = np.random.default_rng(17)
    n = 2000
    t = np.linspace(0, 8 * np.pi, n)
    s1 = np.sin(t)
    s2 = np.sign(np.sin(2.0 * t))          # square wave
    S = np.column_stack([s1, s2])
    S = S - S.mean(axis=0)
    S = S / S.std(axis=0)

    A = np.array([[1.0, 0.7], [0.5, 1.0]])  # mixing
    X = S @ A.T

    # FastICA implementation
    def fastica(X, max_iter=300, tol=1e-6):
        X = X - X.mean(axis=0)
        std = X.std()
        X = X / std
        n, d = X.shape
        W = np.zeros((d, d))
        for i in range(d):
            w = rng.normal(0, 1, d)
            w /= np.linalg.norm(w)
            for _ in range(max_iter):
                u = X @ w
                g    = np.tanh(u)
                g_p  = 1 - np.tanh(u)**2
                w_new = X.T @ g / n - g_p.mean() * w
                # Deflation orthogonalisation
                for j in range(i):
                    w_new -= (w_new @ W[j]) * W[j]
                w_new /= np.linalg.norm(w_new)
                if abs(abs(w_new @ w) - 1.0) < tol:
                    w = w_new; break
                w = w_new
            W[i] = w
        return (X @ W.T) * std, W

    S_hat, W_hat = fastica(X)
    # Correlation (modulo sign) with true sources
    corr1 = max(abs(np.corrcoef(S_hat[:, 0], s1)[0, 1]),
                abs(np.corrcoef(S_hat[:, 0], s2)[0, 1]))
    corr2 = max(abs(np.corrcoef(S_hat[:, 1], s1)[0, 1]),
                abs(np.corrcoef(S_hat[:, 1], s2)[0, 1]))

    print(f"\n  {bold_cyan('Cocktail party: 2 sources mixed, FastICA recovery:')}\n")
    print(f"  Source 1 (sine):         max |corr| with recovered = {green(f'{corr1:.4f}')}")
    print(f"  Source 2 (square wave):  max |corr| with recovered = {green(f'{corr2:.4f}')}")
    print(f"\n  {hint('Correlation near 1.0 means sources recovered (up to sign/permutation).')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("ICA for cocktail party problem", """
import numpy as np
from sklearn.decomposition import FastICA

rng = np.random.default_rng(0)
n = 2000
t = np.linspace(0, 8*np.pi, n)
s1 = np.sin(t)
s2 = np.sign(np.sin(2*t))           # square wave
s3 = rng.laplace(0, 1, n)           # Laplace noise (super-Gaussian)
S = np.column_stack([s1, s2, s3])

A = rng.normal(0, 1, (3, 3))
X = S @ A.T   # mix

ica = FastICA(n_components=3, random_state=0, max_iter=500)
S_hat = ica.fit_transform(X)

# Check recovery quality
for i in range(3):
    best_corr = max(abs(np.corrcoef(S_hat[:,i], S[:,j])[0,1]) for j in range(3))
    print(f"Recovered component {i}: best |corr|={best_corr:.4f}")

print(f"Mixing matrix A (true):\\n{A}")
print(f"Unmixing matrix W (ica.components_):\\n{ica.components_}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "ICA maximises non-Gaussianity (negentropy / kurtosis) to find sources",
        "Central Limit Theorem: mixed signals are more Gaussian than originals",
        "FastICA: Newton step maximising negentropy, deflation removes found components",
        "ICA ⊃ PCA: independence is stronger than uncorrelation (all moments vs 2nd)",
        "Cannot recover Gaussian sources — ICA is identifiable only for non-Gaussian",
        "Whitening X first (PCA) makes W an orthogonal matrix — simplifies search",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("K-Means Clustering",            topic_kmeans),
        ("Gaussian Mixture Model (GMM)",   topic_gmm),
        ("Hierarchical Clustering",        topic_hierarchical),
        ("DBSCAN",                         topic_dbscan),
        ("PCA (Unsupervised View)",         topic_pca_unsupervised),
        ("t-SNE",                          topic_tsne),
        ("Autoencoders",                   topic_autoencoders),
        ("ICA",                            topic_ica),
    ]
    block_menu("b12", "Unsupervised Learning", topics)
