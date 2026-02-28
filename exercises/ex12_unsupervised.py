"""Exercise Set 12: Unsupervised Learning"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title; self.difficulty = difficulty
        self.description = description; self.hint = hint
        self.starter_code = starter_code; self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="K-Means from Scratch",
        difficulty="Beginner",
        description="""
  Implement K-Means clustering from scratch.

  Algorithm:
    1. Initialize K centroids randomly from data points
    2. Assign each point to nearest centroid (Euclidean distance)
    3. Update centroids as mean of assigned points
    4. Repeat 2-3 until convergence

  Test on 3-cluster Gaussian mixture, report:
    - Inertia (within-cluster sum of squared distances)
    - Number of iterations to converge
""",
        hint="""
  distances = np.sqrt(((X[:,np.newaxis] - centroids)**2).sum(axis=2))  # (n,K)
  labels = distances.argmin(axis=1)
  new_centroids = np.array([X[labels==k].mean(axis=0) for k in range(K)])
  convergence: np.allclose(old_centroids, new_centroids)
""",
        starter_code="""
import numpy as np
np.random.seed(42)

def kmeans(X, K, max_iter=100):
    # TODO
    pass

X1 = np.random.randn(100,2) + [0,0]
X2 = np.random.randn(100,2) + [5,0]
X3 = np.random.randn(100,2) + [2.5,4]
X = np.vstack([X1,X2,X3])

labels, centroids, inertia, n_iter = kmeans(X, K=3)
print(f"Inertia: {inertia:.2f}, Iterations: {n_iter}")
""",
        solution_code="""
import numpy as np
np.random.seed(42)

def kmeans(X, K, max_iter=100):
    n = len(X)
    idx = np.random.choice(n, K, replace=False)
    centroids = X[idx].copy()
    for it in range(max_iter):
        dists = np.sqrt(((X[:,np.newaxis] - centroids)**2).sum(axis=2))
        labels = dists.argmin(axis=1)
        new_c = np.array([X[labels==k].mean(axis=0) if (labels==k).any()
                          else centroids[k] for k in range(K)])
        if np.allclose(centroids, new_c, atol=1e-6):
            centroids = new_c
            break
        centroids = new_c
    inertia = sum(((X[labels==k]-centroids[k])**2).sum() for k in range(K))
    return labels, centroids, inertia, it+1

X1 = np.random.randn(100,2)+[0,0]; X2 = np.random.randn(100,2)+[5,0]; X3 = np.random.randn(100,2)+[2.5,4]
X = np.vstack([X1,X2,X3])
labels, centroids, inertia, n_iter = kmeans(X, K=3)
print(f"Inertia: {inertia:.2f}, Iters: {n_iter}")
print(f"Centroids:\\n{centroids.round(3)}")
# True centers approx [0,0],[5,0],[2.5,4]
from sklearn.cluster import KMeans
sk = KMeans(n_clusters=3,random_state=0).fit(X)
print(f"sklearn inertia: {sk.inertia_:.2f}")
"""
    ),

    Exercise(
        title="EM for Gaussian Mixture Model",
        difficulty="Intermediate",
        description="""
  Implement the EM algorithm for a 2-component 1D Gaussian Mixture:
    p(x) = π₁ N(x; μ₁, σ₁²) + π₂ N(x; μ₂, σ₂²)

  E-step: compute responsibilities rᵢₖ = P(z=k|xᵢ)
  M-step: update πₖ, μₖ, σₖ from responsibilities

  Run until log-likelihood change < 1e-6.
  Report: final params, log-likelihood, iterations.
""",
        hint="""
  E-step: rk = pik * Nk(xi) / sum_k pik * Nk(xi)
  M-step:
    Nk = sum(rk)
    mu_k  = sum(rk * x) / Nk
    sig_k = sqrt(sum(rk * (x-mu_k)^2) / Nk)
    pi_k  = Nk / n
""",
        starter_code="""
import numpy as np
from scipy.stats import norm
np.random.seed(42)

x = np.concatenate([np.random.randn(200)*0.8+0, np.random.randn(150)*1.2+5])

def em_gmm(x, K=2, n_iter=200, tol=1e-6):
    # Initialize
    mu  = np.array([-1.0, 6.0])
    sig = np.array([1.0,  1.0])
    pi  = np.array([0.5,  0.5])
    # TODO: EM loop
    pass

mu, sig, pi, ll, n_it = em_gmm(x)
print(f"mu={mu.round(3)}, sig={sig.round(3)}, pi={pi.round(3)}")
print(f"Log-likelihood={ll:.4f}, iterations={n_it}")
""",
        solution_code="""
import numpy as np
from scipy.stats import norm
np.random.seed(42)

x = np.concatenate([np.random.randn(200)*0.8+0, np.random.randn(150)*1.2+5])
n = len(x)

def em_gmm(x, K=2, n_iter=200, tol=1e-6):
    mu  = np.array([-1.0, 6.0])
    sig = np.array([1.0,  1.0])
    pi  = np.array([0.5,  0.5])
    prev_ll = -np.inf
    for it in range(n_iter):
        # E-step
        r = np.column_stack([pi[k]*norm.pdf(x,mu[k],sig[k]) for k in range(K)])
        ll_arr = np.log(r.sum(axis=1)+1e-300)
        ll = ll_arr.sum()
        r /= r.sum(axis=1, keepdims=True)
        # M-step
        Nk = r.sum(axis=0)
        mu  = (r * x[:,np.newaxis]).sum(axis=0) / Nk
        sig = np.sqrt((r * (x[:,np.newaxis]-mu)**2).sum(axis=0) / Nk)
        pi  = Nk / n
        if abs(ll - prev_ll) < tol:
            return mu, sig, pi, ll, it+1
        prev_ll = ll
    return mu, sig, pi, ll, n_iter

mu, sig, pi, ll, n_it = em_gmm(x)
print(f"mu={mu.round(3)}  (true: 0, 5)")
print(f"sig={sig.round(3)} (true: 0.8, 1.2)")
print(f"pi={pi.round(3)}  (true: 0.57, 0.43)")
print(f"Log-likelihood: {ll:.2f}, iters: {n_it}")
"""
    ),

    Exercise(
        title="Implement DBSCAN from Scratch",
        difficulty="Advanced",
        description="""
  Implement DBSCAN density-based clustering.

  Parameters: ε (neighborhood radius), min_samples (core point threshold)
  Algorithm:
    1. For each unvisited point:
       a. Find all ε-neighbors
       b. If |neighbors| >= min_samples → core point → start new cluster
       c. Expand cluster: add neighbors, recursively expand core points
       d. Otherwise → noise point (label = -1)

  Test on make_moons and compare to sklearn DBSCAN.
""",
        hint="""
  def get_neighbors(X, idx, eps):
      dists = np.sqrt(((X - X[idx])**2).sum(axis=1))
      return np.where(dists <= eps)[0]
  
  def expand_cluster(X, labels, idx, neighbors, cluster_id, eps, min_samples):
      labels[idx] = cluster_id
      i = 0
      while i < len(neighbors):
          p = neighbors[i]
          if labels[p] == -2:  # unvisited
              labels[p] = cluster_id
              p_neighbors = get_neighbors(X, p, eps)
              if len(p_neighbors) >= min_samples:
                  neighbors = np.unique(np.concatenate([neighbors, p_neighbors]))
          elif labels[p] == -1:  # noise → border point
              labels[p] = cluster_id
          i += 1
""",
        starter_code="""
import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    # labels: -2=unvisited, -1=noise, >=0 = cluster id
    # TODO: implement DBSCAN
    pass

from sklearn.datasets import make_moons
X, y_true = make_moons(n_samples=200, noise=0.05, random_state=0)
labels = dbscan(X, eps=0.3, min_samples=5)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels==-1).sum()
print(f"Clusters: {n_clusters}, Noise: {n_noise}")
""",
        solution_code="""
import numpy as np

def get_neighbors(X, idx, eps):
    dists = np.sqrt(((X - X[idx])**2).sum(axis=1))
    return np.where(dists <= eps)[0]

def dbscan(X, eps=0.5, min_samples=5):
    n = len(X)
    labels = np.full(n, -2)  # -2 = unvisited
    cluster_id = 0
    for i in range(n):
        if labels[i] != -2: continue
        nbrs = get_neighbors(X, i, eps)
        if len(nbrs) < min_samples:
            labels[i] = -1  # noise
            continue
        labels[i] = cluster_id
        nbrs = list(nbrs)
        j = 0
        while j < len(nbrs):
            p = nbrs[j]
            if labels[p] == -2:
                labels[p] = cluster_id
                p_nbrs = get_neighbors(X, p, eps)
                if len(p_nbrs) >= min_samples:
                    nbrs += [x for x in p_nbrs if x not in set(nbrs)]
            elif labels[p] == -1:
                labels[p] = cluster_id
            j += 1
        cluster_id += 1
    return labels

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN as SkDBSCAN
X, y_true = make_moons(n_samples=200, noise=0.05, random_state=0)

my_labels = dbscan(X, eps=0.3, min_samples=5)
sk_labels = SkDBSCAN(eps=0.3, min_samples=5).fit_predict(X)

print(f"My DBSCAN: {len(set(my_labels))-(1 if -1 in my_labels else 0)} clusters, {(my_labels==-1).sum()} noise")
print(f"sklearn:   {len(set(sk_labels))-(1 if -1 in sk_labels else 0)} clusters, {(sk_labels==-1).sum()} noise")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║     EXERCISES — Block 12: Unsupervised Learning ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m\n")
        for i, ex in enumerate(exercises, 1):
            dc = "\033[92m" if ex.difficulty=="Beginner" else "\033[93m" if ex.difficulty=="Intermediate" else "\033[91m"
            print(f"  {i}. {dc}[{ex.difficulty}]\033[0m {ex.title}")
        print("\n  \033[90m[0] Back\033[0m")
        choice = input("\n\033[96mSelect: \033[0m").strip()
        if choice == "0": break
        try:
            ex = exercises[int(choice)-1]; _run_exercise(ex)
        except (ValueError, IndexError): pass


def _run_exercise(ex):
    dc = "\033[92m" if ex.difficulty=="Beginner" else "\033[93m" if ex.difficulty=="Intermediate" else "\033[91m"
    print(f"\n\033[95m━━━ {ex.title} ━━━\033[0m")
    print(f"  {dc}{ex.difficulty}\033[0m\n{ex.description}")
    while True:
        cmd = input("\n  [h]int [c]ode [r]un [s]olution [b]ack: ").strip().lower()
        if cmd=='b': break
        elif cmd=='h': print(f"\033[93mHINT\033[0m\n{ex.hint}")
        elif cmd=='c': print(f"\033[94mSTARTER\033[0m\n{ex.starter_code}")
        elif cmd=='s': print(f"\033[92mSOLUTION\033[0m\n{ex.solution_code}")
        elif cmd=='r':
            try: exec(compile(ex.solution_code,"<sol>","exec"),{})
            except Exception as e: print(f"\033[91m{e}\033[0m")
