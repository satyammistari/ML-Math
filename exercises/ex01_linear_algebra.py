"""
Exercise Set 01: Linear Algebra
Difficulty levels: Beginner / Intermediate / Advanced
"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title
        self.difficulty = difficulty
        self.description = description
        self.hint = hint
        self.starter_code = starter_code
        self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="Compute Dot Product from Scratch",
        difficulty="Beginner",
        description="""
  Implement the dot product of two vectors WITHOUT using np.dot or @.
  Use only numpy element-wise operations and sum().

  Input:  a = [1, 2, 3],  b = [4, 5, 6]
  Output: 32   (= 1*4 + 2*5 + 3*6)

  Also compute the angle θ between the vectors using:
    cos(θ) = a·b / (||a|| · ||b||)
""",
        hint="""
  Step 1: Element-wise multiply: a * b → [4, 10, 18]
  Step 2: Sum all elements: 4+10+18 = 32
  Step 3: Norm: ||a|| = sqrt(1²+2²+3²) = sqrt(14)
  Step 4: θ = arccos(dot / (||a||·||b||))
""",
        starter_code="""
import numpy as np

def dot_product(a, b):
    # TODO: implement without np.dot or @
    # Hint: use element-wise multiply and sum
    pass

def angle_between(a, b):
    # TODO: compute angle in degrees
    pass

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print(f"Dot product: {dot_product(a, b)}")
print(f"Angle: {angle_between(a, b):.2f} degrees")
""",
        solution_code="""
import numpy as np

def dot_product(a, b):
    return np.sum(a * b)

def angle_between(a, b):
    cos_theta = dot_product(a, b) / (np.sqrt(dot_product(a, a)) * np.sqrt(dot_product(b, b)))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print(f"Dot product: {dot_product(a, b)}")        # 32.0
print(f"Verify: {np.dot(a, b)}")                  # 32.0
print(f"Angle: {angle_between(a, b):.4f}°")       # 12.9332°
""",
        test_cases=[
            {"input": ([1,2,3], [4,5,6]), "expected": 32},
            {"input": ([1,0], [0,1]),     "expected": 0},
        ]
    ),

    Exercise(
        title="Implement PCA from Scratch",
        difficulty="Intermediate",
        description="""
  Implement PCA (Principal Component Analysis) from scratch using numpy.
  Do NOT use sklearn.decomposition.PCA.

  Steps:
    1. Center the data (subtract mean)
    2. Compute covariance matrix C = X.T @ X / (n-1)
    3. Compute eigenvectors of C
    4. Sort by descending eigenvalue
    5. Project data onto top-k eigenvectors

  Test: use a 4D dataset, reduce to 2D.
  Verify: np.dot(components[0], components[1]) ≈ 0 (orthogonal)
""",
        hint="""
  C = np.cov(X.T)   # equivalent to X_centered.T @ X_centered / (n-1)
  eigenvalues, eigenvectors = np.linalg.eigh(C)
  # Sort descending: eigenvalues[::-1], eigenvectors[:, ::-1]
  # Project: Z = X_centered @ V_k  where V_k = top-k eigenvectors
""",
        starter_code="""
import numpy as np

def pca(X, k):
    \"\"\"
    X: (n, d) data matrix
    k: number of components to keep
    Returns: Z (n, k) projected data, components (k, d), explained_var_ratio
    \"\"\"
    # 1. Center
    X_c = None  # TODO: subtract mean

    # 2. Covariance
    C = None  # TODO: covariance matrix

    # 3. Eigendecomposition
    # eigenvalues, eigenvectors = ?

    # 4. Sort descending
    # idx = np.argsort(eigenvalues)[::-1]

    # 5. Project
    # Z = X_c @ V_k

    pass

np.random.seed(42)
X = np.random.randn(100, 4)
X[:, 0] += 2 * X[:, 1]  # add correlation
Z, comps, var_ratio = pca(X, k=2)
print(f"Z shape: {Z.shape}")
print(f"Explained variance: {var_ratio}")
""",
        solution_code="""
import numpy as np

def pca(X, k):
    n, d = X.shape
    X_c = X - X.mean(axis=0)
    C = X_c.T @ X_c / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    V_k = eigenvectors[:, :k]
    Z = X_c @ V_k
    explained = eigenvalues[:k] / eigenvalues.sum()
    return Z, V_k.T, explained

np.random.seed(42)
X = np.random.randn(100, 4)
X[:, 0] += 2 * X[:, 1]
Z, comps, var_ratio = pca(X, k=2)
print(f"Z shape: {Z.shape}")
print(f"Explained variance: {var_ratio.round(4)}")
print(f"Components orthogonal: {abs(comps[0] @ comps[1]) < 1e-10}")
# Verify vs sklearn
from sklearn.decomposition import PCA
pca_sk = PCA(n_components=2).fit_transform(X)
print(f"sklearn var ratio: {PCA(n_components=2).fit(X).explained_variance_ratio_.round(4)}")
"""
    ),

    Exercise(
        title="SVD Image Compression",
        difficulty="Advanced",
        description="""
  Use SVD to compress a matrix (simulated image) and measure:
    1. Compression ratio at each rank k
    2. Frobenius error: ||A - A_k||_F / ||A||_F
    3. Find minimum k such that error < 5%

  A_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]

  Visualize how error decreases with rank k.
  Bonus: compute the optimal truncation point using the Eckart-Young theorem.
""",
        hint="""
  U, S, Vt = np.linalg.svd(A, full_matrices=False)
  A_k = U[:,:k] * S[:k] @ Vt[:k,:]  # broadcasting S as diagonal
  error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
  Eckart-Young: A_k is the BEST rank-k approximation (minimizes Frobenius error)
""",
        starter_code="""
import numpy as np

np.random.seed(77)
# Simulate a 50x40 "image" with low-rank structure
U_true = np.random.randn(50, 5)
V_true = np.random.randn(5, 40)
A = U_true @ V_true + 0.1 * np.random.randn(50, 40)

# TODO: Compute SVD
# U, S, Vt = ?

# TODO: For each k from 1 to 20:
#   Compute A_k
#   Compute relative Frobenius error
#   Compute compression ratio = k*(m+n+1) / (m*n)

# TODO: Find minimum k for <5% error
# TODO: Print table and plot
""",
        solution_code="""
import numpy as np

np.random.seed(77)
U_true = np.random.randn(50, 5)
V_true = np.random.randn(5, 40)
A = U_true @ V_true + 0.1 * np.random.randn(50, 40)
m, n = A.shape
norm_A = np.linalg.norm(A, 'fro')

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"A shape: {A.shape}, rank ≤ {min(m,n)}")
print(f"Singular values: {S[:8].round(2)}")
print(f\"\\n{'k':>4} {'Error%':>8} {'Compress%':>10}\")
print("-" * 26)
for k in range(1, 21):
    A_k = U[:, :k] * S[:k] @ Vt[:k, :]
    err = np.linalg.norm(A - A_k, 'fro') / norm_A * 100
    cr  = k*(m+n+1) / (m*n) * 100  # compression ratio
    marker = " ← <5% error" if err < 5 else ""
    print(f"{k:>4} {err:>8.2f}% {cr:>10.2f}%{marker}")

# Verify Eckart-Young: A_k minimizes ||A - B||_F over all rank-k B
# Best rank-5 uses only top 5 singular values
A_5 = U[:,:5] * S[:5] @ Vt[:5,:]
err_5 = np.linalg.norm(A - A_5, 'fro') / norm_A
print(f\"\\nRank-5 relative error: {err_5:.4f}\")
"""
    ),
]


def run():
    """Interactive exercise runner for Linear Algebra."""
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║     EXERCISES — Block 01: Linear Algebra         ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m\n")
        for i, ex in enumerate(exercises, 1):
            diff_color = "\033[92m" if ex.difficulty == "Beginner" else \
                         "\033[93m" if ex.difficulty == "Intermediate" else "\033[91m"
            print(f"  {i}. {diff_color}[{ex.difficulty}]\033[0m {ex.title}")
        print("\n  \033[90m[0] Back\033[0m")
        choice = input("\n\033[96mSelect exercise: \033[0m").strip()
        if choice == "0":
            break
        try:
            ex = exercises[int(choice) - 1]
            _run_exercise(ex)
        except (ValueError, IndexError):
            pass


def _run_exercise(ex):
    diff_color = "\033[92m" if ex.difficulty == "Beginner" else \
                 "\033[93m" if ex.difficulty == "Intermediate" else "\033[91m"
    print(f"\n\033[95m━━━ {ex.title} ━━━\033[0m")
    print(f"  Difficulty: {diff_color}{ex.difficulty}\033[0m\n")
    print("\033[1mPROBLEM\033[0m")
    print(ex.description)
    while True:
        cmd = input("\n  [h]int  [c]ode  [r]un solution  [s]olution  [b]ack: ").strip().lower()
        if cmd == 'b':
            break
        elif cmd == 'h':
            print(f"\n\033[93mHINT\033[0m\n{ex.hint}")
        elif cmd == 'c':
            print(f"\n\033[94mSTARTER CODE\033[0m\n{ex.starter_code}")
        elif cmd == 's':
            print(f"\n\033[92mSOLUTION\033[0m\n{ex.solution_code}")
        elif cmd == 'r':
            print("\n\033[92mRunning solution...\033[0m")
            try:
                exec(compile(ex.solution_code, "<solution>", "exec"), {})
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")
