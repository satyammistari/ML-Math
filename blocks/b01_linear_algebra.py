"""
blocks/b01_linear_algebra.py
Block 1: Linear Algebra Fundamentals
Topics: Vector spaces, dot products, matrix multiplication, eigenvalues,
        SVD, PCA, norms, Gram-Schmidt.
"""

import numpy as np
import sys
import os

from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold,
                        bold_yellow, bold_magenta, white, hint, red,
                        formula, section, emph, value)
from ui.widgets import (box, section_header, breadcrumb, nav_bar, table,
                         bar_chart, code_block, panel, pager, hr,
                         print_sparkline)
from ui.menu    import topic_nav, clear, block_menu, mark_completed
from viz.ascii_plots import scatter, neural_net_diagram
from viz.terminal_plots import (eigenvalue_spectrum, scree_plot,
                                  distribution_plot)
from viz.matplotlib_plots import (pca_scatter, scatter_classes)
from core.linalg import (dot_product, cosine_similarity, projection,
                          vector_norm, all_norms, eigen_decomp,
                          svd_full, svd_truncated, low_rank_approx,
                          pca, gram_schmidt, is_orthonormal,
                          explained_variance_ratio, matmul_steps,
                          condition_number)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: pause and wait for a key
# ══════════════════════════════════════════════════════════════════════════════
def _pause(msg: str = "  [Enter] to continue..."):
    input(grey(msg))


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — Vector spaces and subspaces
# ══════════════════════════════════════════════════════════════════════════════
def topic_vector_spaces():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Vector Spaces & Subspaces")
    section_header("VECTOR SPACES AND SUBSPACES")
    print()

    # ── THEORY ──────────────────────────────────────────────────────────────
    section_header("1. THEORY")
    print(white("""
  A vector space V over a field F (usually ℝ) is a set of objects — "vectors" —
  along with two operations (addition and scalar multiplication) that satisfy
  eight axioms (commutativity, associativity, distributivity, identity, inverse).
  The most familiar example is ℝⁿ: the set of all n-tuples of real numbers.

  A BASIS of V is a set of linearly independent vectors that spans V.  Any vector
  in V can be written as a unique linear combination of the basis vectors.  The
  number of basis vectors equals the DIMENSION of V.

  A SUBSPACE is a subset of V that is itself a vector space — it must contain the
  zero vector, be closed under addition, and be closed under scalar multiplication.
  Key subspaces of a matrix A: column space (range), row space, null space, left
  null space.  These four are linked by the Fundamental Theorem of Linear Algebra.

  In machine learning, we constantly work in high-dimensional vector spaces.  A
  neural network maps inputs in ℝⁿ through a series of linear subspaces to
  produce outputs.  PCA finds the most important subspace.  SVD reveals the
  intrinsic dimensionality.  Understanding vector spaces is the bedrock.
"""))
    _pause()

    # ── FORMULAS ─────────────────────────────────────────────────────────────
    section_header("2. KEY FORMULAS")
    print()
    print(bold_cyan("  Linear combination:"))
    print(formula("    v = α₁u₁ + α₂u₂ + ··· + αₖuₖ       where αᵢ ∈ ℝ"))
    print()
    print(bold_cyan("  Span:"))
    print(formula("    span{u₁,…,uₖ} = {v : v = Σᵢ αᵢuᵢ,  αᵢ ∈ ℝ}"))
    print()
    print(bold_cyan("  Linear independence:"))
    print(formula("    α₁u₁ + ··· + αₖuₖ = 0  ⟹  α₁=α₂=···=αₖ=0"))
    print()
    print(bold_cyan("  Subspace conditions (all three must hold):"))
    print(formula("    (i)  0 ∈ W"))
    print(formula("    (ii) u, v ∈ W ⟹ u+v ∈ W"))
    print(formula("    (iii) u ∈ W, α ∈ ℝ ⟹ αu ∈ W"))
    print()
    _pause()

    # ── STEP-BY-STEP DERIVATION ───────────────────────────────────────────────
    section_header("3. WORKED EXAMPLE — Two Different Bases of ℝ²")
    print()
    print(white("  Standard basis:"))
    e1 = np.array([1, 0]); e2 = np.array([0, 1])
    print(formula(f"    e₁ = {e1},  e₂ = {e2}"))
    print()
    print(white("  Alternative basis B = {[1,1], [1,-1]}:"))
    b1 = np.array([1, 1]); b2 = np.array([1, -1])
    print(formula(f"    b₁ = {b1},  b₂ = {b2}"))
    print()
    print(white("  Express v = [3, 1] in basis B:"))
    v  = np.array([3, 1])
    # v = α b1 + β b2  →  solve 2x2 system
    A  = np.column_stack([b1, b2])
    coeffs = np.linalg.solve(A, v)
    print(bold_cyan(f"  [1] Set up:  [b₁ | b₂] [α; β] = v"))
    print(bold_cyan(f"  [2] Matrix:  {A}"))
    print(bold_cyan(f"  [3] Solve:   α = {coeffs[0]:.2f},  β = {coeffs[1]:.2f}"))
    print(green(f"  [4] Verify:  {coeffs[0]:.0f}·{b1} + {coeffs[1]:.0f}·{b2} = {(coeffs[0]*b1+coeffs[1]*b2).astype(int)}  ✓"))
    print()
    _pause()

    # ── ASCII VISUALISATION ───────────────────────────────────────────────────
    section_header("4. ASCII VISUALIZATION — Basis Vectors in ℝ²")
    print()
    # Draw a simple ASCII grid showing e1, e2 and b1, b2
    print(cyan("    y"))
    print(cyan("    │"))
    print(cyan("  2 │") + yellow(" b₁=(1,1)"))
    print(cyan("    │") + yellow("  ↗"))
    print(cyan("  1 │") + yellow(" ↗ ") + green("e₂=(0,1)"))
    print(cyan("    │") + green("  ↑"))
    print(cyan("  0 ┼──────────────► x"))
    print(cyan("    0    1    2    "))
    print(cyan("         ↑"))
    print(green("         e₁=(1,0)"))
    print(cyan("                   ") + yellow("b₂=(1,-1) ↘"))
    print()

    # ── PLOTEXT VISUALISATION ─────────────────────────────────────────────────
    try:
        pts = np.array([[0,0],[1,0],[0,0],[0,1],[0,0],[1,1],[0,0],[1,-1]])
        distribution_plot(pts[:,0], pts[:,1],
                          title="Basis vectors (approximate dot chart)")
    except Exception:
        pass
    print()

    # ── CODE SNIPPET ─────────────────────────────────────────────────────────
    section_header("5. PYTHON CODE")
    code_block("Vector Space — Basis Change", """
import numpy as np

# Standard basis
e1, e2 = np.array([1, 0]), np.array([0, 1])

# Alternative basis B
b1, b2 = np.array([1, 1]), np.array([1, -1])

# Target vector
v = np.array([3, 1])

# Express v in basis B: solve [b1 | b2] @ coeffs = v
A      = np.column_stack([b1, b2])
coeffs = np.linalg.solve(A, v)
print(f"v = {coeffs[0]:.2f}·b1 + {coeffs[1]:.2f}·b2")

# Verify independence: det ≠ 0
print(f"det(B) = {np.linalg.det(A):.2f}  (≠ 0 → independent)")
""")
    _pause()

    # ── KEY INSIGHTS ─────────────────────────────────────────────────────────
    section_header("6. KEY INSIGHTS")
    print()
    insights = [
        "A basis is not unique — infinitely many valid bases exist for any space",
        "The dimension of a space is an intrinsic property, independent of basis",
        "Column space of A = all vectors reachable by Ax (the 'reach' of A)",
        "Null space of A = all inputs x where Ax = 0 (the 'blindspot' of A)",
        "rank(A) + nullity(A) = n  (Rank-Nullity Theorem)",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Dot product and projections
# ══════════════════════════════════════════════════════════════════════════════
def topic_dot_product():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Dot Product & Projections")
    section_header("DOT PRODUCT AND PROJECTIONS")
    print()

    section_header("1. THEORY")
    print(white("""
  The DOT PRODUCT (inner product) between two vectors a, b ∈ ℝⁿ is defined
  algebraically as:  a·b = Σᵢ aᵢbᵢ  and geometrically as: a·b = ‖a‖‖b‖cos θ,
  where θ is the angle between them.

  When a·b = 0, the vectors are ORTHOGONAL — they point in completely unrelated
  directions.  This is the geometric basis for many ML concepts: orthogonal
  weights don't interfere; principal components are orthogonal directions of
  variance; attention scores use dot products to measure relevance.

  PROJECTION of vector a onto vector b answers: "how much of a points in the
  direction of b?"  The projection is a vector in b's direction with length
  (a·b/‖b‖).  This is the foundation of least-squares regression, which
  projects the target y onto the column space of X.

  In deep learning, every linear layer computes dot products: output = W @ x + b.
  In attention: scores = Q @ Kᵀ / √d uses dot products to measure query-key
  similarity.  Cosine similarity = dot product of normalised vectors.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print()
    print(formula("  Algebraic:   a·b = Σᵢ aᵢbᵢ = aᵀb"))
    print(formula("  Geometric:   a·b = ‖a‖ · ‖b‖ · cos θ"))
    print(formula("  Cosine sim:  cos θ = a·b / (‖a‖ · ‖b‖)"))
    print(formula("  Projection:  proj_b(a) = (a·b / ‖b‖²) · b"))
    print(formula("  Proj length: scalar_proj = a·b / ‖b‖  =  ‖a‖ cos θ"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Project [3,4] onto [1,0]")
    a = np.array([3.0, 4.0]); b = np.array([1.0, 0.0])
    dp = dot_product(a, b)
    proj = projection(a, b)
    angle = np.degrees(np.arccos(cosine_similarity(a, b)))
    print()
    print(bold_cyan(f"  [1] a = {a},  b = {b}"))
    print(bold_cyan(f"  [2] a·b = {a[0]:.0f}×{b[0]:.0f} + {a[1]:.0f}×{b[1]:.0f} = {dp:.1f}"))
    print(bold_cyan(f"  [3] ‖b‖² = {np.dot(b,b):.1f}"))
    print(bold_cyan(f"  [4] proj_b(a) = ({dp:.1f}/{np.dot(b,b):.1f}) · {b} = {proj}"))
    print(bold_cyan(f"  [5] Angle θ = {angle:.2f}°  (cos⁻¹({cosine_similarity(a,b):.4f}))"))
    print(green(f"\n  Geometric meaning: projecting [3,4] onto the x-axis gives [3,0]"))
    print(green("  — the 'shadow' of a when light shines perpendicular to b."))
    print()

    # ASCII visualisation
    section_header("4. ASCII VISUALIZATION")
    print()
    print(cyan("       y"))
    print(cyan("     4 │") + yellow("  * a=(3,4)"))
    print(cyan("       │") + yellow(" /"))
    print(cyan("     2 │") + yellow("/  "))
    print(cyan("       │") + yellow("/     "))
    print(cyan("     0 ┼──────────" + "────►") + cyan(" x"))
    print(cyan("       0    3    "))
    print(yellow("            ↑"))
    print(yellow("          proj=(3,0)"))
    print(grey("  Note: proj is the foot of the perpendicular from a to the x-axis"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Dot Product & Projection", """
import numpy as np

a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])

# Algebraic dot product
dp = np.dot(a, b)                         # = 3.0

# Geometric interpretation  
norm_a = np.linalg.norm(a)                # = 5.0
norm_b = np.linalg.norm(b)                # = 1.0
cos_theta = dp / (norm_a * norm_b)        # = 0.6
theta_deg = np.degrees(np.arccos(cos_theta))  # ≈ 53.1°

# Projection of a onto b
proj = (np.dot(a, b) / np.dot(b, b)) * b  # = [3, 0]

# Cosine similarity (for unit vectors → pure angle measurement)
a_hat = a / norm_a                         # = [0.6, 0.8]
b_hat = b / norm_b                         # = [1.0, 0.0]
cos_sim = np.dot(a_hat, b_hat)            # = 0.6

print(f"dot(a,b)       = {dp}")
print(f"angle θ        = {theta_deg:.1f}°")
print(f"projection     = {proj}")
print(f"cosine_sim     = {cos_sim:.4f}")
""")
    print()

    section_header("6. KEY INSIGHTS")
    insights = [
        "cos θ = 1 → vectors point same direction;  -1 → opposite;  0 → orthogonal",
        "Projection is the ML least-squares idea: minimize distance from data to subspace",
        "Attention = softmax(QKᵀ/√d): each row is a dot product of query with all keys",
        "Dot product of centred data = covariance — foundation of PCA",
        "High-dimensional dot products concentrate near 0 (blessing/curse of dimensionality)",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Matrix multiplication
# ══════════════════════════════════════════════════════════════════════════════
def topic_matmul():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Matrix Multiplication")
    section_header("MATRIX MULTIPLICATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Matrix multiplication is NOT element-wise.  The entry (AB)ᵢⱼ is the dot
  product of the i-th ROW of A with the j-th COLUMN of B.

  This models function composition: if A represents transformation f and B
  represents transformation g, then AB represents "first apply g, then f"
  (note the reversed order).  Every forward pass in a neural network is a
  sequence of matrix multiplications.

  Dimensions must be compatible: A is m×k, B is k×n → AB is m×n.  The "inner"
  dimensions k must match.  The result has the "outer" dimensions m×n.

  Four equivalent interpretations of AB = C:
  (a) Each entry Cᵢⱼ = dot(rowᵢ(A), colⱼ(B))
  (b) Each column of C = A times a column of B
  (c) Each row of C = a row of A times B
  (d) C = sum of rank-1 outer products: Σₖ colₖ(A) ⊗ rowₖ(B)
  Interpretation (d) is used in transformer attention and low-rank adapters!
"""))
    _pause()

    section_header("2. KEY FORMULA")
    print()
    print(formula("  (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ"))
    print(formula("  dimensions: (m×k) @ (k×n) = (m×n)"))
    print(formula("  C = Σₖ aₖ bₖᵀ  (sum of k rank-1 outer products)"))
    print()
    _pause()

    section_header("3. STEP-BY-STEP: 2×3 times 3×2")
    print()
    A = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=float)
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=float)
    C, steps = matmul_steps(A, B)
    print(bold_cyan(f"  A =\n{A}\n"))
    print(bold_cyan(f"  B =\n{B}\n"))
    for step in steps:
        print(f"  {yellow(step)}")
    print()
    print(green(f"  Result C = A@B =\n{C}\n"))
    print(bold_cyan("  ASCII 'sliding window' view:"))
    print(grey("""
  C[0,0]: row [1,2,3] · col [7,9,11]  = 1·7+2·9+3·11 = 58
  C[0,1]: row [1,2,3] · col [8,10,12] = 1·8+2·10+3·12 = 64
   ...
  (row i slides across A, column j slides down B)
"""))
    _pause()

    # ASCII sliding window diagram
    section_header("4. ASCII VISUALIZATION — Sliding Window")
    print()
    print(cyan("   A (2×3)          B (3×2)          C[0,0]"))
    print(cyan("  ┌─────────┐       ┌───────┐         ┌───┐"))
    print(yellow("  │→ 1 2 3 →│") + cyan("   ×   ") + yellow("│↓ 7 │") + cyan("  =  ") + yellow("│58 │"))
    print(yellow("  │       │") + cyan("       ") + yellow("│↓ 9 │") + cyan("     ") + cyan("└───┘"))
    print(cyan("  │  4 5 6 │") + cyan("       ") + yellow("│↓11 │"))
    print(cyan("  └─────────┘       └────   ┘"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Matrix Multiplication", """
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Standard matrix multiplication
C = A @ B          # or np.matmul(A, B)
print("A @ B =")
print(C)           # [[58, 64], [139, 154]]

# Verify via rank-1 outer product sum
C_verify = sum(np.outer(A[:, k], B[k, :]) for k in range(3))
print("Outer product sum =")
print(C_verify.astype(int))

# Check shapes
print(f"A: {A.shape}, B: {B.shape} → C: {C.shape}")

# Element-wise (Hadamard) — different!
A_sq = A[:2, :2]
B_sq = B[:2, :2]
print("Element-wise A*B =", A_sq * B_sq)
print("Matrix A@B =",       A_sq @ B_sq)
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    insights = [
        "AB ≠ BA in general — matrix multiplication is NOT commutative",
        "Each forward pass in a neural net: h = ReLU(Wx + b) — W is the matrix",
        "Rank-1 decomposition C = Σₖ aₖbₖᵀ is used in LoRA: ΔW = AB (low-rank)",
        "Strassen algorithm: O(n^2.81) vs naive O(n³) — matters for large matrices",
        "GPU matrix multiplication (GEMM) is THE bottleneck operation in deep learning",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Eigenvalue decomposition
# ══════════════════════════════════════════════════════════════════════════════
def topic_eigenvalues():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Eigenvalue Decomposition")
    section_header("EIGENVALUE DECOMPOSITION")
    print()

    section_header("1. THEORY")
    print(white("""
  An EIGENVECTOR of a square matrix A is a non-zero vector v that, when
  transformed by A, only gets SCALED — it does not rotate.  The scaling factor λ
  is called the EIGENVALUE:  Av = λv.

  Think of a transformation as stretching and rotating a shape.  Eigenvectors
  are the special directions that only stretch (or shrink), not rotate.  For a
  covariance matrix, eigenvectors are the principal directions of spread.

  The eigendecomposition A = QΛQᵀ (for symmetric A) decomposes A into
  rotations (Q) and scalings (Λ).  This is exactly what PCA does: it finds the
  eigenvectors of the covariance matrix — i.e., the directions of maximum
  variance.

  In deep learning: the Hessian's eigenvalues reveal the curvature of the loss
  landscape (large eigenvalue = sharp direction = hard to optimise).  In graph
  neural networks, graph Laplacian eigenvectors define the spectral domain.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print()
    print(formula("  Eigenvalue equation:  Av = λv  ⟺  (A - λI)v = 0"))
    print(formula("  Characteristic eqn:   det(A - λI) = 0"))
    print(formula("  Eigendecomposition:   A = Q Λ Q⁻¹  (diagonalisation)"))
    print(formula("  Symmetric case:       A = Q Λ Qᵀ   (Q is orthogonal)"))
    print(formula("  Trace:  Σᵢ λᵢ = tr(A)    Determinant: Πᵢ λᵢ = det(A)"))
    print()
    _pause()

    section_header("3. FULL EXAMPLE: A = [[4,1],[2,3]]")
    print()
    A = np.array([[4.0, 1.0],
                  [2.0, 3.0]])
    vals, vecs = eigen_decomp(A)
    print(bold_cyan(f"  [1] A = {A}"))
    print()
    print(bold_cyan(f"  [2] Characteristic polynomial:  det(A-λI) = 0"))
    print(yellow(f"       = (4-λ)(3-λ) - (1)(2)"))
    print(yellow(f"       = λ² - 7λ + 10 = 0"))
    print(yellow(f"       = (λ-5)(λ-2) = 0"))
    print()
    print(bold_cyan(f"  [3] Eigenvalues:  λ₁ = {vals[0]:.4g},  λ₂ = {vals[1]:.4g}"))
    print()
    for i, (lam, vec) in enumerate(zip(vals, vecs.T), 1):
        Av   = A @ vec
        lv   = lam * vec
        ok   = np.allclose(Av, lv, atol=1e-8)
        print(bold_cyan(f"  [4] v{i} = {vec.round(4)}"))
        print(yellow(f"       A·v{i} = {Av.round(4)}"))
        print(yellow(f"       λ{i}·v{i} = {lv.round(4)}"))
        print(green(f"       Verify: A·v{i} = λ{i}·v{i}  → {ok}  ✓"))
        print()
    recon = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
    print(green(f"  [5] Reconstruction A = QΛQ⁻¹:\n{recon.round(6)}\n  ✓"))
    _pause()

    # ASCII visualisation — circle → ellipse
    section_header("4. ASCII VISUALIZATION — Eigenvectors Don't Rotate")
    print()
    print(cyan("  BEFORE (unit circle)     AFTER (A applied)"))
    print(cyan("                            — ellipse with eigenvector axes"))
    print()
    theta = np.linspace(0, 2*np.pi, 16)
    circle_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse_pts = (A @ circle_pts.T).T
    scatter(circle_pts[:,0]*20+30, circle_pts[:,1]*8+10,
            title="Unit circle (·) → ellipse (×)",
            width=50, height=18, char="·")
    _pause()

    # plotext
    section_header("5. PLOTEXT — Eigenvalue Spectrum")
    eigenvalue_spectrum(np.abs(vals).tolist(), title="Eigenvalue Magnitudes")
    print()

    # matplotlib
    section_header("6. MATPLOTLIB — Eigenvector Arrows")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        theta = np.linspace(0, 2*np.pi, 200)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        ellipse = (A @ circle.T).T
        axes[0].plot(circle[:,0], circle[:,1], "b-"); axes[0].set_title("Unit Circle")
        axes[0].set_aspect("equal")
        axes[1].plot(ellipse[:,0], ellipse[:,1], "r-"); axes[1].set_title("Transformed (Ellipse)")
        for i, (lam, vec) in enumerate(zip(vals, vecs.T)):
            axes[1].annotate("", xy=lam*vec, xytext=[0,0],
                             arrowprops=dict(arrowstyle="->", color=f"C{i}", lw=2))
            axes[1].text(*(lam*vec*1.1), f"λ={lam:.1f} v{i+1}", fontsize=9)
        axes[1].set_aspect("equal")
        plt.suptitle("Eigenvalue Decomposition: A transforms circle → ellipse\n"
                     "Eigenvectors = axes of ellipse")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed — skipping window plot"))
    print()

    section_header("7. PYTHON CODE")
    code_block("Eigenvalue Decomposition", """
import numpy as np

A = np.array([[4, 1],
              [2, 3]], dtype=float)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)          # [5, 2]
print("Eigenvectors (columns):")
print(eigenvectors)

# Verify: A @ v = λ @ v
for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ vec
    lv = lam * vec
    print(f"λ{i+1}={lam:.2f}, A·v=(Av≈λv): {np.allclose(Av, lv)}")

# Reconstruct A = Q Λ Q⁻¹
Q     = eigenvectors
Lam   = np.diag(eigenvalues)
Q_inv = np.linalg.inv(Q)
A_recon = Q @ Lam @ Q_inv
print("A reconstructed:", np.allclose(A, A_recon))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Eigenvectors of covariance matrix = principal components (directions of max variance)",
        "Symmetric matrices have REAL eigenvectors; orthogonal eigenvectors for distinct λ",
        "Power method: repeated A @ v / ‖A@v‖ converges to the dominant eigenvector",
        "Hessian eigenvalues tell you the curvature of loss: large → sharp, hard to cross",
        "PageRank is the dominant eigenvector of a web graph's transition matrix",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Singular Value Decomposition
# ══════════════════════════════════════════════════════════════════════════════
def topic_svd():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Singular Value Decomposition")
    section_header("SINGULAR VALUE DECOMPOSITION (SVD)")
    print()

    section_header("1. THEORY")
    print(white("""
  The Singular Value Decomposition is arguably the most important matrix
  factorization in all of machine learning.  Every matrix A (not just square,
  not just symmetric) can be written as A = UΣVᵀ where:
    • U (m×m): orthogonal matrix — "output directions" (left singular vectors)
    • Σ (m×n): diagonal matrix of non-negative SINGULAR VALUES σ₁≥σ₂≥···≥0
    • Vᵀ (n×n): orthogonal matrix — "input directions" (right singular vectors)

  The singular values σᵢ measure how much A stretches in each direction.  The
  singular vectors u_i, v_i form orthonormal bases for the range and domain.

  Key insight: A = Σᵢ σᵢ uᵢ vᵢᵀ — a sum of rank-1 matrices weighted by
  singular values.  Keeping only the top-k terms gives the BEST rank-k
  approximation of A (Eckart-Young theorem).  This is the mathematical
  foundation of compression, PCA, collaborative filtering, and NLP word
  vectors.

  Relationship to eigendecomposition: σᵢ = √λᵢ(AᵀA) and vᵢ are eigenvectors
  of AᵀA.  SVD works on ANY matrix; eigendecomposition requires squareness.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print()
    print(formula("  A = U Σ Vᵀ         (full SVD)"))
    print(formula("  A ≈ Uₖ Σₖ Vₖᵀ      (truncated rank-k approximation)"))
    print(formula("  AᵀA = V Σ² Vᵀ       (eigendecomp of AᵀA)"))
    print(formula("  AAᵀ = U Σ² Uᵀ       (eigendecomp of AAᵀ)"))
    print(formula("  σᵢ = √λᵢ(AᵀA)"))
    print(formula("  ‖A‖_F = √(Σᵢ σᵢ²)  (Frobenius norm)"))
    print(formula("  rank(A) = number of non-zero singular values"))
    print()
    _pause()

    section_header("3. FULL SVD EXAMPLE: 3×2 matrix")
    print()
    A = np.array([[3, 2],
                  [2, 3],
                  [1, 1]], dtype=float)
    U, s, Vt = svd_full(A)
    print(bold_cyan(f"  A =\n{A}\n"))
    print(bold_cyan(f"  U =\n{U.round(4)}\n  (left singular vectors; U·Uᵀ = I)"))
    print(bold_cyan(f"\n  σ = {s.round(4)}\n  (singular values — σ₁ ≥ σ₂ ≥ 0)"))
    print(bold_cyan(f"\n  Vᵀ =\n{Vt.round(4)}\n  (right singular vectors; Vᵀ·V = I)"))
    # Reconstruction
    Sigma = np.zeros_like(A)
    Sigma[:min(A.shape), :min(A.shape)] = np.diag(s)
    A_recon = U @ Sigma @ Vt
    print(green(f"\n  Reconstruction U Σ Vᵀ =\n{A_recon.round(6)}"))
    print(green(f"  ‖A - UΣVᵀ‖_F = {np.linalg.norm(A - A_recon):.2e}  ✓"))
    print()
    # Rank-1 approximation
    A_rank1 = low_rank_approx(A, 1)
    err1 = np.linalg.norm(A - A_rank1, "fro")
    evr  = explained_variance_ratio(s)
    print(bold_cyan(f"  Rank-1 approximation:"))
    print(yellow(f"  A ≈ σ₁·u₁·v₁ᵀ =\n{A_rank1.round(3)}"))
    print(yellow(f"  Explained variance: {evr[0]*100:.1f}%   Error: {err1:.4f}"))
    _pause()

    section_header("4. ASCII VISUALIZATION — Singular Value Magnitudes")
    print()
    for i, (sv, ev) in enumerate(zip(s, evr), 1):
        bar_pct = int(ev * 40)
        print(f"  σ{i} = {sv:.4f}  " +
              yellow("█" * bar_pct + "░" * (40 - bar_pct)) +
              grey(f"  {ev*100:.1f}%"))
    print()
    _pause()

    section_header("5. PLOTEXT")
    scree_plot(evr, cumulative=True)

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(A, cmap="Blues", aspect="auto")
        axes[0].set_title("Original A (3×2)")
        for k, ax in zip([1, 2], axes[1:]):
            Ak = low_rank_approx(A, k)
            ax.imshow(Ak, cmap="Blues", aspect="auto", vmin=A.min(), vmax=A.max())
            err = np.linalg.norm(A - Ak, "fro")
            ax.set_title(f"Rank-{k} approx\n(err={err:.3f})")
        plt.suptitle("SVD Low-Rank Approximation")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    print()

    section_header("7. PYTHON CODE")
    code_block("SVD — Full and Truncated", """
import numpy as np

A = np.array([[3, 2],
              [2, 3],
              [1, 1]], dtype=float)

# Full SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)
print("Singular values:", s)

# Reconstruct A from SVD
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, s)
A_recon = U @ Sigma @ Vt
print("Reconstruction error:", np.linalg.norm(A - A_recon))

# Rank-1 approximation (best rank-1 approx by Eckart-Young theorem)
A_rank1 = s[0] * np.outer(U[:, 0], Vt[0, :])
print("Rank-1 error:", np.linalg.norm(A - A_rank1, 'fro'))

# Explained variance ratio
evr = (s**2) / (s**2).sum()
print("Explained variance:", evr)

# Condition number = σ_max / σ_min
cond = s[0] / s[-1]
print("Condition number:", cond)
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "SVD always exists — for any matrix, regardless of shape or rank",
        "Top singular values dominate: often 90%+ variance in first few components",
        "SVD of term-document matrix → LSA (Latent Semantic Analysis) in NLP",
        "SVD of user-item matrix → collaborative filtering (Netflix Prize!)",
        "Condition number σ_max/σ_min measures numerical stability of the matrix",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Principal Component Analysis
# ══════════════════════════════════════════════════════════════════════════════
def topic_pca():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "PCA")
    section_header("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print()

    section_header("1. THEORY")
    print(white("""
  PCA is the most widely used dimensionality reduction method.  Given data
  X ∈ ℝⁿˣᵈ, PCA finds a k-dimensional subspace that retains as much variance
  as possible.  The key idea: project data onto the directions (principal
  components) along which the data varies the most.

  Mathematically, PCA solves: find the unit vector w₁ that maximises the
  variance of the projected data: w₁ = argmax_‖w‖=1 Var(Xw).  The solution is
  the top eigenvector of the covariance matrix C = XᵀX/(n-1).  Successive
  components w₂, w₃, ... are eigenvectors of C with decreasing eigenvalues.

  Via SVD: if X is centred, then X = UΣVᵀ, and principal components are the
  rows of Vᵀ.  The projected data is Z = UΣ = XV.  This is numerically more
  stable than computing XᵀX explicitly.

  Applications: face recognition (eigenfaces), noise reduction, visualisation
  of high-dimensional data, preprocessing before ML (whitening), compression.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print()
    print(formula("  Centre data:  X̄ = X - mean(X)"))
    print(formula("  Cov matrix:   C = X̄ᵀX̄ / (n-1)"))
    print(formula("  Eigen eqn:    C·vₖ = λₖ·vₖ"))
    print(formula("  Projection:   Z = X̄ Vₖ   where Vₖ contains top k PC vectors"))
    print(formula("  Exp. var.:    EVR_k = λₖ / Σᵢ λᵢ"))
    print(formula("  Reconstruct:  X̂ = Z Vₖᵀ + mean(X)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 4D → 2D")
    print()
    np.random.seed(42)
    n = 100
    # Generate correlated 4D data
    cov_true = np.array([[4,3,2,1],[3,4,2,1],[2,2,2,1],[1,1,1,1]], dtype=float)
    X4 = np.random.multivariate_normal([0,0,0,0], cov_true, n)
    Z, comps, ev, evr = pca(X4, n_components=2)
    print(bold_cyan(f"  Original data: {X4.shape}"))
    print(bold_cyan(f"  Reduced data:  {Z.shape}"))
    print(bold_cyan(f"  Eigenvalues:   {ev.round(3)}"))
    print(bold_cyan(f"  Explained var: {evr.round(3)} (sum = {evr.sum():.3f})"))
    print()
    cum = np.cumsum(evr)
    print(yellow(f"  PC1 explains {evr[0]*100:.1f}%"))
    print(yellow(f"  PC1+PC2 explains {cum[1]*100:.1f}%"))
    print()
    # Reconstruction error
    X_mean = X4.mean(axis=0)
    X_recon = Z @ comps + X_mean
    err = np.linalg.norm(X4 - X_recon, "fro")
    print(green(f"  Reconstruction error (Frobenius): {err:.4f}"))
    _pause()

    section_header("4. ASCII VISUALIZATION — PCA Projection")
    scatter(Z[:, 0], Z[:, 1], title="2D PCA Projection",
            width=55, height=18, char="·", color_fn=yellow)
    print()
    _pause()

    section_header("5. PLOTEXT — Scree Plot")
    scree_plot(evr)
    print()

    section_header("6. MATPLOTLIB — Scatter with Principal Axes")
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA as skPCA
        pca_sk = skPCA(n_components=2)
        pca_sk.fit(X4)
        Z_sk = pca_sk.transform(X4)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(X4[:, 0], X4[:, 1], alpha=0.5, s=20)
        axes[0].set_title("Original (first 2 dims)")
        # Draw principal axes
        for i, (ev_i, comp) in enumerate(zip(pca_sk.explained_variance_, pca_sk.components_)):
            arrow_scale = np.sqrt(ev_i)
            axes[0].annotate("", xy=arrow_scale*comp[:2]*2,
                             xytext=np.mean(X4[:,:2], axis=0),
                             arrowprops=dict(arrowstyle="->", color=f"C{i}", lw=2))
        axes[1].scatter(Z_sk[:, 0], Z_sk[:, 1], alpha=0.5, s=20)
        axes[1].set_xlabel(f"PC1 ({pca_sk.explained_variance_ratio_[0]*100:.1f}%)")
        axes[1].set_ylabel(f"PC2 ({pca_sk.explained_variance_ratio_[1]*100:.1f}%)")
        axes[1].set_title("PCA Projection")
        plt.suptitle("PCA: Finding Directions of Maximum Variance")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib / scikit-learn not installed"))
    print()

    section_header("7. PYTHON CODE")
    code_block("PCA from scratch + sklearn comparison", """
import numpy as np
from sklearn.decomposition import PCA

# Generate data
np.random.seed(42)
X = np.random.multivariate_normal([0,0,0,0],
    [[4,3,2,1],[3,4,2,1],[2,2,2,1],[1,1,1,1]], 100)

# ── PCA from scratch ──────────────────────────────────────────
X_c     = X - X.mean(axis=0)                  # Centre
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
Z_scratch = X_c @ Vt[:2].T                    # Project to 2D
ev_ratio  = (s**2) / (s**2).sum()
print("Explained variance (scratch):", ev_ratio[:2].round(3))

# ── sklearn PCA ───────────────────────────────────────────────
pca = PCA(n_components=2)
Z_sk = pca.fit_transform(X)
print("Explained variance (sklearn):", pca.explained_variance_ratio_.round(3))
print("Shapes agree:", Z_scratch.shape == Z_sk.shape)

# Reconstruction
X_recon = Z_scratch @ Vt[:2] + X.mean(axis=0)
err = np.linalg.norm(X - X_recon, 'fro')
print(f"Reconstruction error: {err:.4f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "PCA is optimal for linear dimensionality reduction — proved by Eckart-Young",
        "Always centre (subtract mean); normalise if features have different scales",
        "Scree plot elbow heuristic: keep components before the 'kink' in eigenvalues",
        "PCA does NOT consider class labels — it's unsupervised (LDA does use labels)",
        "Whitening (dividing by √λᵢ) makes components unit variance: used before ICA",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Vector Norms
# ══════════════════════════════════════════════════════════════════════════════
def topic_norms():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Vector Norms")
    section_header("VECTOR NORMS")
    print()

    section_header("1. THEORY")
    print(white("""
  A NORM is a function ‖·‖: Vℝⁿ → ℝ that measures the "size" or "length" of a
  vector.  It must satisfy three properties:
    (1) Non-negativity:   ‖x‖ ≥ 0, and ‖x‖ = 0 iff x = 0
    (2) Homogeneity:      ‖αx‖ = |α|·‖x‖
    (3) Triangle ineq.:   ‖x + y‖ ≤ ‖x‖ + ‖y‖

  Different norms create different "shapes" of unit balls — the set of all
  vectors with ‖x‖ ≤ 1.  The L2 norm gives a circle (sphere in higher dims).
  The L1 norm gives a diamond (rotated square).  The L∞ norm gives a square.

  In ML, the choice of norm determines the geometry of regularisation:
  L2 (Ridge/Tikhonov) penalises large weights but keeps all non-zero.
  L1 (Lasso) induces sparsity — many weights go exactly to zero — because the
  diamond shape has corners, and the regularised optimum often lands at a corner.
  L0 counts non-zero elements — the theoretical sparsity measure, but NP-hard
  to optimise directly.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print()
    print(formula("  Lₚ norm:  ‖x‖ₚ = (Σᵢ |xᵢ|ᵖ)^(1/p)"))
    print(formula("  L0 norm:  ‖x‖₀ = #{i : xᵢ ≠ 0}  (count of non-zeros)"))
    print(formula("  L1 norm:  ‖x‖₁ = Σᵢ |xᵢ|"))
    print(formula("  L2 norm:  ‖x‖₂ = √(Σᵢ xᵢ²)"))
    print(formula("  L∞ norm:  ‖x‖∞ = max |xᵢ|"))
    print(formula("  Frobenius: ‖A‖_F = √(Σᵢⱼ aᵢⱼ²) = √(tr(AᵀA)) = ‖σ‖₂"))
    print()
    _pause()

    section_header("3. NUMERICAL EXAMPLE: x = [2, 3, -1, 0]")
    x = np.array([2.0, 3.0, -1.0, 0.0])
    norms = all_norms(x)
    print(bold_cyan(f"  x = {x}"))
    print()
    for name, val in norms.items():
        bar_pct = int(val / 7 * 30)
        print(f"  {yellow(name.ljust(25))} = {value(f'{val:.4f}')}"
              f"  {cyan('█' * bar_pct + '░' * (30 - bar_pct))}")
    print()
    _pause()

    section_header("4. ASCII VISUALIZATION — L1 vs L2 Unit Balls")
    print()
    print(cyan("     L1 (diamond)                L2 (circle)"))
    print(cyan("  y ┌──────────────┐         y ┌──────────────┐"))
    print(cyan("  1 │") + yellow("      *      ") + cyan("│") +
          cyan("        │") + green("    * * *    ") + cyan("│"))
    print(cyan("    │") + yellow("    * · *    ") + cyan("│") +
          cyan("        │") + green("  *       *  ") + cyan("│"))
    print(cyan("  0 │") + yellow("  * · · · *  ") + cyan("│") +
          cyan("      0 │") + green("*           *") + cyan("│"))
    print(cyan("    │") + yellow("    * · *    ") + cyan("│") +
          cyan("        │") + green("  *       *  ") + cyan("│"))
    print(cyan(" -1 │") + yellow("      *      ") + cyan("│") +
          cyan("        │") + green("    * * *    ") + cyan("│"))
    print(cyan("    └──────────────┘         └──────────────┘"))
    print(cyan("      -1  0   1                 -1  0   1   "))
    print()
    print(grey("  L1 diamond corners → regularised solutions land at sparse points"))
    print()
    _pause()

    section_header("5. MATPLOTLIB — All Norm Balls")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(14, 3))
        theta = np.linspace(0, 2*np.pi, 200)
        configs = [(0.5, "L0.5"), (1, "L1"), (2, "L2"), (10, "L∞ approx")]
        for ax, (p, lbl) in zip(axes, configs):
            pts = []
            for t in theta:
                x_t = np.cos(t); y_t = np.sin(t)
                v   = np.array([x_t, y_t])
                v   = v / np.linalg.norm(v, ord=p)
                pts.append(v)
            pts = np.array(pts)
            ax.fill(pts[:,0], pts[:,1], alpha=0.4)
            ax.plot(pts[:,0], pts[:,1])
            ax.set_title(f"{lbl} unit ball")
            ax.set_aspect("equal"); ax.set_xlim(-1.8,1.8); ax.set_ylim(-1.8,1.8)
            ax.grid(alpha=0.3)
        plt.suptitle("Lₚ Unit Balls: p=0.5, 1, 2, ∞")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    print()

    section_header("6. PYTHON CODE")
    code_block("Vector Norms", """
import numpy as np

x = np.array([2.0, 3.0, -1.0, 0.0])

# Different norms
l0   = np.sum(x != 0)                      # = 3
l1   = np.linalg.norm(x, ord=1)            # = 6.0
l2   = np.linalg.norm(x, ord=2)            # ≈ 3.742
linf = np.linalg.norm(x, ord=np.inf)       # = 3.0

print(f"L0 = {l0}")
print(f"L1 = {l1:.4f}")
print(f"L2 = {l2:.4f}")
print(f"L∞ = {linf:.4f}")

# For a 2D matrix A: Frobenius norm
A = np.array([[1,2],[3,4]])
frob = np.linalg.norm(A, 'fro')      # = sqrt(1+4+9+16) = sqrt(30)
nuc  = np.linalg.norm(A, 'nuc')      # nuclear norm = sum of singular values
print(f"‖A‖_F = {frob:.4f}")

# L2 regularisation (Ridge):  loss + λ‖w‖₂²
# L1 regularisation (Lasso):  loss + λ‖w‖₁
w = np.array([0.5, -0.2, 0.3])
lambda_reg = 0.01
ridge_penalty = lambda_reg * np.sum(w**2)
lasso_penalty = lambda_reg * np.sum(np.abs(w))
print(f"Ridge penalty: {ridge_penalty:.6f}")
print(f"Lasso penalty: {lasso_penalty:.6f}")
""")
    _pause()

    section_header("7. KEY INSIGHTS")
    insights = [
        "L1 regularisation = MAP estimation with Laplace prior on weights",
        "L2 regularisation = MAP estimation with Gaussian prior on weights",
        "Nuclear norm (sum of σᵢ) is the convex relaxation of rank — used in matrix completion",
        "Spectral norm (max σ) bounds how much a layer can stretch the input",
        "Gradient norms are used in gradient clipping: if ‖g‖₂ > threshold, scale g down",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Gram-Schmidt Orthogonalisation
# ══════════════════════════════════════════════════════════════════════════════
def topic_gram_schmidt():
    clear()
    breadcrumb("mlmath", "Linear Algebra", "Gram-Schmidt")
    section_header("GRAM-SCHMIDT ORTHOGONALISATION")
    print()

    section_header("1. THEORY")
    print(white("""
  The Gram-Schmidt process converts any set of linearly independent vectors
  into an ORTHONORMAL basis — vectors that are mutually perpendicular and each
  have unit length.  This is computationally powerful because:
    • Matrix operations with orthogonal matrices are numerically stable
    • QᵀQ = I means the inverse is just the transpose (O(n²) vs O(n³))
    • Gram-Schmidt is the foundation of QR decomposition
    • QR decomposition is used in solving least-squares problems (QR is more
      stable than the normal equations!), the QR algorithm for eigenvalues,
      and Gram-Schmidt is implicit in Householder transformations.

  The idea: take each new vector, subtract its projections onto all previously
  computed orthonormal vectors, then normalise the remainder.  What remains
  is exactly the "new" information that vector adds beyond the existing basis.

  In deep learning, orthogonal weight initialisation (Saxe et al. 2013) gives
  perfect conditioning at initialisation.  Orthogonalisation appears in GAN
  training (spectral norm) and policy gradient (natural policy gradient).
"""))
    _pause()

    section_header("2. KEY FORMULA")
    print()
    print(formula("  Given v₁, v₂, ..., vₖ (linearly independent):"))
    print(formula("  u₁ = v₁ / ‖v₁‖"))
    print(formula("  ũₖ = vₖ - Σⱼ<ₖ (vₖ·uⱼ) uⱼ     (subtract projections)"))
    print(formula("  uₖ = ũₖ / ‖ũₖ‖               (normalise)"))
    print()
    print(formula("  Verify: uᵢ·uⱼ = δᵢⱼ  (Kronecker delta)"))
    print(formula("  Matrix form:  Q = [u₁ | u₂ | ... | uₖ],  QᵀQ = I"))
    print()
    _pause()

    section_header("3. STEP-BY-STEP: 3 Vectors → Orthonormal Basis")
    print()
    V = np.array([[1.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0]]).T   # 3×3, columns are vectors
    Q, steps = gram_schmidt(V)
    print(bold_cyan("  Input vectors (columns of V):"))
    for i, col in enumerate(V.T, 1):
        print(f"    v{i} = {col}")
    print()
    for step in steps:
        print(f"  {yellow(step)}")
    print()
    print(bold_cyan("  Result Q (orthonormal columns):"))
    for i, col in enumerate(Q.T, 1):
        print(f"    q{i} = {col.round(5)}")
    print()
    print(green(f"  Verify QᵀQ = I:"))
    QtQ = Q.T @ Q
    print(f"  {QtQ.round(6)}")
    print(green(f"  is_orthonormal: {is_orthonormal(Q)}  ✓"))
    _pause()

    section_header("4. ASCII VISUALIZATION — Orthogonalisation")
    print()
    print(yellow("  Step 1:  u₁ = v₁ / ‖v₁‖  (just normalise v₁)"))
    print(cyan("           v₁ ───────────→ u₁ (unit length)"))
    print()
    print(yellow("  Step 2:  ũ₂ = v₂ - (v₂·u₁)u₁  (subtract projection)"))
    print(cyan("           v₂  ⤵"))
    print(cyan("           proj(v₂→u₁) subtracted → remainder ⊥ u₁"))
    print(cyan("           ũ₂ / ‖ũ₂‖ = u₂"))
    print()
    print(yellow("  Step 3:  ũ₃ = v₃ - (v₃·u₁)u₁ - (v₃·u₂)u₂"))
    print(cyan("           subtract projections onto u₁ and u₂"))
    print(cyan("           whatever remains ⊥ u₁ and u₂ → normalise → u₃"))
    print()

    section_header("5. PYTHON CODE")
    code_block("Gram-Schmidt from Scratch", """
import numpy as np

def gram_schmidt(V):
    '''Orthonormalise columns of V.'''
    Q = np.zeros_like(V, dtype=float)
    for k in range(V.shape[1]):
        q = V[:, k].astype(float)
        for j in range(k):
            q -= np.dot(Q[:, j], q) * Q[:, j]    # subtract projection
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            Q[:, k] = q / norm
    return Q

# Example
V = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q = gram_schmidt(V)
print("Q =")
print(Q.round(5))
print("QᵀQ =")
print((Q.T @ Q).round(10))

# QR decomposition via Gram-Schmidt
Q_np, R_np = np.linalg.qr(V)
print("numpy QR — same columns up to sign:", 
      np.allclose(np.abs(Q), np.abs(Q_np)))
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    insights = [
        "Numerically, modified Gram-Schmidt is more stable than classical GS",
        "Gram-Schmidt ≡ QR decomposition: V = QR where R is upper triangular",
        "QᵀQ = I means orthogonal matrices preserve lengths and angles",
        "Loss of orthogonality in iterative methods → instability; re-orthogonalise!",
        "Orthogonal initialisation (He et al.) maintains gradient norms at init",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Vector Spaces and Subspaces",      topic_vector_spaces),
        ("Dot Product and Projections",       topic_dot_product),
        ("Matrix Multiplication",             topic_matmul),
        ("Eigenvalue Decomposition",          topic_eigenvalues),
        ("Singular Value Decomposition",      topic_svd),
        ("Principal Component Analysis",      topic_pca),
        ("Vector Norms",                      topic_norms),
        ("Gram-Schmidt Orthogonalisation",    topic_gram_schmidt),
    ]
    block_menu("b01", "Linear Algebra Fundamentals", topics)
    mark_completed("b01")
