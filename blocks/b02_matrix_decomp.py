"""
Block 02 — Matrix Decompositions
LU, QR, Cholesky, Condition Number, Solving Ax=b
"""
import numpy as np

from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold, white,
                       formula, value, section, emph, hint, red, bold_magenta)
from ui.widgets import box, section_header, breadcrumb, nav_bar, table, bar_chart, code_block, panel, pager, hr, print_sparkline
from ui.menu import topic_nav, clear, block_menu, mark_completed
from viz.ascii_plots import scatter, line_plot, multi_line, comp_graph, heatmap


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ─────────────────────────────────────────────────────────────────────────────
# 1. LU Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def topic_lu():
    clear()
    breadcrumb("mlmath", "Matrix Decompositions", "LU Decomposition")
    section_header("LU DECOMPOSITION")

    section_header("1. THEORY")
    print(white("  LU decomposition factors a square matrix A into a lower triangular matrix L"))
    print(white("  and an upper triangular matrix U, such that A = LU (or PA = LU with pivoting)."))
    print()
    print(white("  Lower triangular L has 1s on its diagonal and zeros above. Upper triangular U"))
    print(white("  has arbitrary entries on and above the diagonal, zeros below. This factored"))
    print(white("  form makes solving Ax = b cheap: forward substitution for Ly = b, then"))
    print(white("  backward substitution for Ux = y — each O(n²) instead of O(n³)."))
    print()
    print(white("  Partial pivoting (PA = LU) reorders rows for numerical stability; P is a"))
    print(white("  permutation matrix. Without pivoting, small pivots cause catastrophic")  )
    print(white("  cancellation. Gaussian elimination literally constructs LU step by step."))
    print()
    print(white("  Key uses: solving linear systems, computing determinants (det A = det L · det U"),)
    print(white("  = product of diagonal of U), inverting matrices, and computing log-determinants"))
    print(white("  in Gaussian processes and variational inference."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Factorisation:     A  =  P⁻¹ L U   (with partial pivoting)"))
    print(formula("  Determinant:       det(A) = det(U) = ∏ Uᵢᵢ  (if no pivoting)"))
    print(formula("  Forward sub:       Ly = Pb  →  yᵢ = (bᵢ - Σⱼ<ᵢ Lᵢⱼyⱼ) / Lᵢᵢ"))
    print(formula("  Backward sub:      Ux = y   →  xᵢ = (yᵢ - Σⱼ>ᵢ Uᵢⱼxⱼ) / Uᵢᵢ"))
    print(formula("  Lᵢⱼ (i>j):         mᵢⱼ = Aᵢⱼ / Ajⱼ  (multiplier)"))
    print(formula("  Complexity:        O(n³) for factorisation, O(n²) per solve"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    A = np.array([[2., 1., 1.],
                  [4., 3., 3.],
                  [8., 7., 9.]])
    print(white("  Input matrix A:"))
    for row in A:
        print(f"  {cyan('[')}  {'  '.join(value(f'{v:6.2f}') for v in row)}  {cyan(']')}")
    print()

    try:
        from scipy.linalg import lu
        P, L, U = lu(A)
    except ImportError:
        # fallback manual LU without pivoting
        n = len(A)
        L = np.eye(n); U = A.copy(); P = np.eye(n)
        for j in range(n):
            for i in range(j+1, n):
                if abs(U[j, j]) < 1e-12:
                    continue
                m = U[i, j] / U[j, j]
                L[i, j] = m
                U[i, :] -= m * U[j, :]

    print(white("  Permutation matrix P:"))
    for row in P:
        print(f"    {[int(v) for v in row]}")
    print()
    print(white("  Lower triangular L:"))
    for row in L:
        print(f"  {'  '.join(value(f'{v:6.3f}') for v in row)}")
    print()
    print(white("  Upper triangular U:"))
    for row in U:
        print(f"  {'  '.join(value(f'{v:6.3f}') for v in row)}")
    print()

    PA   = P @ A
    LU_  = L @ U
    err  = np.max(np.abs(PA - LU_))
    print(green(f"  ✓ Verification  max|PA - LU| = {err:.2e}"))
    print()

    b    = np.array([1., 2., 3.])
    x    = np.linalg.solve(A, b)
    print(white(f"  Solving Ax = b with b = {b}"))
    print(green(f"  Solution x = {np.round(x, 4)}"))
    print(white(f"  Residual ‖Ax-b‖ = {np.linalg.norm(A @ x - b):.2e}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Triangular structure of L and U:"))
    print()
    n = 4
    print(cyan("  L (lower triangular)") + "          " + cyan("  U (upper triangular)"))
    for i in range(n):
        row_l = "  [ " + "  ".join(("1  " if j == i else ("*  " if j < i else ".  ")) for j in range(n)) + "]"
        row_u = "  [ " + "  ".join(("*  " if j >= i else ".  ") for j in range(n)) + "]"
        print(green(row_l) + "         " + yellow(row_u))
    print()
    print(grey("  * = non-zero entry   . = structural zero"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import bar
        nnz_L = [i + 1 for i in range(4)]
        nnz_U = [4 - i for i in range(4)]
        bar(nnz_L, title="Non-zeros per row of L", xlabel="Row", ylabel="Count")
        bar(nnz_U, title="Non-zeros per row of U", xlabel="Row", ylabel="Count")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, mat, title in zip(axes, [P, L, U], ["P (permutation)", "L (lower)", "U (upper)"]):
            im = ax.imshow(mat, cmap="Blues")
            ax.set_title(title); fig.colorbar(im, ax=ax)
        fig.suptitle("LU Decomposition of A")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("LU Decomposition", """\
import numpy as np
from scipy.linalg import lu

A = np.array([[2., 1., 1.],
              [4., 3., 3.],
              [8., 7., 9.]])

P, L, U = lu(A)          # scipy partial-pivot LU
print("PA == LU:", np.allclose(P @ A, L @ U))

# Solve Ax = b using the factorisation
from scipy.linalg import lu_factor, lu_solve
lu_fac = lu_factor(A)
b = np.array([1., 2., 3.])
x = lu_solve(lu_fac, b)  # O(n²)
print("x =", x)
print("Residual:", np.linalg.norm(A @ x - b))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "PA = LU with partial pivoting is stable; pure LU without pivoting can fail.",
        "Factorisation costs O(n³); each subsequent solve for new b costs only O(n²).",
        "det(A) = sign(P) · ∏ U_ii — LU gives determinants for free.",
        "L has unit diagonal; the n² multipliers used in elimination fill L's lower half.",
        "Sparse LU (AMD reordering) powers interior-point solvers in convex optimisation.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 2. QR Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def topic_qr():
    clear()
    breadcrumb("mlmath", "Matrix Decompositions", "QR Decomposition")
    section_header("QR DECOMPOSITION")

    section_header("1. THEORY")
    print(white("  QR decomposition writes an m×n matrix (m≥n) as A = QR, where Q is an m×m"))
    print(white("  orthogonal matrix (Qᵀ=Q⁻¹, columns are orthonormal) and R is m×n upper"))
    print(white("  triangular. The 'economy' (thin) QR keeps only the first n columns of Q."))
    print()
    print(white("  Two classical algorithms exist:  Gram-Schmidt orthogonalisation (intuitive,"))
    print(white("  numerically fragile) and Householder reflections (numerically stable, used"))
    print(white("  in practice). LAPACK's dgeqrf uses Householder internally."))
    print()
    print(white("  The primary ML application is least-squares regression: given overdetermined"))
    print(white("  Ax ≈ b (more equations than unknowns), the normal equations Aᵀᴬx = Aᵀb"))
    print(white("  are equivalent to Rx = Qᵀb, which is solved by back-substitution — stable"))
    print(white("  and efficient without explicitly forming AᵀA (which squares the condition number)"))
    print()
    print(white("  QR is also the engine of the QR algorithm for computing eigenvalues."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Decomposition:      A  =  Q R          (Q orthogonal, R upper-triangular)"))
    print(formula("  Orthogonality:      Qᵀ Q = Iₙ          (thin Q)"))
    print(formula("  Least-squares:      min ‖Ax - b‖²  →  x* = R⁻¹ Qᵀ b"))
    print(formula("  Gram-Schmidt step:  uₖ = aₖ - Σⱼ<ₖ (aₖ·qⱼ) qⱼ,  qₖ = uₖ/‖uₖ‖"))
    print(formula("  Householder:        H = I - 2vvᵀ/vᵀv  (reflection)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    A = np.array([[1., 1.],
                  [1., 2.],
                  [1., 3.],
                  [1., 4.]])
    b = np.array([1., 2., 2., 3.])
    print(white("  4×2 design matrix A (column 1 = ones, column 2 = x):"))
    for row in A:
        print(f"    {row}")
    print(white(f"\n  Target vector b = {b}"))
    print()

    Q, R = np.linalg.qr(A, mode='reduced')
    print(white("  Thin Q (4×2):"))
    for row in Q:
        print(f"    {np.round(row, 4)}")
    print(white("\n  R (2×2):"))
    for row in R:
        print(f"    {np.round(row, 4)}")

    x_star = np.linalg.solve(R, Q.T @ b)
    print(white(f"\n  Least-squares solution x* = R⁻¹Qᵀb = {np.round(x_star, 4)}"))
    print(white(f"  Residual ‖Ax*-b‖ = {np.linalg.norm(A @ x_star - b):.4f}"))

    print(white("\n  Gram-Schmidt (minimal):"))
    cols = [A[:, j].copy().astype(float) for j in range(A.shape[1])]
    qs = []
    for v in cols:
        for q in qs:
            v -= np.dot(v, q) * q
        v /= np.linalg.norm(v)
        qs.append(v)
    Q_gs = np.column_stack(qs)
    print(white(f"  Max diff Q vs Gram-Schmidt: {np.max(np.abs(np.abs(Q) - np.abs(Q_gs))):.2e}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Structure of thin QR:"))
    m, n = 4, 2
    print(cyan("  A (4×2)") + "      =    " + green("Q (4×2)") + "      ×   " + yellow("R (2×2)"))
    for i in range(m):
        a_s  = cyan(f"  [ {' '.join('*' for _ in range(n))} ]")
        q_s  = green(f"  [ {' '.join('q' for _ in range(n))} ]")
        r_s  = yellow(f"  [ {' '.join(('r' if j >= i else '.') for j in range(n))} ]") if i < n else grey("  (continuation)")
        print(f"{a_s}      {q_s}      {r_s if i < n else ''}")
    print()
    print(grey("  q = orthonormal columns  r = upper-triangular entries  . = structural zero"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import scatter_plot
        x_pts = A[:, 1]
        y_pred = A @ x_star
        scatter_plot(x_pts.tolist(), b.tolist(), title="Least-Squares Fit (QR)",
                     xlabel="x", ylabel="y")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs = np.linspace(0.5, 4.5, 100)
        ys = x_star[0] + x_star[1] * xs
        plt.figure(figsize=(6, 4))
        plt.scatter(A[:, 1], b, color='steelblue', zorder=5, label='Data', s=60)
        plt.plot(xs, ys, color='tomato', lw=2, label=f'Fit: {x_star[0]:.2f} + {x_star[1]:.2f}x')
        plt.xlabel("x"); plt.ylabel("y")
        plt.title("QR Least-Squares Fit"); plt.legend(); plt.tight_layout()
        plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("QR Decomposition & Least Squares", """\
import numpy as np

A = np.array([[1., 1.],
              [1., 2.],
              [1., 3.],
              [1., 4.]])
b = np.array([1., 2., 2., 3.])

# Thin QR
Q, R = np.linalg.qr(A, mode='reduced')

# Least-squares via QR: Rx = Qᵀb
x_star = np.linalg.solve(R, Q.T @ b)
print("Coefficients:", x_star)
print("Residual:", np.linalg.norm(A @ x_star - b))

# Gram-Schmidt from scratch
def gram_schmidt(A):
    Q = A.astype(float).copy()
    for j in range(Q.shape[1]):
        for i in range(j):
            Q[:, j] -= np.dot(Q[:, j], Q[:, i]) * Q[:, i]
        Q[:, j] /= np.linalg.norm(Q[:, j])
    return Q

Q_gs = gram_schmidt(A)
print("Orthogonality check:", np.allclose(Q_gs.T @ Q_gs, np.eye(2)))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "QR for least-squares is more numerically stable than the normal equations AᵀAx=Aᵀb.",
        "Householder QR costs 2mn²-2n³/3 flops; Gram-Schmidt is equivalent but less stable.",
        "Thin QR retains only n orthonormal columns — sufficient for least-squares.",
        "QR + column pivoting reveals the numerical rank of a matrix.",
        "The QR algorithm (iteratively applying QR shifts) computes all eigenvalues of A.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cholesky Decomposition
# ─────────────────────────────────────────────────────────────────────────────
def topic_cholesky():
    clear()
    breadcrumb("mlmath", "Matrix Decompositions", "Cholesky Decomposition")
    section_header("CHOLESKY DECOMPOSITION")

    section_header("1. THEORY")
    print(white("  For a symmetric positive-definite (SPD) matrix A, Cholesky gives A = LLᵀ"))
    print(white("  where L is lower-triangular with positive diagonal entries. It is unique."))
    print()
    print(white("  Cholesky is exactly twice as fast as LU because it exploits symmetry — only"))
    print(white("  the lower triangle is referenced. Cost: n³/3 flops vs 2n³/3 for LU."))
    print()
    print(white("  In ML, Cholesky is ubiquitous: Gaussian processes compute log-marginal"))
    print(white("  likelihood as -½·(2Σ log Lᵢᵢ + n log 2π + yᵀ K⁻¹ y); variational"))
    print(white("  autoencoders reparameterise z = μ + L·ε; multivariate Gaussian sampling"))
    print(white("  draws ε ~ N(0,I) then returns L·ε + μ."))
    print()
    print(white("  If Cholesky fails (negative pivot), the matrix is NOT positive-definite."))
    print(white("  This is the cheapest SPD test: O(n³) but with a tiny constant."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Factorisation:  A  =  L Lᵀ    (L lower-triangular, diagonal > 0)"))
    print(formula("  Diagonal:       Lⱼⱼ = √(Aⱼⱼ - Σₖ<ⱼ Lⱼₖ²)"))
    print(formula("  Off-diagonal:   Lᵢⱼ = (Aᵢⱼ - Σₖ<ⱼ LᵢₖLⱼₖ) / Lⱼⱼ   (i > j)"))
    print(formula("  Log-det trick:  log det(A) = 2 Σᵢ log Lᵢᵢ"))
    print(formula("  GP log-lik:     log p(y) = -½(yᵀK⁻¹y + log|K| + n log 2π)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    A = np.array([[4., 2., 2.],
                  [2., 5., 3.],
                  [2., 3., 6.]])
    print(white("  SPD matrix A:"))
    for row in A:
        print(f"    {row}")

    eigs = np.linalg.eigvalsh(A)
    print(white(f"\n  Eigenvalues (all > 0 → SPD): {np.round(eigs, 4)}"))

    L = np.linalg.cholesky(A)
    print(white("\n  Cholesky factor L:"))
    for row in L:
        print(f"    {np.round(row, 4)}")

    recon = L @ L.T
    err = np.max(np.abs(recon - A))
    print(green(f"\n  ✓ max|LLᵀ - A| = {err:.2e}"))

    log_det_chol = 2 * np.sum(np.log(np.diag(L)))
    log_det_np   = np.log(np.linalg.det(A))
    print(white(f"\n  log det(A) via Cholesky diag: {log_det_chol:.6f}"))
    print(white(f"  log det(A) via numpy:          {log_det_np:.6f}"))
    print()

    print(white("  Manual Cholesky (from scratch):"))
    n = 3; Lm = np.zeros((n, n))
    for j in range(n):
        Lm[j, j] = np.sqrt(A[j, j] - np.sum(Lm[j, :j]**2))
        for i in range(j+1, n):
            Lm[i, j] = (A[i, j] - np.sum(Lm[i, :j] * Lm[j, :j])) / Lm[j, j]
    print(white("  Manual L:"))
    for row in Lm:
        print(f"    {np.round(row, 4)}")
    print(green(f"  ✓ matches numpy: {np.allclose(L, Lm)}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Structure of L and Lᵀ:"))
    n = 4
    print(cyan("   L                  ") + "   ×   " + yellow("   Lᵀ"))
    for i in range(n):
        row_l = "  [ " + "  ".join(("L  " if j <= i else ".  ") for j in range(n)) + "]"
        row_r = "  [ " + "  ".join(("L  " if j >= i else ".  ") for j in range(n)) + "]"
        print(cyan(row_l) + "         " + yellow(row_r))
    print(grey("\n  L = lower triangular  .  = structural zero"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import bar
        diag_vals = np.diag(L).tolist()
        bar(diag_vals, title="Diagonal of L (Cholesky factor)", xlabel="Index", ylabel="Lᵢᵢ")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, mat, title in zip(axes, [A, L, L @ L.T], ["A (original)", "L (Cholesky)", "LLᵀ (reconstruction)"]):
            im = ax.imshow(mat, cmap="Oranges")
            ax.set_title(title); fig.colorbar(im, ax=ax)
        fig.suptitle("Cholesky Decomposition")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Cholesky Decomposition", """\
import numpy as np

A = np.array([[4., 2., 2.],
              [2., 5., 3.],
              [2., 3., 6.]])

# numpy Cholesky
L = np.linalg.cholesky(A)
print("L =\\n", L)
print("LLᵀ == A:", np.allclose(L @ L.T, A))

# Efficient solve: Ax = b using Cholesky
b = np.array([1., 2., 3.])
y = np.linalg.solve(L, b)          # forward substitution
x = np.linalg.solve(L.T, y)        # backward substitution
print("x =", x)
print("Residual:", np.linalg.norm(A @ x - b))

# Log-determinant (numerically stable)
log_det = 2 * np.sum(np.log(np.diag(L)))
print("log det(A) =", log_det)

# Multivariate Gaussian sampling
rng = np.random.default_rng(42)
eps = rng.standard_normal(3)
mu  = np.zeros(3)
sample = mu + L @ eps              # reparameterisation trick
print("Sample from N(0, A):", sample)
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Cholesky requires A to be SPD; it is the cheapest way to verify positive-definiteness.",
        "Cholesky is 2× faster than LU by exploiting symmetry — costs n³/3 flops.",
        "log det(A) = 2 Σ log Lᵢᵢ avoids catastrophic cancellation in direct det computation.",
        "Reparameterisation trick: x ~ N(μ, Σ) ↔ x = μ + L·ε, ε ~ N(0,I), differentiable in μ,L.",
        "Cholesky updates O(n²) allow sequential Bayesian updates without full refactorisation.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Condition Number & Numerical Stability
# ─────────────────────────────────────────────────────────────────────────────
def topic_condition_number():
    clear()
    breadcrumb("mlmath", "Matrix Decompositions", "Condition Number")
    section_header("CONDITION NUMBER & NUMERICAL STABILITY")

    section_header("1. THEORY")
    print(white("  The condition number κ(A) = σ_max / σ_min (ratio of largest to smallest"))
    print(white("  singular value) measures how much a small change in b amplifies error in x"))
    print(white("  when solving Ax = b. Well-conditioned: κ ≈ 1; ill-conditioned: κ >> 1."))
    print()
    print(white("  A rule of thumb: if κ(A) ≈ 10^k, you lose roughly k decimal digits of"))
    print(white("  accuracy in the solution. With double precision (~15 digits), a matrix with"))
    print(white("  κ = 10^12 gives only 3 reliable digits in the solution."))
    print()
    print(white("  The Hilbert matrix H_ij = 1/(i+j-1) is a classic ill-conditioned example."))
    print(white("  κ(H₁₀) ≈ 10^13 — inverting it in double precision is essentially meaningless."))
    print()
    print(white("  Remedies: scaling/preconditioning, regularisation (Tikhonov: solve (AᵀA+λI)x"),)
    print(white("  = Aᵀb), using higher-precision arithmetic, or switching to iterative solvers."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Condition number:    κ₂(A) = ‖A‖₂ · ‖A⁻¹‖₂ = σ_max / σ_min"))
    print(formula("  Error amplification: ‖δx‖/‖x‖  ≤  κ(A) · ‖δb‖/‖b‖"))
    print(formula("  Digits of accuracy:  d ≈ 15 - log₁₀(κ(A))   (double precision)"))
    print(formula("  Hilbert matrix:      H_ij = 1/(i+j-1),  κ(H_n) ≈ e^(3.5n)"))
    print(formula("  Tikhonov reg.:       x* = (AᵀA + λI)⁻¹ Aᵀb,  κ → κ/(1 + λ/σ_min²)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def hilbert(n):
        return np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])

    print(white("  Condition numbers of Hilbert matrices H_n:"))
    print()
    kappas = []
    ns = [2, 3, 4, 5, 6, 8, 10]
    for n in ns:
        H = hilbert(n)
        kappa = np.linalg.cond(H)
        kappas.append(kappa)
        print(f"  {white(f'H_{n}:')}  κ = {value(f'{kappa:.3e}')}  (≈ 10^{np.log10(kappa):.1f})")
    print()

    print(white("  Well-conditioned vs ill-conditioned — solving with perturbed b:"))
    print()
    A_good = np.array([[2., 1.], [1., 3.]])
    A_bad  = hilbert(5)
    rng    = np.random.default_rng(42)

    for Amat, label in [(A_good, "Good κ="+f"{np.linalg.cond(A_good):.1f}"),
                        (A_bad,  "Bad  κ="+f"{np.linalg.cond(A_bad):.1e}")]:
        n   = Amat.shape[0]
        x   = np.ones(n)
        b   = Amat @ x
        db  = rng.standard_normal(n) * 1e-6 * np.linalg.norm(b)
        x2  = np.linalg.solve(Amat, b + db)
        rel = np.linalg.norm(x2 - x) / np.linalg.norm(x)
        print(f"  [{label}]  ‖δx‖/‖x‖ = {rel:.2e}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Condition number spectrum (Hilbert matrices):"))
    print()
    max_kappa = max(kappas)
    bar_width  = 30
    for n_val, k in zip(ns, kappas):
        bar_len = max(1, int(bar_width * np.log10(max(k, 1)) / np.log10(max_kappa)))
        bar_str = "█" * bar_len
        print(f"  H_{n_val}  {green(bar_str):<35}  {value(f'κ={k:.1e}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import scatter_plot
        log_kappas = [np.log10(k) for k in kappas]
        scatter_plot(ns, log_kappas, title="log₁₀(κ) vs matrix size for Hilbert matrices",
                     xlabel="n", ylabel="log₁₀(κ)")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].semilogy(ns, kappas, 'o-', color='tomato')
        axes[0].set_xlabel("n"); axes[0].set_ylabel("κ(H_n)")
        axes[0].set_title("Condition Number of Hilbert Matrices")

        lambdas = np.logspace(-8, 0, 100)
        k_reg   = [np.linalg.cond(A_bad.T @ A_bad + l * np.eye(5)) for l in lambdas]
        axes[1].loglog(lambdas, k_reg, color='steelblue')
        axes[1].set_xlabel("λ (regularisation)"); axes[1].set_ylabel("κ(AᵀA + λI)")
        axes[1].set_title("Tikhonov Regularisation Effect")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Condition Number Analysis", """\
import numpy as np

def hilbert(n):
    return np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])

# Condition numbers
for n in [3, 5, 8, 10]:
    H = hilbert(n)
    kappa = np.linalg.cond(H)
    print(f"κ(H_{n}) = {kappa:.3e}  ({15 - np.log10(kappa):.1f} reliable digits)")

# Tikhonov regularisation
A = hilbert(8)
b = np.ones(8)
lam = 1e-6
x_reg = np.linalg.solve(A.T @ A + lam * np.eye(8), A.T @ b)
print("Tikhonov solution:", x_reg)

# SVD-based condition number
U, s, Vt = np.linalg.svd(A)
print("σ_max/σ_min =", s[0]/s[-1])
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "κ(A) = σ_max/σ_min; each power of 10 in κ costs one decimal digit of accuracy.",
        "The Hilbert matrix is the textbook example: κ(H₁₀) ≈ 10¹³ — useless in double.",
        "Preconditioning (left-multiply by M⁻¹) replaces A with M⁻¹A, lowering κ.",
        "Regularisation λI effectively raises σ_min to max(σ_min, √λ), reducing κ.",
        "Always check condition numbers before trusting least-squares or inversion results.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Solving Ax=b
# ─────────────────────────────────────────────────────────────────────────────
def topic_solving_axb():
    clear()
    breadcrumb("mlmath", "Matrix Decompositions", "Solving Ax=b")
    section_header("SOLVING LINEAR SYSTEMS Ax = b")

    section_header("1. THEORY")
    print(white("  Solving Ax = b is arguably the most common task in scientific computing."))
    print(white("  Methods divide into direct (factorise once, solve in O(n²)) and iterative"))
    print(white("  (update approximation, converge in O(kn) with k << n iterations)."))
    print()
    print(white("  Direct methods: numpy.linalg.solve (LAPACK DGESV, uses LU), Cholesky for SPD"))
    print(white("  (scipy.linalg.cho_solve), QR for least-squares (lstsq). Cost: O(n³) — fast"))
    print(white("  for n < 10,000 but memory-intensive for large sparse systems."))
    print()
    print(white("  Iterative methods: Conjugate Gradient (CG) for SPD Ax=b converges in at"))
    print(white("  most n steps, typically k << n with preconditioning. GMRES for non-symmetric."))
    print(white("  Ideal for sparse or matrix-free settings where A need only be applied as Av."))
    print()
    print(white("  Rule of thumb: use direct for dense n < 10^4, iterative for sparse or n >> 10^4."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Direct (LU):      PA=LU, cost O(2n³/3) flops once, O(n²) per rhs"))
    print(formula("  Cholesky:         A=LLᵀ (SPD only), cost O(n³/3) flops"))
    print(formula("  CG step:          αₖ = rₖᵀrₖ / pₖᵀApₖ"))
    print(formula("                    xₖ₊₁ = xₖ + αₖpₖ"))
    print(formula("                    rₖ₊₁ = rₖ - αₖApₖ"))
    print(formula("  CG convergence:   ‖eₖ‖_A ≤ 2((√κ-1)/(√κ+1))^k ‖e₀‖_A"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    rng = np.random.default_rng(7)
    n   = 4
    B   = rng.standard_normal((n, n))
    A   = B.T @ B + 0.5 * np.eye(n)   # SPD
    x_true = np.array([1., 2., 3., 4.])
    b   = A @ x_true

    print(white("  4×4 SPD system:"))
    print(white(f"  A =\n{np.round(A, 3)}"))
    print(white(f"  b = {np.round(b, 4)}"))
    print(white(f"  True x = {x_true}"))
    print()

    # Method 1: numpy solve (LU)
    x1 = np.linalg.solve(A, b)
    print(white("  Method 1 — numpy.linalg.solve (LU):"))
    print(f"    x = {np.round(x1, 6)},  err = {np.linalg.norm(x1 - x_true):.2e}")

    # Method 2: Cholesky
    L2 = np.linalg.cholesky(A)
    y2 = np.linalg.solve(L2, b)
    x2 = np.linalg.solve(L2.T, y2)
    print(white("  Method 2 — Cholesky:"))
    print(f"    x = {np.round(x2, 6)},  err = {np.linalg.norm(x2 - x_true):.2e}")

    # Method 3: Conjugate Gradient (from scratch)
    def cg(A, b, tol=1e-10, maxiter=1000):
        x = np.zeros_like(b)
        r = b - A @ x; p = r.copy(); rs_old = r @ r
        for _ in range(maxiter):
            Ap = A @ p; alpha = rs_old / (p @ Ap)
            x += alpha * p; r -= alpha * Ap; rs_new = r @ r
            if np.sqrt(rs_new) < tol: break
            p = r + (rs_new / rs_old) * p; rs_old = rs_new
        return x

    x3 = cg(A, b)
    print(white("  Method 3 — Conjugate Gradient:"))
    print(f"    x = {np.round(x3, 6)},  err = {np.linalg.norm(x3 - x_true):.2e}")

    # Method 4: scipy CG
    try:
        from scipy.sparse.linalg import cg as scipy_cg
        x4, info = scipy_cg(A, b, tol=1e-10)
        print(white("  Method 4 — scipy CG:"))
        print(f"    x = {np.round(x4, 6)},  err = {np.linalg.norm(x4 - x_true):.2e}")
    except ImportError:
        pass
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Method comparison table:"))
    rows = [
        ["Method",     "Cost",    "Memory",   "SPD only?", "Error"],
        ["LU (numpy)", "O(2n³/3)","O(n²)",    "No",        "< 1e-10"],
        ["Cholesky",   "O(n³/3)", "O(n²)",    "Yes",       "< 1e-10"],
        ["CG",         "O(kn)",   "O(n)",     "Yes",       "< 1e-10"],
        ["GMRES",      "O(kn²)",  "O(kn)",    "No",        "< 1e-8"],
    ]
    col_w = [12, 10, 10, 12, 12]
    header = rows[0]; divider = "+" + "+".join("-"*(w+2) for w in col_w) + "+"
    print(grey(divider))
    print("| " + " | ".join(cyan(h).ljust(w + 9) for h, w in zip(header, col_w)) + " |")
    print(grey(divider))
    for row in rows[1:]:
        print("| " + " | ".join(white(c).ljust(w + 9) for c, w in zip(row, col_w)) + " |")
    print(grey(divider))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import loss_curve
        # CG residuals
        residuals = []
        x = np.zeros_like(b); r = b - A @ x; p = r.copy(); rs_old = r @ r
        for _ in range(n):
            Ap = A @ p; alpha = rs_old / (p @ Ap)
            x += alpha * p; r -= alpha * Ap; rs_new = r @ r
            residuals.append(float(np.sqrt(rs_new)))
            if rs_new < 1e-20: break
            p = r + (rs_new / rs_old) * p; rs_old = rs_new
        loss_curve(residuals, title="CG Residual Convergence", ylabel="‖r‖")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        residuals_gd = []
        x_gd = np.zeros_like(b)
        for i in range(50):
            grad = A @ x_gd - b
            lr   = 2.0 / (np.linalg.eigvalsh(A)[-1] + np.linalg.eigvalsh(A)[0])
            x_gd -= lr * grad
            residuals_gd.append(np.linalg.norm(A @ x_gd - b))
        plt.figure(figsize=(7, 4))
        plt.semilogy(residuals_gd, label="Gradient Descent", color='tomato')
        plt.semilogy(residuals + [residuals[-1]]*(50 - len(residuals)), label="CG", color='steelblue')
        plt.xlabel("Iteration"); plt.ylabel("‖Ax-b‖ (log)")
        plt.title("Solver Convergence: GD vs CG"); plt.legend(); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Solving Ax=b: Three Methods", """\
import numpy as np

# 4×4 SPD system
A = np.array([[4., 1., 0., 0.],
              [1., 3., 1., 0.],
              [0., 1., 3., 1.],
              [0., 0., 1., 2.]])
b = np.array([1., 2., 3., 4.])

# 1. Direct: LU via numpy
x1 = np.linalg.solve(A, b)
print("LU solution:", x1)

# 2. Cholesky
L = np.linalg.cholesky(A)
y  = np.linalg.solve(L, b)       # forward sub
x2 = np.linalg.solve(L.T, y)     # backward sub
print("Cholesky solution:", x2)

# 3. Conjugate Gradient (scratch)
def cg(A, b, tol=1e-12):
    x = np.zeros_like(b, dtype=float)
    r = b - A @ x; p = r.copy(); rs = r @ r
    for _ in range(len(b)):
        Ap = A @ p; a = rs / (p @ Ap)
        x += a * p; r -= a * Ap; rs_new = r @ r
        if np.sqrt(rs_new) < tol: break
        p = r + (rs_new / rs) * p; rs = rs_new
    return x

x3 = cg(A, b)
print("CG solution:", x3)
print("All solutions agree:", np.allclose(x1, x2) and np.allclose(x1, x3))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "numpy.linalg.solve is always preferable to np.linalg.inv(A) @ b — never invert!",
        "Cholesky is 2× faster than LU for SPD systems and detects non-SPD matrices.",
        "CG converges in ≤ n steps exactly; preconditioned CG can converge in very few.",
        "For sparse A, iterative solvers avoid the O(n²) memory fill-in of direct methods.",
        "Condition number determines convergence speed of CG: k ≈ O(√κ) iterations.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Block runner
# ─────────────────────────────────────────────────────────────────────────────
def run():
    topics = [
        ("LU Decomposition",         topic_lu),
        ("QR Decomposition",         topic_qr),
        ("Cholesky Decomposition",   topic_cholesky),
        ("Condition Number",         topic_condition_number),
        ("Solving Ax = b",           topic_solving_axb),
    ]
    block_menu("b02", "Matrix Decompositions", topics)
    mark_completed("b02")
