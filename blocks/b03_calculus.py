"""
Block 03 — Calculus & Differentiation
Partial derivatives, Gradient, Jacobian, Hessian,
Chain rule, Taylor series, Lagrange multipliers, Autodiff
"""
import numpy as np

from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold, white,
                       formula, value, section, emph, hint, red, bold_magenta)
from ui.widgets import box, section_header, breadcrumb, nav_bar, table, bar_chart, code_block, panel, pager, hr, print_sparkline
from ui.menu import topic_nav, clear, block_menu, mark_completed
from viz.ascii_plots import scatter, line_plot, multi_line


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Partial Derivatives
# ─────────────────────────────────────────────────────────────────────────────
def topic_partial_derivatives():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Partial Derivatives")
    section_header("PARTIAL DERIVATIVES & SYMPY")

    section_header("1. THEORY")
    print(white("  A partial derivative ∂f/∂xᵢ measures the rate of change of f with respect"))
    print(white("  to xᵢ while all other variables are held constant. It is computed exactly"))
    print(white("  like an ordinary derivative — treat every other variable as a constant."))
    print()
    print(white("  The gradient ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ collects ALL partial"))
    print(white("  derivatives into a vector that points in the direction of steepest increase."))
    print()
    print(white("  Two ways to compute: symbolically (exact, using computer algebra) or"))
    print(white("  numerically (approximate). The finite-difference approximation:"))
    print(white("  ∂f/∂xᵢ ≈ [f(x+hₑᵢ) - f(x-hₑᵢ)] / 2h  (central differences, O(h²) error)"))
    print()
    print(white("  In ML, every gradient computation ultimately reduces to partial derivatives,"))
    print(white("  assembled efficiently by backpropagation via the chain rule."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Limit definition:   ∂f/∂x = lim_{h→0} [f(x+h,y) - f(x,y)] / h"))
    print(formula("  Central difference: ∂f/∂xᵢ ≈ [f(x+hₑᵢ) - f(x-hₑᵢ)] / 2h"))
    print(formula("  Error order:        forward diff O(h),  central diff O(h²)"))
    print(formula("  Gradient vector:    ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  f(x, y) = x²y + sin(xy)"))
    print()
    print(white("  Analytical:"))
    print(white("    ∂f/∂x = 2xy + y·cos(xy)"))
    print(white("    ∂f/∂y = x²  + x·cos(xy)"))
    print()

    def f(x, y):    return x**2 * y + np.sin(x * y)
    def df_dx(x, y): return 2*x*y + y*np.cos(x*y)
    def df_dy(x, y): return x**2 + x*np.cos(x*y)

    x0, y0 = 1.5, 2.0
    h = 1e-5
    fd_x = (f(x0+h, y0) - f(x0-h, y0)) / (2*h)
    fd_y = (f(x0, y0+h) - f(x0, y0-h)) / (2*h)

    print(white(f"  At (x, y) = ({x0}, {y0}):"))
    print(f"  {white('∂f/∂x:')}  analytic = {value(f'{df_dx(x0,y0):.6f}')},  numerical = {value(f'{fd_x:.6f}')},  err = {grey(f'{abs(df_dx(x0,y0)-fd_x):.2e}')}")
    print(f"  {white('∂f/∂y:')}  analytic = {value(f'{df_dy(x0,y0):.6f}')},  numerical = {value(f'{fd_y:.6f}')},  err = {grey(f'{abs(df_dy(x0,y0)-fd_y):.2e}')}")
    print()

    print(white("  Symbolic partial derivatives via sympy:"))
    try:
        import sympy as sp
        x_s, y_s = sp.symbols('x y')
        f_s = x_s**2 * y_s + sp.sin(x_s * y_s)
        dfdx_s = sp.diff(f_s, x_s)
        dfdy_s = sp.diff(f_s, y_s)
        print(f"    {white('∂f/∂x =')} {value(str(dfdx_s))}")
        print(f"    {white('∂f/∂y =')} {value(str(dfdy_s))}")
    except ImportError:
        print(grey("  sympy not installed — analytical forms shown above"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Partial derivative: ∂f/∂x at y=2 (cross-section)"))
    xs = np.linspace(-2, 2, 40)
    vals_f   = [f(x, 2.0) for x in xs]
    vals_dfdx = [df_dx(x, 2.0) for x in xs]
    max_f = max(abs(v) for v in vals_f)
    max_d = max(abs(v) for v in vals_dfdx)
    norm_f = [v / max(max_f, 1e-6) for v in vals_f]
    norm_d = [v / max(max_d, 1e-6) for v in vals_dfdx]
    try:
        multi_line([norm_f, norm_d], labels=["f(x,2)", "∂f/∂x(x,2)"], title="f and its x-derivative at y=2")
    except Exception:
        for nf, nd in zip(norm_f[::5], norm_d[::5]):
            bar_f = "█" * int(abs(nf) * 12)
            bar_d = "░" * int(abs(nd) * 12)
            print(f"  {cyan(bar_f):<16}  {yellow(bar_d)}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        multi_loss([vals_f, vals_dfdx], labels=["f(x,2)", "∂f/∂x(x,2)"],
                   title="f(x,2) and ∂f/∂x at y=2")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-2, 2, 60); ys_m = np.linspace(-2, 2, 60)
        Xg, Yg = np.meshgrid(xs_m, ys_m)
        Zg = Xg**2 * Yg + np.sin(Xg * Yg)
        fig = plt.figure(figsize=(8, 5))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.7)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
        ax.set_title("f(x,y) = x²y + sin(xy)")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Partial Derivatives", """\
import numpy as np

def f(x, y):
    return x**2 * y + np.sin(x * y)

# Central-difference partial derivatives
def partial(f, x, y, wrt='x', h=1e-5):
    if wrt == 'x':
        return (f(x+h, y) - f(x-h, y)) / (2*h)
    return (f(x, y+h) - f(x, y-h)) / (2*h)

x0, y0 = 1.5, 2.0
print("∂f/∂x =", partial(f, x0, y0, 'x'))   # ≈ 2xy + y·cos(xy)
print("∂f/∂y =", partial(f, x0, y0, 'y'))   # ≈ x²  + x·cos(xy)

# Symbolic via sympy
import sympy as sp
x, y = sp.symbols('x y')
fs   = x**2 * y + sp.sin(x * y)
print("∂f/∂x =", sp.diff(fs, x))
print("∂f/∂y =", sp.diff(fs, y))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Partial derivatives hold other variables fixed — conceptually simple, computationally powerful.",
        "Central differences (O(h²)) are far more accurate than forward differences (O(h)).",
        "Gradient vector ∇f is normal (perpendicular) to level curves of f.",
        "Sympy gives exact symbolic derivatives; useful for deriving closed-form gradients.",
        "Numerical gradients are the gold standard for checking backpropagation implementations.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient Vector
# ─────────────────────────────────────────────────────────────────────────────
def topic_gradient():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Gradient Vector")
    section_header("GRADIENT VECTOR")

    section_header("1. THEORY")
    print(white("  The gradient ∇f(x) ∈ ℝⁿ is the vector of all first partial derivatives."))
    print(white("  It points in the direction of steepest ASCENT of f. Its magnitude ‖∇f‖"))
    print(white("  equals the maximum rate of change in any direction."))
    print()
    print(white("  Directional derivative: D_v f(x) = ∇f(x)·v̂ — the rate of change of f"))
    print(white("  in unit direction v̂. Maximised when v̂ = ∇f/‖∇f‖."))
    print()
    print(white("  Level curves (f=c) are always perpendicular to ∇f. This is why gradient"))
    print(white("  descent (moving opposite to ∇f) efficiently minimises f — it descends"))
    print(white("  steepest descent, crossing level curves orthogonally."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Gradient:          ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ"))
    print(formula("  Directional deriv: D_v f = ∇f · v̂   (v̂ unit vector)"))
    print(formula("  Steepest ascent:   achieved when v̂ = ∇f/‖∇f‖"))
    print(formula("  Gradient descent:  xₜ₊₁ = xₜ - α ∇f(xₜ)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def f(xy): return xy[0]**2 + 2*xy[1]**2
    def grad_f(xy): return np.array([2*xy[0], 4*xy[1]])

    pts = [np.array([1.0, 2.0]), np.array([3.0, 1.0]), np.array([-2.0, 1.5])]
    print(white("  f(x,y) = x² + 2y²"))
    print()
    print(f"  {'Point':<20} {'f(x,y)':<12} {'∇f':<20} {'‖∇f‖'}")
    print(grey("  " + "-"*60))
    for pt in pts:
        g = grad_f(pt)
        print(f"  {value(str(np.round(pt,2))):<29} {value(f'{f(pt):.3f}'):<21} {value(str(np.round(g,3))):<29} {value(f'{np.linalg.norm(g):.3f}')}")
    print()

    x0 = np.array([1.0, 2.0])
    print(white(f"  At x₀ = {x0}:  ∇f = {grad_f(x0)}"))
    print(white(f"  Directional derivative in direction [1,0]: {np.dot(grad_f(x0), [1,0]):.3f}"))
    print(white(f"  Directional derivative in direction [0,1]: {np.dot(grad_f(x0), [0,1]):.3f}"))
    v_unit = grad_f(x0) / np.linalg.norm(grad_f(x0))
    print(white(f"  Max rate of change = ‖∇f‖ = {np.linalg.norm(grad_f(x0)):.3f} in direction {np.round(v_unit,3)}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Gradient field ∇f(x,y) for f(x,y) = x² + 2y²  (downsampled)"))
    print()
    grid = [-2, -1, 0, 1, 2]
    for y in reversed(grid):
        row = f"  y={y:+d} "
        for x in grid:
            gx, gy = 2*x, 4*y
            norm = np.sqrt(gx**2 + gy**2)
            if norm < 1e-9:
                arrow = " ·"
            else:
                angle = np.arctan2(gy, gx)
                arrows = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
                idx = int((angle + np.pi) / (2*np.pi) * 8 + 0.5) % 8
                arrow = arrows[idx]
            row += f" {cyan(arrow)}"
        print(row)
    print(grey("\n  Arrows show direction of steepest ascent"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import distribution_plot
        # gradient magnitude along diagonal
        ts = np.linspace(-3, 3, 50)
        pts_diag  = [np.array([t, t]) for t in ts]
        mags = [np.linalg.norm(grad_f(p)) for p in pts_diag]
        distribution_plot(mags, title="‖∇f‖ along diagonal (x=y)", xlabel="t", n_bins=20)
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-3, 3, 200); ys_m = np.linspace(-3, 3, 200)
        Xg, Yg = np.meshgrid(xs_m, ys_m)
        Zg = Xg**2 + 2*Yg**2
        xq = np.linspace(-2.5, 2.5, 10); yq = np.linspace(-2.5, 2.5, 10)
        Xq, Yq = np.meshgrid(xq, yq)
        Uq, Vq = 2*Xq, 4*Yq
        plt.figure(figsize=(7, 5))
        plt.contour(Xg, Yg, Zg, levels=15, cmap='Blues')
        plt.quiver(Xq, Yq, Uq, Vq, color='tomato', alpha=0.7)
        plt.title("Level curves with gradient field ∇f"); plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Gradient Vector", """\
import numpy as np

def f(xy):
    x, y = xy
    return x**2 + 2*y**2

def grad_f(xy):
    x, y = xy
    return np.array([2*x, 4*y])

# Numerical gradient via central differences
def numerical_grad(f, x, h=1e-5):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2*h)
    return g

x0 = np.array([1.0, 2.0])
print("Analytic ∇f:", grad_f(x0))
print("Numerical ∇f:", numerical_grad(f, x0))
print("Max diff:", np.max(np.abs(grad_f(x0) - numerical_grad(f, x0))))

# Directional derivative
v = np.array([1., 1.]) / np.sqrt(2)
print("D_v f:", np.dot(grad_f(x0), v))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "∇f is always perpendicular to level curves — moving along a level curve costs zero gradient.",
        "Steepest descent -∇f/‖∇f‖ is optimal local direction but not globally efficient.",
        "Gradient magnitude ‖∇f‖ = 0 is a necessary condition for a local minimum.",
        "In ML, ∇_θ L(θ) is a gradient w.r.t. millions of parameters, computed via backprop.",
        "Inexact gradients (mini-batch stochastic) still converge — noise can even help escape saddles.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Jacobian Matrix
# ─────────────────────────────────────────────────────────────────────────────
def topic_jacobian():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Jacobian Matrix")
    section_header("JACOBIAN MATRIX")

    section_header("1. THEORY")
    print(white("  The Jacobian generalises the gradient to vector-valued functions f: ℝⁿ → ℝᵐ."))
    print(white("  It is an m×n matrix whose (i,j) entry is ∂fᵢ/∂xⱼ."))
    print()
    print(white("  Geometrically, the Jacobian is the best linear approximation of f near x:"))
    print(white("  f(x+δ) ≈ f(x) + J(x)·δ. Its determinant (for square n=m) measures local"))
    print(white("  volume change — a change of variables in an integral multiplies by |det J|."))
    print()
    print(white("  In deep learning, each layer's transformation has a Jacobian; the full"))
    print(white("  network gradient is a product of Jacobians (chain rule, reverse mode)."))
    print(white("  Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs) are"))
    print(white("  the primitives of forward-mode and reverse-mode autodiff respectively."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Jacobian:   J_f(x)_ij = ∂fᵢ/∂xⱼ   — m×n matrix"))
    print(formula("  Shape rule: f: ℝⁿ → ℝᵐ  ⟹  J ∈ ℝᵐˣⁿ"))
    print(formula("  JVP:        J·v  (forward mode, O(n) per v)"))
    print(formula("  VJP:        vᵀ·J (reverse mode, O(m) per v) — powers backprop"))
    print(formula("  Chain rule: J_{f∘g}(x) = J_f(g(x)) · J_g(x)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def F(xy):
        x, y = xy
        return np.array([x**2 + y, np.sin(x) + y**2])

    def J_analytic(xy):
        x, y = xy
        return np.array([[2*x, 1.],
                         [np.cos(x), 2*y]])

    def J_numerical(F, x, h=1e-5):
        n = len(x); m = len(F(x))
        J = np.zeros((m, n))
        for j in range(n):
            xp, xm = x.copy(), x.copy()
            xp[j] += h; xm[j] -= h
            J[:, j] = (F(xp) - F(xm)) / (2*h)
        return J

    x0 = np.array([1.0, 2.0])
    Ja = J_analytic(x0)
    Jn = J_numerical(F, x0)

    print(white("  f: ℝ² → ℝ²    f(x,y) = [x²+y,  sin(x)+y²]"))
    print(white(f"\n  At (x,y) = {x0}:"))
    print(white("\n  Analytic Jacobian:"))
    for row in Ja:
        print(f"    {np.round(row, 4)}")
    print(white("\n  Numerical Jacobian:"))
    for row in Jn:
        print(f"    {np.round(row, 4)}")
    print(green(f"\n  ✓ Max |J_analytic - J_numerical| = {np.max(np.abs(Ja - Jn)):.2e}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Jacobian shape rules:"))
    print()
    rows_data = [
        ["Function type",      "Input dim", "Output dim", "Jacobian shape"],
        ["Scalar → scalar",    "1",         "1",          "1×1  (= derivative)"],
        ["Vector → scalar",    "n",         "1",          "1×n  (= row gradient)"],
        ["Scalar → vector",    "1",         "m",          "m×1  (= column vector)"],
        ["Vector → vector",    "n",         "m",          "m×n  (= full Jacobian)"],
        ["NN layer (batch B)", "Bxn",       "Bxm",        "m×n per sample"],
    ]
    col_w = [22, 12, 12, 22]
    divider = "  +" + "+".join("-"*(w+2) for w in col_w) + "+"
    print(grey(divider))
    print("  | " + " | ".join(cyan(c).ljust(w+9) for c, w in zip(rows_data[0], col_w)) + " |")
    print(grey(divider))
    for row in rows_data[1:]:
        print("  | " + " | ".join(white(c).ljust(w+9) for c, w in zip(row, col_w)) + " |")
    print(grey(divider))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import histogram
        # distribution of Jacobian entries over a grid
        pts = [(x, y) for x in np.linspace(-2, 2, 10) for y in np.linspace(-2, 2, 10)]
        j_norms = [np.linalg.norm(J_analytic(np.array(p)), 'fro') for p in pts]
        histogram(j_norms, title="Distribution of ‖J‖_F over grid", xlabel="Frobenius norm", bins=15)
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-2, 2, 30); ys_m = np.linspace(-2, 2, 30)
        Xg, Yg = np.meshgrid(xs_m, ys_m)
        Jnorm = np.array([[np.linalg.norm(J_analytic(np.array([x, y])), 'fro')
                           for x in xs_m] for y in ys_m])
        plt.figure(figsize=(7, 5))
        cf = plt.contourf(Xg, Yg, Jnorm, levels=20, cmap='plasma')
        plt.colorbar(cf); plt.title("‖J(x,y)‖_F  for f(x,y)=[x²+y, sin(x)+y²]")
        plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Jacobian Matrix", """\
import numpy as np

def F(xy):
    x, y = xy
    return np.array([x**2 + y, np.sin(x) + y**2])

def jacobian(F, x, h=1e-5):
    \"\"\"Numerical Jacobian via central differences.\"\"\"
    n = len(x); m = len(F(x)); J = np.zeros((m, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += h; xm[j] -= h
        J[:, j] = (F(xp) - F(xm)) / (2*h)
    return J

x0 = np.array([1.0, 2.0])
J  = jacobian(F, x0)
print("J =\\n", J)
print("Shape:", J.shape)   # (2, 2)

# Linear approximation: f(x₀ + δ) ≈ f(x₀) + J·δ
delta = np.array([0.01, 0.01])
approx = F(x0) + J @ delta
exact  = F(x0 + delta)
print("Approx:", approx)
print("Exact: ", exact)
print("Error: ", np.linalg.norm(approx - exact))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Jacobian = best linear approximation of f near x: f(x+δ) ≈ f(x) + J·δ.",
        "VJP (row vector × J) is the core primitive of reverse-mode autodiff (backprop).",
        "JVP (J × column vector) is the core primitive of forward-mode autodiff.",
        "For loss function L: ℝⁿ→ℝ, the Jacobian is just the gradient (1×n row vector).",
        "Jacobian-free Newton-Krylov methods approximate J·v via directional finite differences.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hessian Matrix & Curvature
# ─────────────────────────────────────────────────────────────────────────────
def topic_hessian():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Hessian Matrix")
    section_header("HESSIAN MATRIX & CURVATURE")

    section_header("1. THEORY")
    print(white("  The Hessian H(x) is the n×n matrix of second partial derivatives of a scalar"))
    print(white("  function f: ℝⁿ → ℝ.  H_ij = ∂²f / ∂xᵢ∂xⱼ.  It captures curvature."))
    print()
    print(white("  The Hessian is symmetric (Schwarz's theorem: mixed partials commute) and its"))
    print(white("  eigenvalues encode the curvature in each eigen-direction:"))
    print(white("    all λ > 0 → positive definite → local minimum"))
    print(white("    all λ < 0 → negative definite → local maximum"))
    print(white("    mixed signs              → saddle point"))
    print()
    print(white("  In second-order optimisation (Newton's method, L-BFGS), the Hessian or its"))
    print(white("  inverse is used to rescale gradient steps. The Fisher information matrix"))
    print(white("  (in natural gradient methods) is the expected Hessian of log-likelihood."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Definition:      H_f(x)_ij = ∂²f / ∂xᵢ∂xⱼ"))
    print(formula("  Symmetry:        H = Hᵀ (Schwarz theorem)"))
    print(formula("  Taylor 2nd order:f(x+δ) ≈ f(x) + ∇fᵀδ + ½δᵀHδ"))
    print(formula("  SPD ↔ local min: ∇f=0 and H ≻ 0 → x is a local minimum"))
    print(formula("  Saddle:          ∇f=0 and H has both + and − eigenvalues"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def f(xy): return xy[0]**2 - xy[1]**2
    def grad_f(xy): return np.array([2*xy[0], -2*xy[1]])
    def hess_f(): return np.array([[2., 0.], [0., -2.]])

    x0 = np.array([0., 0.])
    H  = hess_f()
    eigs = np.linalg.eigvalsh(H)

    print(white("  f(x,y) = x² − y²  (saddle function)"))
    print()
    print(white(f"  At origin (0,0):"))
    print(white(f"  ∇f = {grad_f(x0)}  (zero — critical point)"))
    print(white(f"\n  Hessian H:"))
    for row in H:
        print(f"    {row}")
    print(white(f"\n  Eigenvalues of H: {eigs}"))
    print(white(f"  Mixed signs → {red('saddle point')}"))
    print()

    # Numerical Hessian
    def num_hessian(f, x, h=1e-4):
        n = len(x); H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ep, em = x.copy(), x.copy()
                ep[i] += h; ep[j] += h
                em[i] += h; em[j] -= h
                fp, fm = x.copy(), x.copy()
                fp[i] -= h; fp[j] += h
                fm[i] -= h; fm[j] -= h
                H[i,j] = (f(ep)-f(em)-f(fp)+f(fm)) / (4*h**2)
        return H

    Hn = num_hessian(f, x0)
    print(white("  Numerical Hessian:"))
    for row in Hn:
        print(f"    {np.round(row, 4)}")
    print(green(f"  ✓ max diff = {np.max(np.abs(H - Hn)):.2e}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Second-order landscape around critical point:"))
    grid = np.linspace(-2, 2, 9)
    print(cyan("  f(x,y) = x² - y²                 Curvature type by eigenvalue sign"))
    print()
    for y in reversed(grid[::2]):
        row = f"  y={y:+.1f}  "
        for x in grid[::2]:
            val = f(np.array([x, y]))
            if abs(val) < 0.5:
                ch = green("·")
            elif val > 0:
                ch = yellow("+")
            else:
                ch = red("-")
            row += ch + " "
        print(row)
    print(grey("\n  + = positive region   - = negative region   · ≈ zero"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import eigenvalue_spectrum
        eigenvalue_spectrum(eigs.tolist(), title="Eigenvalues of Hessian at origin")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-2, 2, 50); ys_m = np.linspace(-2, 2, 50)
        Xg, Yg = np.meshgrid(xs_m, ys_m); Zg = Xg**2 - Yg**2
        fig = plt.figure(figsize=(7, 5))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xg, Yg, Zg, cmap='coolwarm', alpha=0.8)
        ax.scatter([0], [0], [0], s=100, c='red', zorder=5, label='Saddle point')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
        ax.set_title("f(x,y) = x² − y²  (Saddle)"); ax.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Hessian Matrix", """\
import numpy as np

def f(xy):
    return xy[0]**2 - xy[1]**2

def numerical_hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ep = x.copy(); ep[i] += h; ep[j] += h
            em = x.copy(); em[i] += h; em[j] -= h
            fp = x.copy(); fp[i] -= h; fp[j] += h
            fm = x.copy(); fm[i] -= h; fm[j] -= h
            H[i,j] = (f(ep) - f(em) - f(fp) + f(fm)) / (4*h**2)
    return H

x0 = np.array([0., 0.])
H  = numerical_hessian(f, x0)
print("Hessian:\\n", H)

eigs = np.linalg.eigvalsh(H)
print("Eigenvalues:", eigs)
if np.all(eigs > 0):
    print("→ Positive definite: local minimum")
elif np.all(eigs < 0):
    print("→ Negative definite: local maximum")
else:
    print("→ Indefinite: saddle point")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Hessian encodes curvature; eigenvalues tell whether a critical point is min, max, or saddle.",
        "In n dimensions, a random critical point has ~50% chance of each eigenvalue being positive.",
        "H ≻ 0 (positive definite) guarantees unique local minimum in a convex neighbourhood.",
        "Hessian is expensive (O(n²) entries); L-BFGS approximates H⁻¹ without computing it.",
        "Gradient clipping and loss landscape sharpness are Hessian-related concepts in deep learning.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chain Rule
# ─────────────────────────────────────────────────────────────────────────────
def topic_chain_rule():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Chain Rule")
    section_header("CHAIN RULE — FOUNDATION OF BACKPROPAGATION")

    section_header("1. THEORY")
    print(white("  The chain rule computes derivatives of composed functions. For h = f ∘ g:"))
    print(white("  dh/dx = (dh/df) · (df/dx) — multiply local derivatives along the path."))
    print()
    print(white("  In neural networks, the forward pass computes output from input through"))
    print(white("  many function compositions. Backpropagation applies the chain rule in"))
    print(white("  REVERSE (reverse-mode AD), accumulating gradient from output back to input."))
    print()
    print(white("  Key insight: reverse mode computes ∂L/∂x for ALL inputs x in O(cost of"))
    print(white("  one forward pass). This is why backprop is so efficient — one backward pass"))
    print(white("  gives gradients for all weights simultaneously."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Scalar chain rule:   dh/dx = (dh/du) · (du/dx)"))
    print(formula("  Vector chain rule:   ∂h/∂x = (∂h/∂u) · (∂u/∂x) = J_u · J_x"))
    print(formula("  Multi-variable:      ∂z/∂x = Σₖ (∂z/∂yₖ)(∂yₖ/∂x)"))
    print(formula("  Backprop layer l:    δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ'(zˡ)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  h(x) = sin(x²)  =  f(g(x))  with  g(x)=x², f(u)=sin(u)"))
    print()
    print(white("  Step-by-step:"))
    print(white("    g(x) = x²          →   g'(x) = 2x"))
    print(white("    f(u) = sin(u)       →   f'(u) = cos(u)"))
    print(white("    h'(x) = f'(g(x))·g'(x) = cos(x²) · 2x"))
    print()
    def dh_dx_analytic(x): return np.cos(x**2) * 2 * x
    def h(x): return np.sin(x**2)

    for x0 in [0.5, 1.0, 2.0]:
        an = dh_dx_analytic(x0)
        nu = (h(x0 + 1e-6) - h(x0 - 1e-6)) / 2e-6
        print(f"  x = {x0:.1f}:  analytic = {value(f'{an:.6f}')},  numerical = {value(f'{nu:.6f}')},  err = {grey(f'{abs(an-nu):.2e}')}")
    print()

    print(white("  Two-variable chain rule (neural network layer):"))
    print(white("  y = W·x + b,  z = σ(y)  →  ∂z/∂x = diag(σ'(y)) · W"))
    W  = np.array([[1., 2.], [3., 4.]])
    x  = np.array([0.5, -0.5])
    b  = np.array([0.1, 0.2])
    y  = W @ x + b
    def sigmoid(v):        return 1 / (1 + np.exp(-v))
    def sigmoid_prime(v):  return sigmoid(v) * (1 - sigmoid(v))
    dz_dx = np.diag(sigmoid_prime(y)) @ W
    print(white(f"  W = {W},  x = {x}"))
    print(white(f"  y = Wx+b = {np.round(y,4)},  σ(y) = {np.round(sigmoid(y),4)}"))
    print(white(f"  ∂z/∂x (Jacobian) =\n  {np.round(dz_dx, 4)}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Computation graph for h(x) = sin(x²):"))
    print()
    print(f"  {cyan('x')}")
    print(f"  │")
    print(f"  ▼")
    print(f"  {yellow('g = x²')}    (local gradient: dg/dx = 2x)")
    print(f"  │")
    print(f"  ▼")
    print(f"  {yellow('f = sin(g)')} (local gradient: df/dg = cos(g))")
    print(f"  │")
    print(f"  ▼")
    print(f"  {green('h = sin(x²)')}")
    print()
    print(cyan("  Forward pass  ─────────────────────────▶"))
    print(red("  Backward pass ◀─────────────────────────"))
    print()
    print(white("  Chain: dh/dx = df/dg · dg/dx = cos(x²) · 2x"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        xs = np.linspace(-np.pi, np.pi, 80)
        hs = np.sin(xs**2)
        dhs = np.cos(xs**2) * 2 * xs
        multi_loss([hs.tolist(), dhs.tolist()], labels=["h(x)=sin(x²)", "h'(x)=2x·cos(x²)"],
                   title="Function and its chain-rule derivative")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-np.pi, np.pi, 300)
        plt.figure(figsize=(8, 4))
        plt.plot(xs_m, np.sin(xs_m**2), label="h(x) = sin(x²)", color='steelblue')
        plt.plot(xs_m, np.cos(xs_m**2)*2*xs_m, label="h'(x) = 2x·cos(x²)", color='tomato', ls='--')
        plt.axhline(0, color='grey', lw=0.5); plt.legend()
        plt.title("Chain Rule: h=sin(x²)"); plt.xlabel("x"); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Chain Rule & Backpropagation", """\
import numpy as np

# Scalar chain rule
def h(x):    return np.sin(x**2)
def dh_dx(x): return np.cos(x**2) * 2 * x     # chain rule

x0 = 1.5
print(f"h({x0}) = {h(x0):.6f}")
print(f"h'({x0}) = {dh_dx(x0):.6f}")

# Two-layer neural network (manual backprop)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x):    return np.maximum(0, x)
def drelu(x):   return (x > 0).astype(float)

# Forward pass
W1 = np.random.randn(4, 3); b1 = np.zeros(4)
W2 = np.random.randn(2, 4); b2 = np.zeros(2)
x  = np.array([1., 2., 3.])

z1 = W1 @ x + b1; a1 = relu(z1)
z2 = W2 @ a1 + b2; a2 = sigmoid(z2)

# Backward pass (chain rule)
dL_da2 = a2 - np.array([1., 0.])    # example MSE gradient
dL_dz2 = dL_da2 * sigmoid(z2) * (1 - sigmoid(z2))
dL_dW2 = np.outer(dL_dz2, a1)
dL_da1 = W2.T @ dL_dz2
dL_dz1 = dL_da1 * drelu(z1)
dL_dW1 = np.outer(dL_dz1, x)

print("Gradient of W2:", dL_dW2.shape)
print("Gradient of W1:", dL_dW1.shape)
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Chain rule = multiply local derivatives — backprop is just systematic application of this.",
        "Reverse mode (backprop) computes gradients of all n parameters in O(1 forward pass cost).",
        "Forward mode computes directional derivative Jv in O(1 forward pass) — useful for small n.",
        "Vanishing gradients occur when products of local derivatives are < 1 repeatedly.",
        "Residual connections (ResNets) ensure chain-rule product includes identity, preventing vanishing.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Taylor Series
# ─────────────────────────────────────────────────────────────────────────────
def topic_taylor_series():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Taylor Series")
    section_header("TAYLOR SERIES APPROXIMATIONS")

    section_header("1. THEORY")
    print(white("  Taylor series expresses a smooth function as an infinite polynomial around"))
    print(white("  a point a: f(x) = f(a) + f'(a)(x-a) + f''(a)/2!(x-a)² + ..."))
    print()
    print(white("  In optimisation, quadratic approximation (order 2) is the basis of Newton's"))
    print(white("  method. In physics and engineering, linearisation (order 1) gives local models."))
    print()
    print(white("  Key insight: within radius of convergence, truncated Taylor series are"))
    print(white("  excellent local approximations. The error of an n-term approximation is"))
    print(white("  O((x-a)^{n+1}) — each additional term buys one more order of accuracy."))
    print()
    print(white("  Famous series: sin(x)=x-x³/6+x⁵/120-...,  eˣ=1+x+x²/2+...,  ln(1+x)=x-x²/2+..."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Taylor series:   f(x) = Σₙ₌₀^∞  f⁽ⁿ⁾(a)/n! · (x-a)ⁿ"))
    print(formula("  Maclaurin:       a = 0  →  f(x) = Σₙ f⁽ⁿ⁾(0)/n! · xⁿ"))
    print(formula("  sin(x) =  x − x³/3! + x⁵/5! − x⁷/7! + ..."))
    print(formula("  eˣ     =  1 + x + x²/2! + x³/3! + ..."))
    print(formula("  ln(1+x)=  x − x²/2 + x³/3 − ... (|x| < 1)"))
    print(formula("  Newton step: x* ≈ x − [f''(x)]⁻¹ f'(x)  (2nd-order Taylor minimisation)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    from math import factorial
    def taylor_sin(x, n_terms):
        result = 0
        for k in range(n_terms):
            result += ((-1)**k * x**(2*k+1)) / factorial(2*k+1)
        return result

    x_test = 1.0
    print(white(f"  sin({x_test}) = {np.sin(x_test):.8f}"))
    print()
    print(f"  {'Order':<8} {'Approximation':<20} {'Error'}")
    print(grey("  " + "-"*45))
    for n in range(1, 8, 2):
        approx = taylor_sin(x_test, (n+1)//2)
        err    = abs(approx - np.sin(x_test))
        print(f"  {value(str(n)):<17} {value(f'{approx:.8f}'):<29} {grey(f'{err:.2e}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  sin(x) Taylor approximations at x = π/4:"))
    xs = np.linspace(0, 2*np.pi, 60)
    true_sin = np.sin(xs)
    for n_terms in [1, 2, 3, 4]:
        order = 2*n_terms - 1
        approx = np.array([taylor_sin(x, n_terms) for x in xs])
        errors = np.abs(approx - true_sin)
        mean_err = np.mean(errors[xs <= np.pi])
        bar_len = max(1, min(30, int(30 - np.log10(max(mean_err, 1e-10)) * 4)))
        bar_str = "█" * bar_len
        print(f"  Order {order}: {green(bar_str):<40} mean err = {value(f'{mean_err:.1e}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        xs_p = np.linspace(-np.pi, np.pi, 100)
        true_s = np.sin(xs_p).tolist()
        approx_series = []
        labels_s = ["sin(x)"]
        for n_terms in [1, 2, 3, 4]:
            approx_series.append([taylor_sin(x, n_terms) for x in xs_p])
            labels_s.append(f"Order {2*n_terms-1}")
        multi_loss([true_s] + approx_series, labels=labels_s,
                   title="sin(x) Taylor Approximations")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-np.pi, np.pi, 300)
        colors = ['tomato', 'orange', 'gold', 'limegreen']
        plt.figure(figsize=(8, 5))
        plt.plot(xs_m, np.sin(xs_m), 'k-', lw=2, label='sin(x)')
        for n_terms, col in zip([1, 2, 3, 4], colors):
            approx = np.array([taylor_sin(x, n_terms) for x in xs_m])
            approx = np.clip(approx, -5, 5)
            plt.plot(xs_m, approx, '--', color=col, label=f'Order {2*n_terms-1}', alpha=0.8)
        plt.ylim(-2, 2); plt.legend(); plt.title("sin(x) Taylor Approximations")
        plt.xlabel("x"); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Taylor Series Approximation", """\
import numpy as np
from math import factorial

def taylor_sin(x, n_terms):
    \"\"\"Taylor approximation of sin(x) around 0.\"\"\"
    return sum((-1)**k * x**(2*k+1) / factorial(2*k+1)
               for k in range(n_terms))

def taylor_exp(x, n_terms):
    \"\"\"Taylor approximation of e^x around 0.\"\"\"
    return sum(x**k / factorial(k) for k in range(n_terms))

# Test accuracy
import numpy as np
for x in [0.5, 1.0, 2.0]:
    for n in [3, 5, 7]:
        approx = taylor_sin(x, (n+1)//2)
        print(f"sin({x}) order-{n}: approx={approx:.6f}  err={abs(approx-np.sin(x)):.2e}")

# Newton's method as quadratic Taylor optimisation
# f(x) = x^2 - 2  → minimum at x=0, roots at ±√2
def f(x):  return x**2 - 2
def fp(x): return 2*x
def fpp(x): return 2.0

x = 3.0
for i in range(5):
    x = x - fp(x) / fpp(x)    # Newton step from 2nd-order Taylor
    print(f"  iter {i+1}: x = {x:.8f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Each additional Taylor term buys O(h^{n+1}) accuracy — rapid improvement near expansion point.",
        "Gradient descent = 1st-order Taylor min; Newton = 2nd-order Taylor min (quadratic convergence).",
        "Taylor remainder theorem: |error| ≤ M|x-a|^{n+1}/(n+1)! where M bounds the (n+1)th derivative.",
        "Softmax, sigmoid, and tanh are often approximated via Taylor in hardware (TPUs, FPGAs).",
        "Radius of convergence matters: sin/cos converge everywhere; ln(1+x) only for |x| < 1.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lagrange Multipliers
# ─────────────────────────────────────────────────────────────────────────────
def topic_lagrange_multipliers():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Lagrange Multipliers")
    section_header("LAGRANGE MULTIPLIERS")

    section_header("1. THEORY")
    print(white("  Lagrange multipliers solve constrained optimisation: optimise f(x) subject"))
    print(white("  to g(x) = 0. The key insight: at the optimum, ∇f and ∇g are parallel."))
    print(white("  Otherwise we could move along the constraint surface to improve f."))
    print()
    print(white("  The Lagrangian L(x,λ) = f(x) − λg(x) combines objective and constraint."))
    print(white("  Setting ∂L/∂x = 0 and ∂L/∂λ = 0 (= g(x) = 0) gives the KKT conditions."))
    print()
    print(white("  In ML: SVMs use Lagrange multipliers; Lagrangian duality powers many convex"))
    print(white("  optimisation solvers. Policy gradient and constrained RL use Lagrangian relaxation."))
    print(white("  Maximum entropy models (MaxEnt) find distributions satisfying moment constraints."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Lagrangian:   L(x,λ) = f(x) − λ g(x)"))
    print(formula("  Stationarity: ∂L/∂x = 0  →  ∇f(x) = λ ∇g(x)"))
    print(formula("  Feasibility:  ∂L/∂λ = 0  →  g(x) = 0"))
    print(formula("  Inequality:   KKT: λ ≥ 0, g(x) ≤ 0, λg(x) = 0  (complementary slackness)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  Maximise f(x,y) = xy  subject to  x + y = 1"))
    print()
    print(white("  Lagrangian: L = xy − λ(x + y − 1)"))
    print(white("  ∂L/∂x = y − λ = 0   →  y = λ"))
    print(white("  ∂L/∂y = x − λ = 0   →  x = λ"))
    print(white("  ∂L/∂λ = −(x+y−1)=0  →  x + y = 1"))
    print(white("  From x = y and x+y=1:  x = y = 1/2"))
    print(white("  Maximum value: f(1/2, 1/2) = 1/4"))
    print()

    x_sol, y_sol = 0.5, 0.5
    lam = y_sol  # = x_sol
    print(green(f"  ✓ Solution: x* = {x_sol}, y* = {y_sol}, λ* = {lam}"))
    print(green(f"  ✓ f(x*,y*) = {x_sol * y_sol:.4f}"))
    print(green(f"  ✓ Constraint: x+y = {x_sol+y_sol:.4f} = 1 ✓"))
    print()

    print(white("  Verify it's a maximum by scanning constraint:"))
    ts = np.linspace(0, 1, 50)
    f_vals = [t * (1 - t) for t in ts]
    max_idx = int(np.argmax(f_vals))
    print(white(f"  Numerical max: x={ts[max_idx]:.3f}, f={max(f_vals):.6f} ✓"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Level curves of f(x,y)=xy and constraint x+y=1:"))
    grid = np.linspace(-0.5, 1.5, 9)
    print()
    for y in reversed(grid):
        row = f"  y={y:+.2f}  "
        for x in grid:
            on_constraint = abs(x + y - 1) < 0.2
            f_val = x * y
            if abs(x - 0.5) < 0.1 and abs(y - 0.5) < 0.1:
                row += green("★") + " "
            elif on_constraint:
                row += cyan("·") + " "
            elif f_val > 0.15:
                row += yellow("+") + " "
            else:
                row += grey(".") + " "
        print(row)
    print(grey("\n  ★ = solution  · = constraint line  + = high objective"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import scatter_plot
        ts_p = np.linspace(0, 1, 60)
        f_vals_p = [t * (1 - t) for t in ts_p]
        scatter_plot(ts_p.tolist(), f_vals_p, title="f(x, 1-x) = x(1-x) along constraint",
                     xlabel="x", ylabel="f")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-0.2, 1.2, 100); ys_m = np.linspace(-0.2, 1.2, 100)
        Xg, Yg = np.meshgrid(xs_m, ys_m); Zg = Xg * Yg
        plt.figure(figsize=(6, 5))
        plt.contour(Xg, Yg, Zg, levels=15, cmap='Blues')
        plt.plot([0, 1], [1, 0], 'r-', lw=2, label='x+y=1')
        plt.scatter([0.5], [0.5], s=100, c='gold', zorder=5, label='Optimal (½,½)')
        plt.xlabel("x"); plt.ylabel("y"); plt.legend()
        plt.title("Maximise xy s.t. x+y=1"); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Lagrange Multipliers", """\
import numpy as np
from scipy.optimize import minimize

# Maximise f(x,y) = xy  s.t.  x + y = 1
# (negate for minimisation)
f = lambda xy: -xy[0] * xy[1]
constraint = {'type': 'eq', 'fun': lambda xy: xy[0] + xy[1] - 1}
result = minimize(f, [0.3, 0.7], constraints=constraint)
print("Solution:", result.x)
print("Max value:", -result.fun)

# Manual Lagrange from scratch
# ∇f = λ∇g  →  [y, x] = λ[1, 1]  →  x = y = λ
# x + y = 1  →  x = y = 0.5, λ = 0.5
x_opt, y_opt, lam = 0.5, 0.5, 0.5
print(f"\\nAnalytic: x*={x_opt}, y*={y_opt}, λ*={lam}")
print(f"∇f = [{y_opt}, {x_opt}]  =  {lam}·[1,1] = λ∇g ✓")
print(f"Constraint: {x_opt} + {y_opt} = 1 ✓")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "At the constrained optimum, ∇f = λ∇g — objective and constraint gradients are parallel.",
        "Lagrange multiplier λ measures how much the optimum improves if the constraint is relaxed.",
        "SVMs maximise margin (a Lagrangian problem); support vectors are the active constraintpoints.",
        "KKT conditions generalise Lagrange to inequality constraints with complementary slackness.",
        "MaxEnt models: maximise entropy subject to moment constraints — solved via Lagrangians.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Automatic Differentiation
# ─────────────────────────────────────────────────────────────────────────────
def topic_autodiff():
    clear()
    breadcrumb("mlmath", "Calculus & Differentiation", "Automatic Differentiation")
    section_header("AUTOMATIC DIFFERENTIATION")

    section_header("1. THEORY")
    print(white("  Automatic differentiation (autodiff) computes derivatives exactly (to machine"))
    print(white("  precision) by mechanically applying the chain rule to elementary operations."))
    print()
    print(white("  Two modes:"))
    print(white("  • Forward mode: propagates (value, derivative) pairs — dual numbers."))
    print(white("    Cost: one forward pass per input dimension. Efficient when n << m."))
    print(white("  • Reverse mode: records computation graph (Wengert tape), then propagates"))
    print(white("    adjoint (gradient) backwards. Cost: one pass for all inputs. Efficient"))
    print(white("    when m = 1 (typical loss function) — this IS backpropagation."))
    print()
    print(white("  NOT the same as:"))
    print(white("  • Symbolic differentiation (algebraic expression manipulation — combinatorial)"))
    print(white("  • Numerical differentiation (finite differences — O(h²) error, 2 evals/input)"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Dual number:  (a + bε),  ε² = 0,  ε ≠ 0"))
    print(formula("  Forward eval: f(a+bε) = f(a) + f'(a)b·ε"))
    print(formula("  Composition:  f(g(a+ε)) = f(g(a)) + f'(g(a))g'(a)·ε"))
    print(formula("  Reverse mode: adjoint ā = ∂L/∂a,  propagated backwards"))
    print(formula("  Operations:   +,-,*,/ and all elementwise functions"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  Dual number implementation:"))
    print()

    class Dual:
        """Forward-mode AD via dual numbers."""
        def __init__(self, val, deriv=0.0):
            self.val = val; self.deriv = deriv
        def __add__(self, other):
            if isinstance(other, (int, float)): other = Dual(other)
            return Dual(self.val + other.val, self.deriv + other.deriv)
        def __radd__(self, other): return self.__add__(Dual(other))
        def __mul__(self, other):
            if isinstance(other, (int, float)): other = Dual(other)
            return Dual(self.val * other.val,
                        self.val * other.deriv + self.deriv * other.val)
        def __rmul__(self, other): return self.__mul__(Dual(other))
        def __pow__(self, n):
            return Dual(self.val**n, n * self.val**(n-1) * self.deriv)
        def sin(self):
            return Dual(np.sin(self.val), np.cos(self.val) * self.deriv)
        def cos(self):
            return Dual(np.cos(self.val), -np.sin(self.val) * self.deriv)
        def __repr__(self):
            return f"Dual({self.val:.6f} + {self.deriv:.6f}ε)"

    def f_dual(x):
        return x**3 + 2*x**2 + x

    def f_np(x): return x**3 + 2*x**2 + x
    def df_analytic(x): return 3*x**2 + 4*x + 1

    x_val = 3.0
    x_d = Dual(x_val, 1.0)   # seed derivative = 1
    result = f_dual(x_d)

    print(white(f"  f(x) = x³ + 2x² + x   at x = {x_val}"))
    print(white(f"  f'(x) = 3x² + 4x + 1  analytically = {df_analytic(x_val)}"))
    print()
    print(white(f"  Dual number: x = {x_d}"))
    print(white(f"  f(x) = {result}"))
    print(green(f"  ✓ Derivative = {result.deriv}  (analytic: {df_analytic(x_val)})"))
    print()

    # Check at multiple points
    print(white("  Verifying at multiple points:"))
    for x0 in [0.0, 1.0, 2.0, 3.0, -1.0]:
        d = f_dual(Dual(x0, 1.0))
        an = df_analytic(x0)
        nu = (f_np(x0 + 1e-6) - f_np(x0 - 1e-6)) / 2e-6
        print(f"  x={x0:+.1f}:  dual={value(f'{d.deriv:.6f}')},  analytic={value(f'{an:.6f}')},  numerical={value(f'{nu:.6f}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Forward-mode vs Reverse-mode comparison:"))
    print()
    print(white("  Forward mode (Dual numbers):"))
    print(f"  {cyan('Input')}: x = a + 1·ε")
    print(f"   ↓  g = x²     → {yellow('(a², 2a·ε)')}")
    print(f"   ↓  h = sin(g) → {yellow('(sin(a²), cos(a²)·2a·ε)')}")
    print(f"  {green('Output')}: f = h.val,  f\' = h.deriv")
    print()
    print(white("  Reverse mode (Backprop):"))
    print(f"  {cyan('Forward')}: x → g=x² → h=sin(g) → loss L")
    print(f"  {red('Backward')}: L̄=1 → h̄=1 → ḡ=cos(g)·h̄ → x̄=2x·ḡ")
    print(f"  {green('Result')}: ∂L/∂x = x̄")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        xs_p = np.linspace(-3, 3, 80)
        f_vals_p = [f_np(x) for x in xs_p]
        df_vals_p = [df_analytic(x) for x in xs_p]
        dual_vals_p = [f_dual(Dual(x, 1.0)).deriv for x in xs_p]
        multi_loss([f_vals_p, df_vals_p, dual_vals_p],
                   labels=["f(x)", "f'(x) analytic", "f'(x) dual"],
                   title="Autodiff (Dual Numbers) vs Analytic Derivative")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-3, 3, 200)
        fv   = np.array([f_np(x) for x in xs_m])
        df_a = np.array([df_analytic(x) for x in xs_m])
        df_d = np.array([f_dual(Dual(x, 1.0)).deriv for x in xs_m])
        plt.figure(figsize=(8, 4))
        plt.plot(xs_m, fv,   label="f(x)=x³+2x²+x",     color='steelblue')
        plt.plot(xs_m, df_a, label="f'(x) analytic",     color='tomato',    lw=2)
        plt.plot(xs_m, df_d, label="f'(x) dual numbers", color='gold', ls='--', lw=1.5)
        plt.legend(); plt.title("Autodiff: Dual Numbers")
        plt.axhline(0, color='grey', lw=0.5); plt.xlabel("x"); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Automatic Differentiation — Dual Numbers", """\
import numpy as np

class Dual:
    \"\"\"Forward-mode AD via dual numbers: (value, derivative).\"\"\"
    def __init__(self, val, deriv=0.0):
        self.val = float(val); self.deriv = float(deriv)
    def __add__(self, o):
        if not isinstance(o, Dual): o = Dual(o)
        return Dual(self.val + o.val, self.deriv + o.deriv)
    def __radd__(self, o):  return self.__add__(o)
    def __mul__(self, o):
        if not isinstance(o, Dual): o = Dual(o)
        return Dual(self.val*o.val, self.val*o.deriv + self.deriv*o.val)
    def __rmul__(self, o):  return self.__mul__(o)
    def __pow__(self, n):
        return Dual(self.val**n, n*self.val**(n-1)*self.deriv)
    def sin(self):
        return Dual(np.sin(self.val), np.cos(self.val)*self.deriv)

# Compute f(x) = x^3 + 2x^2 + x and f'(x) simultaneously
def f(x): return x**3 + 2*x**2 + x

x = Dual(3.0, 1.0)    # seed: dx/dx = 1
y = f(x)
print(f"f(3)  = {y.val}")          # value
print(f"f'(3) = {y.deriv}")        # derivative (analytic: 3*9+12+1=40)

# Using JAX for production autodiff (pip install jax)
try:
    import jax
    import jax.numpy as jnp
    grad_f = jax.grad(lambda x: x**3 + 2*x**2 + x)
    print("JAX grad at 3:", grad_f(3.0))
except ImportError:
    print("(JAX not installed — using dual numbers above)")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Forward mode (dual numbers) computes (f, f') in one pass — exact to machine precision.",
        "Reverse mode (backprop) computes ∂L/∂all_inputs in one backward pass — O(forward cost).",
        "Numerical differentiation (finite diff) introduces truncation error and costs 2 evals/param.",
        "JAX, PyTorch, TensorFlow all use reverse-mode AD under the hood for neural network training.",
        "Second-order autodiff: differentiate through the backward pass to get Hessian-vector products.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Block runner
# ─────────────────────────────────────────────────────────────────────────────
def run():
    topics = [
        ("Partial Derivatives",      topic_partial_derivatives),
        ("Gradient Vector",          topic_gradient),
        ("Jacobian Matrix",          topic_jacobian),
        ("Hessian & Curvature",      topic_hessian),
        ("Chain Rule & Backprop",    topic_chain_rule),
        ("Taylor Series",            topic_taylor_series),
        ("Lagrange Multipliers",     topic_lagrange_multipliers),
        ("Automatic Differentiation",topic_autodiff),
    ]
    block_menu("b03", "Calculus & Differentiation", topics)
    mark_completed("b03")
