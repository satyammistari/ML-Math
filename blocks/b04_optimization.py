"""
Block 04 — Optimisation
GD, SGD, Momentum, Adam, LR Schedulers, Convexity,
Saddle Points, Newton's Method, CG, L-BFGS
"""
import numpy as np

from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold, white,
                       formula, value, section, emph, hint, red, bold_magenta)
from ui.widgets import box, section_header, breadcrumb, nav_bar, table, bar_chart, code_block, panel, pager, hr, print_sparkline
from ui.menu import topic_nav, clear, block_menu, mark_completed
from viz.ascii_plots import scatter, line_plot, multi_line, comp_graph


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Rosenbrock function
# ─────────────────────────────────────────────────────────────────────────────
def _rosenbrock(xy):
    x, y = xy
    return (1 - x)**2 + 100 * (y - x**2)**2

def _rosenbrock_grad(xy):
    x, y = xy
    gx = -2*(1 - x) - 400*x*(y - x**2)
    gy = 200*(y - x**2)
    return np.array([gx, gy])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gradient Descent
# ─────────────────────────────────────────────────────────────────────────────
def topic_gradient_descent():
    clear()
    breadcrumb("mlmath", "Optimisation", "Gradient Descent")
    section_header("GRADIENT DESCENT")

    section_header("1. THEORY")
    print(white("  Gradient descent is the workhorse of ML optimisation. Starting from an initial"))
    print(white("  point x₀, it iteratively moves in the direction of steepest descent −∇f(x)."))
    print()
    print(white("  Learning rate α controls step size. Too large → diverge (overshoot minimum);"))
    print(white("  too small → converge very slowly. Optimal α = 2/(L_min + L_max) for quadratic"))
    print(white("  f where L_min, L_max are extremal curvatures (eigenvalues of Hessian)."))
    print()
    print(white("  Convergence: for L-smooth, μ-strongly-convex f, GD converges at geometric"))
    print(white("  rate: f(xₖ) − f* ≤ (1 − μ/L)^k [f(x₀) − f*]. The ratio κ = L/μ (condition"))
    print(white("  number) controls convergence speed."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Update:          xₜ₊₁ = xₜ − α ∇f(xₜ)"))
    print(formula("  Convergence:     f(xₖ) − f* ≤ (1 − 2αμL/(μ+L))^k [f(x₀) − f*]"))
    print(formula("  Optimal step:    α* = 2/(μ + L)  for strongly-convex f"))
    print(formula("  Fixed point:     ∇f(x*) = 0  (necessary condition for minimum)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def f(xy): return (xy[0]-3)**2 + (xy[1]-2)**2
    def grad_f(xy): return np.array([2*(xy[0]-3), 2*(xy[1]-2)])

    x = np.array([0.0, 0.0]); alpha = 0.1
    trajectory = [x.copy()]
    losses = []
    for _ in range(50):
        g = grad_f(x); x = x - alpha * g
        trajectory.append(x.copy()); losses.append(f(x))

    print(white("  Minimise f(x,y) = (x-3)² + (y-2)²   (true minimum at (3,2))"))
    print(white(f"  Start: (0,0)   α = {alpha}   50 steps"))
    print()
    for i in [0, 4, 9, 24, 49]:
        pos = trajectory[i+1]
        print(f"  Step {i+1:>3}:  x = ({value(f'{pos[0]:.4f}')}, {value(f'{pos[1]:.4f}')}),  f = {value(f'{losses[i]:.6f}')}")
    print(green(f"\n  ✓ Converged to ≈ {np.round(trajectory[-1], 4)}  (true: [3.0, 2.0])"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Loss curve — GD on (x-3)² + (y-2)²:"))
    max_loss = losses[0]; step = max(1, len(losses) // 30)
    for i, loss in enumerate(losses[::step]):
        bar_len = int(30 * loss / max_loss)
        bar_str = "█" * max(bar_len, 1)
        print(f"  Step {i*step:>3}  {cyan(bar_str):<35}  {value(f'{loss:.4f}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import loss_curve
        loss_curve(losses, title="GD Convergence on f(x,y)=(x-3)²+(y-2)²", ylabel="f(x)")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-0.5, 4.5, 100); ys_m = np.linspace(-0.5, 3.5, 100)
        Xg, Yg = np.meshgrid(xs_m, ys_m)
        Zg = (Xg-3)**2 + (Yg-2)**2
        traj = np.array(trajectory)
        plt.figure(figsize=(7, 5))
        plt.contour(Xg, Yg, Zg, levels=20, cmap='Blues')
        plt.plot(traj[:,0], traj[:,1], 'ro-', ms=3, lw=1.5, label='GD trajectory')
        plt.scatter([3], [2], s=100, c='gold', zorder=5, label='Minimum')
        plt.title("Gradient Descent Trajectory"); plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Gradient Descent", """\
import numpy as np

def f(xy):      return (xy[0]-3)**2 + (xy[1]-2)**2
def grad_f(xy): return np.array([2*(xy[0]-3), 2*(xy[1]-2)])

def gradient_descent(f, grad_f, x0, lr=0.1, n_steps=100):
    x = x0.copy(); history = [x.copy()]
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        history.append(x.copy())
    return x, history

x_opt, history = gradient_descent(f, grad_f, np.array([0., 0.]))
print("Optimal x:", x_opt)
print("f(x*):", f(x_opt))

# Learning rate sensitivity
for lr in [0.01, 0.1, 0.5, 1.1]:
    x, hist = gradient_descent(f, grad_f, np.array([0., 0.]), lr=lr, n_steps=50)
    print(f"lr={lr:.2f}: final f={f(x):.4f}", "DIVERGED" if f(x) > 100 else "")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "GD converges geometrically for strongly-convex, smooth f; rate = 1 − μ/L.",
        "Learning rate too large → oscillation or divergence; too small → impractical slow convergence.",
        "Gradient descent steps are perpendicular to level curves of f.",
        "GD on a quadratic minimises in one step if α = 1/L (Lipschitz constant of gradient).",
        "In deep learning, 'full-batch GD' is rarely used — replaced by mini-batch SGD.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SGD & Mini-batch
# ─────────────────────────────────────────────────────────────────────────────
def topic_sgd():
    clear()
    breadcrumb("mlmath", "Optimisation", "SGD & Mini-batch")
    section_header("SGD & MINI-BATCH GRADIENT DESCENT")

    section_header("1. THEORY")
    print(white("  Stochastic Gradient Descent (SGD) replaces the full gradient ∇f = (1/N)Σᵢ∇fᵢ"))
    print(white("  with a mini-batch estimate: ĝ = (1/B) Σᵢ∈B ∇fᵢ. This reduces cost per step"))
    print(white("  from O(N) to O(B), enabling many more parameter updates per epoch."))
    print()
    print(white("  Intuition: the gradient noise from subsampling acts as a regulariser — gradients"))
    print(white("  from different mini-batches disagree, preventing over-convergence to sharp minima."))
    print(white("  Sharp minima generalise poorly; noise in SGD helps find flatter (better) ones."))
    print()
    print(white("  Batch size tradeoffs:"))
    print(white("  B=1 (pure SGD):   maximum noise, maximum regularisation, slowest wall-clock"))
    print(white("  B=N (full batch):  minimum noise, no regularisation, slow convergence rate"))
    print(white("  B=32-256:          sweet spot for most deep learning tasks"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Mini-batch gradient:  ĝ = (1/B) Σᵢ∈B ∇fᵢ(x)"))
    print(formula("  SGD update:           xₜ₊₁ = xₜ − α ĝₜ"))
    print(formula("  Noise variance:       Var[ĝ] = (1/B) Var[∇fᵢ]  (decreases with B)"))
    print(formula("  Convergence (convex): E[f(xₜ)] − f* ≤ O(1/√T)   (non-convex: O(1/T))"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    rng = np.random.default_rng(42)
    N = 100
    X  = rng.standard_normal((N, 2))
    w_true = np.array([2.0, -1.5])
    y  = X @ w_true + 0.1 * rng.standard_normal(N)

    def mse_loss(w, X, y):    return np.mean((X @ w - y)**2)
    def mse_grad(w, X, y):    return 2 * X.T @ (X @ w - y) / len(y)

    print(white(f"  Linear regression with N={N} samples, w_true = {w_true}"))
    print()

    results = {}
    for B in [1, 16, N]:
        w = np.zeros(2); alpha = 0.01; losses = []
        for epoch in range(20):
            idx = rng.permutation(N)
            for start in range(0, N, B):
                batch = idx[start:start+B]
                g = mse_grad(w, X[batch], y[batch])
                w = w - alpha * g
            losses.append(mse_loss(w, X, y))
        results[B] = (w.copy(), losses)
        label = f"B={B:>3}"
        print(f"  {label}: final w = {np.round(w, 3)},  loss = {mse_loss(w, X, y):.4f}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Loss curves for different batch sizes (20 epochs):"))
    for B, (_, losses) in results.items():
        vals = losses
        max_l = max(losses[0] for _, (_, l) in results.items())
        print(f"\n  B={B:>3}: ", end="")
        for l in vals[::2]:
            bar_len = max(1, int(20 * l / max_l))
            print(cyan("█") if B == 16 else (yellow("█") if B == 1 else green("█")), end="")
        print(f"  {value(f'{vals[-1]:.5f}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        all_losses = [results[b][1] for b in [1, 16, N]]
        multi_loss(all_losses, labels=["B=1 (SGD)", "B=16 (mini-batch)", "B=N (full-batch)"],
                   title="Loss vs Epochs for Different Batch Sizes")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        colors = ['tomato', 'steelblue', 'gold']
        plt.figure(figsize=(7, 4))
        for (B, (_, losses)), col in zip(results.items(), colors):
            plt.semilogy(losses, label=f"B={B}", color=col)
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss (log)")
        plt.title("SGD vs Mini-batch vs Full-batch"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Mini-batch SGD", """\
import numpy as np

rng = np.random.default_rng(42)
N = 200; X = rng.standard_normal((N, 3))
w_true = np.array([1., -2., 0.5])
y = X @ w_true + 0.1 * rng.standard_normal(N)

def sgd(X, y, batch_size=32, lr=0.01, epochs=50):
    w = np.zeros(X.shape[1]); losses = []
    for _ in range(epochs):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            b = idx[start:start+batch_size]
            grad = 2 * X[b].T @ (X[b] @ w - y[b]) / len(b)
            w -= lr * grad
        losses.append(np.mean((X @ w - y)**2))
    return w, losses

w, losses = sgd(X, y, batch_size=32)
print("Estimated w:", np.round(w, 3))
print("True      w:", w_true)
print("Final loss:", losses[-1])
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Mini-batch SGD is the de-facto standard: balance computation efficiency and gradient noise.",
        "Noise from B=1 SGD acts as implicit regularisation, favouring flat minima that generalise.",
        "Linear scaling rule: if you double B, double learning rate to maintain training dynamics.",
        "Gradient noise variance is O(1/B) — halving B doubles the noise but doubles update frequency.",
        "Data loading and GPU memory, not math, often determine practical batch size choices.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Momentum & Nesterov
# ─────────────────────────────────────────────────────────────────────────────
def topic_momentum():
    clear()
    breadcrumb("mlmath", "Optimisation", "Momentum & Nesterov")
    section_header("MOMENTUM & NESTEROV ACCELERATED GRADIENT")

    section_header("1. THEORY")
    print(white("  Gradient descent oscillates in ravines (directions with different curvature))"))
    print(white("  because it takes large steps across the narrow dimension and small steps along"))
    print(white("  the valley. Momentum fixes this by accumulating an exponential moving average"))
    print(white("  of past gradients — oscillations cancel, consistent directions accumulate."))
    print()
    print(white("  Nesterov momentum evaluates the gradient at the 'lookahead' position x + βv"))
    print(white("  (approximately where momentum will take us). This provides a correction that"))
    print(white("  improves the asymptotic convergence rate from O(1/k) to O(1/k²) for convex f."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Heavy ball:  vₜ = β vₜ₋₁ + (1-β) ∇f(xₜ)"))
    print(formula("                xₜ = xₜ₋₁ − α vₜ"))
    print(formula("  Nesterov:    ŷₜ = xₜ + β(xₜ − xₜ₋₁)"))
    print(formula("                xₜ₊₁ = ŷₜ − α ∇f(ŷₜ)"))
    print(formula("  Convergence: momentum achieves O(1/k²) vs O(1/k) for GD on convex f"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def gd_run(f, gradf, x0, lr, n=200):
        x = x0.copy(); losses = [f(x)]
        for _ in range(n):
            x = x - lr * gradf(x)
            losses.append(f(x))
        return x, losses

    def momentum_run(f, gradf, x0, lr, beta=0.9, n=200):
        x = x0.copy(); v = np.zeros_like(x); losses = [f(x)]
        for _ in range(n):
            g = gradf(x); v = beta*v + (1-beta)*g; x = x - lr*v
            losses.append(f(x))
        return x, losses

    x0 = np.array([-1.5, -1.0])
    lr_gd  = 0.001
    lr_mom = 0.001

    x_gd,  l_gd  = gd_run(      _rosenbrock, _rosenbrock_grad, x0, lr_gd,  n=500)
    x_mom, l_mom = momentum_run( _rosenbrock, _rosenbrock_grad, x0, lr_mom, n=500)

    print(white("  Minimise Rosenbrock f(x,y) = (1-x)² + 100(y-x²)²"))
    print(white("  (minimum at (1,1), global basin shape like a banana valley)"))
    print()
    print(f"  {'Method':<15} {'Final x':<30} {'f(x*)'}")
    print(grey("  " + "-"*55))
    print(f"  {white('GD'):<24} {value(str(np.round(x_gd,4))):<39} {value(f'{_rosenbrock(x_gd):.4f}')}")
    print(f"  {white('Momentum β=0.9'):<24} {value(str(np.round(x_mom,4))):<39} {value(f'{_rosenbrock(x_mom):.4f}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Loss comparison (500 steps, sampled every 50):"))
    step = 25
    for label, losses, col in [("GD      ", l_gd, cyan), ("Momentum", l_mom, yellow)]:
        print(f"  {white(label)}: ", end="")
        for l in losses[::step]:
            clamped = min(l, 2000)
            bar_len = max(1, int(20 * clamped / 2000))
            print(col("█"), end="")
        print(f"  {value(f'{losses[-1]:.2f}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        multi_loss([l_gd[:200], l_mom[:200]], labels=["GD", "Momentum β=0.9"],
                   title="Rosenbrock: GD vs Momentum")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.semilogy(l_gd[:200],  label="Gradient Descent",  color='tomato')
        plt.semilogy(l_mom[:200], label="Momentum β=0.9",    color='steelblue')
        plt.xlabel("Iteration"); plt.ylabel("f(x) (log scale)")
        plt.title("Rosenbrock: GD vs Momentum"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Momentum Optimizer", """\
import numpy as np

def rosenbrock(xy):
    x, y = xy
    return (1-x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    return np.array([-2*(1-x) - 400*x*(y-x**2),
                      200*(y - x**2)])

def momentum_gd(grad, x0, lr=0.001, beta=0.9, n=1000):
    x = x0.copy(); v = np.zeros_like(x); history = [x.copy()]
    for _ in range(n):
        g = grad(x)
        v = beta * v + (1 - beta) * g
        x = x - lr * v
        history.append(x.copy())
    return x, history

x_opt, hist = momentum_gd(rosenbrock_grad, np.array([-1.5, -1.0]))
print(f"Converged to: {x_opt}")
print(f"True minimum: [1.0, 1.0]")
print(f"Final loss: {rosenbrock(x_opt):.6f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Momentum cancels gradient oscillations across narrow dimensions, accelerating progress.",
        "Physical analogy: momentum = ball rolling down a hill with inertia.",
        "Nesterov's accelerated gradient achieves optimal O(1/k²) rate for convex smooth functions.",
        "β ≈ 0.9 is a common default; β → 1 gives more inertia (useful for noisy gradients).",
        "Effective learning rate with momentum is α/(1-β) — scale α down when increasing β.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adam Optimizer
# ─────────────────────────────────────────────────────────────────────────────
def topic_adam():
    clear()
    breadcrumb("mlmath", "Optimisation", "Adam Optimizer")
    section_header("ADAM OPTIMIZER — FULL DERIVATION")

    section_header("1. THEORY")
    print(white("  Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning"))
    print(white("  rates. It combines: (1) momentum (exponential MA of gradients = 1st moment)"))
    print(white("  and (2) RMSProp (exponential MA of squared gradients = 2nd moment)."))
    print()
    print(white("  Bias correction: at step t=1, m₁ = (1-β₁)g₁ ≈ 0 regardless of g₁ magnitude."))
    print(white("  Adam divides by (1-β₁ᵗ) and (1-β₂ᵗ) to correct this initialisation bias,"))
    print(white("  giving correct scale estimates from step 1."))
    print()
    print(white("  Result: parameters with large gradients get smaller effective LR; parameters"))
    print(white("  with small/sparse gradients get larger effective LR. Adam self-tunes!"))
    print(white("  Default hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8) work for most tasks."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  1st moment: mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ"))
    print(formula("  2nd moment: vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²"))
    print(formula("  Bias corr:  m̂ₜ = mₜ/(1-β₁ᵗ),   v̂ₜ = vₜ/(1-β₂ᵗ)"))
    print(formula("  Update:     θₜ = θₜ₋₁ − α · m̂ₜ / (√v̂ₜ + ε)"))
    print(formula("  Defaults:   β₁=0.9, β₂=0.999, ε=1e-8, α=0.001"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def adam_run(f, gradf, x0, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, n=500):
        x = x0.copy(); m = np.zeros_like(x); v = np.zeros_like(x); losses = [f(x)]
        for t in range(1, n+1):
            g = gradf(x)
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            m_hat = m / (1 - b1**t)
            v_hat = v / (1 - b2**t)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            losses.append(f(x))
        return x, losses

    def gd_run_simple(f, gradf, x0, lr, n):
        x = x0.copy(); losses = [f(x)]
        for _ in range(n):
            x = x - lr * gradf(x)
            losses.append(f(x))
        return x, losses

    def mom_run(f, gradf, x0, lr, beta=0.9, n=500):
        x = x0.copy(); v = np.zeros_like(x); losses = [f(x)]
        for _ in range(n):
            g = gradf(x); v = beta*v + (1-beta)*g; x -= lr*v
            losses.append(f(x))
        return x, losses

    x0 = np.array([-1.5, -0.5])
    x_adam,  l_adam  = adam_run( _rosenbrock, _rosenbrock_grad, x0)
    x_gd,    l_gd    = gd_run_simple(_rosenbrock, _rosenbrock_grad, x0.copy(), 0.001, 500)
    x_mom,   l_mom   = mom_run(_rosenbrock, _rosenbrock_grad, x0.copy(), 0.001)

    print(white("  Minimise Rosenbrock's banana function (minimum at [1,1]):"))
    print()
    print(f"  {'Optimizer':<15} {'Converged to':<30} {'f(x*)'}")
    print(grey("  " + "-"*60))
    for label, xopt, loss in [("GD", x_gd, l_gd[-1]),
                               ("Momentum", x_mom, l_mom[-1]),
                               ("Adam",     x_adam, l_adam[-1])]:
        print(f"  {white(label):<24} {value(str(np.round(xopt,4))):<39} {value(f'{loss:.4f}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Loss at key iterations (Adam on Rosenbrock):"))
    checkpoints = [0, 10, 50, 100, 200, 499]
    for i in checkpoints:
        loss = l_adam[i]
        bar_len = max(1, min(35, int(35 * np.log10(max(loss, 1)) / np.log10(max(l_adam[0], 1)))))
        bar_str = "█" * bar_len
        print(f"  Step {i:>4}:  {yellow(bar_str):<40}  {value(f'{loss:.4f}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        multi_loss([l_gd[:300], l_mom[:300], l_adam[:300]],
                   labels=["GD", "Momentum", "Adam"],
                   title="Rosenbrock: GD vs Momentum vs Adam")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.semilogy(l_gd[:300],   label="Gradient Descent", color='tomato')
        plt.semilogy(l_mom[:300],  label="Momentum β=0.9",   color='orange')
        plt.semilogy(l_adam[:300], label="Adam",             color='steelblue')
        plt.xlabel("Iteration"); plt.ylabel("f(x) (log scale)")
        plt.title("Optimiser Comparison on Rosenbrock"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Adam Optimizer from Scratch", """\
import numpy as np

class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None; self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def rosenbrock_grad(xy):
    x, y = xy
    return np.array([-2*(1-x) - 400*x*(y-x**2), 200*(y-x**2)])

opt = Adam(lr=0.01)
x   = np.array([-1.5, -0.5])
for i in range(500):
    g = rosenbrock_grad(x)
    x = opt.step(x, g)
    if i % 100 == 0:
        print(f"  step {i}: x = {np.round(x,4)}")
print("Converged to:", x)
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Adam divides gradient by √variance — parameters with rare large gradients get smaller steps.",
        "Bias correction ensures moment estimates are unbiased from step 1 (not just asymptotically).",
        "Adam can fail to converge in some convex settings — AMSGrad fixes this with running max of v.",
        "β₁=0.9 means ~10 steps to 'forget' old gradient direction; β₂=0.999 means ~1000 steps for scale.",
        "Adam often reaches a good solution faster than SGD, but SGD often finds better final accuracy.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Learning Rate Schedulers
# ─────────────────────────────────────────────────────────────────────────────
def topic_lr_schedulers():
    clear()
    breadcrumb("mlmath", "Optimisation", "LR Schedulers")
    section_header("LEARNING RATE SCHEDULERS")

    section_header("1. THEORY")
    print(white("  A constant learning rate is rarely optimal. Early training: large LR helps"))
    print(white("  explore the landscape quickly. Late training: small LR for fine convergence."))
    print()
    print(white("  Warmup: starting with a very small LR and gradually increasing prevents"))
    print(white("  large gradients in the first steps (when weights are near random init) from"))
    print(white("  pushing the model into a bad basin. Critical for transformers."))
    print()
    print(white("  Cosine annealing: smoothly decreases LR from max to min following a cosine"))
    print(white("  curve. Can be combined with warm restarts (SGDR) to escape local minima."))
    print(white("  Step decay: multiply LR by γ < 1 every fixed number of epochs — simple, fast"))
    print(white("  to tune, widely used in computer vision."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Step decay:     ηₜ = η₀ · γ^⌊t/s⌋"))
    print(formula("  Cosine:         ηₜ = ηₘᵢₙ + ½(ηₘₐₓ-ηₘᵢₙ)(1 + cos(πt/T))"))
    print(formula("  Linear warmup:  ηₜ = ηₘₐₓ · t/T_warmup  for t ≤ T_warmup"))
    print(formula("  Exponential:    ηₜ = η₀ · exp(−λt)"))
    print(formula("  Polynomial:     ηₜ = η₀ · (1 − t/T)^p"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    T = 100; eta_min = 1e-4; eta_max = 0.1; gamma = 0.5; step_size = 20
    T_warmup = 10

    def step_decay(t):      return eta_max * (gamma ** (t // step_size))
    def cosine(t):          return eta_min + 0.5*(eta_max - eta_min)*(1 + np.cos(np.pi*t/T))
    def linear_warmup(t):   return eta_max * min(t / T_warmup, 1.0) * cosine(max(t - T_warmup, 0)) / eta_max
    def exponential(t):     return eta_max * np.exp(-5 * t / T)

    ts = range(T)
    schedules = {
        "Step decay":      [step_decay(t)    for t in ts],
        "Cosine":          [cosine(t)         for t in ts],
        "Linear warmup":   [linear_warmup(t)  for t in ts],
        "Exponential":     [exponential(t)    for t in ts],
    }
    print(white("  LR schedules over 100 steps:"))
    print(f"  {'Schedule':<20} {'t=0':<12} {'t=25':<12} {'t=50':<12} {'t=99'}")
    print(grey("  " + "-"*60))
    for name, vals in schedules.items():
        print(f"  {white(name):<29} {value(f'{vals[0]:.4f}'):<21} {value(f'{vals[25]:.4f}'):<21} {value(f'{vals[50]:.4f}'):<21} {value(f'{vals[99]:.4f}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  LR schedules visualised:"))
    print()
    for name, vals in schedules.items():
        max_v = max(vals); row = f"  {name:<20}  "
        for v in vals[::5]:
            bar_len = max(1, int(20 * v / max_v))
            row += "█" * bar_len + "░" * (20 - bar_len) + " "
        print(row[:80])
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        multi_loss(list(schedules.values()), labels=list(schedules.keys()),
                   title="Learning Rate Schedules (100 steps)")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        colors = ['tomato', 'steelblue', 'gold', 'limegreen']
        plt.figure(figsize=(8, 4))
        for (name, vals), col in zip(schedules.items(), colors):
            plt.plot(list(ts), vals, label=name, color=col)
        plt.xlabel("Step"); plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedulers"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Learning Rate Schedulers", """\
import numpy as np

T = 100; eta_max = 0.1; eta_min = 1e-4

# Step decay
def step_decay(t, eta0=eta_max, gamma=0.5, step=20):
    return eta0 * gamma ** (t // step)

# Cosine annealing
def cosine_anneal(t, T=T, eta_min=eta_min, eta_max=eta_max):
    return eta_min + 0.5*(eta_max - eta_min)*(1 + np.cos(np.pi * t / T))

# Linear warmup + cosine
def warmup_cosine(t, T_warmup=10, T=T, eta_max=eta_max, eta_min=eta_min):
    if t < T_warmup:
        return eta_max * t / T_warmup
    t2 = t - T_warmup; T2 = T - T_warmup
    return eta_min + 0.5*(eta_max - eta_min)*(1 + np.cos(np.pi * t2 / T2))

# Print schedule
for t in [0, 10, 25, 50, 75, 99]:
    print(f"t={t:>3}: step={step_decay(t):.4f}  "
          f"cosine={cosine_anneal(t):.4f}  "
          f"warmup={warmup_cosine(t):.4f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Warmup prevents large gradient updates when weights are far from good initialisations.",
        "Cosine annealing smoothly decreases LR, avoiding abrupt step-decay discontinuities.",
        "ReduceLROnPlateau detects stagnation and halves LR — useful when optimisation plateau.",
        "Cyclic learning rates (CLR) periodically increase LR to escape sharp local minima.",
        "Learning rate is often the most impactful hyperparameter — schedule can matter as much.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convexity
# ─────────────────────────────────────────────────────────────────────────────
def topic_convexity():
    clear()
    breadcrumb("mlmath", "Optimisation", "Convexity")
    section_header("CONVEXITY")

    section_header("1. THEORY")
    print(white("  A function f is convex iff for all x, y and λ ∈ [0,1]:"))
    print(white("  f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)"))
    print(white("  The chord between any two points lies ABOVE (or on) the curve."))
    print()
    print(white("  Why convexity is paradise for optimisation:"))
    print(white("  • Every local minimum is a global minimum"))
    print(white("  • Gradient descent converges to the global optimum"))
    print(white("  • First-order conditions ∇f=0 are sufficient (not just necessary)"))
    print(white("  • Duality gap is zero (strong duality holds)"))
    print()
    print(white("  Convex functions in ML: squared loss, logistic loss, hinge loss, L1/L2"))
    print(white("  regularisation. Neural networks are NOT convex — but we train them anyway!"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Definition:   f(λx+(1-λ)y) ≤ λf(x)+(1-λ)f(y)  ∀x,y, λ∈[0,1]"))
    print(formula("  1st order:    f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)  (tangent plane lower bound)"))
    print(formula("  2nd order:    H(x) ≽ 0  (Hessian positive semidefinite)"))
    print(formula("  Strict convex: > in definition; unique minimum"))
    print(formula("  α-strong cvx: f(y) ≥ f(x)+∇fᵀ(y-x) + (α/2)‖y-x‖²"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def check_convex(f, a, b, n=20):
        xs = np.linspace(a, b, n); ys = np.linspace(a, b, n)
        violations = 0
        for x in xs:
            for y in ys:
                for lam in [0.2, 0.5, 0.8]:
                    mid = lam*x + (1-lam)*y
                    lhs = f(mid)
                    rhs = lam*f(x) + (1-lam)*f(y)
                    if lhs > rhs + 1e-8:
                        violations += 1
        return violations == 0

    funcs = [
        ("f(x) = x²",        lambda x: x**2),
        ("f(x) = |x|",       lambda x: abs(x)),
        ("f(x) = exp(x)",    lambda x: np.exp(x)),
        ("f(x) = x³",        lambda x: x**3),
        ("f(x) = sin(x)",    lambda x: np.sin(x)),
        ("f(x) = x²-x³/10", lambda x: x**2 - x**3/10),
    ]
    print(white("  Convexity check by sampling the definition:"))
    for name, fn in funcs:
        cvx = check_convex(fn, -2, 2)
        mark = green("✓ CONVEX") if cvx else red("✗ non-convex")
        print(f"  {white(name):<25}  {mark}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Convex vs Non-convex shapes:"))
    print()
    print(cyan("  Convex: f(x)=x²          ") + yellow("  Non-convex: f(x)=x³-3x"))
    xs = np.linspace(-2, 2, 12)
    for row_idx in range(5, -1, -1):
        row_cvx = "  "
        row_ncvx = "  "
        for x in xs:
            val_c  = x**2
            val_nc = x**3 - 3*x
            # normalize
            if abs(val_c - row_idx/2.5) < 0.3:
                row_cvx += cyan("█") + " "
            else:
                row_cvx += "  "
            if abs(val_nc/2 - (row_idx - 2)) < 0.6:
                row_ncvx += yellow("█") + " "
            else:
                row_ncvx += "  "
        print(row_cvx + "          " + row_ncvx)
    print(grey("\n  Convex: all chords lie above the curve"))
    print(grey("  Non-convex: some chords dip below the curve"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        xs_p = np.linspace(-2, 2, 80)
        cvx  = (xs_p**2).tolist()
        ncvx = (xs_p**3 - 3*xs_p).tolist()
        multi_loss([cvx, ncvx], labels=["x² (convex)", "x³-3x (non-convex)"],
                   title="Convex vs Non-convex Function")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-2, 2, 200)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(xs_m, xs_m**2, 'steelblue', lw=2)
        x1, x2 = -1.5, 1.0; lam = 0.4
        xm = lam*x1 + (1-lam)*x2
        axes[0].plot([x1, x2], [x1**2, x2**2], 'ro-', lw=2, label='Chord')
        axes[0].scatter([xm], [xm**2], s=80, c='gold', zorder=5)
        axes[0].set_title("Convex: f(x)=x²"); axes[0].legend()
        axes[1].plot(xs_m, xs_m**3 - 3*xs_m, 'tomato', lw=2)
        axes[1].set_title("Non-convex: f(x)=x³-3x")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Convexity Check", """\
import numpy as np

def is_convex_sample(f, a, b, n=30, tol=1e-8):
    \"\"\"Check convexity by sampling Jensen's inequality.\"\"\"
    xs = np.linspace(a, b, n)
    for x in xs:
        for y in xs:
            for lam in [0.1, 0.3, 0.5, 0.7, 0.9]:
                mid = lam*x + (1-lam)*y
                if f(mid) > lam*f(x) + (1-lam)*f(y) + tol:
                    return False
    return True

def is_convex_hessian(f, x, h=1e-4):
    \"\"\"Check convexity via positive semidefinite Hessian (scalar case).\"\"\"
    second_deriv = (f(x+h) - 2*f(x) + f(x-h)) / h**2
    return second_deriv >= -1e-8

funcs = [("x²", lambda x: x**2),
         ("|x|", np.abs),
         ("exp(x)", np.exp),
         ("sin(x)", np.sin)]

for name, fn in funcs:
    print(f"{name}: convex={is_convex_sample(fn, -2, 2)}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Convexity ↔ every chord above the curve ↔ tangent plane lower bound ↔ semidefinite Hessian.",
        "For convex f, gradient descent with α ≤ 1/L converges: f(xₖ) − f* ≤ O(1/k).",
        "Strong convexity (f ≻ αI) gives exponential convergence rate (1 − α/L)^k.",
        "Logistic regression is convex (no local minima) — if it fails to converge, it's a hyper issue.",
        "Neural network loss is non-convex, but over-parameterisation and SGD noise help navigate it.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Saddle Points
# ─────────────────────────────────────────────────────────────────────────────
def topic_saddle_points():
    clear()
    breadcrumb("mlmath", "Optimisation", "Saddle Points in High Dimensions")
    section_header("SADDLE POINTS IN HIGH DIMENSIONS")

    section_header("1. THEORY")
    print(white("  In n dimensions, a critical point (∇f=0) is a saddle if H has both positive"))
    print(white("  and negative eigenvalues. In 2D, saddle points are relatively rare. In n"))
    print(white("  dimensions, the probability of all n eigenvalues being positive is (1/2)^n"))
    print(white("  for random matrices — exponentially small! So most critical points are saddles."))
    print()
    print(white("  This is actually GOOD news for training. GD can escape saddles (with noise from"))
    print(white("  SGD or perturbation), and saddle points are not stuck-points. The bad critical"))
    print(white("  points — sharp local minima that don't generalise — are increasingly understood"))
    print(white("  to be far less common than feared in high-dimensional loss landscapes."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Saddle point: ∇f(x*) = 0, H(x*) has both + and − eigenvalues"))
    print(formula("  Escape rate:  along negative-curvature direction v (H·v = λ·v, λ < 0)"))
    print(formula("                gradient grows as |∇f| ~ |λ| · ‖x − x*‖ → GD escapes"))
    print(formula("  In high dim:  P(all eigenvalues > 0) ≈ (1/2)^n → exponentially unlikely"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def f_saddle(xy): return xy[0]**2 - xy[1]**2
    def grad_saddle(xy): return np.array([2*xy[0], -2*xy[1]])
    H = np.array([[2., 0.], [0., -2.]])
    eigs = np.linalg.eigvalsh(H)

    print(white("  f(x,y) = x² − y²   (classic saddle)"))
    print(white(f"\n  ∇f(0,0) = {grad_saddle(np.array([0., 0.]))}")  )
    print(white(f"  H(0,0) =\n  {H}"))
    print(white(f"  Eigenvalues: {eigs}  → {red('mixed signs = saddle')}"))
    print()
    print(white("  GD trajectory starting near saddle with small perturbation:"))
    x = np.array([0.01, 0.02]); alpha = 0.1; history = [x.copy()]
    for i in range(20):
        x = x - alpha * grad_saddle(x)
        history.append(x.copy())
    print(white(f"  Start: {history[0]}"))
    for i in [4, 9, 14, 19]:
        pos = history[i+1]
        print(f"  Step {i+1:>2}:  ({value(f'{pos[0]:.4f}')}, {value(f'{pos[1]:.4f}')})   f = {value(f'{f_saddle(pos):.4f}')}")
    print(white("\n  → GD moves away once gradient builds up (escapes saddle along negative-curvature direction)"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Surface of f(x,y) = x² − y²  (viewed from above):"))
    grid = np.linspace(-2, 2, 11)
    for y in reversed(grid[::2]):
        row = f"  y={y:+.1f}  "
        for x in grid[::2]:
            val = f_saddle(np.array([x, y]))
            if abs(x) < 0.15 and abs(y) < 0.15:
                row += green("✦") + " "
            elif val > 1.5:
                row += yellow("▲") + " "
            elif val < -1.5:
                row += red("▼") + " "
            else:
                row += grey("·") + " "
        print(row)
    print(grey("\n  ✦ = saddle point  ▲ = positive region  ▼ = negative region"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import eigenvalue_spectrum
        eigenvalue_spectrum(eigs.tolist(), title="Hessian eigenvalues at saddle point (0,0)")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        xs_m = np.linspace(-2, 2, 50); ys_m = np.linspace(-2, 2, 50)
        Xg, Yg = np.meshgrid(xs_m, ys_m); Zg = Xg**2 - Yg**2
        traj = np.array(history)
        fig = plt.figure(figsize=(8, 5))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xg, Yg, Zg, cmap='coolwarm', alpha=0.7)
        zs = [f_saddle(p) for p in traj]
        ax.plot(traj[:,0], traj[:,1], zs, 'ko-', ms=4, lw=2, label='GD trajectory')
        ax.scatter([0], [0], [0], s=100, c='gold', zorder=10, label='Saddle')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
        ax.set_title("Saddle Point: x² − y²"); ax.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Saddle Point Analysis", """\
import numpy as np

def f(xy):    return xy[0]**2 - xy[1]**2
def grad(xy): return np.array([2*xy[0], -2*xy[1]])

# Hessian and eigenvalue check
H    = np.array([[2., 0.], [0., -2.]])
eigs = np.linalg.eigvalsh(H)
print("Eigenvalues:", eigs)
print("Critical point type:", end=" ")
if np.all(eigs > 0):   print("Local minimum")
elif np.all(eigs < 0): print("Local maximum")
else:                  print("Saddle point")

# GD escape from near-saddle (with perturbation)
x = np.array([0.001, 0.002])   # tiny perturbation from origin
for i in range(15):
    x = x - 0.1 * grad(x)
    if i % 5 == 4:
        print(f"  step {i+1}: x={np.round(x,4)}, f={f(x):.4f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "In n-dim, P(random critical point is local min) ≈ (1/2)^n — saddles dominate.",
        "Saddle escape speed scales with |λ_min| (magnitude of most negative eigenvalue).",
        "SGD noise naturally escapes saddles; pure GD may need explicit perturbation.",
        "Strict saddle property: near saddle, there exists direction with negative curvature — find it.",
        "Modern neural networks: overparameterisation smooths the landscape, saddles are less sticky.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Newton's Method
# ─────────────────────────────────────────────────────────────────────────────
def topic_newtons_method():
    clear()
    breadcrumb("mlmath", "Optimisation", "Newton's Method")
    section_header("NEWTON'S METHOD")

    section_header("1. THEORY")
    print(white("  Newton's method uses curvature information (Hessian) to take better steps."))
    print(white("  It minimises the local quadratic Taylor approximation exactly:"))
    print(white("  q(δ) = f(x) + ∇f·δ + ½δᵀHδ  →  minimised at δ* = −H⁻¹∇f"))
    print()
    print(white("  Convergence: quadratic near the minimum (number of correct digits doubles"))
    print(white("  each step), versus linear for gradient descent. In practice: 5-10 steps"))
    print(white("  vs thousands for GD."))
    print()
    print(white("  Drawbacks: computing H costs O(n²) memory and O(n³) for the solve. For"))
    print(white("  neural networks with n~10^9 parameters, exact Newton is impossible."))
    print(white("  Quasi-Newton methods (BFGS, L-BFGS) approximate H⁻¹ cheaply."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Newton step:    δ* = −H(x)⁻¹ ∇f(x)"))
    print(formula("  Update:         xₜ₊₁ = xₜ − H(xₜ)⁻¹ ∇f(xₜ)"))
    print(formula("  Quadratic conv: ‖xₜ₊₁ − x*‖ ≤ C‖xₜ − x*‖²"))
    print(formula("  Cost per step:  O(n²) space for H,  O(n³) for solve"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    def f1(x):   return x**4 - 4*x**2      # local mins at ±√2, local max at 0
    def df1(x):  return 4*x**3 - 8*x
    def d2f1(x): return 12*x**2 - 8

    print(white("  Minimise f(x) = x⁴ − 4x²  (local minima at x = ±√2 ≈ ±1.414)"))
    print()

    # GD
    x_gd = 2.0; alpha_gd = 0.01
    losses_gd = []
    for _ in range(100):
        x_gd -= alpha_gd * df1(x_gd)
        losses_gd.append(f1(x_gd))

    # Newton
    x_nw = 2.0
    losses_nw = []
    for i in range(15):
        h = d2f1(x_nw)
        if abs(h) < 1e-12: break
        x_nw -= df1(x_nw) / h
        losses_nw.append(f1(x_nw))

    print(white(f"  GD ({len(losses_gd)} steps):     x = {x_gd:.8f},  f = {f1(x_gd):.8f}"))
    print(white(f"  Newton ({len(losses_nw)} steps): x = {x_nw:.8f},  f = {f1(x_nw):.8f}"))
    print(white(f"  True minimum: x* = {np.sqrt(2):.8f},  f* = {f1(np.sqrt(2)):.8f}"))
    print()
    print(white("  Newton convergence (first 10 steps):"))
    x_nw2 = 2.0
    for i in range(10):
        x_nw2 -= df1(x_nw2) / d2f1(x_nw2)
        err = abs(x_nw2 - np.sqrt(2))
        print(f"  Step {i+1}: x = {value(f'{x_nw2:.8f}')},  err = {grey(f'{err:.2e}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Convergence: Newton vs GD (errors):"))
    xs_v = np.linspace(-3, 3, 60); fs_v = [f1(x) for x in xs_v]
    print(white("\n  f(x) = x⁴ − 4x² landscape:"))
    max_f = max(abs(v) for v in fs_v)
    for x, fv in zip(xs_v[::5], [f1(x) for x in xs_v[::5]]):
        h = int(8 * fv / max_f) + 8
        col = green if fv < -3 else (grey if abs(fv) < 1 else yellow)
        print(f"  x={x:+.1f} {'│' + col('█') * max(0, h)}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_loss
        pad = max(0, len(losses_gd) - len(losses_nw))
        nw_padded = losses_nw + [losses_nw[-1]] * pad
        multi_loss([losses_gd, nw_padded], labels=["GD (100 steps)", "Newton (<15 steps)"],
                   title="Newton vs GD Convergence on x⁴-4x²")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(losses_gd, color='tomato', label="GD")
        axes[0].set_title("GD (100 steps)"); axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("f(x)")
        axes[1].semilogy(losses_nw, color='steelblue', label="Newton", marker='o')
        axes[1].set_title("Newton (quadratic convergence)"); axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("f(x) log")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Newton's Method", """\
import numpy as np

def f(x):   return x**4 - 4*x**2
def df(x):  return 4*x**3 - 8*x
def d2f(x): return 12*x**2 - 8

def newton_1d(f, df, d2f, x0, tol=1e-12, max_iter=50):
    x = x0; history = [x]
    for i in range(max_iter):
        g = df(x); h = d2f(x)
        if abs(h) < 1e-14: break
        x = x - g / h
        history.append(x)
        if abs(g) < tol: break
    return x, history

x_opt, hist = newton_1d(f, df, d2f, x0=2.0)
print(f"Converged to x* = {x_opt:.10f}")
print(f"True min at ±√2 = ±{np.sqrt(2):.10f}")
print(f"Iterations: {len(hist)-1}")

# Multi-dim Newton (2D)
def f2(xy): return (xy[0]-1)**2 + 10*(xy[1]-xy[0]**2)**2 * 0.1
# ... extend with Hessian inverse for n-dim
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Newton converges quadratically (digits double each step) vs GD's linear convergence.",
        "Newton is scale-invariant — H⁻¹ normalises each direction by its own curvature.",
        "In ill-conditioned problems, Newton's natural scaling is a huge advantage over GD.",
        "Quasi-Newton (BFGS): builds H⁻¹ approximation from gradient differences, costs O(n²).",
        "L-BFGS: only stores last m vector pairs — O(mn) memory, O(mn) per step, enables large-scale.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Conjugate Gradient
# ─────────────────────────────────────────────────────────────────────────────
def topic_conjugate_gradient():
    clear()
    breadcrumb("mlmath", "Optimisation", "Conjugate Gradient")
    section_header("CONJUGATE GRADIENT")

    section_header("1. THEORY")
    print(white("  CG solves Ax=b for SPD A without computing A⁻¹ or factorising A. It maintains"))
    print(white("  search directions pₖ that are 'A-conjugate': pᵢᵀApⱼ = 0 for i≠j."))
    print()
    print(white("  Key property: n A-conjugate search directions span ℝⁿ, so CG finds the exact"))
    print(white("  solution in at most n steps (finite termination). In practice, with good"))
    print(white("  preconditioning, CG converges in far fewer steps."))
    print()
    print(white("  Memory: CG only needs to store x, r, p — three vectors. Contrast with direct"))
    print(white("  methods that need O(n²) for the factorisation. CG is matrix-free: only"))
    print(white("  matrix-vector products Ap are needed, enabling implicit A definitions."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Residual:     rₖ = b − Axₖ"))
    print(formula("  Step size:    αₖ = rₖᵀrₖ / pₖᵀApₖ"))
    print(formula("  Update x:     xₖ₊₁ = xₖ + αₖpₖ"))
    print(formula("  Update r:     rₖ₊₁ = rₖ − αₖApₖ"))
    print(formula("  β coefficient: βₖ = rₖ₊₁ᵀrₖ₊₁ / rₖᵀrₖ"))
    print(formula("  Update p:     pₖ₊₁ = rₖ₊₁ + βₖpₖ"))
    print(formula("  Convergence:  ‖eₖ‖_A / ‖e₀‖_A ≤ 2((√κ-1)/(√κ+1))^k"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    rng = np.random.default_rng(123)
    n = 5
    B = rng.standard_normal((n, n))
    A = B.T @ B + 0.5 * np.eye(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true

    def cg(A, b, tol=1e-12, maxiter=1000):
        x = np.zeros_like(b); r = b - A @ x; p = r.copy()
        residuals = [np.linalg.norm(r)]
        rs_old = r @ r
        for _ in range(maxiter):
            Ap = A @ p; alpha = rs_old / (p @ Ap)
            x += alpha * p; r -= alpha * Ap; rs_new = r @ r
            residuals.append(np.sqrt(rs_new))
            if np.sqrt(rs_new) < tol: break
            beta = rs_new / rs_old
            p = r + beta * p; rs_old = rs_new
        return x, residuals

    x_cg, residuals = cg(A, b)
    print(white(f"  5×5 SPD system:  κ(A) = {np.linalg.cond(A):.2f}"))
    print(white(f"  True x:   {np.round(x_true, 4)}"))
    print(white(f"  CG x:     {np.round(x_cg, 4)}"))
    print(green(f"  ✓ Error = {np.linalg.norm(x_cg - x_true):.2e}"))
    print()
    print(white("  Residual per iteration:"))
    for i, r in enumerate(residuals):
        bar = "█" * max(1, int(30 * r / residuals[0]))
        print(f"  Iter {i}: {cyan(bar):<35}  {value(f'{r:.2e}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  CG search directions are A-conjugate:"))
    print(white("  p₀ = r₀ (steepest descent)"))
    print(white("  p₁ = r₁ + β₀p₀  (modified to be A-conjugate to p₀)"))
    print(white("  ..."))
    print()
    print(white("  Each pₖ is optimal in the Krylov subspace span{r₀, Ar₀, ..., Aᵏr₀}."))
    print(white("  CG finds optimal x in ℝⁿ in exactly n steps."))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import loss_curve
        loss_curve(residuals, title="CG Residual ‖rₖ‖ — Convergence", ylabel="‖rₖ‖")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.semilogy(residuals, 'o-', color='steelblue', label="CG residual")
        plt.xlabel("Iteration"); plt.ylabel("‖rₖ‖ (log scale)")
        plt.title("Conjugate Gradient Convergence"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Conjugate Gradient from Scratch", """\
import numpy as np

def conjugate_gradient(A, b, tol=1e-12):
    \"\"\"Solve Ax=b for SPD A using Conjugate Gradient.\"\"\"
    n = len(b); x = np.zeros(n)
    r = b - A @ x; p = r.copy(); rs_old = r @ r
    for k in range(n):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            print(f"  Converged at iteration {k+1}")
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# Example
rng = np.random.default_rng(42)
n   = 50
B   = rng.standard_normal((n, n))
A   = B.T @ B + 0.1 * np.eye(n)   # SPD
b   = rng.standard_normal(n)
x   = conjugate_gradient(A, b)
print("Residual:", np.linalg.norm(A @ x - b))
print("Solution x[:5]:", np.round(x[:5], 4))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "CG solves n×n SPD systems in exactly n steps; preconditioned CG in << n steps.",
        "CG only needs matrix-vector products — enables matrix-free solvers for huge systems.",
        "Convergence rate depends on √κ(A); clustering eigenvalues near 1 enables fast convergence.",
        "CG minimises f(x)=½xᵀAx−bᵀx over Krylov subspace — perfectly optimal per iteration.",
        "Lanczos algorithm = CG without solution tracking; used for eigenvalue computation.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 10. L-BFGS
# ─────────────────────────────────────────────────────────────────────────────
def topic_lbfgs():
    clear()
    breadcrumb("mlmath", "Optimisation", "L-BFGS")
    section_header("L-BFGS (LIMITED-MEMORY BFGS)")

    section_header("1. THEORY")
    print(white("  L-BFGS approximates the inverse Hessian H⁻¹ using the last m gradient"))
    print(white("  difference pairs {(sₖ, yₖ)} where sₖ = xₖ − xₖ₋₁, yₖ = ∇fₖ − ∇fₖ₋₁."))
    print()
    print(white("  BFGS update: Hₖ₊₁⁻¹ = Vₖᵀ Hₖ⁻¹ Vₖ + ρₖ sₖsₖᵀ  (rank-2 update)"))
    print(white("  L-BFGS: only stores last m pairs → O(mn) memory vs O(n²) for full BFGS."))
    print(white("  Typical m = 5–20. The two-loop recursion computes H⁻¹g without forming H⁻¹."))
    print()
    print(white("  SuperLinear convergence: L-BFGS is typically much faster than first-order"))
    print(white("  methods for smooth, non-stochastic objectives. It is the standard choice"))
    print(white("  for full-batch fine-tuning and scientific computing optimisation."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Curvature pair: sₖ = xₖ₊₁ − xₖ,   yₖ = ∇fₖ₊₁ − ∇fₖ"))
    print(formula("  Curvature cond: sₖᵀyₖ > 0 (required for positive-definite update)"))
    print(formula("  BFGS update:    Hₖ₊₁⁻¹ = Vₖᵀ Hₖ⁻¹ Vₖ + ρₖ sₖsₖᵀ"))
    print(formula("  ρₖ = 1/(yₖᵀsₖ),   Vₖ = I − ρₖ yₖsₖᵀ"))
    print(formula("  Memory cost: O(mn)  vs  O(n²)  for full BFGS"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  Minimise Rosenbrock's function using scipy L-BFGS-B:"))
    import time
    x0 = np.array([-1.5, -0.5])

    # scipy L-BFGS-B
    try:
        from scipy.optimize import minimize
        res = minimize(_rosenbrock, x0, jac=_rosenbrock_grad, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-15})
        print(white(f"  L-BFGS-B result: x = {np.round(res.x, 6)},  f = {res.fun:.2e}"))
        print(white(f"  Iterations: {res.nit},  Function evals: {res.nfev}"))
        print(green(f"  ✓ Converged: {res.success}"))
        print(white(f"  Message: {res.message}"))
    except ImportError:
        print(grey("  scipy not available"))

    print()

    # Compare with GD
    x_gd = x0.copy(); alpha = 0.001
    it_gd = 0
    for _ in range(5000):
        g = _rosenbrock_grad(x_gd)
        x_gd -= alpha * g; it_gd += 1
        if np.linalg.norm(g) < 1e-6: break
    print(white(f"  GD iterations needed:      {it_gd}  (α=0.001)"))
    print(white(f"  L-BFGS iterations needed:  {res.nit}"))
    print(white(f"  Speedup:                   ≈ {it_gd // max(res.nit,1)}×"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Two-loop L-BFGS recursion (sketch):"))
    print()
    print(cyan("  First loop (backward):"))
    print(white("    for i = k, k-1, ..., k-m+1:"))
    print(white("        αᵢ = ρᵢ sᵢᵀ q"))
    print(white("        q  = q − αᵢ yᵢ"))
    print()
    print(cyan("  Scale:  r = H₀⁻¹ q  (usually r = (sₖᵀyₖ / yₖᵀyₖ) q)"))
    print()
    print(cyan("  Second loop (forward):"))
    print(white("    for i = k-m+1, ..., k:"))
    print(white("        β  = ρᵢ yᵢᵀ r"))
    print(white("        r  = r + sᵢ(αᵢ − β)"))
    print()
    print(white("  Output: H⁻¹g = r"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import convergence_plot
        import scipy.optimize as _so
        history = []
        def callback(x): history.append(float(_rosenbrock(x)))
        _so.minimize(_rosenbrock, x0.copy(), jac=_rosenbrock_grad,
                     method='L-BFGS-B', callback=callback,
                     options={'maxiter': 100})
        convergence_plot(history, title="L-BFGS-B Convergence on Rosenbrock", ylabel="f(x)")
    except Exception as e:
        try:
            from viz.terminal_plots import loss_curve
            loss_curve(history, title="L-BFGS-B Convergence", ylabel="f(x)")
        except Exception:
            print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        history_gd = []
        x_g = x0.copy()
        for _ in range(500):
            x_g -= 0.001 * _rosenbrock_grad(x_g)
            history_gd.append(_rosenbrock(x_g))
        plt.figure(figsize=(8, 4))
        plt.semilogy(history_gd,         label="GD (0.001)", color='tomato')
        if history:
            plt.semilogy(history,         label="L-BFGS-B",  color='steelblue')
        plt.xlabel("Iteration"); plt.ylabel("f(x) (log)")
        plt.title("GD vs L-BFGS-B on Rosenbrock"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("L-BFGS via scipy", """\
import numpy as np
from scipy.optimize import minimize

def rosenbrock(xy):
    x, y = xy
    return (1-x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    return np.array([-2*(1-x) - 400*x*(y-x**2),
                      200*(y - x**2)])

# L-BFGS-B (handles bounds; same algorithm without bounds = L-BFGS)
result = minimize(
    rosenbrock, x0=[-1.5, -0.5],
    jac=rosenbrock_grad,
    method='L-BFGS-B',
    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-8}
)

print("Converged:", result.success)
print("x* =", result.x)
print("f* =", result.fun)
print("Iterations:", result.nit)
print("Gradient evals:", result.njev)

# Also works for any smooth function:
from scipy.optimize import minimize as sci_min
res2 = sci_min(lambda x: np.sum((x - 1)**2 + (x+1)**2),
               x0=np.zeros(100), method='L-BFGS-B')
print("\\n100-dim quadratic:", res2.nit, "iters")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "L-BFGS uses O(mn) memory (m≈10); full BFGS needs O(n²) — critical for n > 10,000.",
        "Superlinear convergence: gradient norms drop faster than geometric (quadratic near solution).",
        "L-BFGS is the standard choice for full-batch smooth optimisation (scipy, torch LBFGS).",
        "For stochastic objectives (SGD regime), L-BFGS breaks down — curvature estimates are noisy.",
        "Two-loop recursion computes H⁻¹g implicitly in O(mn) — never form H⁻¹ explicitly.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Block runner
# ─────────────────────────────────────────────────────────────────────────────
def run():
    topics = [
        ("Gradient Descent",         topic_gradient_descent),
        ("SGD & Mini-batch",         topic_sgd),
        ("Momentum & Nesterov",      topic_momentum),
        ("Adam Optimizer",           topic_adam),
        ("LR Schedulers",            topic_lr_schedulers),
        ("Convexity",                topic_convexity),
        ("Saddle Points",            topic_saddle_points),
        ("Newton's Method",          topic_newtons_method),
        ("Conjugate Gradient",       topic_conjugate_gradient),
        ("L-BFGS",                   topic_lbfgs),
    ]
    block_menu("b04", "Optimisation", topics)
    mark_completed("b04")
