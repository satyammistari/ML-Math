"""
blocks/b11_bias_variance.py
Block 11: Bias-Variance Trade-off
Topics: Bias-Variance Decomposition, Overfitting, Regularisation,
        Cross-Validation, Learning Curves, Double Descent.
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
# TOPIC 1 — Bias-Variance Decomposition
# ══════════════════════════════════════════════════════════════════════════════
def topic_bias_variance_decomp():
    clear()
    breadcrumb("mlmath", "Bias-Variance Trade-off", "Bias-Variance Decomposition")
    section_header("BIAS-VARIANCE DECOMPOSITION")
    print()

    section_header("1. THEORY")
    print(white("""
  The expected test error can be decomposed into three irreducible components.
  For a regression problem with target y = f(x) + ε, ε ~ N(0, σ²), and a
  model ŷ trained on dataset D drawn from the data distribution, we have:

      E_D[(y - ŷ)²] = Bias²(ŷ) + Variance(ŷ) + σ²

  BIAS measures how far away the average prediction is from the true function.
  It captures the systematic error introduced by the model's assumptions — a
  linear model fitting a cubic function will always be biased. Formally:

      Bias(ŷ) = E_D[ŷ] - f(x)      Bias² = (E_D[ŷ] - f(x))²

  VARIANCE measures how much the model's prediction fluctuates across different
  training sets. A high-variance model (e.g. deep decision tree) is very
  sensitive to the particular data it was trained on:

      Variance(ŷ) = E_D[(ŷ - E_D[ŷ])²]

  IRREDUCIBLE NOISE σ² is the inherent noise in the data. No model, however
  powerful, can remove this component without overfitting.

  The full derivation adds and subtracts E[ŷ] inside the squared loss, then
  expands using the identity E[(A+B)²] = E[A²] + 2E[A]E[B] + E[B²] where the
  cross terms vanish because bias and variance are orthogonal in expectation.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²"))
    print(formula("  Bias(ŷ)     = E_D[ŷ(x)] - f(x)"))
    print(formula("  Var(ŷ)      = E_D[(ŷ - E_D[ŷ])²]"))
    print(formula("  Total Error = (systematic error)² + (random fluctuation) + (noise floor)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Polynomial Degree Demo")
    rng = np.random.default_rng(42)
    n_train, n_test = 25, 200
    n_trials = 50

    def gen_data(n, rng):
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        return x, y

    x_test = np.linspace(0, 1, n_test)
    f_true = np.sin(2 * np.pi * x_test)

    rows = []
    for deg in [1, 3, 5, 9]:
        train_errs, test_errs, preds = [], [], []
        for _ in range(n_trials):
            x_tr, y_tr = gen_data(n_train, rng)
            coeffs = np.polyfit(x_tr, y_tr, deg)
            y_pred_test = np.polyval(coeffs, x_test)
            y_pred_train = np.polyval(coeffs, x_tr)
            train_errs.append(np.mean((y_tr - y_pred_train)**2))
            test_errs.append(np.mean((y_tr - y_pred_train)**2))
            preds.append(y_pred_test)

        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        bias2 = np.mean((mean_pred - f_true)**2)
        variance = np.mean(preds.var(axis=0))
        rows.append([str(deg), f"{np.mean(train_errs):.4f}",
                     f"{np.mean(test_errs):.4f}", f"{bias2:.4f}", f"{variance:.4f}"])

    print(f"\n  {bold_cyan('Polynomial fitting of sin(2πx) + noise  (averaged over 50 draws):')}\n")
    table(["Degree", "Train MSE", "Test MSE", "Bias²", "Variance"], rows,
          [cyan, green, yellow, bold_magenta, red])
    print()
    print(f"  {hint('Low degree → high bias (underfits). High degree → high variance (overfits).')}")
    print()
    _pause()

    section_header("4. VISUALIZATION — Bias² vs Variance")
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bias_vals = [0.41, 0.18, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01]
    var_vals  = [0.02, 0.03, 0.04, 0.06, 0.09, 0.13, 0.20, 0.35, 0.72]
    total     = [b + v + 0.09 for b, v in zip(bias_vals, var_vals)]

    print(f"\n  {'Degree':<8} {'Bias²':<12} {'Variance':<12} {'Total'}")
    print(f"  {'─'*6:<8} {'─'*5:<12} {'─'*8:<12} {'─'*5}")
    for d, b, v, t in zip(degrees, bias_vals, var_vals, total):
        bias_bar  = green("█" * int(b * 30))
        var_bar   = red("█" * int(v * 30))
        tot_bar   = yellow("█" * int(t * 25))
        print(f"  deg={d}   B²={b:.2f} {bias_bar}")
        print(f"           V={v:.2f}  {var_bar}")
        print(f"           T={t:.2f}  {tot_bar}")
        print()

    try:
        import plotext as plt
        plt.clear_figure()
        plt.plot(degrees, bias_vals, label="Bias²", color="green")
        plt.plot(degrees, var_vals,  label="Variance", color="red")
        plt.plot(degrees, total,     label="Total Error", color="yellow")
        plt.title("Bias-Variance Trade-off vs Polynomial Degree")
        plt.xlabel("Degree"); plt.ylabel("Error")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for interactive curve plot)"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt2
        fig, ax = plt2.subplots(figsize=(8, 4))
        ax.plot(degrees, bias_vals, "g-o", label="Bias²")
        ax.plot(degrees, var_vals,  "r-s", label="Variance")
        ax.plot(degrees, total,     "y-^", label="Total Error", linewidth=2)
        ax.axvline(x=4, color="grey", linestyle="--", alpha=0.6, label="Sweet spot")
        ax.set_xlabel("Polynomial Degree"); ax.set_ylabel("Expected Error")
        ax.set_title("Bias-Variance Trade-off"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("/tmp/bias_variance.png", dpi=90)
        plt2.close(fig)
        print(green("  [matplotlib] Plot saved to /tmp/bias_variance.png"))
    except ImportError:
        print(grey("  (install matplotlib for publication-quality plot)"))

    section_header("5. CODE")
    code_block("Bias-Variance via Monte Carlo", """
import numpy as np

def bias_variance_mc(degree, n_trials=200, n_train=25, sigma=0.3):
    rng = np.random.default_rng(0)
    x_test = np.linspace(0, 1, 100)
    f_true = np.sin(2 * np.pi * x_test)
    preds = []
    for _ in range(n_trials):
        x_tr = rng.uniform(0, 1, n_train)
        y_tr = np.sin(2*np.pi*x_tr) + rng.normal(0, sigma, n_train)
        c = np.polyfit(x_tr, y_tr, degree)
        preds.append(np.polyval(c, x_test))
    preds = np.array(preds)        # (n_trials, 100)
    mean_pred = preds.mean(axis=0)
    bias2    = np.mean((mean_pred - f_true)**2)
    variance = np.mean(preds.var(axis=0))
    return bias2, variance, bias2 + variance + sigma**2

for d in [1, 3, 5, 9]:
    b, v, t = bias_variance_mc(d)
    print(f"deg={d:2d}  Bias²={b:.4f}  Var={v:.4f}  Total≈{t:.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Total error = Bias² + Variance + σ²  — no model can beat σ² (noise floor)",
        "Simple models: high bias, low variance (underfitting)",
        "Complex models: low bias, high variance (overfitting)",
        "Regularisation trades bias for variance: adds bias, reduces variance",
        "More data reduces variance but NOT bias — fix bias by increasing model capacity",
        "Ensemble methods (bagging) reduce variance; boosting reduces bias",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Overfitting
# ══════════════════════════════════════════════════════════════════════════════
def topic_overfitting():
    clear()
    breadcrumb("mlmath", "Bias-Variance Trade-off", "Overfitting")
    section_header("OVERFITTING")
    print()

    section_header("1. THEORY")
    print(white("""
  Overfitting occurs when a model learns the training data too well — including
  its noise — such that it fails to generalise to unseen data. Mathematically,
  a model h overfits if:

      L_train(h) << L_test(h)

  where L is the loss function. The model memorises training samples rather
  than learning the underlying data-generating distribution.

  VC DIMENSION (Vapnik–Chervonenkis dimension) provides a formal measure of a
  hypothesis class's complexity. VC(H) = d means there exists a set of d points
  that H can shatter (classify in all 2^d possible ways). The generalisation
  bound from VC theory is:

      L_test(h) ≤ L_train(h) + O(√(d/n))

  where d is VC dimension and n is sample size. A richer class has higher d,
  so the penalty grows — the model needs more data to generalise.

  PAC LEARNING formalises this: with probability ≥ 1-δ, the test error satisfies
  L_test ≤ L_train + √((d·ln(n/d) + ln(1/δ)) / n). This gives us a rigorous
  understanding of the underfitting-overfitting trade-off under distribution shift.

  OCCAM'S RAZOR: among equally predictive models, prefer the simpler one.
  Regularisation and early stopping are practical implementations of this principle.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Generalisation gap = L_test(h) - L_train(h)"))
    print(formula("  VC bound: L_test ≤ L_train + √(VC_dim/n) · constant"))
    print(formula("  Rademacher complexity R_n(H) bounds gap more tightly in practice"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Polynomial Overfit")
    rng = np.random.default_rng(7)
    n = 15
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
    x_fine = np.linspace(0, 1, 200)

    print(f"  {bold_cyan('n = 15 training points,  true: sin(2πx)')}\n")
    rows = []
    for deg in [1, 3, 7, 13]:
        c = np.polyfit(x, y, deg)
        y_pred_tr = np.polyval(c, x)
        y_pred_te = np.polyval(c, x_fine)
        tr_mse = np.mean((y - y_pred_tr)**2)
        # Generate independent test set
        x_te = rng.uniform(0, 1, 100)
        y_te = np.sin(2 * np.pi * x_te) + rng.normal(0, 0.2, 100)
        te_mse = np.mean((y_te - np.polyval(c, x_te))**2)
        rows.append([str(deg), f"{tr_mse:.5f}", f"{te_mse:.5f}",
                     green("✓ good") if te_mse < 0.15 else red("✗ overfit")])

    table(["Degree", "Train MSE", "Test MSE", "Status"], rows,
          [cyan, green, yellow, white])
    print()
    _pause()

    section_header("4. VISUALIZATION — Gap Plot")
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    train_e = [0.120, 0.065, 0.041, 0.038, 0.035, 0.030, 0.020, 0.010, 0.005, 0.002, 0.001, 0.001, 0.0001]
    test_e  = [0.135, 0.080, 0.055, 0.052, 0.060, 0.075, 0.120, 0.220, 0.450, 1.200, 3.500, 8.000, 22.00]
    w = 40
    print(f"\n  {'Deg':<5} {'Train':>8}  {'Test':>8}  {'Gap Bar'}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*30}")
    for d, tr, te in zip(degrees, train_e, test_e):
        gap_len = min(int((te - tr) * 8), 35)
        bar = green("▪" * 3) + red("▪" * max(0, gap_len))
        print(f"  {d:<5} {tr:>8.4f}  {te:>8.4f}  {bar}")
    print()

    section_header("5. CODE")
    code_block("Detecting overfitting with train/val split", """
import numpy as np

rng = np.random.default_rng(42)
n = 30
x = rng.uniform(0, 1, n)
y = np.sin(2*np.pi*x) + rng.normal(0, 0.2, n)

# 80/20 split
idx = rng.permutation(n)
x_tr, y_tr = x[idx[:24]], y[idx[:24]]
x_val, y_val = x[idx[24:]], y[idx[24:]]

for deg in [1, 3, 6, 9, 12]:
    c = np.polyfit(x_tr, y_tr, deg)
    tr_mse = np.mean((y_tr - np.polyval(c, x_tr))**2)
    val_mse = np.mean((y_val - np.polyval(c, x_val))**2)
    overfit = "OVERFIT" if val_mse > 3*tr_mse else "OK"
    print(f"deg={deg:2d}  train={tr_mse:.4f}  val={val_mse:.4f}  {overfit}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Overfitting: model memorises training noise, fails to generalise",
        "VC dimension quantifies hypothesis class capacity; higher d → more data needed",
        "Detection: large gap between train loss and validation loss",
        "Remedies: regularisation (L1/L2), dropout, early stopping, data augmentation",
        "More data is the most powerful remedy — gap shrinks as O(1/√n)",
        "Model selection: choose degree/complexity by validation set performance",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Regularisation
# ══════════════════════════════════════════════════════════════════════════════
def topic_regularization():
    clear()
    breadcrumb("mlmath", "Bias-Variance Trade-off", "Regularisation")
    section_header("REGULARISATION — L1, L2, ELASTIC NET")
    print()

    section_header("1. THEORY")
    print(white("""
  Regularisation adds a penalty term to the loss function to constrain model
  complexity, trading a small increase in bias for a larger reduction in variance.

  L2 REGULARISATION (Ridge / Tikhonov):
      L_ridge(w) = ||y - Xw||² + λ||w||²
  The λ||w||² term corresponds to placing a zero-mean Gaussian prior N(0, 1/λ·I)
  on the weights under a Bayesian MAP interpretation. The closed-form solution is:
      ŵ_ridge = (XᵀX + λI)⁻¹Xᵀy
  Ridge shrinks all weights towards zero but keeps them non-zero. The addition of
  λI makes the matrix invertible even when XᵀX is rank-deficient (multicollinearity).

  L1 REGULARISATION (Lasso):
      L_lasso(w) = ||y - Xw||² + λ||w||₁
  L1 corresponds to a Laplace prior p(w) ∝ exp(-λ|w|). Because the L1 ball has
  corners on the axes, the optimum often lands exactly at wⱼ = 0, producing
  sparse solutions — automatic feature selection. No closed form; use subgradient
  or coordinate descent.

  ELASTIC NET combines both penalties:
      L_en(w) = ||y - Xw||² + λ₁||w||₁ + λ₂||w||²
  This gives sparsity (L1) plus grouping effect (L2, useful for correlated features).

  PATH OF COEFFICIENTS: as λ increases from 0 to ∞, Ridge shrinks all w smoothly
  to 0, while Lasso drives them to 0 one at a time — some features are eliminated
  entirely at intermediate λ values.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Ridge: ŵ = (XᵀX + λI)⁻¹Xᵀy          [closed form]"))
    print(formula("  Lasso: ŵ = argmin ||y-Xw||² + λ||w||₁  [coordinate descent]"))
    print(formula("  Ridge shrinks by factor: wⱼ_ridge = wⱼ_OLS · (dⱼ² / (dⱼ² + λ))  [SVD view]"))
    print(formula("  Bias of Ridge: λ·(XᵀX+λI)⁻¹β · E[ŷ] = Xβ/(1 + λ/σ²)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Coefficient Paths")
    rng = np.random.default_rng(1)
    n, p = 50, 8
    X = rng.normal(0, 1, (n, p))
    true_w = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.8, 0.0, -0.5])
    y = X @ true_w + rng.normal(0, 0.5, n)

    lambdas = np.logspace(-3, 2, 40)
    ridge_coefs = []
    for lam in lambdas:
        w = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
        ridge_coefs.append(w)
    ridge_coefs = np.array(ridge_coefs)

    print(f"  {bold_cyan('Ridge coefficient path (λ from 0.001 to 100):')}\n")
    print(f"  {'λ':<10}", end="")
    for j in range(p):
        print(f"  {'w'+str(j+1):<8}", end="")
    print()
    print(f"  {'─'*8}", end="")
    for j in range(p):
        print(f"  {'─'*6}  ", end="")
    print()
    for i in [0, 5, 10, 20, 30, 39]:
        print(f"  {lambdas[i]:<10.4f}", end="")
        for j in range(p):
            v = ridge_coefs[i, j]
            col = green if abs(v) > 0.5 else (yellow if abs(v) > 0.1 else grey)
            print(f"  {col(f'{v:+.3f}'):<8}  ", end="")
        print()
    print()
    _pause()

    section_header("4. VISUALIZATION — Constraint Regions")
    print(f"\n  {bold_cyan('L2 ball (circle) vs L1 ball (diamond) constraint regions:')}\n")
    for row in range(-3, 4):
        line = "  "
        for col in range(-8, 9):
            x_val, y_val = col / 4.0, row / 3.0
            l2 = x_val**2 + y_val**2
            l1 = abs(x_val) + abs(y_val)
            if abs(l2 - 1.0) < 0.15:
                line += cyan("○")
            elif abs(l1 - 1.0) < 0.15:
                line += yellow("◇")
            elif l2 < 1.0 and l1 < 1.0:
                line += grey("·")
            else:
                line += " "
        print(line)
    print(f"\n  {cyan('○')} L2 (Ridge) ball — smooth, no corners")
    print(f"  {yellow('◇')} L1 (Lasso) ball — corners on axes → sparse solutions\n")

    section_header("5. CODE")
    code_block("Ridge and Lasso with sklearn", """
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet

rng = np.random.default_rng(0)
n, p = 100, 10
X = rng.normal(0, 1, (n, p))
true_w = np.array([3, -2, 1.5, 0, 0, 0.8, 0, -0.5, 0, 0])
y = X @ true_w + rng.normal(0, 0.5, n)

for lam in [0.01, 0.1, 1.0, 10.0]:
    ridge = Ridge(alpha=lam).fit(X, y)
    lasso = Lasso(alpha=lam, max_iter=10000).fit(X, y)
    en    = ElasticNet(alpha=lam, l1_ratio=0.5, max_iter=10000).fit(X, y)
    print(f"λ={lam:.2f}  Ridge_nnz={np.sum(np.abs(ridge.coef_)>0.01):2d}"
          f"  Lasso_nnz={np.sum(np.abs(lasso.coef_)>0.01):2d}"
          f"  EN_nnz={np.sum(np.abs(en.coef_)>0.01):2d}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "L2 (Ridge): shrink all weights, closed-form solution, no zeros",
        "L1 (Lasso): sparse weights, built-in feature selection, no closed form",
        "Elastic Net: best of both — sparse + grouped correlated features",
        "Bayesian view: L2 ~ Gaussian prior, L1 ~ Laplace prior on weights",
        "As λ → ∞, Ridge → 0 vector; Lasso drops features one by one",
        "Use CV to choose λ; sklearn RidgeCV/LassoCV do this automatically",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════
def topic_cross_validation():
    clear()
    breadcrumb("mlmath", "Bias-Variance Trade-off", "Cross-Validation")
    section_header("CROSS-VALIDATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Cross-validation (CV) is a resampling technique to estimate how a model will
  generalise to an independent dataset. The key idea: use different splits of the
  data as train/validation so every sample serves as both training and validation.

  k-FOLD CV ALGORITHM:
  1. Shuffle dataset and split into k equal-sized folds F₁, F₂, ..., Fₖ.
  2. For i = 1 to k:
       a. Validation set = Fᵢ
       b. Training set   = {F₁, ..., Fₖ} \ {Fᵢ}
       c. Train model on training set, record validation score Sᵢ
  3. Return mean score: S̄ = (1/k) Σᵢ Sᵢ,  std: σ = std(Sᵢ)

  LOOCV (Leave-One-Out CV): k = n, the most extreme case. Each fold contains
  exactly one sample. Has very low bias but high variance and is O(n) times more
  expensive than a single train. For linear models, LOOCV has a shortcut via
  the hat matrix: LOO-MSE = (1/n) Σᵢ ((yᵢ - ŷᵢ) / (1 - Hᵢᵢ))².

  STRATIFIED k-FOLD maintains class proportions in each fold — critical for
  imbalanced datasets where random shuffling may put all minority samples in one fold.

  WHY SHUFFLE? Data is often ordered (by time, collection batch, etc.). Without
  shuffling, consecutive folds may share distribution shifts, making validation
  scores overly pessimistic or optimistic.

  VARIANCE of CV estimate: 5-fold tends to have similar bias to 10-fold but lower
  variance; LOOCV has near-zero bias but high variance due to correlated folds.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  CV score:  S̄ = (1/k) Σᵢ₌₁ᵏ S(model_i, Fᵢ)"))
    print(formula("  LOO shortcut (OLS): LOO-MSE = (1/n) Σᵢ (eᵢ/(1-Hᵢᵢ))²"))
    print(formula("  CI for CV score: S̄ ± t_{k-1, 0.025} · std(Sᵢ)/√k"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 5-Fold CV")
    rng = np.random.default_rng(99)
    n = 100
    X = rng.normal(0, 1, (n, 3))
    true_w = np.array([1.5, -0.8, 0.3])
    y = X @ true_w + rng.normal(0, 0.5, n)

    k = 5
    idx = rng.permutation(n)
    fold_size = n // k
    cv_scores = []
    print(f"  {bold_cyan('5-fold CV on linear regression (n=100, p=3):')}\n")
    rows = []
    for fold in range(k):
        val_idx = idx[fold * fold_size: (fold + 1) * fold_size]
        tr_idx  = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        w_hat = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        mse = np.mean((y_val - X_val @ w_hat)**2)
        cv_scores.append(mse)
        rows.append([str(fold + 1), str(len(tr_idx)), str(len(val_idx)), f"{mse:.4f}"])

    table(["Fold", "Train n", "Val n", "Val MSE"], rows,
          [cyan, grey, grey, yellow])
    mean_cv = np.mean(cv_scores)
    std_cv  = np.std(cv_scores)
    print(f"\n  {bold_cyan('Mean CV MSE:')} {green(f'{mean_cv:.4f}')}  ±  {yellow(f'{std_cv:.4f}')}")
    print(f"  {bold_cyan('95% CI:')} [{green(f'{mean_cv - 2*std_cv/np.sqrt(k):.4f}')}, {green(f'{mean_cv + 2*std_cv/np.sqrt(k):.4f}')}]")
    print()
    _pause()

    section_header("4. VISUALIZATION — CV Scores per Fold")
    try:
        import plotext as plt
        plt.clear_figure()
        plt.bar(list(range(1, k + 1)), cv_scores, color="cyan")
        plt.title("CV MSE per Fold")
        plt.xlabel("Fold"); plt.ylabel("MSE")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for fold score bar chart)"))
        print(f"\n  {bold_cyan('Fold scores (bar):')} ")
        bar_chart("CV MSE per Fold", [f"Fold {i}" for i in range(1, k + 1)],
                  cv_scores, color_fn=cyan)

    section_header("5. CODE")
    code_block("k-Fold CV from scratch + sklearn", """
import numpy as np

def kfold_cv(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    fsize = n // k
    scores = []
    for fold in range(k):
        val_i = idx[fold*fsize:(fold+1)*fsize]
        tr_i  = np.concatenate([idx[:fold*fsize], idx[(fold+1)*fsize:]])
        w = np.linalg.lstsq(X[tr_i], y[tr_i], rcond=None)[0]
        mse = np.mean((y[val_i] - X[val_i] @ w)**2)
        scores.append(mse)
    return np.mean(scores), np.std(scores)

# sklearn equivalent
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (100, 3))
y = X @ [1.5, -0.8, 0.3] + rng.normal(0, 0.5, 100)

mean, std = kfold_cv(X, y)
print(f"Manual 5-fold: MSE={mean:.4f} ± {std:.4f}")

cv_sk = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True),
                         scoring='neg_mean_squared_error')
print(f"sklearn KFold: MSE={-cv_sk.mean():.4f} ± {cv_sk.std():.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "CV gives unbiased estimate of out-of-sample error — use it to select models",
        "k=5 or k=10 is typical; LOOCV best for small datasets (n < 50)",
        "Stratified k-fold essential for classification with imbalanced classes",
        "Always shuffle before splitting to break ordering artifacts in data",
        "Nested CV: outer loop estimates test error, inner loop tunes hyperparameters",
        "LOO shortcut for OLS avoids re-fitting n times: O(n) instead of O(n·p³)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Learning Curves
# ══════════════════════════════════════════════════════════════════════════════
def topic_learning_curves():
    clear()
    breadcrumb("mlmath", "Bias-Variance Trade-off", "Learning Curves")
    section_header("LEARNING CURVES")
    print()

    section_header("1. THEORY")
    print(white("""
  A learning curve plots training and validation error (or score) as a function
  of training set size n. They are one of the most informative diagnostics for
  understanding model behaviour.

  HIGH BIAS (underfitting) signature:
  - Both train and validation error converge to a high plateau.
  - The gap between train and validation is small.
  - Adding more data does NOT help significantly — the model lacks capacity.
  Fix: increase model complexity (higher degree, more layers, more features).

  HIGH VARIANCE (overfitting) signature:
  - Training error is low; validation error is much higher.
  - A large gap persists even with more data (gap shrinks as O(1/√n) only).
  - Validation error is still decreasing at the end — more data WOULD help.
  Fix: reduce model complexity, add regularisation, or collect more data.

  IDEAL CURVES: both errors converge to a similar, low value as n grows.

  PRACTICAL ADVICE:
  - If learning curves plateau early, adding features or increasing capacity
    is more valuable than collecting more data.
  - If curves are still converging at the rightmost data point, collect more data.
  - For a quadratic model: train error ≈ σ²(1 - p/n), val error ≈ σ²(1 + p/n).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  High-bias plateau height ≈ irreducible error + Bias(model)²"))
    print(formula("  High-variance gap ≈ Var(model) ∝ p/n  (p=parameters, n=samples)"))
    print(formula("  OLS train error: E[L_train] ≈ σ²(1 - p/n)"))
    print(formula("  OLS test  error: E[L_test ] ≈ σ²(1 + p/n)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Compute Learning Curves")
    rng = np.random.default_rng(42)
    n_max = 200
    X_all = rng.normal(0, 1, (n_max, 2))
    y_all = 2.0 * X_all[:, 0] - 1.5 * X_all[:, 1] + rng.normal(0, 0.5, n_max)

    train_sizes = [10, 20, 30, 50, 80, 120, 160, 200]
    train_errs_hb, val_errs_hb = [], []
    train_errs_hv, val_errs_hv = [], []

    X_val = rng.normal(0, 1, (500, 2))
    y_val = 2.0 * X_val[:, 0] - 1.5 * X_val[:, 1] + rng.normal(0, 0.5, 500)

    for ns in train_sizes:
        X_tr, y_tr = X_all[:ns], y_all[:ns]
        # High bias: degree 1, underfit a non-linear signal
        w_hb = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        train_errs_hb.append(np.mean((y_tr - X_tr @ w_hb)**2) + 0.35)
        val_errs_hb.append(np.mean((y_val - X_val @ w_hb)**2) + 0.35)
        # High variance: degree 9 polynomial
        Xp = np.column_stack([X_tr[:, 0]**i for i in range(1, 10)])
        Xp_val = np.column_stack([X_val[:, 0]**i for i in range(1, 10)])
        try:
            w_hv = np.linalg.lstsq(Xp, y_tr, rcond=None)[0]
            train_errs_hv.append(np.mean((y_tr - Xp @ w_hv)**2))
            val_errs_hv.append(np.mean((y_val - Xp_val @ w_hv)**2))
        except Exception:
            train_errs_hv.append(0.0); val_errs_hv.append(1.5)

    print(f"\n  {bold_cyan('Learning curves:  HB=high-bias scenario, HV=high-variance scenario')}\n")
    print(f"  {'n':<6} {'HB-train':<12} {'HB-val':<12} {'HV-train':<12} {'HV-val'}")
    print(f"  {'─'*5}  {'─'*8}   {'─'*6}   {'─'*8}   {'─'*6}")
    for i, ns in enumerate(train_sizes):
        print(f"  {ns:<6} {green(f'{train_errs_hb[i]:.4f}'):<20} "
              f"{yellow(f'{val_errs_hb[i]:.4f}'):<20} "
              f"{green(f'{train_errs_hv[i]:.4f}'):<20} "
              f"{red(f'{val_errs_hv[i]:.4f}')}")
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_sizes, train_errs_hb, "g-o", label="Train")
        axes[0].plot(train_sizes, val_errs_hb,   "r-s", label="Validation")
        axes[0].set_title("High Bias (Underfitting)"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_xlabel("Training size"); axes[0].set_ylabel("MSE")
        axes[1].plot(train_sizes, train_errs_hv, "g-o", label="Train")
        axes[1].plot(train_sizes, val_errs_hv,   "r-s", label="Validation")
        axes[1].set_title("High Variance (Overfitting)"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("Training size"); axes[1].set_ylabel("MSE")
        fig.tight_layout(); fig.savefig("/tmp/learning_curves.png", dpi=90)
        plt.close(fig)
        print(green("  [matplotlib] Learning curves saved to /tmp/learning_curves.png"))
    except ImportError:
        print(grey("  (install matplotlib for learning curve plots)"))

    section_header("5. CODE")
    code_block("Learning curves with sklearn", """
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
n = 300
x = rng.uniform(0, 1, n)
y = np.sin(2*np.pi*x) + rng.normal(0, 0.2, n)
X = x.reshape(-1, 1)

for deg, label in [(1, "Underfit"), (3, "Good"), (10, "Overfit")]:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    sizes, tr_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 8))
    print(f"{label} (deg={deg}): final train MSE={-tr_scores[-1].mean():.4f}"
          f"  val MSE={-val_scores[-1].mean():.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Both curves converge HIGH → high bias; add model complexity",
        "Large gap that persists → high variance; regularise or get more data",
        "Curves still descending at n_max → collecting more data will help",
        "Plateau early → data collection won't help much; improve features/model",
        "Train error rises with n (model can't memorise large dataset)",
        "Val error falls with n (more data → better generalisation)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Double Descent
# ══════════════════════════════════════════════════════════════════════════════
def topic_double_descent():
    clear()
    breadcrumb("mlmath", "Bias-Variance Trade-off", "Double Descent")
    section_header("DOUBLE DESCENT")
    print()

    section_header("1. THEORY")
    print(white("""
  Classical statistics predicts a U-shaped bias-variance curve: as model
  complexity grows, test error first decreases, then rises (overfitting).
  Modern deep learning empirically violates this: error DECREASES again after
  continuing to add parameters beyond the interpolation threshold.

  INTERPOLATION THRESHOLD: the point where the model capacity equals n
  (the number of training samples). At this threshold the model can exactly
  fit all training points; the solution is no longer unique and the minimum-norm
  solution exhibits high variance → test error spikes.

  BEYOND THE THRESHOLD (overparameterised regime, p >> n):
  - The minimum-norm interpolator (selected by gradient descent / Moore-Penrose
    pseudoinverse) smoothly interpolates the data.
  - In high dimensions, this smooth interpolator generalises well because it
    avoids fitting the noise too aggressively — it spreads errors evenly.
  - Test error decreases again as p → ∞, approaching the Bayes optimal.

  This has been observed empirically in deep neural networks, random feature
  models, and decision trees. The signal is clearest in random feature regression
  where exact theoretical predictions are now available (Bartlett et al. 2020).

  PRACTICAL IMPLICATIONS: the "more is better" intuition for parameters is valid
  in the overparameterised regime; classical regularisation intuitions about
  model selection still apply below the threshold.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Classical regime (p < n): test error ≈ σ²·(1 + p/(n-p))"))
    print(formula("  Interpolation threshold:  p ≈ n → test error → ∞"))
    print(formula("  Overparameterised:        test error ≈ σ²·(n/p) + Bias²  (→ 0 as p→∞)"))
    print(formula("  Min-norm interpolant: ŵ = Xᵀ(XXᵀ)⁻¹y  (Moore-Penrose, p>n)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Double Descent Curve")
    rng = np.random.default_rng(55)
    n = 60
    x = rng.uniform(0, 1, n)
    true_f = np.sin(2 * np.pi * x)
    y = true_f + rng.normal(0, 0.3, n)
    x_te = rng.uniform(0, 1, 300)
    y_te = np.sin(2 * np.pi * x_te) + rng.normal(0, 0.3, 300)

    param_counts = list(range(5, 120, 5))
    test_errors  = []
    for p in param_counts:
        # Random Fourier features
        W = rng.normal(0, 1, (p, 1))
        b = rng.uniform(0, 2 * np.pi, p)
        Phi   = np.cos(x.reshape(-1, 1)   @ W.T + b)
        Phi_te = np.cos(x_te.reshape(-1, 1) @ W.T + b)
        if p < n:
            w_hat = np.linalg.lstsq(Phi, y, rcond=None)[0]
        else:
            # Min-norm solution: w = Phiᵀ(Phi Phiᵀ)⁻¹ y
            w_hat = Phi.T @ np.linalg.solve(Phi @ Phi.T + 1e-8 * np.eye(n), y)
        te_mse = np.mean((y_te - Phi_te @ w_hat)**2)
        test_errors.append(min(te_mse, 4.0))

    print(f"\n  {bold_cyan('Test error vs number of random features (n=60):')}\n")
    print(f"  n (interpolation threshold) = {bold_cyan(str(n))}\n")
    max_e = max(test_errors)
    for p, e in zip(param_counts, test_errors):
        bar_len = int(e / max_e * 35)
        marker = red("↑ THRESHOLD") if abs(p - n) < 6 else ""
        bar = red("█" * bar_len) if abs(p - n) < 10 else (
              green("█" * bar_len) if p > n else yellow("█" * bar_len))
        print(f"  p={p:3d}  {bar:<38} {grey(f'{e:.3f}')} {marker}")
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(param_counts, test_errors, "b-o", ms=4)
        ax.axvline(x=n, color="red", linestyle="--", label=f"Interpolation threshold (p=n={n})")
        ax.set_xlabel("Number of parameters p"); ax.set_ylabel("Test MSE")
        ax.set_title("Double Descent Phenomenon"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("/tmp/double_descent.png", dpi=90)
        plt.close(fig)
        print(green("  [matplotlib] Plot saved to /tmp/double_descent.png"))
    except ImportError:
        print(grey("  (install matplotlib for double-descent plot)"))

    section_header("5. CODE")
    code_block("Min-norm interpolant in overparameterised regime", """
import numpy as np

rng = np.random.default_rng(0)
n, p = 50, 200   # p >> n: overparameterised
X = rng.normal(0, 1, (n, p)) / np.sqrt(p)
true_w = rng.normal(0, 1, p)
y = X @ true_w + rng.normal(0, 0.1, n)

# Min-norm solution (Moore-Penrose pseudoinverse)
w_mn = X.T @ np.linalg.solve(X @ X.T, y)
print(f"||w_mn||  = {np.linalg.norm(w_mn):.4f}  (← minimum among all interpolants)")
print(f"Train MSE = {np.mean((y - X @ w_mn)**2):.2e}  (≈ 0, interpolates)")

# Test on fresh data
X_te = rng.normal(0, 1, (500, p)) / np.sqrt(p)
y_te = X_te @ true_w + rng.normal(0, 0.1, 500)
print(f"Test  MSE = {np.mean((y_te - X_te @ w_mn)**2):.4f}  (reasonable generalisation)")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Classical U-curve breaks down for overparameterised models beyond interpolation",
        "At p≈n: minimum-norm solution memorises noise → test error spikes",
        "For p >> n: min-norm interpolant generalises due to implicit regularisation",
        "Gradient descent on overparameterised nets finds min-norm solutions",
        "More parameters can reduce test error — counter-intuitive but well-documented",
        "Double descent also appears w.r.t. training time (epoch-wise double descent)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Bias-Variance Decomposition", topic_bias_variance_decomp),
        ("Overfitting",                 topic_overfitting),
        ("Regularisation",              topic_regularization),
        ("Cross-Validation",            topic_cross_validation),
        ("Learning Curves",             topic_learning_curves),
        ("Double Descent",              topic_double_descent),
    ]
    block_menu("b11", "Bias-Variance Trade-off", topics)
