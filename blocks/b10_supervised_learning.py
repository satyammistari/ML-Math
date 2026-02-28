"""
blocks/b10_supervised_learning.py
Block 10: Supervised Learning Algorithms
Topics: Linear Regression, Ridge/Lasso, Logistic Regression, SVM, Decision Trees,
        Random Forest, Gradient Boosting, Naive Bayes, KNN, Perceptron.
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
from viz.ascii_plots import scatter, neural_net_diagram
from viz.terminal_plots import distribution_plot, loss_curve_plot
from viz.matplotlib_plots import plot_decision_boundary, show_heatmap


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — Linear Regression
# ══════════════════════════════════════════════════════════════════════════════
def topic_linear_regression():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Linear Regression")
    section_header("LINEAR REGRESSION — OLS")
    print()

    section_header("1. THEORY")
    print(white("""
  Model: ŷ = Xβ,  where X ∈ ℝⁿˣᵖ (design matrix), β ∈ ℝᵖ.

  OBJECTIVE: minimise squared residual sum
      L(β) = ||y - Xβ||² = (y-Xβ)ᵀ(y-Xβ)

  NORMAL EQUATIONS:
  Set ∂L/∂β = -2Xᵀ(y-Xβ) = 0 → Xᵀy = XᵀXβ
      β̂ = (XᵀX)⁻¹Xᵀy

  GEOMETRIC INTERPRETATION:
  ŷ = Xβ̂ is the orthogonal projection of y onto column space of X (col(X)).
  Residual ê = y - ŷ ⊥ col(X):    Xᵀ(y - Xβ̂) = 0

  GAUSS-MARKOV THEOREM:
  Under homoscedastic, uncorrelated errors with mean 0, OLS is BLUE:
  Best Linear Unbiased Estimator (minimum variance among all linear unbiased estimators).

  ASSUMPTIONS:
  1. Linearity: E[y|X] = Xβ
  2. Independence: observations are independent
  3. Homoscedasticity: Var(εᵢ) = σ² for all i
  4. No perfect multicollinearity: rank(X) = p
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  β̂ = (XᵀX)⁻¹Xᵀy"))
    print(formula("  ŷ = Xβ̂ = X(XᵀX)⁻¹Xᵀy = Hy  (H = hat/projection matrix)"))
    print(formula("  Var(β̂) = σ²(XᵀX)⁻¹"))
    print(formula("  R² = 1 - SS_res/SS_tot = 1 - ||y-ŷ||²/||y-ȳ||²"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    rng = np.random.default_rng(0)
    n = 20
    x_raw = rng.uniform(0, 10, n)
    y_true = 2.5 * x_raw + 4.0
    y_obs  = y_true + rng.normal(0, 1.5, n)

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x_raw])
    beta_hat = np.linalg.lstsq(X, y_obs, rcond=None)[0]
    y_hat = X @ beta_hat
    ss_res = np.sum((y_obs - y_hat)**2)
    ss_tot = np.sum((y_obs - y_obs.mean())**2)
    r2 = 1.0 - ss_res / ss_tot

    print(bold_cyan(f"  True model: y = 2.5x + 4.0 + ε,  ε ~ N(0, 1.5²)"))
    print(bold_cyan(f"  n = {n} samples"))
    print()
    print(f"  Estimated intercept: {green(f'{beta_hat[0]:.4f}')} (true: 4.0)")
    print(f"  Estimated slope:     {green(f'{beta_hat[1]:.4f}')} (true: 2.5)")
    print(f"  R² = {green(f'{r2:.4f}')}  (1.0 = perfect fit)")
    print(f"  RMSE = {green(f'{np.sqrt(ss_res/n):.4f}')}")
    print()

    # ASCII scatter
    print(cyan("  ASCII scatter: ● data,  ─ fit line"))
    xs_line = np.linspace(0, 10, 20)
    print()
    for yi, xi in sorted(zip(y_obs, x_raw)):
        xi_norm = int(xi / 10 * 30)
        yi_norm = int((yi - y_obs.min()) / (y_obs.max() - y_obs.min()) * 15)
        print("  " + " " * xi_norm + cyan("●"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("OLS via Normal Equations and sklearn", """
import numpy as np

def ols(X, y):
    '''Ordinary Least Squares: beta = (XtX)^{-1} Xt y'''
    return np.linalg.solve(X.T @ X, X.T @ y)

# Generate data
rng = np.random.default_rng(42)
n = 100
x = rng.uniform(0, 10, n)
y = 2.5*x + 4.0 + rng.normal(0, 1.5, n)
X = np.column_stack([np.ones(n), x])

beta = ols(X, y)
print(f"intercept={beta[0]:.3f},  slope={beta[1]:.3f}")

# Equivalent with sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(x.reshape(-1,1), y)
print(f"sklearn: intercept={lr.intercept_:.3f}, slope={lr.coef_[0]:.3f}")

# Hat matrix (projection)
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"H is idempotent (H²=H): {np.allclose(H @ H, H)}")
print(f"rank(H) = {np.linalg.matrix_rank(H)} = p = 2")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Normal equation β̂=(XᵀX)⁻¹Xᵀy is exact but O(p³) — slow for high-dim",
        "ŷ = Hy is orthogonal projection onto col(X); residuals ⊥ col(X)",
        "Gauss-Markov: OLS is BLUE under correct assumptions",
        "If XᵀX is ill-conditioned (multicollinearity), use Ridge regularisation",
        "R² can increase just by adding features — use adj-R² or cross-validation",
        "Gradient descent converges to same solution: update β ← β + (α/n)Xᵀ(y-Xβ)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Ridge / Lasso
# ══════════════════════════════════════════════════════════════════════════════
def topic_ridge_lasso():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Ridge and Lasso")
    section_header("RIDGE (L2) AND LASSO (L1) REGRESSION")
    print()

    section_header("1. THEORY")
    print(white("""
  REGULARISED REGRESSION adds a penalty on the coefficient magnitude:

  RIDGE (Tikhonov, L2 penalty):
      L_ridge(β) = ||y - Xβ||² + λ||β||²
  Closed-form solution:
      β̂_ridge = (XᵀX + λI)⁻¹Xᵀy
  Effect: shrinks all coefficients toward 0, never exactly 0.
  Useful when: many small effects, multicollinearity.

  LASSO (L1 penalty):
      L_lasso(β) = ||y - Xβ||² + λ||β||₁
  No closed form — solved with subgradient or proximal methods.
  Effect: many coefficients become EXACTLY 0 → automatic feature selection.
  Useful when: sparse signal, feature selection needed.

  GEOMETRY:
  Constraint view: minimise ||y-Xβ||² subject to ||β||₂ ≤ t (Ridge, circular ball)
                                               or ||β||₁ ≤ t (Lasso, diamond)
  The corners of the L1 diamond are on the axes → likely intersection has βⱼ=0.
  The L2 circle has no corners → solutions typically don't have exact zeros.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Ridge: β̂ = (XᵀX + λI)⁻¹Xᵀy"))
    print(formula("  Ridge: β̂ⱼ = β̂_ols · dⱼ² / (dⱼ² + λ)   (SVD form, dⱼ = singular values)"))
    print(formula("  Lasso: no closed form — coordinate descent update:"))
    print(formula("    β̂ⱼ ← S(rⱼ, λ) / ||Xⱼ||²"))
    print(formula("    S(z,λ) = sign(z)·max(|z|-λ, 0)  (soft-thresholding)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Ridge Shrinkage Path")
    rng = np.random.default_rng(7)
    n, p = 50, 10
    X = rng.standard_normal((n, p))
    # True: only first 3 features matter
    beta_true = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta_true + rng.normal(0, 0.5, n)

    lambdas = [0.001, 0.1, 1.0, 10.0, 100.0]
    print(bold_cyan("  Ridge coefficient shrinkage as λ increases:"))
    print(bold_cyan(f"  {'λ':>8}  {'β₀':>8}  {'β₁':>8}  {'β₂':>8}  {'β₃ (true=0)':>14}"))
    print(grey("  " + "─"*55))
    for lam in lambdas:
        beta_r = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
        spars = f"{np.sum(np.abs(beta_r) < 0.01):>3} zeros"
        print(f"  {lam:>8.3f}  {beta_r[0]:>8.4f}  {beta_r[1]:>8.4f}  {beta_r[2]:>8.4f}  {beta_r[3]:>14.6f}  ({spars})")
    print()
    print(green("  Note: Ridge never reaches exact 0; Lasso would zero out β₃…β₉"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Ridge and Lasso Regression", """
import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV

rng = np.random.default_rng(42)
n, p = 100, 20
X = rng.standard_normal((n, p))
beta_true = np.zeros(p); beta_true[:5] = [3, -2, 1.5, -1, 0.5]
y = X @ beta_true + rng.normal(0, 1, n)

# Ridge
ridge = Ridge(alpha=1.0).fit(X, y)
print(f"Ridge non-zero coefs: {np.sum(np.abs(ridge.coef_) > 0.01)} / {p}")

# Lasso
lasso = Lasso(alpha=0.1).fit(X, y)
print(f"Lasso non-zero coefs: {np.sum(np.abs(lasso.coef_) > 0.01)} / {p}")

# Cross-validated lambda selection
ridgecv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X, y)
print(f"Best Ridge alpha: {ridgecv.alpha_:.4f}")

lassocv = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5).fit(X, y)
print(f"Best Lasso alpha: {lassocv.alpha_:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Ridge: closed form solution (XᵀX+λI)⁻¹Xᵀy — handles multicollinearity",
        "Lasso: soft-thresholding produces EXACT zeros — built-in feature selection",
        "Elastic Net: α||β||₁ + (1-α)||β||₂² combines both (sklearn ElasticNet)",
        "LASSO sparsity geometry: L1 ball has corners on axes where βⱼ=0",
        "Ridge bias-variance trade-off: higher λ → more bias, less variance",
        "Choose λ via cross-validation — RidgeCV/LassoCV does this efficiently",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
def topic_logistic_regression():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Logistic Regression")
    section_header("LOGISTIC REGRESSION")
    print()

    section_header("1. THEORY")
    print(white("""
  Logistic regression models P(y=1|x) = σ(wᵀx + b) where σ = sigmoid.

  LOG-LIKELIHOOD (binary cross-entropy):
      ℓ(w) = Σᵢ [yᵢ log σ(wᵀxᵢ) + (1-yᵢ) log(1 - σ(wᵀxᵢ))]

  GRADIENT (elegant!):
      ∂ℓ/∂w = Σᵢ (yᵢ - σ(wᵀxᵢ)) xᵢ  = Xᵀ(y - ŷ)
  where ŷᵢ = σ(wᵀxᵢ).  Gradient = error × feature.

  HESSIAN (for Newton-Raphson):
      H = -XᵀSX,  where S = diag(ŷᵢ(1-ŷᵢ))
  Hessian is negative semi-definite → log-likelihood is CONCAVE → unique maximum.

  DECISION BOUNDARY:
  σ(wᵀx) = 0.5 ↔ wᵀx = 0 — a linear hyperplane.
  Logistic regression is a LINEAR classifier.

  MULTINOMIAL (softmax) EXTENSION:
  P(y=k|x) = softmax(Wx + b)ₖ — one weight vector per class.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  P(y=1|x) = σ(wᵀx + b)"))
    print(formula("  ℓ(w) = Σᵢ [yᵢ log ŷᵢ + (1-yᵢ) log(1-ŷᵢ)]"))
    print(formula("  ∇ℓ = Xᵀ(y - ŷ)   (same form as linear regression!)"))
    print(formula("  Update (SGD): w ← w + η (y-ŷ) x"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Gradient Descent Training")
    rng = np.random.default_rng(43)
    n, p = 200, 2
    X_raw = rng.standard_normal((n, p))
    w_true = np.array([1.5, -2.0])
    b_true = 0.5
    probs_true = 1.0 / (1.0 + np.exp(-(X_raw @ w_true + b_true)))
    y_labels = (rng.uniform(size=n) < probs_true).astype(float)

    # Add bias column
    X_train = np.column_stack([np.ones(n), X_raw])
    w = np.zeros(p + 1)

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))
    def bce(yt, yp): return -np.mean(yt * np.log(yp + 1e-9) + (1 - yt) * np.log(1 - yp + 1e-9))

    lr = 0.1
    print(bold_cyan("  Training logistic regression via gradient ascent"))
    print(bold_cyan(f"  {'Epoch':>6}  {'Log-loss':>12}  {'Accuracy':>10}"))
    print(grey("  " + "─"*35))
    for epoch in range(300):
        y_hat = sigmoid(X_train @ w)
        grad  = X_train.T @ (y_labels - y_hat) / n
        w    += lr * grad
        if epoch % 50 == 0:
            loss = bce(y_labels, y_hat)
            acc  = np.mean((y_hat > 0.5) == y_labels.astype(bool))
            print(f"  {epoch:>6}  {loss:>12.6f}  {acc:>10.4f}")

    print()
    print(green(f"  Final weights:  [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]"))
    print(green(f"  True (+ bias):  [{b_true:.3f}, {w_true[0]:.3f}, {w_true[1]:.3f}]"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Logistic Regression from Scratch", """
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression(X, y, lr=0.1, n_iter=500):
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        yhat = sigmoid(X @ w)
        grad = X.T @ (y - yhat) / len(y)
        w   += lr * grad
    return w

def predict_proba(X, w): return sigmoid(X @ w)
def predict(X, w):       return (predict_proba(X, w) >= 0.5).astype(int)

# With sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=5, random_state=0)
model = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
print(f"sklearn accuracy: {model.score(X, y):.3f}")
print(f"Coefficients: {np.round(model.coef_[0], 3)}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Gradient ∇ℓ = Xᵀ(y-ŷ): same form as linear regression — elegant symmetry",
        "Log-likelihood is CONCAVE → unique global maximum, no local optima",
        "Decision boundary is LINEAR (hyperplane wᵀx = 0)",
        "C in sklearn = 1/λ — LARGER C means LESS regularisation",
        "For multi-class: multinomial (one-vs-rest or softmax with label smoothing)",
        "Newton-Raphson converges in ~5-10 iterations; SGD needs more but scales better",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — SVM
# ══════════════════════════════════════════════════════════════════════════════
def topic_svm():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "SVM")
    section_header("SUPPORT VECTOR MACHINE")
    print()

    section_header("1. THEORY")
    print(white("""
  SVM finds the maximum-margin hyperplane separating two classes.

  HARD MARGIN (linearly separable):
  Minimise  (1/2)||w||²  subject to  yᵢ(wᵀxᵢ + b) ≥ 1  for all i.
  The margin is 2/||w||.  Support vectors: points with yᵢ(wᵀxᵢ+b) = 1.

  SOFT MARGIN (introduces C):
  Minimise  (1/2)||w||² + C Σᵢ ξᵢ
  subject to  yᵢ(wᵀxᵢ+b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
  C controls trade-off: large C → hard-margin (low bias, high variance),
                        small C → wide margin (high bias, low variance).

  DUAL PROBLEM & KERNEL TRICK:
  Dual: maximise Σᵢαᵢ - (1/2)Σᵢ,ⱼ αᵢαⱼ yᵢyⱼ xᵢᵀxⱼ
        subject to 0 ≤ αᵢ ≤ C,  Σᵢ αᵢyᵢ = 0
  The data appears only as inner products xᵢᵀxⱼ.
  KERNEL TRICK: replace xᵢᵀxⱼ with K(xᵢ,xⱼ) to map to higher dimensions
  without computing the mapping explicitly!

  Common kernels:
  • Linear:  K(x,z) = xᵀz
  • RBF:     K(x,z) = exp(-γ||x-z||²)   most common non-linear choice
  • Poly:    K(x,z) = (xᵀz + c)ᵈ
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Margin = 2/||w||  — maximise by minimising (1/2)||w||²"))
    print(formula("  KKT: αᵢ[yᵢ(wᵀxᵢ+b) - 1] = 0  → αᵢ>0 only at support vectors"))
    print(formula("  w = Σᵢ αᵢyᵢxᵢ  (weight as sum of support vectors)"))
    print(formula("  Predict: sign(Σᵢ αᵢyᵢ K(xᵢ,x) + b)"))
    print(formula("  RBF kernel: K(x,z)=exp(-γ||x-z||²)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — SVM with sklearn")
    from sklearn.svm import SVC
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X_data, y_data = make_moons(n_samples=200, noise=0.2, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    print(bold_cyan("  Comparison: Linear vs RBF SVM on 'moons' dataset"))
    print(bold_cyan(f"  {'Kernel':>10}  {'C':>6}  {'gamma':>8}  {'TrainAcc':>10}  {'TestAcc':>10}  {'#SVs':>6}"))
    print(grey("  " + "─"*60))
    for kernel, C, gamma in [("linear", 1.0, "scale"), ("linear", 0.01, "scale"),
                               ("rbf",   1.0, 0.5),    ("rbf",    10.0,  2.0)]:
        clf = SVC(kernel=kernel, C=C, gamma=gamma).fit(Xtr_s, ytr)
        tr_acc = clf.score(Xtr_s, ytr)
        te_acc = clf.score(Xte_s, yte)
        n_sv   = clf.support_vectors_.shape[0]
        print(f"  {kernel:>10}  {C:>6.2f}  {str(gamma):>8}  {tr_acc:>10.4f}  {te_acc:>10.4f}  {n_sv:>6}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("SVM with Kernel Trick", """
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
X = StandardScaler().fit_transform(X)

# Linear SVM (cannot separate circles)
lin = SVC(kernel='linear', C=1.0).fit(X, y)
print(f"Linear SVM accuracy: {lin.score(X, y):.3f}")

# RBF SVM (kernel trick maps to high-dim where circles are separable)
rbf = SVC(kernel='rbf', C=1.0, gamma=1.0).fit(X, y)
print(f"RBF SVM accuracy:    {rbf.score(X, y):.3f}")
print(f"Number of support vectors: {rbf.support_vectors_.shape[0]}")
print(f"Support vector weights (alpha): {rbf.dual_coef_.shape}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "SVM maximises margin — width 2/||w||: only support vectors define the boundary",
        "Kernel trick: implicitly maps to high-dim feature space via K(x,z) inner products",
        "C controls bias-variance: large C = smaller margin = potential overfit",
        "RBF kernel is the default choice when data is not linearly separable",
        "γ in RBF: large γ = very local, complex boundary; small γ = smoother",
        "SVMs work well in high-dim, low-n settings; scale poorly to n > 100k",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Decision Trees
# ══════════════════════════════════════════════════════════════════════════════
def topic_decision_trees():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Decision Trees")
    section_header("DECISION TREES")
    print()

    section_header("1. THEORY")
    print(white("""
  Decision trees recursively partition the feature space with axis-aligned splits.

  SPLITTING CRITERIA:
  • Information Gain (ID3, C4.5):  IG = H(parent) - Σₖ (|Dₖ|/|D|) H(Dₖ)
    where H(D) = -Σₚ pⱼ log₂ pⱼ  (Shannon entropy)
  • Gini Impurity (CART):  Gini(D) = 1 - Σⱼ pⱼ²
    ΔGini = Gini(parent) - Σₖ (|Dₖ|/|D|) Gini(Dₖ)
  • Both measure the reduction in impurity after the split.

  STOPPING CRITERIA:
  • All samples in leaf have same class
  • max_depth exceeded
  • min_samples_leaf not satisfied
  • ΔImpurity < threshold

  OVERFITTING:
  Fully grown trees memorise training data (overfit).
  PRUNING: cost-complexity pruning, min_samples_split, max_leaf_nodes.

  PROPERTIES:
  • Handles non-linear boundaries (axis-aligned step functions)
  • Handles mixed feature types and missing data (with modifications)
  • Very fast prediction: O(depth) per sample
  • Unstable — small data changes can produce very different trees (high variance)
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Entropy:  H(D) = -Σⱼ pⱼ log₂ pⱼ"))
    print(formula("  Gini:     Gini(D) = 1 - Σⱼ pⱼ²"))
    print(formula("  Info Gain: IG = H(parent) - Σₖ wₖ H(Dₖ)"))
    print(formula("  MSE (regression): loss = (1/|D|) Σᵢ (yᵢ - ȳ)²"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Manual Gini Split")
    # 10 samples, features: [x1, x2], labels: [0,1]
    rng = np.random.default_rng(22)
    data_x = rng.uniform(0, 4, (10, 1))
    data_y = (data_x.flatten() > 2.0).astype(int)

    def gini(labels):
        if len(labels) == 0: return 0.0
        p = np.bincount(labels, minlength=2) / len(labels)
        return 1.0 - np.sum(p**2)

    thresholds = np.linspace(0.5, 3.5, 10)
    print(bold_cyan(f"  {'Threshold':>12}  {'Left Gini':>12}  {'Right Gini':>12}  {'Weighted ΔGini':>16}")   )
    print(grey("  " + "─"*60))
    parent_gini = gini(data_y)
    best_thresh, best_gain = None, -np.inf
    for t in thresholds:
        mask_l = data_x.flatten() <= t
        mask_r = ~mask_l
        g_l = gini(data_y[mask_l]) if mask_l.sum() > 0 else 0
        g_r = gini(data_y[mask_r]) if mask_r.sum() > 0 else 0
        wg  = (mask_l.sum() * g_l + mask_r.sum() * g_r) / len(data_y)
        gain = parent_gini - wg
        mark = green("← best") if gain > best_gain else ""
        if gain > best_gain:
            best_gain, best_thresh = gain, t
        print(f"  x1 ≤ {t:<8.2f}  {g_l:>12.4f}  {g_r:>12.4f}  {gain:>16.4f} {mark}")
    print()
    print(bold_cyan(f"  Best split: x1 ≤ {best_thresh:.2f},  ΔGini = {best_gain:.4f}"))
    print()
    _pause()

    section_header("4. ASCII TREE DIAGRAM")
    print()
    print(cyan("  Decision tree on x1 ≤ 2.0:"))
    print()
    print(f"  {bold('              [x1 ≤ 2.0?]')}")
    print(f"  {bold('             /            \\')}")
    print(f"  {green(' YES (left)'):<20}  {red('NO (right)')}")
    print(f"  {green('[Predict: 0]'):<20}  {red('[Predict: 1]')}")
    print(f"  {green('[Gini = 0.00]'):<20}  {red('[Gini = 0.00]')}")
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Decision Tree with sklearn", """
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

dt = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
dt.fit(X, y)

print(f"Training accuracy: {dt.score(X, y):.4f}")
print(f"Number of leaves:  {dt.get_n_leaves()}")
print(f"Tree depth:        {dt.get_depth()}")

# Print ASCII tree
tree_text = export_text(dt, feature_names=iris.feature_names)
print(tree_text)

# Feature importance
for name, imp in zip(iris.feature_names, dt.feature_importances_):
    print(f"  {name:30s}: {imp:.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Gini and entropy give nearly identical splits in practice",
        "Gini: faster (no log), range [0, 0.5]; Entropy: range [0, 1]",
        "Decision trees are high-variance — small data changes → different tree",
        "Pruning (max_depth, min_samples_leaf) reduces overfitting",
        "Feature importances = mean decrease in impurity (Gini/entropy)",
        "Trees handle non-linear boundaries but only with axis-aligned steps",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Random Forest
# ══════════════════════════════════════════════════════════════════════════════
def topic_random_forest():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Random Forest")
    section_header("RANDOM FOREST — BAGGING + FEATURE RANDOMNESS")
    print()

    section_header("1. THEORY")
    print(white("""
  Random Forest combines two variance-reduction techniques:

  1. BAGGING (Bootstrap Aggregating — Breiman 1996):
     Train T trees on T bootstrap samples of size n (sample with replacement).
     Aggregate predictions by majority vote (classification) or mean (regression).
     Each bootstrap sample contains ~63.2% unique training examples.

  VARIANCE REDUCTION PROOF:
  If T trees have variance σ² and correlation ρ between any pair:
      Var(mean) = ρσ² + (1-ρ)/T · σ²
  As T → ∞: Var → ρσ²  (irreducible variance from tree correlation)
  To reduce variance further, we must REDUCE ρ between trees!

  2. FEATURE RANDOMNESS (Breiman 2001):
  At each node, only consider a random subset of m ≤ p features.
  • Classification: m = √p  (typical default)
  • Regression:     m = p/3  (typical default)
  This de-correlates trees, reducing ρ and thus total variance.

  OUT-OF-BAG (OOB) ERROR:
  Each sample not in bootstrap i is OOB for tree i (~36.8% per tree).
  Predict using only trees where sample was OOB → free validation estimate!
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Var(bagged) = ρσ² + (1-ρ)/T · σ²"))
    print(formula("  OOB fraction: P(not selected in n draws) = (1-1/n)ⁿ → 1/e ≈ 36.8%"))
    print(formula("  Feature importance: E[ΔImpurity] averaged over all trees & nodes"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Bias-Variance Decomposition")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    X_rf, y_rf = make_classification(n_samples=500, n_features=20, n_informative=10,
                                      noise_factor=0.0, random_state=0)

    print(bold_cyan("  Comparing single tree vs random forest on 20-feature classification:"))
    print(bold_cyan(f"  {'Model':>35}  {'CV mean':>10}  {'CV std':>10}  {'Complexity'}"))
    print(grey("  " + "─"*75))
    for Model, kwargs, label in [
        (DecisionTreeClassifier, dict(max_depth=None, random_state=0), "DecisionTree (full, no pruning)"),
        (DecisionTreeClassifier, dict(max_depth=5, random_state=0),    "DecisionTree (max_depth=5)"),
        (RandomForestClassifier, dict(n_estimators=10,  max_features='sqrt', random_state=0), "RF (10 trees)"),
        (RandomForestClassifier, dict(n_estimators=100, max_features='sqrt', random_state=0), "RF (100 trees)"),
        (RandomForestClassifier, dict(n_estimators=100, max_features='log2', random_state=0), "RF (log2 features)"),
    ]:
        scores = cross_val_score(Model(**kwargs), X_rf, y_rf, cv=5)
        print(f"  {label:>35}  {scores.mean():>10.4f}  {scores.std():>10.4f}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Random Forest with OOB Error", """
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',   # √p features at each split
    oob_score=True,        # enable OOB error estimate
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)

print(f"OOB accuracy: {rf.oob_score_:.4f}")  # free validation, no held-out set!
print(f"Training accuracy: {rf.score(X, y):.4f}")

# Feature importances (MDI — mean decrease impurity)
importances = rf.feature_importances_
top5 = importances.argsort()[-5:][::-1]
print("Top 5 features:")
for i in top5:
    print(f"  Feature {i:2d}: {importances[i]:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Bootstrap sampling + feature randomness de-correlates trees → variance drops",
        "As T→∞, variance → ρσ² (irreducible); more trees never hurt, just slow",
        "OOB error ≈ leave-one-out CV with no extra computation cost",
        "Feature importance (MDI) can be biased toward high-cardinality features",
        "Permutation importance (sklearn's permutation_importance) is more reliable",
        "Random forests often match gradient boosting with less hyperparameter tuning",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Gradient Boosting
# ══════════════════════════════════════════════════════════════════════════════
def topic_gradient_boosting():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Gradient Boosting")
    section_header("GRADIENT BOOSTING MACHINES")
    print()

    section_header("1. THEORY")
    print(white("""
  Gradient boosting builds an additive model:
      F_M(x) = Σₘ₌₀ᴹ ηhₘ(x)

  Starting from F₀(x) = ȳ, at each step m:
  1. Compute PSEUDO-RESIDUALS (negative gradient of loss w.r.t. Fₘ₋₁):
         rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]_{F=Fₘ₋₁}
     For MSE loss: rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)  (actual residuals)
     For log-loss: rᵢₘ = yᵢ - p̂ᵢ  (probability residuals)
  2. Fit a weak learner hₘ(x) to {(xᵢ, rᵢₘ)}.  Typically depth-3 trees.
  3. Line search: γₘ = argmin_γ Σᵢ L(yᵢ, Fₘ₋₁(xᵢ) + γhₘ(xᵢ))
  4. Update: Fₘ(x) = Fₘ₋₁(x) + η · γₘ · hₘ(x)

  This is GRADIENT DESCENT in FUNCTION SPACE:
  Each tree hₘ points in the direction of steepest descent of the loss.

  KEY HYPERPARAMETERS:
  η (learning rate): smaller → more trees needed, better generalisation
  max_depth: tree complexity (3-6 is typical)
  n_estimators: number of boosting rounds
  subsample: fraction of training data per round (stochastic GB → less overfit)
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Pseudo-residuals: rᵢₘ = -∂L(yᵢ, F(xᵢ))/∂F"))
    print(formula("  MSE pseudo-residuals = yᵢ - Fₘ₋₁(xᵢ)"))
    print(formula("  Log-loss pseudo-residuals = yᵢ - σ(Fₘ₋₁(xᵢ))"))
    print(formula("  Update: Fₘ = Fₘ₋₁ + η · hₘ  (function-space gradient step)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Manual Gradient Boosting (MSE)")
    rng = np.random.default_rng(5)
    n = 30
    x_gb = rng.uniform(0, 5, n)
    y_gb = np.sin(x_gb) + rng.normal(0, 0.1, n)

    # We'll use depth-1 stumps (split at median)
    class Stump:
        def __init__(self):
            self.threshold = 0; self.left = 0; self.right = 0
        def fit(self, x, r):
            best_loss, best_t = np.inf, 0
            for t in np.percentile(x, range(10, 91, 10)):
                l = r[x <= t].mean() if (x <= t).any() else 0
                ri = r[x > t].mean() if (x > t).any() else 0
                loss = np.sum((r[x <= t] - l)**2) + np.sum((r[x > t] - ri)**2)
                if loss < best_loss:
                    best_loss, best_t = loss, t
                    self.threshold, self.left, self.right = t, l, ri
        def predict(self, x):
            return np.where(x <= self.threshold, self.left, self.right)

    F = np.full(n, y_gb.mean())
    eta = 0.5
    print(bold_cyan("  Gradient Boosting with stumps (MSE loss):"))
    print(bold_cyan(f"  {'Step':>5}  {'Train RMSE':>12}  {'F(x=2.5)':>12}  (true sin(2.5)≈{np.sin(2.5):.3f})"))
    print(grey("  " + "─"*45))
    print(f"  {0:>5}  {np.sqrt(np.mean((y_gb - F)**2)):>12.6f}  {F.mean():>12.6f}")
    stumps = []
    for step in range(1, 11):
        residuals = y_gb - F
        stump = Stump()
        stump.fit(x_gb, residuals)
        F += eta * stump.predict(x_gb)
        stumps.append(stump)
        if step in [1, 3, 5, 7, 10]:
            rmse = np.sqrt(np.mean((y_gb - F)**2))
            pred25 = y_gb.mean() + sum(eta * s.predict(np.array([2.5])) for s in stumps)[0]
            print(f"  {step:>5}  {rmse:>12.6f}  {pred25:>12.6f}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Gradient Boosting with sklearn and XGBoost concept", """
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,   # η
    max_depth=3,         # weak learner depth
    subsample=0.8,       # stochastic GB
    random_state=42
)
gb.fit(Xtr, ytr)
print(f"Test accuracy: {gb.score(Xte, yte):.4f}")

# Staged prediction (see convergence)
from sklearn.metrics import accuracy_score
for n_trees in [10, 50, 100, 200]:
    pred = gb.staged_predict(Xte)
    for i, p in enumerate(pred):
        if i+1 == n_trees:
            print(f"  After {n_trees:3d} trees: {accuracy_score(yte, p):.4f}")
            break
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Gradient boosting = gradient descent in function space (each tree is a step)",
        "Pseudo-residuals from MSE loss ARE actual residuals — hence the name",
        "Small η + more trees: better generalisation but slower training",
        "XGBoost/LightGBM add second-order (Newton) steps → faster convergence",
        "Subsample < 1.0 makes it 'stochastic GB' — reduces overfitting",
        "Gradient boosting is high-bias-friendly: adds complexity step by step",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Naive Bayes
# ══════════════════════════════════════════════════════════════════════════════
def topic_naive_bayes():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Naive Bayes")
    section_header("NAIVE BAYES CLASSIFIER")
    print()

    section_header("1. THEORY")
    print(white("""
  Naive Bayes applies Bayes' theorem with the NAIVE conditional independence assumption:

      P(y|x) ∝ P(y) · Π_{j=1}^p P(xⱼ|y)

  Despite the naive assumption (features independent given class), it often works well
  in text classification and high-dimensional settings.

  VARIANTS:
  • Gaussian NB: P(xⱼ|y) = N(μⱼᵧ, σⱼᵧ²) — continuous features
  • Bernoulli NB: P(xⱼ|y) = θⱼᵧˣʲ (1 - θⱼᵧ)^{1-xⱼ} — binary features (text BOW)
  • Multinomial NB: P(xⱼ|y) = θⱼᵧˣʲ / Σxⱼ! — word count features (text BOW)

  DECISION RULE (log-space for numerical stability):
      ŷ = argmax_{y} [log P(y) + Σⱼ log P(xⱼ|y)]

  LAPLACE SMOOTHING:
  To avoid zero probabilities when a feature-class combination is not in training:
      P(xⱼ=v|y) = (count(xⱼ=v, y) + α) / (count(y) + α·V)
  where V = vocabulary size, α = 1 (Laplace) or small smoothing constant.

  WHY IT WORKS:
  Even with violated independence, the argmax decision is often correct
  because we only need the class with the highest score, not calibrated probabilities.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  P(y|x) ∝ P(y) Πⱼ P(xⱼ|y)"))
    print(formula("  log P(y|x) = log P(y) + Σⱼ log P(xⱼ|y)"))
    print(formula("  Gaussian: P(xⱼ|y)=N(μⱼᵧ,σⱼᵧ²)  → compute class mean/var from data"))
    print(formula("  Laplace: P(xⱼ=v|y)=(count+α)/(count_y+α·V)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Gaussian NB from Scratch")
    rng = np.random.default_rng(11)
    n_per_class = 40
    # Class 0: N([0,0], I),  Class 1: N([2,2], I)
    X0 = rng.normal([0, 0], 1.0, (n_per_class, 2))
    X1 = rng.normal([2, 2], 1.0, (n_per_class, 2))
    X_nb = np.vstack([X0, X1])
    y_nb = np.array([0]*n_per_class + [1]*n_per_class)

    # Fit Gaussian NB
    classes = [0, 1]
    log_prior = {c: np.log(np.mean(y_nb == c)) for c in classes}
    mean_c    = {c: X_nb[y_nb == c].mean(axis=0) for c in classes}
    var_c     = {c: X_nb[y_nb == c].var(axis=0)  for c in classes}

    def gaussian_nb_predict(x_new):
        log_posts = []
        for c in classes:
            logp = log_prior[c] + np.sum(
                -0.5 * np.log(2 * np.pi * var_c[c]) -
                 0.5 * (x_new - mean_c[c])**2 / var_c[c]
            )
            log_posts.append(logp)
        return np.argmax(log_posts)

    preds = np.array([gaussian_nb_predict(X_nb[i]) for i in range(len(y_nb))])
    acc = np.mean(preds == y_nb)
    print(bold_cyan(f"  Gaussian NB from scratch: accuracy = {acc:.4f}"))
    print()
    print(bold_cyan(f"  Class 0 estimated mean: {mean_c[0]}"))
    print(bold_cyan(f"  Class 1 estimated mean: {mean_c[1]}"))
    print()
    for test_x in [np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([1.0, 1.0])]:
        pred = gaussian_nb_predict(test_x)
        print(f"  x={test_x}  → predicted class: {green(str(pred))}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Naive Bayes with sklearn (Gaussian and Multinomial)", """
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Gaussian NB on Iris
iris = load_iris()
gnb = GaussianNB().fit(iris.data, iris.target)
print(f"Gaussian NB on Iris: {gnb.score(iris.data, iris.target):.4f}")

# Multinomial NB for text classification
from sklearn.datasets import make_classification
import numpy as np

# Simulate word count matrix (non-negative integers)
rng = np.random.default_rng(42)
X_text = rng.integers(0, 10, (200, 100))  # 200 docs, 100 words
y_text = rng.integers(0, 2, 200)
mnb = MultinomialNB(alpha=1.0).fit(X_text, y_text)  # alpha = Laplace smoothing
print(f"Multinomial NB (text): {mnb.score(X_text, y_text):.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Naive Bayes is fast: O(n·p) training, O(p) prediction",
        "Independence assumption often wrong yet model often works — especially text",
        "Log-space computation prevents underflow from multiplying many small probs",
        "Laplace smoothing (α=1) prevents zero-probability catastrophe",
        "Calibrated probabilities from NB can be poor; use isotonic regression for calibration",
        "Gaussian NB: estimate mean & variance per feature per class; Σ = diagonal",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 9 — KNN
# ══════════════════════════════════════════════════════════════════════════════
def topic_knn():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "k-Nearest Neighbours")
    section_header("K-NEAREST NEIGHBOURS (KNN)")
    print()

    section_header("1. THEORY")
    print(white("""
  KNN is a non-parametric, instance-based learner — no model is trained.
  At prediction time, find the k training points closest to the query x,
  then aggregate their labels.

      KNN classification: ŷ = majority_vote({y_{i₁}, …, y_{iₖ}})
      KNN regression:     ŷ = (1/k) Σ y_{iⱼ}

  DISTANCE METRICS:
  • Euclidean: d(x,z) = ||x-z||₂    (standard for continuous features)
  • Manhattan: d(x,z) = ||x-z||₁    (robust to outliers)
  • Minkowski: d(x,z) = (Σ|xⱼ-zⱼ|^p)^{1/p}  (p=2: Euclidean, p=1: Manhattan)
  • Cosine:    1 - (x·z)/(||x||·||z||)   (for text/sparse)

  CURSE OF DIMENSIONALITY:
  In high dimensions, all points become approximately equidistant:
      E[d(x,z)] → constant as p→∞  (if features are iid)
  This makes "nearest" nearly meaningless for large p.
  Rule of thumb: KNN degrades significantly above ~20 features.

  CHOOSING k:
  • Small k: low bias, high variance (sensitive to noise)
  • Large k: high bias, low variance (over-smooths)
  • Cross-validation to find optimal k
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ŷ = majority({y : x ∈ kNN(x_query)})"))
    print(formula("  Euclidean: d = √(Σ (xⱼ - zⱼ)²)"))
    print(formula("  Minkowski: d = (Σ|xⱼ-zⱼ|^p)^{1/p}"))
    print(formula("  Decision boundary: Voronoi diagram over training data"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Effect of k")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import cross_val_score

    X_knn, y_knn = make_moons(n_samples=300, noise=0.25, random_state=0)

    print(bold_cyan("  Effect of k on bias-variance tradeoff (5-fold CV on moons):"))
    print(bold_cyan(f"  {'k':>6}  {'Train acc':>12}  {'CV acc':>12}  {'CV std':>10}  {'Assessment'}"))
    print(grey("  " + "─"*65))
    for k in [1, 3, 5, 10, 20, 50, 100, 150]:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv = cross_val_score(knn, X_knn, y_knn, cv=5)
        knn.fit(X_knn, y_knn)
        tr = knn.score(X_knn, y_knn)
        assessment = (red("overfit") if tr - cv.mean() > 0.05 else
                      yellow("underfits") if cv.mean() < 0.80 else green("good"))
        print(f"  {k:>6}  {tr:>12.4f}  {cv.mean():>12.4f}  {cv.std():>10.4f}  {assessment}")
    print()
    _pause()

    section_header("4. ASCII — Curse of Dimensionality")
    print(cyan("  Fraction of volume in a unit hypercube WITHIN distance 0.1 of a corner:\n"))
    for d in [1, 2, 5, 10, 20, 50, 100]:
        # P(within 0.1 Euclidean ball in d dims) ≈ volume of ball / volume of cube
        # Rough estimate: most data is in the shell, not near the centre
        edge_fraction = (0.1) ** d  # L∞ ball
        bar = "█" * max(1, int(edge_fraction * 1e6)) if d <= 5 else ""
        print(f"  d={d:>3}:  {edge_fraction:.2e}  {cyan(bar)}")
    print(grey("\n  The 0.1 corner region occupies 0 fraction for d>6 — 'empty space'!"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("KNN Classification and Cross-Validation", """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)

# CRITICAL: scale features — KNN is distance-based!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Grid search over k
param_grid = {'n_neighbors': list(range(1, 30, 2))}
gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
gs.fit(X_scaled, y)

print(f"Best k: {gs.best_params_['n_neighbors']}")
print(f"Best CV accuracy: {gs.best_score_:.4f}")

# KNN with different metrics
for metric in ['euclidean', 'manhattan', 'cosine']:
    knn = KNeighborsClassifier(n_neighbors=gs.best_params_['n_neighbors'], metric=metric)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(knn, X_scaled, y, cv=5).mean()
    print(f"  {metric:12s}: CV acc = {cv:.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "KNN stores ALL training data — high memory, slow inference O(n·p) each query",
        "Always scale features before KNN — distance is sensitive to scale",
        "k=1: memorises data (overfit); k=n: always predicts majority class (underfit)",
        "Curse of dimensionality: use PCA or feature selection first for high-dim",
        "KD-trees or ball-trees speed up nearest-neighbour search to O(log n)",
        "KNN is a smooth Parzen-window density estimator as k/n → Bayes optimal",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 10 — Perceptron
# ══════════════════════════════════════════════════════════════════════════════
def topic_perceptron():
    clear()
    breadcrumb("mlmath", "Supervised Learning", "Perceptron")
    section_header("THE PERCEPTRON ALGORITHM")
    print()

    section_header("1. THEORY")
    print(white("""
  The perceptron (Rosenblatt, 1957) is the simplest neural unit and the oldest
  online learning algorithm.

  MODEL:
      ŷ = sign(wᵀx + b) ∈ {-1, +1}

  UPDATE RULE (only triggers on misclassification):
      if yᵢ(wᵀxᵢ + b) ≤ 0:   # misclassified
          w ← w + η yᵢ xᵢ
          b ← b + η yᵢ

  CONVERGENCE THEOREM (Novikoff 1962):
  If data is LINEARLY SEPARABLE with margin γ = min_i yᵢ(wᵀxᵢ)/||w||, the
  number of mistakes before convergence is bounded by:
      T ≤ (R/γ)²
  where R = max_i ||xᵢ|| (radius of data).

  RELATION TO LOGISTIC REGRESSION:
  • Perceptron uses HARD threshold (sign); logistic uses SOFT (sigmoid)
  • Logistic minimises cross-entropy (smooth, globally optimised)
  • Perceptron minimises 0/1 loss (non-smooth, online updates only on errors)
  • Both are linear classifiers; both use gradient w.r.t. linear combination
  • Perceptron gradient: (y - ŷ)x, Logistic gradient: (y - σ(wᵀx))x

  LIMITATION:
  Cannot solve XOR — not linearly separable. This led to the AI winter.
  Solution: multi-layer perceptrons (MLP) with hidden units — universal approximators.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ŷ = sign(wᵀx + b)"))
    print(formula("  Update: w ← w + η · yᵢ · xᵢ  (only if misclassified)"))
    print(formula("  Convergence bound: T ≤ (R/γ)²"))
    print(formula("  Perceptron vs Logistic: hard vs soft threshold"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Perceptron Training")
    rng = np.random.default_rng(77)
    n_percep = 50
    # Linearly separable data
    X_pos = rng.normal([1, 1], 0.5, (n_percep//2, 2))
    X_neg = rng.normal([-1, -1], 0.5, (n_percep//2, 2))
    X_p = np.vstack([X_pos, X_neg])
    y_p = np.array([1]*(n_percep//2) + [-1]*(n_percep//2), dtype=float)

    w = np.zeros(2); b = 0.0; eta = 1.0
    mistake_counts = []
    print(bold_cyan("  Perceptron training on linearly separable data:"))
    print(bold_cyan(f"  {'Epoch':>6}  {'Mistakes':>10}  {'w₀':>10}  {'w₁':>10}  {'b':>6}"))
    print(grey("  " + "─"*50))
    for epoch in range(10):
        mistakes = 0
        for i in range(n_percep):
            if y_p[i] * (np.dot(w, X_p[i]) + b) <= 0:
                w += eta * y_p[i] * X_p[i]
                b += eta * y_p[i]
                mistakes += 1
        mistake_counts.append(mistakes)
        if epoch < 6 or mistakes == 0:
            print(f"  {epoch+1:>6}  {mistakes:>10}  {w[0]:>10.4f}  {w[1]:>10.4f}  {b:>6.3f}")
        if mistakes == 0:
            print(green(f"\n  Converged at epoch {epoch+1}!"))
            break
    print()

    # Test accuracy
    y_pred = np.sign(X_p @ w + b)
    acc = np.mean(y_pred == y_p)
    print(bold_cyan(f"  Final accuracy: {acc:.4f}"))
    print()
    _pause()

    section_header("4. PLOTEXT — Mistake Convergence")
    try:
        loss_curve_plot(mistake_counts, title="Perceptron Mistakes per Epoch")
    except Exception:
        print(cyan("  Mistakes per epoch:  " + "  ".join(f"{m}" for m in mistake_counts)))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Perceptron from Scratch vs sklearn", """
import numpy as np

class Perceptron:
    def __init__(self, lr=1.0, n_iter=50):
        self.lr, self.n_iter = lr, n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        for epoch in range(self.n_iter):
            for xi, yi in zip(X, y):
                if yi * (self.w_ @ xi + self.b_) <= 0:
                    self.w_ += self.lr * yi * xi
                    self.b_ += self.lr * yi
        return self

    def predict(self, X):
        return np.sign(X @ self.w_ + self.b_)

# Linearly separable example
rng = np.random.default_rng(0)
X = rng.normal(0, 1, (100, 2))
y = np.sign(2*X[:,0] - X[:,1] + 0.3)   # true separator: 2x0 - x1 + 0.3 = 0

perc = Perceptron(lr=0.1, n_iter=100).fit(X, y)
print(f"Accuracy: {np.mean(perc.predict(X) == y):.3f}")

# sklearn Perceptron (equivalent)
from sklearn.linear_model import Perceptron as SkPerc
sp = SkPerc(max_iter=100, eta0=0.1).fit(X, y)
print(f"sklearn Perceptron accuracy: {sp.score(X, y):.3f}")

# Comparison: perceptron vs logistic on same data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X, y)
print(f"Logistic Regression accuracy: {lr.score(X, y):.3f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Perceptron update: w ← w + η·y·x ONLY when misclassified — no gradient for correct",
        "Convergence theorem: T ≤ (R/γ)² mistakes — finite bound when data is separable",
        "Non-separable data: perceptron may oscillate forever; logistic always converges",
        "Hard vs soft: sign() vs sigmoid() — same gradient form, different threshold",
        "Pocket algorithm: keep best weights seen so far — handles non-separable data",
        "Multi-layer perceptron (MLP) overcomes XOR limitation — universal approximation",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Linear Regression",     topic_linear_regression),
        ("Ridge and Lasso",       topic_ridge_lasso),
        ("Logistic Regression",   topic_logistic_regression),
        ("Support Vector Machine",topic_svm),
        ("Decision Trees",        topic_decision_trees),
        ("Random Forest",         topic_random_forest),
        ("Gradient Boosting",     topic_gradient_boosting),
        ("Naive Bayes",           topic_naive_bayes),
        ("K-Nearest Neighbours",  topic_knn),
        ("Perceptron",            topic_perceptron),
    ]
    block_menu("b10", "Supervised Learning", topics)
