"""
blocks/b14_probabilistic_ml.py
Block 14: Probabilistic ML
Topics: Gaussian Processes, Bayesian Linear Regression, Variational Inference,
        MCMC, HMC, Normalising Flows, Kalman Filter.
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
# TOPIC 1 — Gaussian Processes
# ══════════════════════════════════════════════════════════════════════════════
def topic_gaussian_processes():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "Gaussian Processes")
    section_header("GAUSSIAN PROCESSES (GP)")
    print()

    section_header("1. THEORY")
    print(white("""
  A Gaussian Process is a distribution over functions such that any finite
  collection of function values has a joint Gaussian distribution:
      f ~ GP(m(x), k(x, x'))
  where m(x) = E[f(x)] is the mean function and k(x, x') = Cov[f(x), f(x')]
  is the covariance (kernel) function.

  PRIOR: before seeing data, f(X) ~ N(m(X), K(X,X)) where K is the kernel matrix
  Kᵢⱼ = k(xᵢ, xⱼ). Common kernels:
  - RBF (SE): k(x,x') = σ² exp(-||x-x'||²/2ℓ²)  — infinitely differentiable
  - Matérn: controls smoothness via parameter ν (ν=1/2: Ornstein-Uhlenbeck)
  - Periodic: k(x,x') = exp(-2sin²(π|x-x'|/p)/ℓ²)

  POSTERIOR: given observations y = f(X) + ε with ε ~ N(0, σ²_n I):
      p(f* | X, y, x*) = N(μ*, Σ*) where
      μ* = K*ᵀ (K + σ²_n I)⁻¹ y
      Σ* = K** - K*ᵀ (K + σ²_n I)⁻¹ K*
  Here K* = K(X, x*) and K** = k(x*, x*).
  The posterior mean is the model's prediction; the diagonal of Σ* gives the
  predictive uncertainty (larger where data is sparse). Cost: O(n³) inversion.

  HYPERPARAMETERS (ℓ, σ², σ²_n) are typically learnt by maximising the marginal
  likelihood (type-II MML): log p(y|X, θ) = -½ yᵀ(K+σ²I)⁻¹y - ½log|K+σ²I| - n/2 log(2π).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Prior:   f ~ GP(m(x), k(x, x'))"))
    print(formula("  μ* = K(x*,X) [K(X,X)+σ²I]⁻¹ y"))
    print(formula("  Σ* = K(x*,x*) - K(x*,X) [K(X,X)+σ²I]⁻¹ K(X,x*)"))
    print(formula("  RBF: k(x,x') = σ² exp(-||x-x'||²/(2ℓ²))"))
    print(formula("  log p(y|X,θ) = -½yᵀ(K+σ²I)⁻¹y - ½log|K+σ²I| + const"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — GP Posterior")
    rng = np.random.default_rng(1)
    def rbf(X1, X2, ell=1.0, sigma=1.0):
        X1, X2 = np.atleast_2d(X1).T, np.atleast_2d(X2).T
        sqdist = np.sum((X1[:, None] - X2[None])**2, axis=-1)
        return sigma**2 * np.exp(-0.5 * sqdist / ell**2)

    n_train = 8
    x_tr = rng.uniform(-3, 3, n_train)
    y_tr = np.sin(x_tr) + rng.normal(0, 0.2, n_train)
    x_te = np.linspace(-4, 4, 80)
    sigma_n = 0.2; ell = 1.0; sigma_f = 1.0

    K_tr   = rbf(x_tr, x_tr, ell, sigma_f) + sigma_n**2 * np.eye(n_train) + 1e-6*np.eye(n_train)
    K_te_tr = rbf(x_te, x_tr, ell, sigma_f)
    K_te   = rbf(x_te, x_te, ell, sigma_f)
    L = np.linalg.solve(K_tr, K_te_tr.T)
    mu_post = K_te_tr @ np.linalg.solve(K_tr, y_tr)
    Sig_post = K_te - K_te_tr @ L
    std_post = np.sqrt(np.maximum(np.diag(Sig_post), 0))

    print(f"\n  {bold_cyan('GP posterior (n=8 training points, RBF kernel):')}\n")
    print(f"  {'x*':<8} {'μ*':<10} {'±2σ*':<10} {'sin(x)':<10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    for xi, mi, si, ti in zip(x_te[::10], mu_post[::10],
                                std_post[::10], np.sin(x_te[::10])):
        bar = green("█" * int(abs(mi) * 5) if mi >= 0 else "") + \
              red("█" * int(abs(mi) * 5) if mi < 0 else "")
        print(f"  {xi:<8.2f} {green(f'{mi:+.3f}'):<20} {yellow(f'{2*si:.3f}'):<18} {grey(f'{ti:.3f}')}")

    print(f"\n  {hint('Uncertainty (±2σ*) grows in regions far from training data.')}")
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_te, np.sin(x_te), 'k--', alpha=0.4, label="True f=sin(x)")
        ax.plot(x_te, mu_post, 'b-', linewidth=2, label="GP mean")
        ax.fill_between(x_te, mu_post - 2*std_post, mu_post + 2*std_post,
                        alpha=0.25, color='blue', label="±2σ")
        ax.scatter(x_tr, y_tr, c='red', s=60, zorder=5, label="Training data")
        ax.set_title("Gaussian Process Posterior"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("/tmp/gp_posterior.png", dpi=90)
        plt.close(fig)
        print(green("  [matplotlib] GP posterior saved to /tmp/gp_posterior.png"))
    except ImportError:
        print(grey("  (install matplotlib for GP posterior plot)"))

    section_header("5. CODE")
    code_block("Gaussian Process from scratch", """
import numpy as np

def rbf_kernel(X1, X2, ell=1.0, sf=1.0):
    d2 = ((X1[:, None] - X2[None])**2).sum(-1)
    return sf**2 * np.exp(-0.5 * d2 / ell**2)

def gp_posterior(x_tr, y_tr, x_te, ell=1.0, sf=1.0, sn=0.1):
    K   = rbf_kernel(x_tr, x_tr, ell, sf) + sn**2*np.eye(len(x_tr)) + 1e-6*np.eye(len(x_tr))
    Ks  = rbf_kernel(x_te, x_tr, ell, sf)
    Kss = rbf_kernel(x_te, x_te, ell, sf)
    alpha = np.linalg.solve(K, y_tr)
    mu    = Ks @ alpha
    V     = np.linalg.solve(K, Ks.T)
    Sig   = Kss - Ks @ V
    return mu, np.sqrt(np.maximum(np.diag(Sig), 0))

rng = np.random.default_rng(0)
x_tr = rng.uniform(-3, 3, 10)
y_tr = np.sin(x_tr) + rng.normal(0, 0.2, 10)
x_te = np.linspace(-4, 4, 100)

mu, std = gp_posterior(x_tr, y_tr, x_te)
print(f"Max uncertainty at x={x_te[std.argmax()]:.2f}  (far from training data)")
print(f"Min uncertainty at x={x_te[std.argmin()]:.2f}  (near training data)")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "GP = Bayesian non-parametric: distribution over functions, not parameter vectors",
        "Posterior mean = best prediction; posterior variance = uncertainty (epistemic)",
        "Kernel choice = prior over function shape (smooth/rough/periodic)",
        "Uncertainty grows far from data — GP knows what it doesn't know",
        "Cost: O(n³) for inversion, O(n²) memory — use sparse/inducing methods for n>10k",
        "Marginal log-likelihood optimisation automatically balances fit vs complexity",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Bayesian Linear Regression
# ══════════════════════════════════════════════════════════════════════════════
def topic_bayesian_linear_regression():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "Bayesian Linear Regression")
    section_header("BAYESIAN LINEAR REGRESSION")
    print()

    section_header("1. THEORY")
    print(white("""
  Bayesian linear regression places a prior on the weight vector w and maintains
  a posterior distribution — quantifying parameter uncertainty rather than just
  giving a point estimate.

  MODEL:
      Prior:       w ~ N(0, α⁻¹ I)
      Likelihood:  y | X, w ~ N(X w, β⁻¹ I)   where β = 1/σ²_noise

  POSTERIOR: by Bayes' rule, p(w|X,y) ∝ p(y|X,w)·p(w) is also Gaussian:
      p(w|X,y) = N(μ_N, S_N) where
      S_N    = (α I + β Xᵀ X)⁻¹
      μ_N    = β S_N Xᵀ y

  Each new data point updates the Gaussian posterior sequentially.

  PREDICTIVE DISTRIBUTION: integrating out w:
      p(y*|x*, X, y) = N(μ_N · x*, σ²*(x*)) where
      σ²*(x*) = β⁻¹ + x*ᵀ S_N x*
  The predictive variance has two parts:
  - β⁻¹: irreducible observation noise
  - x*ᵀ S_N x*: uncertainty due to finite data (→ 0 as n → ∞)

  CONNECTION TO RIDGE: the MAP estimate μ_N (mode of posterior) equals the
  Ridge regression solution ŵ_ridge with λ = α/β. Bayesian regression adds the
  full posterior distribution on top of ridge's point estimate.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  S_N = (αI + β XᵀX)⁻¹          [posterior covariance]"))
    print(formula("  μ_N = β S_N Xᵀ y               [posterior mean = MAP = Ridge]"))
    print(formula("  p(y*|x*,X,y) = N(μ_N·x*,  β⁻¹ + x*ᵀ S_N x*)"))
    print(formula("  Sequential update: S_{N+1} = (S_N⁻¹ + β xₙxₙᵀ)⁻¹  [online Bayes]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Posterior Draws")
    rng = np.random.default_rng(5)
    n = 20
    x_tr = rng.uniform(-2, 2, n)
    y_tr = 2.0 * x_tr + 0.5 + rng.normal(0, 0.5, n)
    X_tr = np.column_stack([np.ones(n), x_tr])
    alpha, beta = 1.0, 4.0

    S_N = np.linalg.inv(alpha * np.eye(2) + beta * X_tr.T @ X_tr)
    mu_N = beta * S_N @ X_tr.T @ y_tr

    print(f"\n  {bold_cyan('Bayesian Linear Regression (n=20, α=1, β=4):')}\n")
    print(f"  True w = [0.5 (intercept), 2.0 (slope)]")
    print(f"  Posterior mean μ_N = [{green(f'{mu_N[0]:.3f}')}, {green(f'{mu_N[1]:.3f}')}]")
    print(f"  Posterior std (intercept) = {yellow(f'{np.sqrt(S_N[0,0]):.4f}')}")
    print(f"  Posterior std (slope)     = {yellow(f'{np.sqrt(S_N[1,1]):.4f}')}")
    print()

    x_te = np.linspace(-3, 3, 80)
    X_te = np.column_stack([np.ones(80), x_te])
    mu_pred = X_te @ mu_N
    var_pred = 1.0/beta + np.array([x_.T @ S_N @ x_ for x_ in X_te])
    std_pred = np.sqrt(var_pred)

    print(f"  {bold_cyan('Predictive distribution (sample):')}\n")
    print(f"  {'x*':<8} {'pred μ':<12} {'±2σ pred':<12} {'uncertainty'}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*12}")
    for xi, mi, si in zip(x_te[::12], mu_pred[::12], std_pred[::12]):
        bar = yellow("█" * min(int(si * 15), 20))
        print(f"  {xi:<8.1f} {green(f'{mi:+.3f}'):<18} {yellow(f'{2*si:.3f}'):<14} {bar}")
    print()

    # Draw posterior samples and plot
    w_samples = rng.multivariate_normal(mu_N, S_N, 5)
    print(f"\n  {bold_cyan('5 posterior draws of (intercept, slope):')}")
    rows = [[str(i+1), f"{w[0]:.3f}", f"{w[1]:.3f}"] for i, w in enumerate(w_samples)]
    table(["Sample", "Intercept", "Slope"], rows, [cyan, green, yellow])
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_tr, y_tr, c='red', s=40, zorder=5, label="Data")
        ax.plot(x_te, mu_pred, 'b-', lw=2, label="Posterior mean")
        ax.fill_between(x_te, mu_pred - 2*std_pred, mu_pred + 2*std_pred,
                        alpha=0.2, color='blue', label="±2σ predictive")
        for w in w_samples:
            ax.plot(x_te, X_te @ w, 'g-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Bayesian Linear Regression: Posterior Draws + Predictive"); ax.legend()
        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig("/tmp/bayesian_lr.png", dpi=90); plt.close(fig)
        print(green("  [matplotlib] Bayesian LR plot saved to /tmp/bayesian_lr.png"))
    except ImportError:
        print(grey("  (install matplotlib for Bayesian LR plot)"))

    section_header("5. CODE")
    code_block("Bayesian Linear Regression from scratch", """
import numpy as np

def bayesian_lr(X, y, alpha=1.0, beta=4.0):
    n, d = X.shape
    S_N   = np.linalg.inv(alpha*np.eye(d) + beta*X.T@X)
    mu_N  = beta * S_N @ X.T @ y
    return mu_N, S_N

def predict(x_star, mu_N, S_N, beta):
    mu_pred  = x_star @ mu_N
    var_pred = 1/beta + np.einsum('...i,ij,...j', x_star, S_N, x_star)
    return mu_pred, np.sqrt(var_pred)

rng = np.random.default_rng(0)
n = 30
x = rng.uniform(-2, 2, n)
y = 2*x + 0.5 + rng.normal(0, 0.5, n)
X = np.column_stack([np.ones(n), x])

mu_N, S_N = bayesian_lr(X, y)
print(f"Posterior mean: {mu_N}")
print(f"Ridge MAP = Bayesian MAP: {np.allclose(mu_N, np.linalg.solve(X.T@X + 0.25*np.eye(2), X.T@y))}")

# Predictive at x=0
x_star = np.array([[1.0, 0.0]])
mu_p, std_p = predict(x_star, mu_N, S_N, beta=4)
print(f"Prediction at x=0: μ={mu_p[0]:.3f} ± {2*std_p[0]:.3f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Posterior mean = Ridge regression MAP; Bayesian adds uncertainty quantification",
        "Predictive variance β⁻¹ + x*ᵀ S_N x* = noise + parameter uncertainty",
        "More data → S_N → 0 → predictive uncertainty collapses to noise floor β⁻¹",
        "Prior α controls regularisation: large α → shrink w toward 0 (like Ridge)",
        "Marginal likelihood p(y|X,α,β) selects hyperparameters without cross-validation",
        "Sequential (online) update: incorporate each new data point with O(d²) cost",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Variational Inference
# ══════════════════════════════════════════════════════════════════════════════
def topic_variational_inference():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "Variational Inference")
    section_header("VARIATIONAL INFERENCE (VI)")
    print()

    section_header("1. THEORY")
    print(white("""
  Variational Inference (VI) approximates an intractable posterior p(z|x) by
  finding the closest tractable distribution q(z; φ) within a variational family,
  minimising KL(q||p):

      KL(q(z)||p(z|x)) = E_q[log q(z)] - E_q[log p(z|x)]
      = E_q[log q(z)] - E_q[log p(x,z)] + log p(x)

  Since log p(x) is constant w.r.t. q, minimising KL ≡ maximising the ELBO:
      ELBO(q) = E_q[log p(x,z)] - E_q[log q(z)]

  MEAN-FIELD VI: assume q(z) = Πᵢ qᵢ(zᵢ) — products of independent factors.
  The optimal q*ⱼ(zⱼ) satisfies:
      log q*ⱼ(zⱼ) = E_{q_{-j}}[log p(x,z)] + const
  This gives coordinate ascent updates that can be applied cyclically.

  REPARAMETERISATION TRICK: enables backpropagation through stochastic nodes.
  Instead of sampling z ~ q(z) = N(μ, σ²) directly:
      z = μ + σ·ε,    ε ~ N(0,1)
  Now z is a deterministic function of (μ, σ, ε), so gradients flow through
  μ and σ. Used in VAE, Normalising Flows, and modern probabilistic models.

  AMORTISED VI (VAE): train an encoder network qφ(z|x) to output (μ, σ) directly,
  sharing parameters across all data points. The decoder pθ(x|z) is the generative
  model. Optimise ELBO = E[log p(x|z)] - KL(qφ(z|x)||p(z)).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ELBO = E_q[log p(x,z)] - E_q[log q(z)]  = E_q[log p(x|z)] - KL(q||prior)"))
    print(formula("  log q*ⱼ(zⱼ) = E_{-j}[log p(x,z)] + const   [mean-field update]"))
    print(formula("  Reparam: z = μ + σ·ε, ε~N(0,1)  → ∂z/∂μ=1, ∂z/∂σ=ε  [backprop OK]"))
    print(formula("  VAE ELBO: E[log p(x|z)] - KL(N(μ,σ²)||N(0,1))"))
    print(formula("  KL(N(μ,σ²)||N(0,1)) = ½(μ² + σ² - log σ² - 1)  [closed form]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Mean-Field VI on Gaussian Posterior")
    rng = np.random.default_rng(6)
    # True posterior: given θ~N(0,1), x|θ~N(θ,1), observe x=2
    # True posterior: θ|x ~ N(1, 0.5)
    x_obs = 2.0
    true_mu_post = x_obs / 2.0
    true_var_post = 0.5

    # Mean-field VI: minimise KL(N(m,v)||N(1,0.5)) analytically
    # ELBO for Gaussian-Gaussian = -½ log(v) + v + (m-1)² / (2*0.5) + const
    def elbo_gaussian(m, log_v):
        v = np.exp(log_v)
        log_joint_term = -0.5*(m**2 + v) - 0.5*((x_obs - m)**2 + v)
        entropy = 0.5*(1 + log_v + np.log(2*np.pi))
        return log_joint_term + entropy

    m = 0.0; log_v = 0.0; lr = 0.1
    elbos = []
    for _ in range(100):
        elbos.append(elbo_gaussian(m, log_v))
        v = np.exp(log_v)
        dE_dm = -(m) - (m - x_obs)
        dE_dlv = -0.5*v - 0.5*v + 0.5
        m     += lr * dE_dm
        log_v += lr * dE_dlv

    v_final = np.exp(log_v)
    print(f"\n  {bold_cyan('Mean-field VI: optimising N(m,v) to approximate N(1.0, 0.5):')}\n")
    print(f"  True posterior:      N({true_mu_post:.3f}, {true_var_post:.3f})")
    print(f"  VI posterior (100 steps): N({green(f'{m:.3f}')}, {green(f'{v_final:.3f}')})")
    print(f"  KL gap: {yellow(f'{abs(m - true_mu_post):.5f}')} (mean), "
          f"{yellow(f'{abs(v_final - true_var_post):.5f}')} (var)")
    print()
    print(f"  {bold_cyan('ELBO progression:')}")
    print_sparkline(elbos, label="ELBO", color_fn=green)
    print()
    _pause()

    section_header("4. CODE")
    code_block("VAE ELBO with reparameterisation trick (numpy)", """
import numpy as np

def vae_elbo(x, mu, log_sigma, decoder_fn, n_samples=10, seed=0):
    '''
    Compute VAE ELBO = E[log p(x|z)] - KL(q(z|x)||N(0,1))
    mu, log_sigma: encoder outputs (shape: d_latent)
    decoder_fn: z -> reconstructed x
    '''
    rng = np.random.default_rng(seed)
    sigma = np.exp(log_sigma)

    # Reparameterisation: z = mu + sigma * eps, eps ~ N(0,I)
    eps = rng.normal(0, 1, (n_samples, len(mu)))
    z   = mu + sigma * eps           # (n_samples, d_latent)

    # Reconstruction term: E_q[log p(x|z)]
    recon_losses = []
    for zi in z:
        xhat = decoder_fn(zi)
        recon_losses.append(-0.5 * np.sum((x - xhat)**2))  # Gaussian log-lik
    recon_term = np.mean(recon_losses)

    # KL divergence: KL(N(mu,sigma²)||N(0,1)) [closed form]
    kl = 0.5 * np.sum(mu**2 + sigma**2 - 2*log_sigma - 1)

    return recon_term - kl

# Toy: latent dim=2, data dim=5
rng = np.random.default_rng(42)
x   = rng.normal(0, 1, 5)
mu  = rng.normal(0, 0.5, 2)
lsig = rng.normal(-0.5, 0.1, 2)

W_dec = rng.normal(0, 0.3, (5, 2))
decode = lambda z: W_dec @ z

elbo = vae_elbo(x, mu, lsig, decode)
print(f"VAE ELBO = {elbo:.4f}")
print(f"Gradient flows through mu and log_sigma via reparameterisation")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "VI converts intractable posterior inference into optimisation problem",
        "ELBO = reconstruction + negative KL = log p(x) - KL(q||p_posterior)",
        "Mean-field: independent factors → tractable but ignores correlations",
        "Reparameterisation z=μ+σε makes ELBO differentiable w.r.t. φ (encoder params)",
        "VAE: encoder qφ(z|x) + decoder pθ(x|z) trained jointly on ELBO",
        "KL(q||p) vs KL(p||q): expectation-propagation minimises the reverse divergence",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — MCMC
# ══════════════════════════════════════════════════════════════════════════════
def topic_mcmc():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "MCMC")
    section_header("MARKOV CHAIN MONTE CARLO (MCMC)")
    print()

    section_header("1. THEORY")
    print(white("""
  MCMC methods draw samples from a target distribution π(x) ∝ p̃(x) (known up to
  normalisation constant Z = ∫ p̃(x) dx, which may be intractable to compute).

  METROPOLIS-HASTINGS (MH) ALGORITHM:
  1. Initialise x₀ (arbitrary starting point).
  2. Propose: x' ~ q(x'|xₜ)  (e.g. Gaussian N(xₜ, σ²I))
  3. Accept with probability:
         α = min(1,  π(x')·q(xₜ|x') / (π(xₜ)·q(x'|xₜ)))
     If accepted: xₜ₊₁ = x'. Else: xₜ₊₁ = xₜ.
  4. Repeat.

  DETAILED BALANCE: the MH acceptance ratio ensures:
      π(x)·q(x'|x)·α(x→x') = π(x')·q(x|x')·α(x'→x)
  which is sufficient for the Markov chain to have π as its stationary distribution.

  BURN-IN: the first B samples are discarded as the chain moves from the (possibly
  poor) initialisation to the high-probability region of π. Typical B = 1000.

  THINNING: take every k-th sample to reduce autocorrelation between consecutive
  samples. Effective sample size (ESS) = n_samples / (1 + 2Σₖ ρₖ).

  CONVERGENCE DIAGNOSTICS:
  - Trace plots: should look like 'fuzzy caterpillars' — well-mixing chain.
  - R-hat (Gelman-Rubin): multiple chains converge when R̂ ≈ 1.
  - ACF: autocorrelation should decay quickly for efficient sampling.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Acceptance: α = min(1, π(x')q(x|x') / π(x)q(x'|x))"))
    print(formula("  For symmetric proposal q: α = min(1, π(x')/π(x))  [Metropolis]"))
    print(formula("  Detailed balance: π(x)K(x→x') = π(x')K(x'→x)"))
    print(formula("  ESS = n / (1 + 2Σₖ₌₁^∞ ρₖ)  where ρₖ = autocorrelation at lag k"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Metropolis on a Banana Distribution")
    rng = np.random.default_rng(7)

    def log_banana(x, a=1.0, b=0.1):
        x1, x2 = x[0], x[1]
        return -0.5 * (x1**2 / (2*a**2) + (x2 - b*x1**2 + a**2*b)**2 / 0.5**2)

    def metropolis(n_samples=3000, sigma_prop=0.5, burn_in=500):
        x = np.array([0.0, 0.0])
        samples, accepts = [], 0
        for i in range(n_samples + burn_in):
            x_prop = x + rng.normal(0, sigma_prop, 2)
            log_alpha = log_banana(x_prop) - log_banana(x)
            if np.log(rng.random() + 1e-300) < log_alpha:
                x = x_prop; accepts += 1
            if i >= burn_in:
                samples.append(x.copy())
        return np.array(samples), accepts / (n_samples + burn_in)

    samples, acc_rate = metropolis()
    print(f"\n  {bold_cyan('Metropolis on banana distribution (3000 samples):')}\n")
    print(f"  Acceptance rate: {green(f'{acc_rate:.3f}')} (optimal range: 0.234–0.44)")
    print(f"  Sample mean: ({green(f'{samples[:,0].mean():.3f}')}, {green(f'{samples[:,1].mean():.3f}')})")
    print(f"  Sample std:  ({yellow(f'{samples[:,0].std():.3f}')}, {yellow(f'{samples[:,1].std():.3f}')})")
    print()

    # Trace plot ASCII
    print(f"  {bold_cyan('Trace plot (x₁, first 60 samples after burn-in):')}\n")
    trace = samples[:60, 0]
    min_t, max_t = trace.min(), trace.max()
    for i in range(0, 60, 6):
        bar_pos = int((trace[i] - min_t) / (max_t - min_t + 1e-9) * 50)
        print(f"  t={i:3d} " + grey("·") * bar_pos + cyan("●") + grey("·") * (50 - bar_pos))
    print()

    try:
        import plotext as plt
        plt.clear_figure()
        plt.scatter(samples[::5, 0].tolist(), samples[::5, 1].tolist(),
                    color="cyan", marker="dot")
        plt.title("Metropolis: Banana Distribution Samples")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for 2D scatter of MCMC samples)"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt2
        fig, axes = plt2.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(samples[::3, 0], samples[::3, 1], alpha=0.3, s=5)
        axes[0].set_title("MCMC Samples: Banana Distribution"); axes[0].grid(alpha=0.3)
        axes[1].plot(samples[:500, 0]); axes[1].set_title("Trace Plot x₁")
        axes[1].set_xlabel("Iteration"); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("/tmp/mcmc_samples.png", dpi=90); plt2.close(fig)
        print(green("  [matplotlib] MCMC samples saved to /tmp/mcmc_samples.png"))
    except ImportError:
        print(grey("  (install matplotlib for MCMC plots)"))

    section_header("5. CODE")
    code_block("Metropolis-Hastings sampler", """
import numpy as np

def metropolis_hastings(log_target, x0, n_samples=5000,
                         sigma_prop=0.5, burn_in=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float)
    samples = []
    n_accept = 0
    for i in range(n_samples + burn_in):
        x_prop = x + rng.normal(0, sigma_prop, x.shape)
        log_alpha = log_target(x_prop) - log_target(x)
        if np.log(rng.random() + 1e-300) < min(0, log_alpha):
            x = x_prop; n_accept += 1
        if i >= burn_in:
            samples.append(x.copy())
    acc_rate = n_accept / (n_samples + burn_in)
    return np.array(samples), acc_rate

# Sample from N(2, 0.5²)
log_target = lambda x: -0.5 * ((x - 2) / 0.5)**2
samples, acc = metropolis_hastings(log_target, x0=0.0, n_samples=5000)
print(f"Acceptance rate: {acc:.3f}")
print(f"Estimated mean: {samples.mean():.4f}  (true: 2.0)")
print(f"Estimated std:  {samples.std():.4f}   (true: 0.5)")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "MH: accept/reject ensures π is stationary — only needs ratio π(x')/π(x)",
        "No normalisation constant needed — cancels in acceptance ratio",
        "Acceptance rate ≈ 0.234 (d-dim) to 0.44 (1-dim) optimal for RW Metropolis",
        "Burn-in removes influence of initialisation; check with trace plots",
        "High autocorrelation → low ESS → need more samples for same accuracy",
        "Gibbs sampling: MH special case where each coordinate accepted exactly = 1",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — HMC
# ══════════════════════════════════════════════════════════════════════════════
def topic_hmc():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "Hamiltonian Monte Carlo")
    section_header("HAMILTONIAN MONTE CARLO (HMC)")
    print()

    section_header("1. THEORY")
    print(white("""
  HMC introduces an auxiliary momentum variable p to use gradient information
  to make large, correlated moves while maintaining high acceptance rate.

  HAMILTONIAN: H(x, p) = U(x) + K(p) where
      U(x) = -log π(x)         [potential energy, from target]
      K(p) = p·p / 2 = ‖p‖²/2  [kinetic energy, p ~ N(0, I)]

  HAMILTON'S EQUATIONS (dynamics):
      dx/dt = +∂H/∂p = p
      dp/dt = -∂H/∂x = -∂U/∂x = ∇ log π(x)

  Following these dynamics preserves H (total energy) exactly → acceptance rate = 1.
  In practice, we use the LEAPFROG INTEGRATOR (time-reversible, volume-preserving):
      p_{t+ε/2} = p_t - (ε/2)∇U(x_t)
      x_{t+ε}   = x_t + ε·p_{t+ε/2}
      p_{t+ε}   = p_{t+ε/2} - (ε/2)∇U(x_{t+ε})
  Run for L leapfrog steps, then MH accept/reject to correct for discretisation error.

  WHY HMC > MH: MH (random walk) proposes Δx ~ N(0, σ²), moves O(σ) per step.
  HMC uses gradient to guide proposals — moves O(1) per step even in high dimensions.
  MH requires n² steps to cross a distribution; HMC requires O(n^{1/4}) steps.

  NUTS (No-U-Turn Sampler): automatically tunes step size ε and number of steps L
  using the no-U-turn criterion (stop building the trajectory when it doubles back).
  NUTS is the default sampler in Stan, PyMC, and NumPyro.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  H(x,p) = U(x) + ‖p‖²/2,  U(x) = -log π(x)"))
    print(formula("  Leapfrog: p' ← p - (ε/2)∇U(x);  x' ← x + ε·p';  p' ← p' - (ε/2)∇U(x')"))
    print(formula("  MH accept: α = min(1, exp(-H(x',p') + H(x,p)))"))
    print(formula("  HMC efficiency gain: O(d^{1/4}) vs O(d) steps for random walk MH"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — HMC on 2D Gaussian")
    rng = np.random.default_rng(10)
    # Target: N([3,3], [[1,0.9],[0.9,1]])
    Sigma_true = np.array([[1.0, 0.9], [0.9, 1.0]])
    Sigma_inv  = np.linalg.inv(Sigma_true)
    mu_true    = np.array([3.0, 3.0])

    def grad_U(x): return Sigma_inv @ (x - mu_true)
    def U(x): diff = x - mu_true; return 0.5 * diff @ Sigma_inv @ diff

    def hmc_step(x, eps=0.2, L=10):
        p = rng.normal(0, 1, len(x))
        H_old = U(x) + 0.5 * p @ p
        x_new = x.copy()
        p_new = p.copy()
        p_new -= (eps / 2) * grad_U(x_new)
        for _ in range(L):
            x_new  = x_new + eps * p_new
            p_new -= eps * grad_U(x_new)
        p_new -= (eps / 2) * grad_U(x_new)
        H_new = U(x_new) + 0.5 * p_new @ p_new
        if np.log(rng.random() + 1e-300) < H_old - H_new:
            return x_new, True
        return x, False

    n_samples = 1000; burn_in = 200
    x = np.zeros(2)
    samples_hmc = []; accepts = 0
    for i in range(n_samples + burn_in):
        x, acc = hmc_step(x)
        if acc: accepts += 1
        if i >= burn_in: samples_hmc.append(x.copy())
    samples_hmc = np.array(samples_hmc)

    print(f"\n  {bold_cyan('HMC on N([3,3], [[1,0.9],[0.9,1]]):')}\n")
    print(f"  Acceptance rate: {green(f'{accepts/(n_samples+burn_in):.3f}')} (HMC: typically >0.6)")
    print(f"  Est. mean:  {green(f'[{samples_hmc[:,0].mean():.3f}, {samples_hmc[:,1].mean():.3f}]')} (true [3,3])")
    print(f"  Est. cov[0,1]: {green(f'{np.cov(samples_hmc.T)[0,1]:.3f}')} (true 0.9)")
    print()
    _pause()

    section_header("4. CODE")
    code_block("HMC sampler (numpy)", """
import numpy as np

def hmc(log_prob_and_grad, x0, n_samples=1000, eps=0.1, L=20,
        burn_in=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float)
    samples, n_accept = [], 0
    for i in range(n_samples + burn_in):
        p = rng.normal(0, 1, x.shape)
        lp, g = log_prob_and_grad(x)
        H_old = -lp + 0.5 * p @ p
        x_new, p_new = x.copy(), p.copy()
        _, g_new = log_prob_and_grad(x_new)
        p_new += 0.5*eps*g_new
        for _ in range(L):
            x_new += eps * p_new
            lp_n, g_new = log_prob_and_grad(x_new)
            p_new += eps * g_new
        p_new += 0.5*eps*g_new
        H_new = -lp_n + 0.5 * p_new @ p_new
        if np.log(rng.random()+1e-300) < H_old - H_new:
            x = x_new; n_accept += 1
        if i >= burn_in: samples.append(x.copy())
    return np.array(samples), n_accept/(n_samples+burn_in)

# Target: mixture of Gaussians
def log_p(x):
    lp = np.logaddexp(-0.5*((x-2)**2)/0.5**2,
                      -0.5*((x+2)**2)/0.5**2) - np.log(2)
    g  = np.tanh(2*(x-2)/0.5) * (-2/0.5**2) * 0.5  # approx grad
    return lp.sum(), g
samples, acc = hmc(log_p, x0=np.zeros(1))
print(f"Acceptance={acc:.3f}  mean={samples.mean():.3f}  std={samples.std():.3f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "HMC uses gradient ∇log π(x) to make large guided proposals → low autocorrelation",
        "Leapfrog is symplectic: time-reversible + volume-preserving → exact MH correction",
        "High acceptance rate (0.6-0.9) with large moves — efficient in high dimensions",
        "NUTS eliminates need to tune L (trajectory length) — key practical improvement",
        "HMC scales as O(d^{1/4}) vs O(d) for random walk — huge win for deep models",
        "Requires differentiable log-likelihood — use automatic differentiation (JAX/PyTorch)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Normalising Flows
# ══════════════════════════════════════════════════════════════════════════════
def topic_normalising_flows():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "Normalising Flows")
    section_header("NORMALISING FLOWS")
    print()

    section_header("1. THEORY")
    print(white("""
  A Normalising Flow transforms a simple base distribution (e.g. N(0,I)) into
  a complex target distribution through a series of bijective transformations.

  CHANGE OF VARIABLES: if z = f⁻¹(x) (or x = f(z)), the log density of x is:
      log p(x) = log p_z(f⁻¹(x)) + log |det(∂f⁻¹/∂x)|
               = log p_z(z) - log |det(∂f/∂z)|   [via inverse function theorem]

  For a sequence of K flows f = fₖ ∘ ... ∘ f₁:
      log p(x) = log p_z(z₀) - Σₖ log |det(∂fₖ/∂zₖ₋₁)|

  KEY REQUIREMENT: each transformation fₖ must be:
  1. Bijective (invertible)
  2. Differentiable (smooth)
  3. Have a tractable Jacobian determinant

  PLANAR FLOW: f(z) = z + u·tanh(wᵀz + b). The Jacobian is triangular-like,
  |det J| = |1 + uᵀψ(wᵀz + b)| where ψ = tanh'.

  RealNVP COUPLING LAYERS: partition z = [z_a, z_b]:
      x_a = z_a
      x_b = z_b ⊙ exp(s(z_a)) + t(z_a)  [s, t are neural networks]
  Jacobian: triangular with diagonal exp(s(z_a)) → det = exp(Σᵢ s_i(z_a)).
  Inverse is trivial: z_b = (x_b - t(x_a)) ⊙ exp(-s(x_a)).

  APPLICATIONS: density estimation, variational inference, image generation,
  molecular structure generation, calibrated uncertainty in deep learning.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  log p(x) = log p_z(z) - log |det(∂f/∂z)|"))
    print(formula("  Chain: log p_K(xₖ) = log p₀(z₀) - Σₖ log |det Jfₖ|"))
    print(formula("  RealNVP: x_b = z_b ⊙ exp s(z_a) + t(z_a),  det J = exp(Σ sᵢ)"))
    print(formula("  Planar:  f(z) = z + u·tanh(wᵀz+b),  |det J| = |1 + uᵀψ(wᵀz+b)|"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 1D Flow: Gaussian → Bimodal")
    rng = np.random.default_rng(15)

    def apply_planar_flow(z, u, w, b):
        lin = z * w + b
        h = np.tanh(lin)
        x = z + u * h
        psi = 1 - h**2
        log_det = np.log(np.abs(1 + u * w * psi) + 1e-9)
        return x, log_det

    z = rng.normal(0, 1, 1000)
    log_pz = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    w, u, b = 1.5, -2.0, 0.5
    x, log_det = apply_planar_flow(z, u, w, b)
    log_px = log_pz - log_det

    print(f"\n  {bold_cyan('Planar flow: N(0,1) → bimodal distribution'):}\n")
    print(f"  Flow params: w={w}, u={u}, b={b}")
    print(f"  Input z range:  [{z.min():.2f}, {z.max():.2f}]")
    print(f"  Output x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  |log p(x)| mean: {green(f'{np.abs(log_px).mean():.4f}')}")

    # Show density histogram in ASCII
    bins = np.linspace(x.min(), x.max(), 20)
    hist, _ = np.histogram(x, bins=bins)
    max_h = hist.max()
    print(f"\n  {bold_cyan('Transformed density histogram:')}\n")
    for count, left in zip(hist, bins[:-1]):
        bar_len = int(count / max_h * 40)
        print(f"  {left:+.1f}  {cyan('█' * bar_len)}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("RealNVP coupling layer", """
import numpy as np

class CouplingLayer:
    def __init__(self, d, seed=0):
        rng = np.random.default_rng(seed)
        half = d // 2
        # Simple scale/translation networks (linear here)
        self.Ws = rng.normal(0, 0.1, (half, half))
        self.Wt = rng.normal(0, 0.1, (half, half))

    def forward(self, z):
        '''z -> x with log|det J|'''
        d = z.shape[1]
        half = d // 2
        za, zb = z[:, :half], z[:, half:]
        s = np.tanh(za @ self.Ws)       # scale
        t = za @ self.Wt                # translate
        xa = za
        xb = zb * np.exp(s) + t
        x  = np.concatenate([xa, xb], axis=1)
        log_det = s.sum(axis=1)         # Σ sᵢ(za)
        return x, log_det

    def inverse(self, x):
        '''x -> z'''
        d = x.shape[1]
        half = d // 2
        xa, xb = x[:, :half], x[:, half:]
        s = np.tanh(xa @ self.Ws)
        t = xa @ self.Wt
        zb = (xb - t) * np.exp(-s)
        return np.concatenate([xa, zb], axis=1)

rng = np.random.default_rng(0)
z = rng.normal(0, 1, (1000, 4))
layer = CouplingLayer(d=4)
x, log_det = layer.forward(z)
z_rec = layer.inverse(x)

print(f"Forward: z {z.shape} -> x {x.shape}")
print(f"Mean log|det J|: {log_det.mean():.4f}")
print(f"Inverse reconstruction error: {np.abs(z - z_rec).max():.2e}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Flows transform simple distributions → complex via bijections with tractable det J",
        "log p(x) = log p_z(f⁻¹(x)) + log|det J_{f⁻¹}| — exact density evaluation",
        "RealNVP: split input, transform half; Jacobian triangular → det = product of diag",
        "Unlike VAEs: exact log-likelihood (no lower bound), but generation slower",
        "Normalising flows can parameterise variational posteriors in VI — more flexible than Gaussian",
        "Modern: Neural Spline Flows, Glow (image generation), Neural ODE flow",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Kalman Filter
# ══════════════════════════════════════════════════════════════════════════════
def topic_kalman_filter():
    clear()
    breadcrumb("mlmath", "Probabilistic ML", "Kalman Filter")
    section_header("KALMAN FILTER")
    print()

    section_header("1. THEORY")
    print(white("""
  The Kalman Filter is the exact Bayesian filter for a linear-Gaussian dynamical
  system. It maintains a Gaussian posterior over the hidden state xₜ given
  observations y₁, ..., yₜ.

  LINEAR DYNAMICAL SYSTEM:
      State   transition: xₜ = A xₜ₋₁ + wₜ,    wₜ ~ N(0, Q)
      Observation:        yₜ = C xₜ   + vₜ,    vₜ ~ N(0, R)
  where A is the transition matrix, C is the observation matrix,
  Q is process noise, R is observation noise.

  KALMAN FILTER STEPS (two-step recursion):

  PREDICT STEP (prior for time t):
      μₜ|ₜ₋₁ = A · μₜ₋₁|ₜ₋₁
      Pₜ|ₜ₋₁ = A · Pₜ₋₁|ₜ₋₁ · Aᵀ + Q

  UPDATE STEP (posterior after observing yₜ):
      Innovation:      νₜ = yₜ - C·μₜ|ₜ₋₁
      Innovation cov:  Sₜ = C·Pₜ|ₜ₋₁·Cᵀ + R
      Kalman gain:     Kₜ = Pₜ|ₜ₋₁·Cᵀ·Sₜ⁻¹
      State update:    μₜ|ₜ = μₜ|ₜ₋₁ + Kₜ·νₜ
      Cov update:      Pₜ|ₜ = (I - Kₜ·C)·Pₜ|ₜ₋₁

  KALMAN GAIN INTERPRETATION: if R→0 (observation perfect), K→C⁻¹ and we trust
  the observation completely. If Q→0 (dynamics perfect), K→0 and we trust the
  prediction. K balances these two sources of information optimally.

  OPTIMALITY: the Kalman filter is the optimal linear filter (minimum MSE) for
  linear-Gaussian systems. It's also the exact Bayesian filter in this setting.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Predict:  μ⁻ = Aμ,  P⁻ = APAᵀ + Q"))
    print(formula("  Kalman gain: K = P⁻Cᵀ(CP⁻Cᵀ + R)⁻¹"))
    print(formula("  Update:  μ = μ⁻ + K(y - Cμ⁻),  P = (I-KC)P⁻"))
    print(formula("  Log-likelihood: log p(y₁:T) = Σₜ log N(yₜ; Cμₜ₋₁, CPₜ₋₁Cᵀ+R)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Object Tracking")
    rng = np.random.default_rng(19)
    T = 50
    # 1D position tracking with constant velocity
    # State: [position, velocity]
    A = np.array([[1.0, 1.0], [0.0, 1.0]])  # constant velocity model
    C = np.array([[1.0, 0.0]])              # observe position only
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])  # process noise
    R = np.array([[2.0]])                    # observation noise

    # Generate true trajectory
    x_true = np.zeros((T, 2))
    y_obs  = np.zeros((T, 1))
    x_true[0] = [0.0, 0.5]  # start at 0 with velocity 0.5
    for t in range(1, T):
        x_true[t] = A @ x_true[t-1] + rng.multivariate_normal([0,0], Q)
        y_obs[t]  = C @ x_true[t]   + rng.multivariate_normal([0], R)

    # Kalman Filter
    mu = np.zeros(2); P = np.eye(2) * 5.0
    filtered_positions = []
    for t in range(1, T):
        # Predict
        mu = A @ mu; P = A @ P @ A.T + Q
        # Update
        S = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(S)
        nu = y_obs[t] - C @ mu
        mu = mu + K @ nu
        P  = (np.eye(2) - K @ C) @ P
        filtered_positions.append(mu[0])

    filtered_positions = np.array(filtered_positions)
    estimator_mse = np.mean((filtered_positions - x_true[1:, 0])**2)
    naive_mse     = np.mean((y_obs[1:, 0] - x_true[1:, 0])**2)

    print(f"\n  {bold_cyan('Kalman Filter: 1D Position Tracking (T=50 steps):')}\n")
    print(f"  Observation MSE (naive):  {red(f'{naive_mse:.4f}')}")
    print(f"  Kalman Filter MSE:        {green(f'{estimator_mse:.4f}')}")
    print(f"  Improvement factor:        {green(f'{naive_mse/estimator_mse:.2f}')}")
    print()
    print(f"  {'t':<5} {'True x':<12} {'Observed':<12} {'KF est':<12}")
    print(f"  {'─'*4}  {'─'*8}   {'─'*8}   {'─'*8}")
    for t in range(0, min(15, T-1)):
        print(f"  {t:<5} {green(f'{x_true[t+1,0]:+.3f}'):<20} "
              f"{yellow(f'{y_obs[t+1,0]:+.3f}'):<20} "
              f"{cyan(f'{filtered_positions[t]:+.3f}')}")
    print(f"  ... (T={T})")
    print()
    _pause()

    section_header("4. CODE")
    code_block("Kalman Filter from scratch", """
import numpy as np

def kalman_filter(y, A, C, Q, R, mu0, P0):
    T = len(y); n = len(mu0)
    mu, P = mu0.copy(), P0.copy()
    mus, Ps = [mu.copy()], [P.copy()]
    log_lik = 0.0
    for t in range(T):
        # Predict
        mu = A @ mu; P = A @ P @ A.T + Q
        # Innovation
        nu = y[t] - C @ mu
        S  = C @ P @ C.T + R
        # Log-likelihood contribution
        d = len(nu)
        log_lik += -0.5*(d*np.log(2*np.pi) + np.log(np.linalg.det(S)) +
                         nu @ np.linalg.solve(S, nu))
        # Kalman gain and update
        K  = P @ C.T @ np.linalg.inv(S)
        mu = mu + K @ nu
        P  = (np.eye(n) - K @ C) @ P
        mus.append(mu.copy()); Ps.append(P.copy())
    return np.array(mus[1:]), np.array(Ps[1:]), log_lik

rng = np.random.default_rng(0)
T = 100; A = np.array([[1,1],[0,1]]); C = np.array([[1,0]])
Q = np.eye(2)*0.1; R = np.array([[2.0]])
x0 = np.array([0.0, 0.5])
x = np.zeros((T, 2)); x[0] = x0
for t in range(1, T): x[t] = A @ x[t-1] + rng.multivariate_normal([0,0], Q)
y = np.array([C @ x[t] + rng.multivariate_normal([0], R) for t in range(T)])

mus, Ps, ll = kalman_filter(y, A, C, Q, R, np.zeros(2), np.eye(2)*5)
print(f"Filter MSE = {np.mean((mus[:,0] - x[:,0])**2):.4f}")
print(f"Observation MSE = {np.mean((y[:,0] - x[:,0])**2):.4f}")
print(f"Log-likelihood = {ll:.2f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Kalman Filter = exact Bayesian filter for linear-Gaussian state-space models",
        "Kalman gain K balances model prediction vs observation based on noise ratios",
        "K→C⁻¹ when R→0 (trust observation); K→0 when Q→0 (trust prediction)",
        "Kalman Smoother (RTS): runs forward filter + backward smoother for all states",
        "Extensions: EKF (linearise A,C), UKF (sigma points), Particle Filter (non-linear/non-Gaussian)",
        "Online learning (seq. Bayes): new observation → O(d²) update of (μ, P)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Gaussian Processes",         topic_gaussian_processes),
        ("Bayesian Linear Regression",  topic_bayesian_linear_regression),
        ("Variational Inference",       topic_variational_inference),
        ("MCMC",                        topic_mcmc),
        ("Hamiltonian Monte Carlo",     topic_hmc),
        ("Normalising Flows",           topic_normalising_flows),
        ("Kalman Filter",               topic_kalman_filter),
    ]
    block_menu("b14", "Probabilistic ML", topics)
