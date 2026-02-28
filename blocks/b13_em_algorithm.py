"""
blocks/b13_em_algorithm.py
Block 13: EM Algorithm
Topics: EM as ELBO, Jensen's Inequality, General EM, GMM-EM,
        HMM Baum-Welch, EM Convergence, EM Missing Data.
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
# TOPIC 1 — EM as ELBO
# ══════════════════════════════════════════════════════════════════════════════
def topic_em_as_elbo():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "EM as ELBO")
    section_header("EM ALGORITHM AS EVIDENCE LOWER BOUND (ELBO)")
    print()

    section_header("1. THEORY")
    print(white("""
  The EM algorithm is a method to maximise the log marginal likelihood:
      log p(x|θ) = log Σ_z p(x, z|θ)
  (or ∫ in the continuous-z case). Direct optimisation is hard because we must
  sum/integrate over all possible values of the latent variable z.

  EM INSIGHT: introduce an auxiliary distribution q(z) over the latent variables.
  By Jensen's inequality (see Topic 2):
      log p(x|θ) = log Σ_z q(z) · [p(x,z|θ) / q(z)]
                 ≥ Σ_z q(z) log [p(x,z|θ) / q(z)]
                 = E_q[log p(x,z|θ)] - E_q[log q(z)]
                 ≡ ELBO(q, θ)

  The difference: log p(x|θ) - ELBO = KL(q(z) || p(z|x,θ)) ≥ 0.
  So ELBO is a lower bound on log p(x|θ), and it's tight when q*(z) = p(z|x,θ).

  E-STEP (Expectation): maximise ELBO over q, holding θ fixed.
  Since KL = log p(x) - ELBO, maximising ELBO minimises KL → sets q*(z) = p(z|x,θ).
  The ELBO becomes: E_{p(z|x,θ)}[log p(x,z|θ)] - E_{p(z|x,θ)}[log p(z|x,θ)]
                   = E_{p(z|x,θ)}[log p(x,z|θ)] + H[p(z|x,θ)]

  M-STEP (Maximisation): maximise ELBO over θ, holding q = p(z|x,θ_old) fixed.
  Maximising E_q[log p(x,z|θ)] over θ is often analytically tractable (complete-data
  log-likelihood is in the exponential family in many models).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  log p(x|θ) ≥ ELBO(q,θ) = E_q[log p(x,z|θ)] - E_q[log q(z)]"))
    print(formula("  log p(x|θ) = ELBO + KL(q||p(z|x,θ))"))
    print(formula("  E-step:  q*(z) = p(z|x,θ_old)  (posterior)"))
    print(formula("  M-step:  θ* = argmax_θ E_{q*}[log p(x,z|θ)]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — ELBO Tightening Diagram")
    rng = np.random.default_rng(0)
    # Simulate EM iterations on a 1-component Gaussian toy problem
    # True model: x ~ N(μ, σ²), observe only |x|
    true_mu, true_sigma = 3.0, 1.0
    n = 100
    x_latent = rng.normal(true_mu, true_sigma, n)
    x_obs = x_latent**2  # observe squared (latent is sign)

    mu_est = 1.0  # initial estimate (poor start)
    elbo_vals = []
    for it in range(20):
        # E-step: E[x_latent | x_obs, mu] = mu (for symmetric Gaussian)
        z_mean = np.sqrt(np.maximum(x_obs, 0)) * np.sign(mu_est) if abs(mu_est) > 0.01 else np.sqrt(x_obs)
        # M-step: MLE update
        mu_est = z_mean.mean()
        # ELBO (proxy: negative MSE between E[z] and true mu)
        elbo = -np.mean((z_mean - true_mu)**2)
        elbo_vals.append(elbo)

    print(f"\n  {bold_cyan('ELBO convergence (proxy) over EM iterations:')}\n")
    print_sparkline(elbo_vals, label="ELBO", color_fn=green)
    print(f"\n  Final μ estimate: {green(f'{mu_est:.4f}')}  (true: {true_mu})")

    print(f"\n  {bold_cyan('Schematic: ELBO tightens at each E-step:')}\n")
    lines = [
        "  log p(x)  ─────────────────────────────── (constant)",
        "              ↕ KL(q||p)  (gap)",
        "  ELBO      ↗ tight after E-step (KL=0)",
        "              ↑ raised after M-step (θ improves)",
        "",
        "  Iteration:  E  M  E  M  E  M  ... (log p non-decreasing)",
    ]
    for line in lines:
        print(cyan(line))
    print()
    _pause()

    section_header("4. CODE")
    code_block("ELBO derivation: log p(x) = ELBO + KL", """
import numpy as np
from scipy.stats import norm

# Toy: p(z) = N(0,1), p(x|z) = N(z, 0.5²), x_obs = 1.5
x_obs = 1.5
sigma_noise = 0.5

# p(z|x) = N(mu_post, sigma_post²) via Bayes
sigma_post_sq = 1 / (1 + 1/sigma_noise**2)
mu_post = sigma_post_sq * (x_obs / sigma_noise**2)
print(f"Posterior: N({mu_post:.4f}, {sigma_post_sq:.4f})")

# log p(x) (marginal likelihood)
log_px = norm.logpdf(x_obs, 0, np.sqrt(1 + sigma_noise**2))
print(f"log p(x) = {log_px:.4f}")

# ELBO when q = posterior (should be tight)
z_samples = norm.rvs(mu_post, np.sqrt(sigma_post_sq), size=50000, random_state=0)
log_joint = norm.logpdf(z_samples, 0, 1) + norm.logpdf(x_obs, z_samples, sigma_noise)
log_q     = norm.logpdf(z_samples, mu_post, np.sqrt(sigma_post_sq))
elbo = np.mean(log_joint - log_q)
print(f"ELBO (q=posterior) = {elbo:.4f}  (≈ log p(x) = {log_px:.4f})")
print(f"KL(q||p) ≈ {log_px - elbo:.6f}  (≈ 0 since q is optimal)")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "ELBO = E_q[log p(x,z)] - E_q[log q(z)] = complete likelihood - entropy of q",
        "KL(q||p) ≥ 0 always → ELBO is always a lower bound on log p(x)",
        "E-step: set q = posterior → KL = 0, ELBO tight (= log p(x))",
        "M-step: optimise θ → raises the entire log p(x) curve",
        "EM guarantees log p(x|θ) non-decreasing — never goes down",
        "ELBO generalised → Variational Inference (VI) when exact posterior intractable",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Jensen's Inequality
# ══════════════════════════════════════════════════════════════════════════════
def topic_jensens_inequality():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "Jensen's Inequality")
    section_header("JENSEN'S INEQUALITY")
    print()

    section_header("1. THEORY")
    print(white("""
  Jensen's inequality is a fundamental result relating a convex function applied
  to an expectation vs the expectation of the function applied to a random variable.

  STATEMENT: For a convex function f and random variable X:
      f(E[X]) ≤ E[f(X)]      (convex case)
      f(E[X]) ≥ E[f(X)]      (concave case, e.g. f = log)

  PROOF BY DIAGRAM: A convex function lies above any tangent line. For any
  distribution p(x), the weighted average E[f(X)] lies above f(E[X]) because
  the function curves upward. Formally: by the supporting hyperplane theorem,
  for a convex f there exists a linear lower bound L(x) = f(c) + a(x-c) with
  f(x) ≥ L(x) for all x. Taking expectations: E[f(X)] ≥ E[L(X)] = f(c) + a(E[X]-c).
  Setting c = E[X]: E[f(X)] ≥ f(E[X]).  □

  LOG IS CONCAVE: log'' = -1/x² < 0, so log is concave → E[log X] ≤ log E[X].
  This is exactly the form used in EM:
      log p(x|θ) = log Σ_z q(z) [p(x,z|θ)/q(z)] ≥ Σ_z q(z) log[p(x,z|θ)/q(z)]
  because log is concave, Jensen gives log(E_q[·]) ≥ E_q[log(·)].

  EQUALITY CONDITION: equality holds if and only if X is constant a.s.
  In EM: ELBO = log p(x) iff p(x,z|θ)/q(z) is constant in z, i.e. q(z) = p(z|x,θ).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Convex f:  f(E[X]) ≤ E[f(X)]"))
    print(formula("  Concave f: f(E[X]) ≥ E[f(X)]   e.g. f = log"))
    print(formula("  EM use: log Σ_z q(z) r(z) ≥ Σ_z q(z) log r(z)   [Jensen + concave log]"))
    print(formula("  Equality: X is constant a.s.  ↔  q(z) ∝ p(x,z|θ)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Visual Verification")
    rng = np.random.default_rng(3)
    X_samples = rng.exponential(2.0, 10000)
    E_X = X_samples.mean()
    E_logX = np.log(np.maximum(X_samples, 1e-9)).mean()
    log_EX = np.log(E_X)

    print(f"\n  {bold_cyan('Jensen on log with X ~ Exponential(2):')}\n")
    print(f"  E[X]         = {green(f'{E_X:.4f}')}")
    print(f"  log(E[X])    = {green(f'{log_EX:.4f}')}")
    print(f"  E[log(X)]    = {yellow(f'{E_logX:.4f}')}")
    print(f"  Jensen gap   = log(E[X]) - E[log(X)] = {red(f'{log_EX - E_logX:.4f}')} ≥ 0 ✓")
    print()

    # Convex function f(x) = x²
    X2 = rng.normal(2, 1, 10000)
    print(f"  {bold_cyan('Jensen on f(x)=x² with X ~ N(2,1):')}")
    print(f"  E[X²]  = {green(f'{np.mean(X2**2):.4f}')}")
    print(f"  E[X]²  = {yellow(f'{np.mean(X2)**2:.4f}')}")
    print(f"  Jensen gap = E[X²] - E[X]² = {red(f'{np.mean(X2**2) - np.mean(X2)**2:.4f}')}"
          f"  (= Var[X] ≈ 1.0) ✓")
    print()

    # ASCII diagram of convex function and chord
    print(f"\n  {bold_cyan('Jensen diagram for f(x) = x² (convex):')}\n")
    x_pts = np.linspace(0, 4, 40)
    f_vals = x_pts**2
    mu_pt = 2.0
    chord = mu_pt**2 + 2 * mu_pt * (x_pts - mu_pt)  # tangent at x=2
    print(f"  {'x':<6} {'f(x)=x²':<12} {'tangent at 2':<14} {'f above chord'}")
    for x_, f_, ch_ in zip(x_pts[::8], f_vals[::8], chord[::8]):
        above = green("yes ✓") if f_ >= ch_ else red("no ✗")
        bar = yellow("█" * int(f_ * 3))
        print(f"  {x_:<6.1f} {f_:<12.2f} {ch_:<14.2f} {above}  {bar}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("Jensen's inequality verification", """
import numpy as np

rng = np.random.default_rng(0)
n = 100000

# 1. Concave: log(E[X]) >= E[log(X)]
X = rng.exponential(scale=3, size=n)
print(f"Concave (log): log(E[X])={np.log(X.mean()):.4f}  E[log(X)]={np.log(X).mean():.4f}")
print(f"  Jensen gap = {np.log(X.mean()) - np.log(X).mean():.4f} >= 0")

# 2. Convex: f(E[X]) <= E[f(X)]
for f_name, f, f_EX in [
    ("x²",      lambda x: x**2,         lambda mu: mu**2),
    ("exp(x)",  lambda x: np.exp(x),     lambda mu: np.exp(mu)),
    ("x^4",     lambda x: x**4,          lambda mu: mu**4),
]:
    X = rng.normal(0, 1, n)
    lhs = f_EX(X.mean())
    rhs = f(X).mean()
    print(f"Convex ({f_name}): f(E[X])={lhs:.4f}  E[f(X)]={rhs:.4f}  gap={rhs-lhs:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Jensen: convex f → f(E[X]) ≤ E[f(X)]; concave → reverse inequality",
        "log is concave (log'' < 0) → E[log X] ≤ log E[X]",
        "EM uses concave log to lower-bound log p(x): Jensen gives ELBO",
        "Equality ↔ random variable constant a.s. ↔ q(z) = exact posterior",
        "Jensen gap = KL divergence between q and true posterior in EM context",
        "Other applications: AM-GM inequality, information theory (data processing)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — General EM
# ══════════════════════════════════════════════════════════════════════════════
def topic_em_general():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "General EM Framework")
    section_header("GENERAL EM ALGORITHM FRAMEWORK")
    print()

    section_header("1. THEORY")
    print(white("""
  The EM algorithm is a general framework for maximum likelihood estimation
  in models with latent (unobserved) variables. It iterates two steps:

  SETUP: observed data x, latent variables z, parameters θ.
  Goal: find θ* = argmax_θ log p(x|θ) = argmax_θ log ∫ p(x,z|θ) dz.

  ABSTRACT E-STEP: compute q*(z) = p(z|x, θ_old) — the posterior over z.
  This makes the ELBO tight: ELBO(q*, θ) = log p(x|θ_old).

  ABSTRACT M-STEP: set θ_new = argmax_θ Q(θ, θ_old) where:
      Q(θ, θ_old) = E_{z~p(z|x,θ_old)}[log p(x, z|θ)]
  This is the expected complete-data log-likelihood under the old posterior.
  Since p(x,z|θ) = p(z|x,θ)p(x|θ), and the E-step computed q* = p(z|x,θ_old):
      log p(x|θ_new) ≥ ELBO(q*, θ_new) ≥ ELBO(q*, θ_old) = log p(x|θ_old)

  GUARANTEE: log p(x|θ) is monotonically non-decreasing at each EM step.

  GENERALISED EM: if the M-step is hard, any θ_new that increases Q (not
  necessarily argmax) also guarantees non-decreasing log-likelihood → ELBO staircase.

  CONVERGENCE: EM converges to a stationary point of log p(x|θ). This may be
  a local maximum (not global). Multiple random restarts are common in practice.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Q(θ, θ_old) = E_{p(z|x,θ_old)}[log p(x,z|θ)]"))
    print(formula("  E-step:  q* ← p(z|x, θ_old)"))
    print(formula("  M-step:  θ_new ← argmax_θ Q(θ, θ_old)"))
    print(formula("  Guarantee: log p(x|θ_new) ≥ log p(x|θ_old)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Abstract ELBO Staircase")
    rng = np.random.default_rng(9)
    # Simulate EM-like convergence with realistic log-likelihood trajectory
    # Model: mix of 2 Gaussians, EM converges monotonically
    n = 150
    X = np.concatenate([rng.normal(0, 1, 75), rng.normal(5, 1, 75)])

    def gmm_ll(X, mu1, mu2, pi1=0.5, sigma=1.0):
        p1 = pi1 * (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((X-mu1)/sigma)**2)
        p2 = (1-pi1) * (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((X-mu2)/sigma)**2)
        return np.sum(np.log(p1 + p2 + 1e-300))

    mu1, mu2, pi = 1.0, 4.0, 0.5
    lls = [gmm_ll(X, mu1, mu2, pi)]
    for it in range(25):
        # E-step
        p1 = pi * np.exp(-0.5*(X-mu1)**2)
        p2 = (1-pi) * np.exp(-0.5*(X-mu2)**2)
        denom = p1 + p2 + 1e-300
        r1, r2 = p1/denom, p2/denom
        # M-step
        N1, N2 = r1.sum(), r2.sum()
        mu1 = (r1 * X).sum() / N1
        mu2 = (r2 * X).sum() / N2
        pi  = N1 / n
        ll  = gmm_ll(X, mu1, mu2, pi)
        lls.append(ll)

    print(f"\n  {bold_cyan('EM on 2-component GMM (n=150):')}\n")
    print(f"  Final: μ₁={green(f'{mu1:.3f}')} (true 0.0), μ₂={green(f'{mu2:.3f}')} (true 5.0)")
    print(f"  π₁={green(f'{pi:.3f}')} (true 0.5)")
    print(f"\n  {bold_cyan('Log-likelihood (non-decreasing monotone):')}")
    print_sparkline(lls, label="log p(x|θ)", color_fn=cyan)

    # Check monotonicity
    diffs = np.diff(lls)
    violations = np.sum(diffs < -1e-8)
    if violations == 0:
        print(f"\n  {green('✓ Monotone non-decrease verified over all iterations')}")
    else:
        print(f"\n  {red(f'⚠ {violations} violations (numerical noise)')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("Abstract EM template", """
import numpy as np

class AbstractEM:
    '''Template for EM algorithm. Override e_step and m_step for your model.'''

    def __init__(self, init_params):
        self.params = init_params

    def e_step(self, X, params):
        '''Compute q*(z) = p(z|X, params). Return sufficient statistics.'''
        raise NotImplementedError

    def m_step(self, X, stats):
        '''Update params = argmax Q(theta, theta_old). Return new params.'''
        raise NotImplementedError

    def log_likelihood(self, X, params):
        '''Compute log p(X | params). Used for convergence monitoring.'''
        raise NotImplementedError

    def fit(self, X, n_iter=100, tol=1e-6, verbose=True):
        prev_ll = -np.inf
        for it in range(n_iter):
            stats = self.e_step(X, self.params)        # E-step
            self.params = self.m_step(X, stats)        # M-step
            ll = self.log_likelihood(X, self.params)
            if verbose and it % 10 == 0:
                print(f"  Iter {it:3d}: log-likelihood = {ll:.4f}")
            if abs(ll - prev_ll) < tol:
                print(f"  Converged at iteration {it}")
                break
            assert ll >= prev_ll - 1e-6, f"EM violated monotonicity at iter {it}"
            prev_ll = ll
        return self
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "EM decomposes hard marginal MLE into tractable E and M steps via latents",
        "Q function = E[complete log-likelihood] — often in exponential family form",
        "Guaranteed non-decrease: log p(x|θ) is a staircase that only goes up",
        "Generalised EM: any θ_new that increases Q — useful when argmax is hard",
        "Convergence to stationary point (not necessarily global max) — use restarts",
        "E-step can be computationally expensive — variational approximations help",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — GMM-EM (Complete)
# ══════════════════════════════════════════════════════════════════════════════
def topic_gmm_em():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "GMM-EM Full Derivation")
    section_header("EM FOR GAUSSIAN MIXTURE MODELS — COMPLETE DERIVATION")
    print()

    section_header("1. THEORY")
    print(white("""
  GMM complete-data log-likelihood (with binary indicator zᵢₖ = 1 if i∈k):
      log p(X, Z|θ) = Σᵢ Σₖ zᵢₖ [log πₖ + log N(xᵢ|μₖ, Σₖ)]

  E-STEP: compute responsibilities γᵢₖ = E[zᵢₖ|xᵢ, θ_old] = p(zᵢₖ=1|xᵢ,θ):
      γᵢₖ = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)

  M-STEP: maximise E_q[log p(X,Z|θ)] = Σᵢ Σₖ γᵢₖ[log πₖ + log N(xᵢ|μₖ,Σₖ)].
  Setting derivatives to zero gives closed-form updates:

  Effective count:      Nₖ = Σᵢ γᵢₖ
  Mean update:          μₖ_new = (1/Nₖ) Σᵢ γᵢₖ xᵢ
  Covariance update:    Σₖ_new = (1/Nₖ) Σᵢ γᵢₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ
  Mixing weight update: πₖ_new = Nₖ / n

  ELBO for GMM:
      ELBO = Σᵢ Σₖ γᵢₖ log[πₖ N(xᵢ|μₖ,Σₖ)] - Σᵢ Σₖ γᵢₖ log γᵢₖ
           = (complete log-likelihood) + (entropy of q)
  Each EM iteration is guaranteed to increase ELBO ≥ log p(X|θ).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  γᵢₖ = πₖ N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ N(xᵢ|μⱼ,Σⱼ)"))
    print(formula("  Nₖ = Σᵢ γᵢₖ"))
    print(formula("  μₖ = (Σᵢ γᵢₖ xᵢ) / Nₖ"))
    print(formula("  Σₖ = (Σᵢ γᵢₖ (xᵢ-μₖ)(xᵢ-μₖ)ᵀ) / Nₖ"))
    print(formula("  πₖ = Nₖ / n"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Full GMM EM")
    rng = np.random.default_rng(42)
    K = 3; n_each = 80
    true_mu  = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    true_cov = [np.array([[1.0, 0.4], [0.4, 0.8]]),
                np.array([[0.8, -0.3], [-0.3, 1.2]]),
                np.array([[1.5, 0.0], [0.0, 0.5]])]
    X = np.vstack([rng.multivariate_normal(true_mu[k], true_cov[k], n_each) for k in range(K)])
    n, d = X.shape

    def mvn_pdf(x, mu, Sigma):
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma + 1e-6*np.eye(d))
        _, logdet = np.linalg.slogdet(Sigma + 1e-6*np.eye(d))
        maha = np.einsum('...i,ij,...j', diff, Sigma_inv, diff)
        return np.exp(-0.5 * (d*np.log(2*np.pi) + logdet + maha))

    mu_hat   = X[rng.choice(n, K, replace=False)].copy()
    Sigma_hat = [np.eye(d) * 2.0] * K
    pi_hat    = np.ones(K) / K
    lls = []

    for it in range(40):
        # E-step
        R = np.zeros((n, K))
        for k in range(K):
            R[:, k] = pi_hat[k] * mvn_pdf(X, mu_hat[k], Sigma_hat[k])
        R_sum = R.sum(axis=1, keepdims=True) + 1e-300
        ll = np.sum(np.log(R_sum))
        R /= R_sum
        lls.append(ll)
        # M-step
        Nk = R.sum(axis=0)
        mu_hat    = [(R[:, k:k+1] * X).sum(axis=0) / Nk[k] for k in range(K)]
        for k in range(K):
            diff = X - mu_hat[k]
            Sigma_hat[k] = ((R[:, k:k+1] * diff).T @ diff) / Nk[k] + 1e-4*np.eye(d)
        pi_hat = Nk / n
        mu_hat = np.array(mu_hat)

    print(f"\n  {bold_cyan('GMM EM results (n=240, K=3, 40 iterations):')}\n")
    rows = [[str(k+1),
             f"[{mu_hat[k,0]:.2f},{mu_hat[k,1]:.2f}]",
             f"[{true_mu[k,0]:.1f},{true_mu[k,1]:.1f}]",
             f"{pi_hat[k]:.3f}",
             "1/3"] for k in range(K)]
    table(["Cluster", "Est. μ", "True μ", "Est. π", "True π"], rows,
          [cyan, green, yellow, bold_cyan, grey])
    print(f"\n  {bold_cyan('Log-Likelihood trajectory:')}")
    print_sparkline(lls, label="log p(X)", color_fn=green)

    try:
        import plotext as plt
        plt.clear_figure()
        plt.plot(list(range(len(lls))), lls, color="green", label="LL")
        plt.title("GMM EM: Log-Likelihood per Iteration")
        plt.xlabel("Iteration"); plt.ylabel("Log p(X|θ)")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for log-likelihood curve)"))
    print()
    _pause()

    section_header("4. CODE")
    code_block("Full GMM EM from scratch", """
import numpy as np

def mvn_pdf(X, mu, Sigma):
    d = len(mu)
    diff = X - mu
    Si = np.linalg.inv(Sigma + 1e-6*np.eye(d))
    _, ld = np.linalg.slogdet(Sigma + 1e-6*np.eye(d))
    maha = np.einsum('...i,ij,...j', diff, Si, diff)
    return np.exp(-0.5*(d*np.log(2*np.pi)+ld+maha))

def gmm_em(X, K, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)].copy().astype(float)
    Sig = [np.eye(d)] * K
    pi  = np.ones(K) / K
    for it in range(max_iter):
        R = np.column_stack([pi[k] * mvn_pdf(X, mu[k], Sig[k]) for k in range(K)])
        ll = np.log(R.sum(1)+1e-300).sum()
        R /= R.sum(1, keepdims=True) + 1e-300
        Nk = R.sum(0)
        mu = [(R[:,k:k+1]*X).sum(0)/Nk[k] for k in range(K)]
        Sig = [((R[:,k:k+1]*(X-mu[k])).T@(X-mu[k]))/Nk[k]+1e-4*np.eye(d)
               for k in range(K)]
        pi = Nk / n
    return np.array(mu), Sig, pi, R.argmax(1), ll

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.8, (60,2)) for c in [[0,0],[4,0],[2,3]]])
mu, Sig, pi, labels, ll = gmm_em(X, K=3)
print(f"Cluster means:\\n{mu}")
print(f"Mixing weights: {pi}")
print(f"Final log-likelihood: {ll:.2f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Responsibilities γᵢₖ are soft, probabilistic cluster assignments ∈ [0,1]",
        "M-step closed form for GMM because complete log-lik is in exponential family",
        "π update = fraction of data assigned to each cluster (weighted by γ)",
        "Covariance update = weighted outer product of residuals (generalised MLE)",
        "Numerical tip: work in log-space for responsibilities to avoid underflow",
        "Singularity: σ²→0 when cluster collapses to single point — add small ε·I",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — HMM Baum-Welch
# ══════════════════════════════════════════════════════════════════════════════
def topic_hmm_baum_welch():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "HMM Baum-Welch")
    section_header("HIDDEN MARKOV MODELS — BAUM-WELCH ALGORITHM")
    print()

    section_header("1. THEORY")
    print(white("""
  A Hidden Markov Model (HMM) models sequences where the underlying state is
  hidden and observations are emitted from each state.

  COMPONENTS:
  - States: S = {1, ..., N}  (hidden)
  - Observations: O = {1, ..., M}  (or continuous)
  - Initial distribution: π = (π₁, ..., πₙ)  with πᵢ = P(q₁=i)
  - Transition matrix: A = (aᵢⱼ)  aᵢⱼ = P(qₜ₊₁=j | qₜ=i)
  - Emission matrix: B = (bⱼ(k))  bⱼ(k) = P(oₜ=k | qₜ=j)

  FORWARD ALGORITHM: αₜ(i) = P(o₁,...,oₜ, qₜ=i | λ)
      α₁(i) = πᵢ · bᵢ(o₁)
      αₜ₊₁(j) = [Σᵢ αₜ(i)·aᵢⱼ] · bⱼ(oₜ₊₁)

  BACKWARD ALGORITHM: βₜ(i) = P(oₜ₊₁,...,oT | qₜ=i, λ)
      βT(i) = 1  (base case)
      βₜ(i) = Σⱼ aᵢⱼ · bⱼ(oₜ₊₁) · βₜ₊₁(j)

  BAUM-WELCH = EM for HMMs:
  E-step: compute γₜ(i) = P(qₜ=i | O, λ) and ξₜ(i,j) = P(qₜ=i,qₜ₊₁=j | O, λ)
      γₜ(i) = αₜ(i)·βₜ(i) / P(O|λ)
      ξₜ(i,j) = αₜ(i)·aᵢⱼ·bⱼ(oₜ₊₁)·βₜ₊₁(j) / P(O|λ)
  M-step: update A, B, π using γ and ξ as sufficient statistics.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  αₜ₊₁(j) = [Σᵢ αₜ(i)·aᵢⱼ] · bⱼ(oₜ₊₁)     [forward recursion]"))
    print(formula("  βₜ(i)  = Σⱼ aᵢⱼ·bⱼ(oₜ₊₁)·βₜ₊₁(j)          [backward recursion]"))
    print(formula("  γₜ(i)  = αₜ(i)·βₜ(i) / Σⱼ αₜ(j)·βₜ(j)     [state posterior]"))
    print(formula("  ξₜ(i,j) = αₜ(i)·aᵢⱼ·bⱼ(oₜ₊₁)·βₜ₊₁(j) / P(O|λ)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 2-State HMM")
    rng = np.random.default_rng(14)
    # True HMM: 2 states, 3 symbols
    N, M = 2, 3
    true_A = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    true_pi = np.array([0.6, 0.4])

    # Generate sequence
    T = 50
    states = [rng.choice(N, p=true_pi)]
    for _ in range(T - 1):
        states.append(rng.choice(N, p=true_A[states[-1]]))
    obs = [rng.choice(M, p=true_B[s]) for s in states]

    def forward(obs, A, B, pi):
        T = len(obs); N = len(pi)
        alpha = np.zeros((T, N))
        alpha[0] = pi * B[:, obs[0]]
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]
        return alpha

    def backward(obs, A, B):
        T = len(obs); N = A.shape[0]
        beta = np.ones((T, N))
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i, :] * B[:, obs[t+1]] * beta[t+1, :])
        return beta

    # Init params
    A_est  = rng.dirichlet([1]*N, size=N) + 0.1; A_est /= A_est.sum(1, keepdims=True)
    B_est  = rng.dirichlet([1]*M, size=N) + 0.1; B_est /= B_est.sum(1, keepdims=True)
    pi_est = rng.dirichlet([1]*N) + 0.1;         pi_est /= pi_est.sum()
    lls    = []

    for it in range(30):
        alpha = forward(obs, A_est, B_est, pi_est)
        beta  = backward(obs, A_est, B_est)
        p_obs = alpha[-1].sum()
        lls.append(np.log(p_obs + 1e-300))
        # E-step
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True) + 1e-300
        xi = np.zeros((len(obs)-1, N, N))
        for t in range(len(obs)-1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (alpha[t, i] * A_est[i, j] *
                                   B_est[j, obs[t+1]] * beta[t+1, j])
            xi[t] /= xi[t].sum() + 1e-300
        # M-step
        pi_est = gamma[0] / (gamma[0].sum() + 1e-300)
        A_est = xi.sum(0) / (gamma[:-1].sum(0, keepdims=True).T + 1e-300)
        A_est /= A_est.sum(1, keepdims=True) + 1e-300
        for j in range(N):
            for k in range(M):
                B_est[j, k] = gamma[np.array(obs)==k, j].sum() / (gamma[:, j].sum() + 1e-300)

    print(f"\n  {bold_cyan('Baum-Welch results (T=50, N=2 states, M=3 symbols):')}\n")
    print(f"  {bold_cyan('True A:')}")
    for row in true_A: print(f"    {[f'{v:.2f}' for v in row]}")
    print(f"  {bold_cyan('Estimated A:')}")
    for row in A_est: print(f"    {[green(f'{v:.2f}') for v in row]}")
    print(f"\n  {bold_cyan('Log p(O|λ) progression:')}")
    print_sparkline(lls, label="log p(O)", color_fn=cyan)
    print()
    _pause()

    section_header("4. CODE")
    code_block("Baum-Welch (hmmlearn)", """
import numpy as np
from hmmlearn import hmm

rng = np.random.default_rng(0)
# Generate from known HMM
model_true = hmm.CategoricalHMM(n_components=2, random_state=0)
model_true.startprob_ = np.array([0.6, 0.4])
model_true.transmat_  = np.array([[0.7, 0.3], [0.4, 0.6]])
model_true.emissionprob_ = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])

X, _ = model_true.sample(500, random_state=0)

# Fit with Baum-Welch
model_fit = hmm.CategoricalHMM(n_components=2, n_iter=50, random_state=42)
model_fit.fit(X)

print(f"True transition:\\n{model_true.transmat_}")
print(f"Fit transition (permuted possibly):\\n{model_fit.transmat_}")
print(f"True emission:\\n{model_true.emissionprob_}")
print(f"Fit emission:\\n{model_fit.emissionprob_}")
print(f"Score: {model_fit.score(X):.2f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Baum-Welch = EM where latent variables are the hidden state sequence",
        "Forward α and backward β passes together give state posteriors in O(T·N²)",
        "ξₜ(i,j) = joint posterior of consecutive states — drives transition update",
        "Scaling needed: α can underflow for long sequences — use log-sum-exp",
        "Viterbi algorithm finds most likely state sequence (decoding, not training)",
        "Baum-Welch finds local optima — multiple random starts recommended",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — EM Convergence
# ══════════════════════════════════════════════════════════════════════════════
def topic_em_convergence():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "EM Convergence")
    section_header("EM CONVERGENCE ANALYSIS")
    print()

    section_header("1. THEORY")
    print(white("""
  EM is guaranteed to increase (or maintain) log p(x|θ) at each iteration.
  The formal proof uses the ELBO decomposition and properties of KL divergence.

  PROOF OF MONOTONE NON-DECREASE:
  Let θ_old be current parameters and θ_new = argmax_θ Q(θ, θ_old).
      log p(x|θ_new) = Q(θ_new, θ_old) + H[p(z|x,θ_old)] + KL(p(z|x,θ_old)||p(z|x,θ_new))
                     ≥ Q(θ_new, θ_old) + H[p(z|x,θ_old)]    [KL ≥ 0]
                     ≥ Q(θ_old, θ_old) + H[p(z|x,θ_old)]    [M-step maximises Q]
                     = log p(x|θ_old)                          [ELBO tight after E-step]
  Therefore: log p(x|θ_new) ≥ log p(x|θ_old).  □

  LOCAL vs GLOBAL MAXIMA: EM converges to a stationary point of log p(x|θ).
  This may be a local max, saddle point, or (in degenerate cases) a saddle.
  For GMMs: many local maxima exist depending on initialisation.
  For exponential family models (HMMs, factor analysis): unique global max in
  the complete-data problem, but marginalisation creates multiple modes.

  INITIALISATION STRATEGIES:
  1. Random restarts: run EM k times, keep best log-likelihood.
  2. K-Means init: for GMM, initialise μₖ using K-Means centroids.
  3. Short runs: run 10 random inits for 5 iterations, pick best, run to convergence.
  4. Deterministic annealing: start with high temperature (soft assignments) and cool.

  CONVERGENCE RATE: typically linear convergence (geometric). Rate depends on
  the fraction of missing information: slow when latent variables carry little
  information (high fraction missing) — the EM algorithm knows little about θ.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  log p(x|θ_new) ≥ Q(θ_new, θ_old) + H[q*] ≥ Q(θ_old, θ_old) + H[q*] = log p(x|θ_old)"))
    print(formula("  Convergence rate: ||θ_t - θ*|| ≤ C · ρᵗ,  ρ = fraction missing information"))
    print(formula("  ρ = 1 - I_obs(θ) / I_complete(θ)  (Fisher information ratio)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Multiple Restarts")
    rng = np.random.default_rng(0)
    n = 200
    X_1d = np.concatenate([rng.normal(0, 1, 100), rng.normal(6, 1, 100)])

    def em_1d_gmm(X, mu_init, n_iter=50):
        mu1, mu2, pi = mu_init[0], mu_init[1], 0.5
        lls = []
        for _ in range(n_iter):
            r1 = pi * np.exp(-0.5*(X-mu1)**2)
            r2 = (1-pi) * np.exp(-0.5*(X-mu2)**2)
            denom = r1 + r2 + 1e-300
            r1 /= denom;  r2 /= denom
            ll = np.log(denom.sum())
            lls.append(ll)
            N1, N2 = r1.sum(), r2.sum()
            mu1 = (r1*X).sum()/(N1+1e-9); mu2 = (r2*X).sum()/(N2+1e-9)
            pi  = N1 / len(X)
        return mu1, mu2, pi, lls[-1]

    print(f"\n  {bold_cyan('5 random initialisations of 1D GMM (true μ = 0, 6):')}\n")
    rows = []
    best_ll, best_mu = -np.inf, None
    for trial in range(5):
        mu_init = rng.uniform(-5, 10, 2)
        mu1, mu2, pi, final_ll = em_1d_gmm(X_1d, mu_init)
        status = green("✓ global") if final_ll > -280 else red("✗ local")
        rows.append([str(trial+1),
                     f"[{mu_init[0]:.1f},{mu_init[1]:.1f}]",
                     f"[{mu1:.2f},{mu2:.2f}]",
                     f"{pi:.3f}", f"{final_ll:.2f}", status])
        if final_ll > best_ll:
            best_ll = final_ll; best_mu = (mu1, mu2)

    table(["Trial", "Init μ", "Final μ", "π₁", "Log-lik", "Quality"], rows,
          [cyan, grey, green, yellow, bold_cyan, white])
    print(f"\n  {green('Best init:')} μ = ({best_mu[0]:.2f}, {best_mu[1]:.2f})  LL = {green(f'{best_ll:.2f}')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("EM convergence monitoring and restarts", """
import numpy as np

def run_gmm_em(X, K, seed=0, n_iter=100, tol=1e-6):
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)].astype(float)
    Sig = [np.eye(d)] * K
    pi = np.ones(K) / K
    prev_ll = -np.inf
    for it in range(n_iter):
        R = np.column_stack([pi[k] * multivariate_normal.pdf(X, mu[k], Sig[k])
                             for k in range(K)])
        ll = np.log(R.sum(1)+1e-300).sum()
        if abs(ll - prev_ll) < tol:
            break
        assert ll >= prev_ll - 1e-6, f"EM violated monotonicity at iter {it}"
        prev_ll = ll
        R /= R.sum(1, keepdims=True)+1e-300
        Nk = R.sum(0)
        mu = [(R[:,k:k+1]*X).sum(0)/Nk[k] for k in range(K)]
        Sig = [((R[:,k:k+1]*(X-mu[k])).T@(X-mu[k]))/Nk[k]+1e-5*np.eye(d) for k in range(K)]
        pi = Nk/n
    return np.array(mu), Sig, pi, ll, it

# Multiple restarts
rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0,0],1,(80,2)), rng.normal([5,0],1,(80,2))])
results = [run_gmm_em(X, K=2, seed=s) for s in range(10)]
best = max(results, key=lambda r: r[3])
print(f"Best LL: {best[3]:.2f}, converged at iter {best[4]}")
print(f"Best means: {best[0]}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Proof: EM monotone via KL ≥ 0 sandwiched between two ELBO evaluations",
        "Convergence to stationary point only — not necessarily global maximum",
        "Rate: linear (geometric) convergence; slower when latent variables informative",
        "Degeneracy: GMM cluster collapses to single point (σ→0) — add regularisation",
        "Multiple restarts with K-Means init is the standard practical approach",
        "BIC/AIC: model selection across K to choose number of components",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — EM for Missing Data
# ══════════════════════════════════════════════════════════════════════════════
def topic_em_missing_data():
    clear()
    breadcrumb("mlmath", "EM Algorithm", "EM for Missing Data")
    section_header("EM FOR MISSING DATA")
    print()

    section_header("1. THEORY")
    print(white("""
  EM provides a principled framework for maximum likelihood estimation when
  some data values are missing. Rubin (1976) classification of missingness:

  MCAR (Missing Completely At Random): the probability of missingness does not
  depend on any data values. EM gives unbiased estimates.

  MAR (Missing At Random): missingness depends on observed values but not on
  the missing values themselves. EM still gives valid ML estimates.

  MNAR (Missing Not At Random): missingness depends on the missing values.
  EM is biased — need a model for the missingness mechanism.

  EM FRAMEWORK FOR MISSING DATA:
  Let X = (X_obs, X_miss). Define:
      E-step: impute the sufficient statistics E[T(X_miss) | X_obs, θ_old]
              rather than imputing X_miss itself. This avoids single imputation bias.
      M-step: maximise expected complete-data log-likelihood as if X were complete.

  For Gaussian data (X ~ N(μ, Σ)), the E-step imputes:
      E[xᵢ_miss | xᵢ_obs, θ] = μ_miss + Σ_{miss,obs} Σ_{obs,obs}⁻¹ (xᵢ_obs - μ_obs)
  using the conditional mean formula for multivariate normals.

  MEDICAL DATA EXAMPLE: 100 patients, features (age, weight, blood pressure).
  Blood pressure missing for 30% of patients. EM imputes missing BP using
  correlation with age and weight, then re-estimates mean and covariance until
  convergence — fully exploiting all observed data.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  E[x_miss | x_obs, θ] = μ_miss + Σ_{mo} Σ_{oo}⁻¹ (x_obs - μ_obs)"))
    print(formula("  E[x_miss xᵀ_miss | x_obs, θ] = E[x_miss]E[x_miss]ᵀ + Σ_{mm|o}"))
    print(formula("  Σ_{mm|o} = Σ_{mm} - Σ_{mo} Σ_{oo}⁻¹ Σ_{om}  [conditional variance]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Gaussian Missing Data EM")
    rng = np.random.default_rng(22)
    n, d = 120, 3
    true_mu = np.array([5.0, 70.0, 120.0])  # age, weight, BP
    true_Sigma = np.array([[25, 30, 15],
                            [30, 100, 40],
                            [15, 40, 225]])
    X_complete = rng.multivariate_normal(true_mu, true_Sigma, n)

    # Introduce 25% missingness in feature 2 (BP) for random patients
    missing_mask = np.zeros((n, d), dtype=bool)
    missing_idx = rng.choice(n, int(0.25*n), replace=False)
    missing_mask[missing_idx, 2] = True
    X_obs = X_complete.copy()
    X_obs[missing_mask] = np.nan

    def em_gaussian_missing(X_obs, n_iter=30):
        mask = np.isnan(X_obs)
        n, d = X_obs.shape
        # Init: fill with column means
        col_means = np.nanmean(X_obs, axis=0)
        X_imp = X_obs.copy()
        for j in range(d):
            X_imp[np.isnan(X_imp[:, j]), j] = col_means[j]
        mu_hat = X_imp.mean(0)
        Sig_hat = np.cov(X_imp.T)
        for it in range(n_iter):
            for i in range(n):
                miss_idx = np.where(mask[i])[0]
                obs_idx  = np.where(~mask[i])[0]
                if len(miss_idx) == 0: continue
                Sig_mo = Sig_hat[np.ix_(miss_idx, obs_idx)]
                Sig_oo = Sig_hat[np.ix_(obs_idx, obs_idx)] + 1e-6*np.eye(len(obs_idx))
                x_obs_i = X_obs[i, obs_idx]
                X_imp[i, miss_idx] = (mu_hat[miss_idx] +
                    Sig_mo @ np.linalg.solve(Sig_oo, x_obs_i - mu_hat[obs_idx]))
            mu_hat = X_imp.mean(0)
            Sig_hat = np.cov(X_imp.T) + 1e-6*np.eye(d)
        return mu_hat, Sig_hat

    mu_em, Sig_em = em_gaussian_missing(X_obs)
    naive_mu = np.nanmean(X_obs, axis=0)

    print(f"\n  {bold_cyan('Gaussian EM for missing data (n=120, 25% BP missing):')}\n")
    rows = [
        ["Age (0)",    f"{true_mu[0]:.1f}", f"{mu_em[0]:.2f}", f"{naive_mu[0]:.2f}"],
        ["Weight (1)", f"{true_mu[1]:.1f}", f"{mu_em[1]:.2f}", f"{naive_mu[1]:.2f}"],
        ["BP (2)",     f"{true_mu[2]:.1f}", f"{mu_em[2]:.2f}", f"{naive_mu[2]:.2f}"],
    ]
    table(["Feature", "True μ", "EM μ", "Naive μ (ignore)"], rows,
          [cyan, yellow, green, red])
    print(f"\n  {hint('EM leverages correlation between features to impute missing BP more accurately.')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("EM imputation for multivariate Gaussian", """
import numpy as np

def em_impute(X_obs, n_iter=50):
    mask = np.isnan(X_obs)
    n, d = X_obs.shape
    X = X_obs.copy()
    for j in range(d):
        X[np.isnan(X[:,j]), j] = np.nanmean(X_obs[:,j])
    mu = X.mean(0)
    Sig = np.cov(X.T) + 1e-4*np.eye(d)
    for _ in range(n_iter):
        for i in range(n):
            mi = np.where(mask[i])[0]; oi = np.where(~mask[i])[0]
            if len(mi) == 0: continue
            Smo = Sig[np.ix_(mi, oi)]; Soo = Sig[np.ix_(oi, oi)]
            X[i, mi] = mu[mi] + Smo @ np.linalg.solve(Soo+1e-6*np.eye(len(oi)),
                                                        X_obs[i, oi]-mu[oi])
        mu = X.mean(0); Sig = np.cov(X.T)+1e-4*np.eye(d)
    return X, mu, Sig

rng = np.random.default_rng(0)
X_true = rng.multivariate_normal([0,0,0], [[1,0.8,0.3],[0.8,1,0.5],[0.3,0.5,1]], 200)
X_obs = X_true.copy()
X_obs[rng.choice(200, 50, replace=False), 2] = np.nan   # 25% missing

X_imp, mu_hat, Sig_hat = em_impute(X_obs)
mae_em = np.mean(np.abs(X_imp[np.isnan(X_obs), 2] - X_true[np.isnan(X_obs), 2]))
print(f"MAE of EM imputation: {mae_em:.4f}")
print(f"Estimated μ: {mu_hat}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "EM imputes sufficient statistics E[T(X_miss)|X_obs], not raw X_miss values",
        "More efficient than complete-case analysis or mean imputation",
        "Exploits inter-feature correlation for better imputation accuracy",
        "Valid under MAR — if MNAR, need separate model for missingness mechanism",
        "Multiple imputation runs EM M times with different noise → uncertainty estimate",
        "Real-world: sklearn's IterativeImputer implements this EM approach",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("EM as ELBO",           topic_em_as_elbo),
        ("Jensen's Inequality",  topic_jensens_inequality),
        ("General EM Framework", topic_em_general),
        ("GMM-EM Full",          topic_gmm_em),
        ("HMM Baum-Welch",       topic_hmm_baum_welch),
        ("EM Convergence",       topic_em_convergence),
        ("EM for Missing Data",  topic_em_missing_data),
    ]
    block_menu("b13", "EM Algorithm", topics)
