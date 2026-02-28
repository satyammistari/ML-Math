"""
blocks/b06_statistics.py
Block 6: Statistics for Machine Learning
Topics: MLE, MAP, Bayesian Inference, Hypothesis Testing,
        p-Values, Confidence Intervals, Effect Size, Multiple Testing.
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
# TOPIC 1 — Maximum Likelihood Estimation
# ══════════════════════════════════════════════════════════════════════════════
def topic_mle():
    clear()
    breadcrumb("mlmath", "Statistics", "Maximum Likelihood Estimation")
    section_header("MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
    print()

    section_header("1. THEORY")
    print(white("""
  Maximum Likelihood Estimation finds the parameters θ that make the observed
  data most probable. Given data x₁,…,xₙ drawn i.i.d. from p(x|θ), we choose:

      θ_MLE = argmax_θ  ∏ᵢ p(xᵢ|θ)

  Taking logs (a monotone transform that turns products into sums and makes
  optimisation easier), we maximise the log-likelihood:

      ℓ(θ) = Σᵢ log p(xᵢ|θ)

  MLE has excellent large-sample properties: it is consistent (converges to the
  true θ), asymptotically efficient (achieves the Cramér-Rao lower bound), and
  asymptotically normal. For small samples it can overfit — that is where MAP
  and Bayesian methods help.

  KEY DERIVATIONS:
  • Gaussian: ℓ(μ,σ²) = -n/2 log(2πσ²) - Σ(xᵢ-μ)²/(2σ²).
    Setting ∂ℓ/∂μ=0 gives μ_MLE = x̄.
    Setting ∂ℓ/∂σ²=0 gives σ²_MLE = (1/n)Σ(xᵢ-x̄)².  Note: biased!
  • Bernoulli: ℓ(p) = Σxᵢ log p + (n-Σxᵢ) log(1-p) → p_MLE = x̄.
  • Poisson:   ℓ(λ) = Σxᵢ log λ - nλ - Σlog(xᵢ!) → λ_MLE = x̄.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Likelihood:      L(θ) = ∏ᵢ p(xᵢ|θ)"))
    print(formula("  Log-likelihood:  ℓ(θ) = Σᵢ log p(xᵢ|θ)"))
    print(formula("  MLE condition:   ∂ℓ/∂θ = 0   (score equation)"))
    print(formula("  Gaussian MLE:    μ̂ = x̄,   σ̂² = (1/n)Σ(xᵢ-x̄)²"))
    print(formula("  Bernoulli MLE:   p̂ = x̄"))
    print(formula("  Poisson MLE:     λ̂ = x̄"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Gaussian MLE")
    rng = np.random.default_rng(42)
    true_mu, true_sigma = 5.0, 2.0
    data = rng.normal(true_mu, true_sigma, size=50)
    mu_mle = np.mean(data)
    sigma2_mle = np.mean((data - mu_mle) ** 2)
    sigma2_unbiased = np.var(data, ddof=1)
    ll = -len(data)/2 * np.log(2 * np.pi * sigma2_mle) - np.sum((data - mu_mle)**2) / (2 * sigma2_mle)

    print(bold_cyan(f"  [1] True parameters:           μ = {true_mu},  σ = {true_sigma}"))
    print(bold_cyan(f"  [2] Sample size:               n = {len(data)}"))
    print(bold_cyan(f"  [3] MLE mu:                    μ̂ = {mu_mle:.4f}  (true={true_mu})"))
    print(bold_cyan(f"  [4] MLE sigma² (biased):       σ̂² = {sigma2_mle:.4f}"))
    print(bold_cyan(f"  [5] Unbiased sigma² (ddof=1):  s² = {sigma2_unbiased:.4f}"))
    print(bold_cyan(f"  [6] Log-likelihood at MLE:     ℓ = {ll:.4f}"))
    print()
    print(green("  Note: MLE σ² uses 1/n (biased), sample variance uses 1/(n-1)."))
    print(green("  For n=50 the difference is small but for n=5 it matters."))
    print()
    _pause()

    section_header("4. ASCII VISUALIZATION — Log-Likelihood Surface (μ axis)")
    mus = np.linspace(mu_mle - 3, mu_mle + 3, 40)
    lls = [-len(data)/2 * np.log(2*np.pi*sigma2_mle) - np.sum((data - m)**2)/(2*sigma2_mle) for m in mus]
    ll_min, ll_max = min(lls), max(lls)
    print(cyan(f"  Log-likelihood ℓ(μ | data, σ̂²={sigma2_mle:.2f})"))
    print(cyan(f"  Peak at μ = {mu_mle:.4f}"))
    print()
    for i, (m, l) in enumerate(zip(mus[::4], lls[::4])):
        norm = (l - ll_min) / (ll_max - ll_min + 1e-12)
        bar = "█" * int(norm * 30)
        marker = green(" ← MLE") if abs(m - mu_mle) < 0.4 else ""
        print(f"  μ={m:5.2f}  {cyan(bar)}{marker}")
    print()
    _pause()

    section_header("5. PLOTEXT VISUALIZATION")
    try:
        xs = np.linspace(mu_mle - 4, mu_mle + 4, 60)
        ys = np.array([-len(data)/2 * np.log(2*np.pi*sigma2_mle) - np.sum((data - m)**2)/(2*sigma2_mle) for m in xs])
        loss_curve_plot(ys, title=f"Log-likelihood vs μ (peak={mu_mle:.2f})")
    except Exception:
        print(grey("  plotext unavailable — see ASCII plot above"))
    print()
    _pause()

    section_header("6. PYTHON CODE")
    code_block("MLE for Gaussian, Bernoulli, Poisson", """
import numpy as np

rng = np.random.default_rng(0)

# --- Gaussian MLE ---
data_g = rng.normal(loc=5, scale=2, size=100)
mu_mle  = np.mean(data_g)
sig_mle = np.std(data_g, ddof=0)   # biased (1/n)
print(f"Gaussian MLE: mu={mu_mle:.3f}, sigma={sig_mle:.3f}")

# --- Bernoulli MLE ---
data_b = rng.binomial(1, p=0.3, size=200)
p_mle  = np.mean(data_b)
print(f"Bernoulli MLE: p={p_mle:.3f}")

# --- Poisson MLE ---
data_p = rng.poisson(lam=4.5, size=200)
lam_mle = np.mean(data_p)
print(f"Poisson MLE: lambda={lam_mle:.3f}")

# --- Log-likelihood at MLE ---
ll_g = np.sum(-0.5*np.log(2*np.pi*sig_mle**2) - (data_g - mu_mle)**2/(2*sig_mle**2))
print(f"Gaussian log-likelihood at MLE: {ll_g:.2f}")
""")
    _pause()

    section_header("7. KEY INSIGHTS")
    insights = [
        "MLE maximises data probability — it is the 'most consistent with data' estimate",
        "MLE Gaussian σ² is biased (divides by n); unbiased estimator divides by n-1",
        "For exponential families (Gaussian, Bernoulli, Poisson) MLE = sufficient statistic",
        "MLE = Method of Moments for Gaussian; they differ for, e.g., Gamma distribution",
        "With infinite data MLE converges to the true θ (consistency)",
        "MLE is sensitive to outliers — robust alternatives: M-estimators, trimmed MLE",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Maximum A Posteriori
# ══════════════════════════════════════════════════════════════════════════════
def topic_map():
    clear()
    breadcrumb("mlmath", "Statistics", "Maximum A Posteriori (MAP)")
    section_header("MAXIMUM A POSTERIORI ESTIMATION (MAP)")
    print()

    section_header("1. THEORY")
    print(white("""
  MAP adds a PRIOR over parameters to MLE via Bayes' theorem:

      P(θ|data) ∝ P(data|θ) · P(θ)

  We maximise the posterior:
      θ_MAP = argmax_θ [ log P(data|θ) + log P(θ) ]
             = MLE objective + regularisation term

  The log prior acts as a REGULARISER on the log-likelihood.  This means MAP
  is equivalent to penalised maximum likelihood, which is exactly regularisation
  in machine learning!

  PRIOR → REGULARISATION EQUIVALENCE:
  • Gaussian prior N(0,τ²) on weights:
      log P(θ) = -‖θ‖²/(2τ²) + const  →  Ridge (L2) regularisation
      Minimising -log posterior = MSE + λ‖θ‖²  with λ = σ²/τ²

  • Laplace prior Laplace(0,b) on weights:
      log P(θ) = -‖θ‖₁/b + const  →  Lasso (L1) regularisation

  This bridge between Bayesian statistics and regularisation is one of the
  most elegant results in machine learning theory.  Understanding it lets you
  choose the right prior (= right regulariser) for your problem.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Bayes:          P(θ|x) ∝ P(x|θ) · P(θ)"))
    print(formula("  MAP:            θ_MAP = argmax_θ [ℓ(θ) + log P(θ)]"))
    print(formula("  Gaussian prior: log P(θ) = -‖θ‖²/(2τ²)  →  L2 penalty"))
    print(formula("  Laplace prior:  log P(θ) = -‖θ‖₁/b      →  L1 penalty"))
    print(formula("  MAP Gaussian:   θ_MAP = (XᵀX + λI)⁻¹ Xᵀy   (Ridge)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — MAP for Bernoulli with Beta Prior")
    # Beta(α,β) prior + Binomial likelihood → Beta posterior
    alpha_prior, beta_prior = 2.0, 2.0   # prior: symmetric, mild
    n_heads, n_tails = 7, 3              # 10 coin flips, 7 heads
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails
    p_mle = n_heads / (n_heads + n_tails)
    p_map = (alpha_post - 1) / (alpha_post + beta_post - 2)   # mode of Beta
    p_bayes_mean = alpha_post / (alpha_post + beta_post)       # mean of Beta

    print(bold_cyan(f"  [1] Prior: Beta(α={alpha_prior}, β={beta_prior})  →  prior mode = {(alpha_prior-1)/(alpha_prior+beta_prior-2):.2f}"))
    print(bold_cyan(f"  [2] Data:  {n_heads} heads, {n_tails} tails  →  MLE = {p_mle:.2f}"))
    print(bold_cyan(f"  [3] Posterior: Beta(α={alpha_post}, β={beta_post})"))
    print(bold_cyan(f"  [4] MAP (posterior mode):  p = {p_map:.4f}"))
    print(bold_cyan(f"  [5] Bayesian mean (PME):   p = {p_bayes_mean:.4f}"))
    print()
    print(green("  MAP is pulled toward the prior: 0.667 vs MLE 0.700"))
    print(green("  With n→∞ the prior is overwhelmed and MAP → MLE"))
    print()
    _pause()

    section_header("4. PRIOR → REGULARISATION TABLE")
    print()
    table(
        ["Prior", "Distribution", "Penalty", "ML Name", "Effect"],
        [
            ["Gaussian",  "N(0,τ²)",         "λ‖θ‖²",  "Ridge (L2)",  "Shrink all weights"],
            ["Laplace",   "Laplace(0,b)",     "λ‖θ‖₁",  "Lasso (L1)",  "Sparse weights"],
            ["Uniform",   "Uniform",          "none",    "MLE",         "No regularisation"],
            ["Horseshoe", "Horseshoe",        "special", "Sparse Bayes","Adaptive sparsity"],
        ]
    )
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("MAP = Regularised MLE", """
import numpy as np

# MAP for linear regression with Gaussian prior = Ridge
# θ_MAP = (XᵀX + λI)⁻¹ Xᵀy,  where λ = sigma²/tau²

np.random.seed(0)
n, d      = 50, 5
X         = np.random.randn(n, d)
true_w    = np.array([1., -2., 0.5, 0., 3.])
y         = X @ true_w + np.random.randn(n) * 0.5

# lambda corresponds to sigma²/tau² (Gaussian prior strength)
for lam in [0.0, 0.1, 1.0, 10.0]:
    w_map = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    mse   = np.mean((X @ w_map - y)**2)
    print(f"  lambda={lam:5.1f}  w={np.round(w_map,2)}  MSE={mse:.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "MAP = MLE + log prior = penalised likelihood",
        "Gaussian prior ↔ L2 regularisation (Ridge), Laplace ↔ L1 (Lasso)",
        "λ = σ²_noise / τ²_prior  —  stronger prior = larger λ = more regularisation",
        "MAP returns a POINT estimate; full Bayesian uses the whole posterior",
        "With n→∞, data dominates and MAP → MLE regardless of prior",
        "MAP can overfit less than MLE but cannot quantify uncertainty",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Bayesian Inference
# ══════════════════════════════════════════════════════════════════════════════
def topic_bayesian_inference():
    clear()
    breadcrumb("mlmath", "Statistics", "Bayesian Inference")
    section_header("BAYESIAN INFERENCE")
    print()

    section_header("1. THEORY")
    print(white("""
  Bayesian inference treats parameters as RANDOM VARIABLES and maintains a
  full probability distribution over them rather than a single point estimate.

  The cornerstone is Bayes' theorem:
      P(θ|data) = P(data|θ) · P(θ) / P(data)

  where:
    P(θ)        = PRIOR: our beliefs about θ before seeing data
    P(data|θ)   = LIKELIHOOD: how probable the data is given θ
    P(θ|data)   = POSTERIOR: updated beliefs after seeing data
    P(data)     = EVIDENCE (marginal likelihood): normalising constant

  SEQUENTIAL UPDATING: today's posterior becomes tomorrow's prior.  After
  seeing each new datapoint, we update: P(θ|x₁,…,xₙ) ∝ P(xₙ|θ)·P(θ|x₁,…,xₙ₋₁).
  This is computationally equivalent to a single batch update.

  CONJUGATE PRIORS: when prior and posterior are the same family, computation
  is closed-form.  Beta-Binomial is the canonical example.

  CREDIBLE INTERVALS vs CONFIDENCE INTERVALS:
    • 95% Credible interval: "there is 95% probability that θ ∈ [a,b]" (Bayesian)
    • 95% Confidence interval: "if we repeated this experiment infinitely often,
      95% of the intervals would contain the true θ" (frequentist)
  The credible interval is what most people THINK a CI means!
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Bayes:       P(θ|x) = P(x|θ)P(θ) / P(x)"))
    print(formula("  P(x) =       ∫ P(x|θ) P(θ) dθ  (marginal likelihood)"))
    print(formula("  Beta-Bin:    Prior Beta(α,β) + Binom(n,p) → Post Beta(α+k, β+n-k)"))
    print(formula("  Post mean:   E[p|data] = (α+k)/(α+β+n)"))
    print(formula("  Post mode:   (α+k-1)/(α+β+n-2)  (MAP estimate)"))
    print()
    _pause()

    section_header("3. BETA-BINOMIAL WORKED EXAMPLE — Sequential Updating")
    alpha, beta_p = 1.0, 1.0   # flat prior
    coin_flips = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]  # 7 heads, 3 tails
    print(bold_cyan(f"  Prior: Beta(α=1, β=1) — uniform (no prior knowledge)"))
    print(bold_cyan(f"  Flips: {coin_flips}"))
    print()
    print(f"  {'Step':<6} {'Flip':<6} {'α_post':<10} {'β_post':<10} {'Post mean':<12} {'95% CI'}")
    print(f"  {'-'*60}")
    for i, flip in enumerate(coin_flips):
        alpha += flip
        beta_p += (1 - flip)
        post_mean = alpha / (alpha + beta_p)
        # 95% credible interval via Beta quantiles
        from scipy import stats as sc_stats
        lo = sc_stats.beta.ppf(0.025, alpha, beta_p)
        hi = sc_stats.beta.ppf(0.975, alpha, beta_p)
        print(f"  {i+1:<6} {('H' if flip else 'T'):<6} {alpha:<10.0f} {beta_p:<10.0f} {post_mean:<12.4f} [{lo:.3f}, {hi:.3f}]")
    print()
    _pause()

    section_header("4. ASCII BAYESIAN UPDATE")
    print(cyan("  Prior: Beta(1,1) — flat bar"))
    print(cyan("  Likelihood shifts distribution toward observed frequency"))
    print(cyan("  Posterior: Beta(8,4) — peaked near 0.7"))
    print()
    ps = np.linspace(0.01, 0.99, 20)
    from scipy.special import betaln
    def beta_pdf(p, a, b):
        return np.exp((a-1)*np.log(p) + (b-1)*np.log(1-p) - betaln(a, b))
    prior_vals  = beta_pdf(ps, 1, 1)
    post_vals   = beta_pdf(ps, 8, 4)
    p_max = max(post_vals)
    for p, pr, po in zip(ps, prior_vals, post_vals):
        b_prior = "░" * int(pr / p_max * 20)
        b_post  = "█" * int(po / p_max * 20)
        print(f"  p={p:.2f}  {grey(b_prior):<22} {cyan(b_post)}")
    print(f"         {grey('Prior')}                    {cyan('Posterior')}")
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Bayesian Beta-Binomial Updating", """
import numpy as np
from scipy import stats

# Beta-Binomial conjugate update
alpha, beta = 1.0, 1.0      # flat prior

observations = np.random.binomial(1, 0.6, size=20)
print(f"Observed {observations.sum()} heads in {len(observations)} flips")

for x in observations:
    alpha += x
    beta  += (1 - x)

print(f"Posterior: Beta(alpha={alpha:.0f}, beta={beta:.0f})")
print(f"Posterior mean: {alpha/(alpha+beta):.4f}")

# 95% credible interval
lo, hi = stats.beta.ppf([0.025, 0.975], alpha, beta)
print(f"95% Credible interval: [{lo:.4f}, {hi:.4f}]")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Posterior = prior × likelihood (up to normalisation)",
        "Conjugate priors give closed-form posteriors — very convenient",
        "More data → posterior concentrates → prior matters less",
        "Bayesian credible intervals have the intuitive probability interpretation",
        "Full Bayesian inference is often intractable → MCMC, Variational Bayes",
        "Predictive distribution: P(x_new|data) = ∫ P(x_new|θ) P(θ|data) dθ",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Hypothesis Testing
# ══════════════════════════════════════════════════════════════════════════════
def topic_hypothesis_testing():
    clear()
    breadcrumb("mlmath", "Statistics", "Hypothesis Testing")
    section_header("HYPOTHESIS TESTING")
    print()

    section_header("1. THEORY")
    print(white("""
  Hypothesis testing is a formal framework for making decisions from data.
  We state a NULL HYPOTHESIS H₀ (the default, e.g., "the drug has no effect")
  and an ALTERNATIVE HYPOTHESIS H₁ (the claim, e.g., "the drug works").

  THE t-TEST (one-sample):
  Given n observations with sample mean x̄ and sample std s, the t-statistic:
      t = (x̄ - μ₀) / (s / √n)  follows a t-distribution with n-1 dof under H₀.
  We reject H₀ if |t| > t_crit (for significance level α, usually 0.05).

  TWO-SAMPLE t-TEST: tests whether two groups have the same mean.
      t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)

  CHI-SQUARED TEST: tests whether categorical observations match expected counts.
      χ² = Σ (Observed - Expected)² / Expected

  TYPE I ERROR (α): rejecting H₀ when it is true (false positive)
  TYPE II ERROR (β): failing to reject H₀ when H₁ is true (false negative)
  POWER = 1 - β = probability of correctly detecting a true effect.

  The threshold α = 0.05 is conventional, not sacred. Think of it as the
  tolerable rate of false alarms.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  One-sample t:  t = (x̄ - μ₀) / (s/√n),  dof = n-1"))
    print(formula("  Two-sample t:  t = (x̄₁-x̄₂) / √(s₁²/n₁ + s₂²/n₂)"))
    print(formula("  Chi-squared:   χ² = Σᵢ (Oᵢ-Eᵢ)²/Eᵢ,  dof = k-1"))
    print(formula("  Type I:        P(reject H₀ | H₀ true) = α"))
    print(formula("  Type II:       P(fail to reject | H₁ true) = β"))
    print(formula("  Power:         1 - β = P(reject H₀ | H₁ true)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — One-Sample t-Test")
    rng = np.random.default_rng(7)
    mu0 = 0.0
    sample = rng.normal(loc=0.5, scale=1.0, size=30)
    n = len(sample)
    xbar, s = np.mean(sample), np.std(sample, ddof=1)
    t_stat = (xbar - mu0) / (s / np.sqrt(n))
    from scipy import stats as sc
    p_val = 2 * sc.t.sf(abs(t_stat), df=n-1)
    t_crit = sc.t.ppf(0.975, df=n-1)

    print(bold_cyan(f"  H₀: μ = {mu0}  |  H₁: μ ≠ {mu0}  |  α = 0.05"))
    print(bold_cyan(f"  n = {n},  x̄ = {xbar:.4f},  s = {s:.4f}"))
    print(bold_cyan(f"  t-statistic = {t_stat:.4f}"))
    print(bold_cyan(f"  t-critical  = ±{t_crit:.4f}  (dof={n-1}, α=0.05)"))
    print(bold_cyan(f"  p-value     = {p_val:.4f}"))
    if p_val < 0.05:
        print(green(f"  Decision: REJECT H₀ (p={p_val:.4f} < 0.05)"))
    else:
        print(red(f"  Decision: FAIL TO REJECT H₀ (p={p_val:.4f} ≥ 0.05)"))
    print()
    _pause()

    section_header("4. ASCII — Null Distribution + Rejection Regions")
    xs = np.linspace(-4, 4, 60)
    t_pdf = sc.t.pdf(xs, df=n-1)
    t_max_pdf = max(t_pdf)
    print(cyan(f"  t-distribution (dof={n-1})   |  critical = ±{t_crit:.2f}"))
    print()
    for x, p in zip(xs, t_pdf):
        bar_len = int(p / t_max_pdf * 25)
        bar = "█" * bar_len
        if abs(x) > t_crit:
            color_bar = red(bar)
            region = red(" ← reject")
        else:
            color_bar = cyan(bar)
            region = ""
        t_marker = green("  ← t_obs") if abs(x - t_stat) < 0.15 else ""
        print(f"  x={x:5.1f}  {color_bar}{region}{t_marker}")
    print(grey("  Red = rejection region (|t| > t_crit)"))
    print()
    _pause()

    section_header("5. PLOTEXT VISUALIZATION")
    try:
        pdf_vals = sc.t.pdf(np.linspace(-4, 4, 80), df=n-1)
        loss_curve_plot(pdf_vals, title=f"t-distribution (dof={n-1}) — reject red tails")
    except Exception:
        print(grey("  plotext unavailable"))
    print()
    _pause()

    section_header("6. PYTHON CODE")
    code_block("t-Test and Chi-Squared Test with SciPy", """
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# One-sample t-test: is the mean equal to 0?
sample = rng.normal(loc=0.5, scale=1.0, size=30)
t_stat, p_val = stats.ttest_1samp(sample, popmean=0.0)
print(f"One-sample t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Two-sample t-test
group_a = rng.normal(5.0, 1.0, 40)
group_b = rng.normal(5.5, 1.0, 40)
t2, p2 = stats.ttest_ind(group_a, group_b)
print(f"Two-sample t-test: t={t2:.3f}, p={p2:.4f}")

# Chi-squared goodness of fit
observed = np.array([20, 30, 25, 25])
expected = np.array([25, 25, 25, 25])
chi2, p_chi = stats.chisquare(f_obs=observed, f_exp=expected)
print(f"Chi-squared test: χ²={chi2:.3f}, p={p_chi:.4f}")
""")
    _pause()

    section_header("7. KEY INSIGHTS")
    for ins in [
        "Failing to reject H₀ does NOT prove H₀ is true — absence of evidence ≠ evidence of absence",
        "Type I (α) and Type II (β) errors trade off — reducing one increases the other",
        "Power increases with: larger sample size, larger effect size, larger α",
        "t-test assumes normality but is robust for n≥30 (Central Limit Theorem)",
        "Chi-squared requires expected counts ≥ 5 per cell for valid approximation",
        "Always state hypotheses BEFORE seeing data to avoid circular reasoning",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — p-Values
# ══════════════════════════════════════════════════════════════════════════════
def topic_pvalues():
    clear()
    breadcrumb("mlmath", "Statistics", "p-Values")
    section_header("p-VALUES: MEANING AND MISUSE")
    print()

    section_header("1. THEORY")
    print(white("""
  The p-value is the probability of observing a test statistic as extreme as
  (or more extreme than) the one we computed, ASSUMING H₀ is true:

      p-value = P(T ≥ t_obs | H₀)

  WHAT p-values DON'T MEAN:
  • p = 0.03 does NOT mean "there is a 3% chance H₀ is true"
  • p = 0.03 does NOT mean "there is a 97% chance H₁ is true"
  • p = 0.03 does NOT tell you the size of the effect
  • p < 0.05 does NOT mean the result is "important" or "large"
  • p > 0.05 does NOT mean H₀ is true

  P-HACKING: collecting data until p < 0.05 inflates the false positive rate.
  If you test 20 independent true null hypotheses, you expect 1 spurious
  "significant" result at α = 0.05.

  MULTIPLE COMPARISONS PROBLEM: testing 1000 SNPs for disease association,
  50 will appear "significant" by chance alone at α = 0.05.
  Solution: corrections like Bonferroni (see Multiple Testing topic).

  REPLICATION CRISIS: over-reliance on p-values without pre-registration,
  adequate power, or effect size reporting has led to many irreproducible
  results in psychology, medicine, and ML model comparisons.
"""))
    _pause()

    section_header("2. CONCRETE EXAMPLE — p-Hacking Simulation")
    rng = np.random.default_rng(99)
    n_experiments = 1000
    n_per_exp = 20
    alpha = 0.05
    from scipy import stats as sc
    false_positives = 0
    p_values = []
    for _ in range(n_experiments):
        sample = rng.normal(0, 1, n_per_exp)   # true H₀
        _, p = sc.ttest_1samp(sample, 0)
        p_values.append(p)
        if p < alpha:
            false_positives += 1

    fpr = false_positives / n_experiments
    print(bold_cyan(f"  Simulated {n_experiments} experiments where H₀ is TRUE (μ=0)"))
    print(bold_cyan(f"  Using α = {alpha}"))
    print(bold_cyan(f"  False positives: {false_positives}/{n_experiments} = {fpr:.3f}"))
    print(bold_cyan(f"  Expected: ~{alpha*100:.0f}  (Type I error rate = α)"))
    print()
    print(green("  Even when there is NO real effect, 5% of tests appear 'significant'"))
    print(green("  If you select only significant results to report, you are p-hacking!"))
    print()
    p_values_arr = np.array(p_values)
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    counts = np.histogram(p_values_arr, bins=bins)[0]
    print(cyan("  p-value distribution (should be uniform if H₀ true):"))
    for lo, hi, cnt in zip(bins[:-1], bins[1:], counts):
        bar = "█" * (cnt // 5)
        print(f"    [{lo:.2f},{hi:.2f})  {cyan(bar)} {cnt}")
    print()
    _pause()

    section_header("3. PYTHON CODE")
    code_block("p-Value Simulation and Interpretation", """
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# Simulate many experiments under H₀ — how often do we get p<0.05?
n_sims, n_obs = 10000, 25
p_vals_h0 = [stats.ttest_1samp(rng.normal(0,1,n_obs), 0)[1] for _ in range(n_sims)]
fp_rate = np.mean(np.array(p_vals_h0) < 0.05)
print(f"False positive rate at alpha=0.05: {fp_rate:.4f}")   # ≈ 0.05

# Sequential testing (p-hacking): stop when p<0.05
batch_size = 5
sample = []
for step in range(1, 21):
    sample.extend(rng.normal(0,1,batch_size).tolist())  # true null
    if len(sample) >= 10:
        t, p = stats.ttest_1samp(sample, 0)
        if p < 0.05:
            print(f"  Stopped at n={len(sample)}, p={p:.4f}  ← p-hacking!")
            break
else:
    print("  Never reached significance — honest result")
""")
    _pause()

    section_header("4. KEY INSIGHTS")
    for ins in [
        "p-value = P(data this extreme | H₀ true) — NOT P(H₀ true | data)",
        "Under H₀, p-values are uniformly distributed on [0,1]",
        "A tiny p-value could be from a large sample detecting a trivial effect",
        "Always report effect sizes alongside p-values (Cohen's d, r², etc.)",
        "Pre-register your analysis to avoid p-hacking (specify n before collecting)",
        "Consider Bayes factors or estimation approaches as alternatives to NHST",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Confidence Intervals
# ══════════════════════════════════════════════════════════════════════════════
def topic_confidence_intervals():
    clear()
    breadcrumb("mlmath", "Statistics", "Confidence Intervals")
    section_header("CONFIDENCE INTERVALS")
    print()

    section_header("1. THEORY")
    print(white("""
  A 95% CONFIDENCE INTERVAL [L, U] constructed from sample data has the
  property: if we repeated the sampling procedure infinitely many times, 95%
  of the constructed intervals would contain the true parameter.

  IMPORTANT: once computed, the interval either contains θ or it doesn't —
  there is no 95% probability attached to a specific realised interval.
  This is the key distinction from a Bayesian credible interval.

  PARAMETRIC CI (normal approximation):
  For large n, x̄ ± z_{α/2} · σ/√n  where z_{0.025} = 1.96.

  STUDENT's t CI (small n, unknown σ):
  x̄ ± t_{α/2,n-1} · s/√n  where s = sample standard deviation.

  BOOTSTRAP CI (non-parametric):
  1. Resample with replacement B times (B=2000 typical)
  2. Compute statistic on each bootstrap sample
  3. Take the [2.5%, 97.5%] quantiles of the bootstrap distribution
  Bootstrap works for any statistic — median, correlation, AUC — no distributional
  assumptions needed.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Normal CI:     x̄ ± z_{α/2} · σ/√n,   z_{0.025} = 1.96"))
    print(formula("  t CI:          x̄ ± t_{α/2,n-1} · s/√n"))
    print(formula("  Margin:        ME = t_{α/2} · s/√n"))
    print(formula("  Required n:    n ≥ (z_{α/2} · σ / ME)²"))
    print(formula("  Bootstrap CI:  quantile([θ̂*₁,…,θ̂*_B], [α/2, 1-α/2])"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Parametric and Bootstrap CIs")
    rng = np.random.default_rng(3)
    from scipy import stats as sc
    true_mu = 10.0
    sample = rng.normal(true_mu, 3.0, size=25)
    n = len(sample); xbar = np.mean(sample); s = np.std(sample, ddof=1)
    t_crit = sc.t.ppf(0.975, df=n-1)
    me = t_crit * s / np.sqrt(n)
    ci_lo, ci_hi = xbar - me, xbar + me

    # Bootstrap CI for median
    B = 2000
    boot_medians = np.array([np.median(rng.choice(sample, size=n, replace=True)) for _ in range(B)])
    boot_lo, boot_hi = np.percentile(boot_medians, [2.5, 97.5])

    print(bold_cyan(f"  Sample:  n={n},  x̄={xbar:.4f},  s={s:.4f}"))
    print(bold_cyan(f"  t-critical (α=0.05, dof={n-1}): {t_crit:.4f}"))
    print(bold_cyan(f"  Margin of error: {me:.4f}"))
    print(bold_cyan(f"  95% CI (t):      [{ci_lo:.4f}, {ci_hi:.4f}]"))
    print(bold_cyan(f"  True μ={true_mu} is {'inside ✓' if ci_lo <= true_mu <= ci_hi else 'outside ✗'}"))
    print()
    print(bold_cyan(f"  Bootstrap 95% CI for median (B={B}): [{boot_lo:.4f}, {boot_hi:.4f}]"))
    print()

    section_header("4. ASCII — Coverage Simulation")
    n_sims = 30
    covered = 0
    print(cyan("  Simulating 30 CIs — does each contain μ=10?"))
    for i in range(n_sims):
        s_i = rng.normal(true_mu, 3.0, size=25)
        xb = np.mean(s_i); sd = np.std(s_i, ddof=1)
        lo = xb - sc.t.ppf(0.975, 24) * sd / np.sqrt(25)
        hi = xb + sc.t.ppf(0.975, 24) * sd / np.sqrt(25)
        inside = lo <= true_mu <= hi
        if inside: covered += 1
        sym = "█" if inside else "░"
        bar_color = green(sym * 15) if inside else red(sym * 15)
        print(f"  CI {i+1:2d}: [{lo:5.2f},{hi:5.2f}]  {bar_color}  {'✓' if inside else '✗'}")
    print()
    print(bold_cyan(f"  Coverage: {covered}/30 = {covered/n_sims*100:.1f}%  (expected ≈ 95%)"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Confidence Intervals — Parametric and Bootstrap", """
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
data = rng.normal(10, 3, size=30)

# --- Parametric t CI ---
n = len(data)
xbar, s = np.mean(data), np.std(data, ddof=1)
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
ci = (xbar - t_crit*s/np.sqrt(n), xbar + t_crit*s/np.sqrt(n))
print(f"95% t CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# --- Bootstrap CI for median ---
B = 5000
boot = np.array([np.median(rng.choice(data, n, replace=True)) for _ in range(B)])
boot_ci = np.percentile(boot, [2.5, 97.5])
print(f"Bootstrap 95% CI for median: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")

# --- Wilson CI for proportion ---
k, n_b = 17, 30   # 17 successes out of 30
ci_w = stats.proportion.proportion_confint(k, n_b, method='wilson')
print(f"Wilson 95% CI for p: {ci_w}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "CI width ∝ 1/√n — quadruple n to halve the width",
        "Bootstrap CIs need no distributional assumptions — great for complex statistics",
        "Confidence level (95%) refers to the PROCEDURE, not the specific interval",
        "Comparing two non-overlapping 95% CIs ≠ p < 0.05 (common misconception)",
        "Bayesian credible intervals give the intuitive interpretation most people want",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Effect Size
# ══════════════════════════════════════════════════════════════════════════════
def topic_effect_size():
    clear()
    breadcrumb("mlmath", "Statistics", "Effect Size")
    section_header("EFFECT SIZE: PRACTICAL VS STATISTICAL SIGNIFICANCE")
    print()

    section_header("1. THEORY")
    print(white("""
  STATISTICAL SIGNIFICANCE (p < 0.05) tells you whether an effect exists.
  PRACTICAL SIGNIFICANCE (effect size) tells you whether it MATTERS.

  With a large enough sample size, virtually ANY non-zero effect becomes
  statistically significant, even if it is scientifically trivial.
  Example: with n = 1,000,000 you can detect that Group A is 0.001 IQ points
  higher than Group B with p < 0.001 — but who cares?

  COHEN'S d: standardised mean difference
      d = (μ₁ - μ₂) / σ_pooled
  Benchmarks: d=0.2 (small), d=0.5 (medium), d=0.8 (large)

  OTHER EFFECT SIZES:
  • r (Pearson correlation): r=0.1, 0.3, 0.5 → small, medium, large
  • η² (eta-squared, ANOVA): proportion of variance explained
  • Odds ratio (binary outcomes)
  • AUC (classification)

  Always report an effect size alongside any p-value.  "The drug significantly
  reduced cholesterol (p<0.001, d=0.12)" — tiny effect.
  "The training significantly improved performance (p=0.03, d=0.91)" — large effect.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Cohen's d:       d = (μ₁-μ₂) / σ_pooled"))
    print(formula("  σ_pooled:        √[(s₁²(n₁-1) + s₂²(n₂-1)) / (n₁+n₂-2)]"))
    print(formula("  r from d:        r = d / √(d²+4)"))
    print(formula("  R² from r:       r²  (proportion of variance explained)"))
    print(formula("  Required n:      (z_α + z_β)² σ² · 2 / d²  per group"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Large n Makes Everything 'Significant'")
    rng = np.random.default_rng(11)
    from scipy import stats as sc

    true_d = 0.05   # tiny but real effect
    print(bold_cyan(f"  True effect size: d = {true_d} (tiny — below 'small' threshold of 0.2)"))
    print()

    for n in [30, 100, 500, 5000, 50000]:
        sample1 = rng.normal(0.0,    1.0, n)
        sample2 = rng.normal(true_d, 1.0, n)
        t, p    = sc.ttest_ind(sample1, sample2)
        sp      = np.sqrt(((n-1)*np.var(sample1, ddof=1) + (n-1)*np.var(sample2, ddof=1)) / (2*n-2))
        d_obs   = (np.mean(sample2) - np.mean(sample1)) / sp
        sig     = "✓ significant!" if p < 0.05 else "✗ not sig."
        print(f"    n={n:6d}:  p={p:.5f}  d={d_obs:+.4f}  {green(sig) if p<0.05 else grey(sig)}")
    print()
    print(green("  At n=50000 a trivially small effect is overwhelmingly significant."))
    print(green("  Always check effect size — statistical ≠ practical significance."))
    print()
    _pause()

    section_header("4. ASCII — Effect Size Reference")
    print()
    categories = [
        ("Negligible", 0.1,  "░"),
        ("Small",      0.2,  "▒"),
        ("Medium",     0.5,  "▓"),
        ("Large",      0.8,  "█"),
        ("Very Large", 1.2,  "█"),
    ]
    for label, d_val, sym in categories:
        bar = sym * int(d_val * 20)
        print(f"  {label:<12}  d={d_val:.1f}  {cyan(bar)}")
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Computing Cohen's d", """
import numpy as np
from scipy import stats

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pool = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / s_pool

rng = np.random.default_rng(0)
a = rng.normal(100, 15, 200)
b = rng.normal(107, 15, 200)   # 7 IQ points higher

t, p = stats.ttest_ind(a, b)
d = cohens_d(b, a)
print(f"t={t:.3f}, p={p:.4f}")
print(f"Cohen's d = {d:.3f}  (0.2=small, 0.5=medium, 0.8=large)")
print(f"The effect is {'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small'}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Always report effect size alongside p-values — they answer different questions",
        "Cohen's d = 0.2/0.5/0.8 = small/medium/large (conventional benchmarks)",
        "With n>10000, even d=0.01 can give p<0.001 — means nothing practically",
        "Effect size is dimensionless and comparable across studies (meta-analysis)",
        "Power analysis requires specifying the minimum effect size you care about",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Multiple Testing Corrections
# ══════════════════════════════════════════════════════════════════════════════
def topic_multiple_testing():
    clear()
    breadcrumb("mlmath", "Statistics", "Multiple Testing Corrections")
    section_header("MULTIPLE TESTING CORRECTIONS")
    print()

    section_header("1. THEORY")
    print(white("""
  When performing m independent tests simultaneously at level α, the probability
  of at least one false positive (Family-Wise Error Rate, FWER):

      FWER = 1 - (1-α)ᵐ

  For m=100 tests and α=0.05: FWER = 1 - 0.95¹⁰⁰ ≈ 99.4% — virtually guaranteed
  to get at least one false positive!  This is the multiple comparisons problem.

  BONFERRONI CORRECTION:
  Reject H_i if p_i < α/m.  This controls FWER ≤ α.
  Very conservative — many True discoveries will be missed (high Type II error).

  BENJAMINI-HOCHBERG (BH) PROCEDURE:
  Controls the FALSE DISCOVERY RATE (FDR = expected proportion of false discoveries
  among rejections).  Less conservative, more power than Bonferroni.
  Algorithm:
    1. Sort p-values: p_(1) ≤ p_(2) ≤ … ≤ p_(m)
    2. Find largest k such that p_(k) ≤ k·q/m  (q = desired FDR level, e.g. 0.05)
    3. Reject H_(1),…,H_(k)

  FWER vs FDR:
    FWER = P(any false discovery) — use for clinical trials where one false discovery is catastrophic
    FDR  = E[false discoveries / total discoveries] — use for genomics, feature selection
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  FWER (no correction): 1 - (1-α)ᵐ"))
    print(formula("  Bonferroni threshold:  α* = α/m"))
    print(formula("  Šidák correction:      α* = 1-(1-α)^(1/m)  (slightly less conservative)"))
    print(formula("  BH procedure:          reject p_(k) ≤ k·q/m,  find largest k"))
    print(formula("  FDR:   E[V/R]  where V=false discoveries, R=total rejections"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 100 Tests, 10 True Effects")
    rng = np.random.default_rng(17)
    from scipy import stats as sc
    m = 100
    n_true = 10     # 10 tests have real effects
    n_null = m - n_true
    n_obs  = 40

    # Generate p-values
    p_null = [sc.ttest_1samp(rng.normal(0, 1, n_obs), 0)[1] for _ in range(n_null)]
    p_alt  = [sc.ttest_1samp(rng.normal(0.8, 1, n_obs), 0)[1] for _ in range(n_true)]
    all_p  = np.array(p_null + p_alt)
    true_labels = np.array([0]*n_null + [1]*n_true)

    # No correction
    rej_none = all_p < 0.05
    fp_none  = np.sum(rej_none & (true_labels == 0))
    tp_none  = np.sum(rej_none & (true_labels == 1))

    # Bonferroni
    bonf_thresh = 0.05 / m
    rej_bonf = all_p < bonf_thresh
    fp_bonf  = np.sum(rej_bonf & (true_labels == 0))
    tp_bonf  = np.sum(rej_bonf & (true_labels == 1))

    # BH / FDR
    sorted_idx = np.argsort(all_p)
    sorted_p = all_p[sorted_idx]
    thresholds = np.arange(1, m+1) * 0.05 / m
    bh_mask = sorted_p <= thresholds
    if bh_mask.any():
        largest_k = np.where(bh_mask)[0].max()
        bh_reject_sorted = np.zeros(m, dtype=bool)
        bh_reject_sorted[:largest_k+1] = True
    else:
        bh_reject_sorted = np.zeros(m, dtype=bool)
    rej_bh = np.zeros(m, dtype=bool)
    rej_bh[sorted_idx] = bh_reject_sorted
    fp_bh = np.sum(rej_bh & (true_labels == 0))
    tp_bh = np.sum(rej_bh & (true_labels == 1))

    print(bold_cyan(f"  {m} tests: {n_null} truly null, {n_true} truly non-null (d=0.8)"))
    print()
    table(
        ["Method",     "Threshold",       "Rejected", "True Pos", "False Pos", "FWER/FDR"],
        [
            ["No correction", "0.05",           str(rej_none.sum()), str(tp_none), str(fp_none), f"FWER={1-(0.95**m):.2%}"],
            ["Bonferroni",    f"{bonf_thresh:.4f}", str(rej_bonf.sum()), str(tp_bonf), str(fp_bonf), "FWER≤0.05"],
            ["Benjamini-Hochberg", "adaptive",  str(rej_bh.sum()),  str(tp_bh),  str(fp_bh),  "FDR≤0.05"],
        ]
    )
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Bonferroni and Benjamini-Hochberg in Python", """
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng(0)
m = 50

# Generate p-values (20 real effects, 30 null)
p_null = [stats.ttest_1samp(rng.normal(0, 1, 30), 0)[1] for _ in range(30)]
p_alt  = [stats.ttest_1samp(rng.normal(1.0, 1, 30), 0)[1] for _ in range(20)]
pvals  = np.array(p_null + p_alt)

# Bonferroni: controls FWER
rej_b, p_corr_b, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
print(f"Bonferroni rejections: {rej_b.sum()}")

# Benjamini-Hochberg: controls FDR
rej_bh, p_corr_bh, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
print(f"BH (FDR 0.05) rejections: {rej_bh.sum()}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "With m=100 tests at α=0.05, expect ~5 false positives even if all H₀ true",
        "Bonferroni: very safe (controls FWER) but low power — misses many real effects",
        "Benjamini-Hochberg: controls FDR — better power, ideal for exploratory analyses",
        "In genomics (m~1M), use stringent Bonferroni: α* = 5×10⁻⁸",
        "Pre-registration and replication are stronger defences than correction alone",
        "Machine learning: avoid evaluating many models on the same test set — use a holdout",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Maximum Likelihood Estimation", topic_mle),
        ("Maximum A Posteriori (MAP)", topic_map),
        ("Bayesian Inference", topic_bayesian_inference),
        ("Hypothesis Testing", topic_hypothesis_testing),
        ("p-Values", topic_pvalues),
        ("Confidence Intervals", topic_confidence_intervals),
        ("Effect Size", topic_effect_size),
        ("Multiple Testing Corrections", topic_multiple_testing),
    ]
    block_menu("b06", "Statistics", topics)
