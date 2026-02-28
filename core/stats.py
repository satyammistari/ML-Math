"""core/stats.py — MLE, MAP, hypothesis testing, bootstrap, model selection."""

import numpy as np
from typing import Tuple, List, Optional


# ── MLE helpers ───────────────────────────────────────────────────────────────
def mle_gaussian(data: np.ndarray) -> Tuple[float, float]:
    """MLE for Gaussian: μ̂ = mean, σ̂² = variance (biased)."""
    mu  = float(data.mean())
    var = float(((data - mu)**2).mean())
    return mu, np.sqrt(var)

def mle_bernoulli(data: np.ndarray) -> float:
    """MLE for Bernoulli: p̂ = mean."""
    return float(data.mean())

def mle_poisson(data: np.ndarray) -> float:
    """MLE for Poisson: λ̂ = mean."""
    return float(data.mean())

def mle_exponential(data: np.ndarray) -> float:
    """MLE for Exponential: λ̂ = 1/mean."""
    return 1.0 / float(data.mean())

def log_likelihood_gaussian(data: np.ndarray, mu: float, sigma: float) -> float:
    n = len(data)
    return (-n/2 * np.log(2*np.pi*sigma**2)
            - np.sum((data-mu)**2) / (2*sigma**2))


# ── MAP estimation ────────────────────────────────────────────────────────────
def map_gaussian_known_sigma(data: np.ndarray, sigma: float,
                              mu0: float, sigma0: float) -> float:
    """
    MAP for Gaussian likelihood with Gaussian prior.
    μ_MAP = (μ₀/σ₀² + Σxᵢ/σ²) / (1/σ₀² + n/σ²)
    """
    n     = len(data)
    prec0 = 1 / sigma0**2
    prec  = n / sigma**2
    return (prec0*mu0 + prec*data.mean()) / (prec0 + prec)


# ── Hypothesis testing ────────────────────────────────────────────────────────
def one_sample_t_test(data: np.ndarray, mu0: float = 0.0
                       ) -> Tuple[float, float]:
    """Two-tailed one-sample t-test. Returns (t_stat, p_value)."""
    from scipy import stats
    t, p = stats.ttest_1samp(data, mu0)
    return float(t), float(p)

def two_sample_t_test(a: np.ndarray, b: np.ndarray,
                       equal_var: bool = True) -> Tuple[float, float]:
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    return float(t), float(p)

def chi_squared_test(observed: np.ndarray,
                      expected: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import chisquare
    chi2, p = chisquare(observed, f_exp=expected)
    return float(chi2), float(p)

def anova_f_test(*groups) -> Tuple[float, float]:
    from scipy.stats import f_oneway
    F, p = f_oneway(*groups)
    return float(F), float(p)


# ── Confidence intervals ──────────────────────────────────────────────────────
def ci_mean_parametric(data: np.ndarray, alpha: float = 0.05
                        ) -> Tuple[float, float]:
    """Parametric CI for the mean using t-distribution."""
    from scipy import stats
    n    = len(data)
    mean = data.mean()
    se   = stats.sem(data)
    t    = stats.t.ppf(1 - alpha/2, df=n-1)
    return float(mean - t*se), float(mean + t*se)

def ci_bootstrap(data: np.ndarray, stat_fn=None,
                  B: int = 2000, alpha: float = 0.05,
                  rng_seed: int = 42) -> Tuple[float, float]:
    """Bootstrap percentile CI for any statistic."""
    rng  = np.random.default_rng(rng_seed)
    fn   = stat_fn or np.mean
    boot = np.array([fn(rng.choice(data, len(data), replace=True))
                     for _ in range(B)])
    lo   = float(np.percentile(boot, 100 * alpha/2))
    hi   = float(np.percentile(boot, 100 * (1 - alpha/2)))
    return lo, hi


# ── Effect size ────────────────────────────────────────────────────────────���───
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_std"""
    pooled = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 +
                      (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# ── Multiple testing ───────────────────────────────────────────────────────────
def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    corrected = p_values * len(p_values)
    return np.minimum(corrected, 1.0)

def benjamini_hochberg(p_values: np.ndarray,
                        alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """BH FDR procedure. Returns (rejected, adjusted_p)."""
    n    = len(p_values)
    idx  = np.argsort(p_values)
    adj  = np.empty(n)
    adj[idx] = p_values[idx] * n / (np.arange(1, n+1))
    adj  = np.minimum.accumulate(adj[::-1])[::-1]
    adj  = np.minimum(adj, 1.0)
    return adj <= alpha, adj


# ── Model selection ───────────────────────────────────────────────────────────
def aic(log_lik: float, n_params: int) -> float:
    """AIC = 2k - 2·log L"""
    return 2*n_params - 2*log_lik

def bic(log_lik: float, n_params: int, n_obs: int) -> float:
    """BIC = k·log(n) - 2·log L"""
    return n_params * np.log(n_obs) - 2*log_lik

def dickey_fuller(ts: np.ndarray) -> Tuple[float, float]:
    """Augmented Dickey-Fuller test for stationarity."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(ts)
        return float(result[0]), float(result[1])
    except ImportError:
        return float("nan"), float("nan")


# ── Regression metrics ─────────────────────────────────────────────────────────
def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res / ss_tot)

def adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray,
                        n_features: int) -> float:
    n  = len(y_true)
    r2 = r_squared(y_true, y_pred)
    return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred)**2))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
