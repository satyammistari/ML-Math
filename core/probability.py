"""core/probability.py — Probability distributions and information-theoretic quantities."""

import numpy as np
from typing import Tuple, List, Optional


# ── Gaussian ──────────────────────────────────────────────────────────────────
def gaussian_pdf(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_log_pdf(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma) ** 2

def gaussian_cdf(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    from scipy.special import erf
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


# ── Bernoulli / Binomial ───────────────────────────────────────────────────────
def bernoulli_pmf(k: np.ndarray, p: float) -> np.ndarray:
    return np.where(k == 1, p, 1 - p)

def binomial_pmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    from scipy.special import comb
    return comb(n, k, exact=False) * p**k * (1-p)**(n-k)


# ── Poisson ───────────────────────────────────────────────────────────────────
def poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    from scipy.special import factorial
    return lam**k * np.exp(-lam) / factorial(k)


# ── Beta ──────────────────────────────────────────────────────────────────────
def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.special import beta as beta_fn
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return x**(a-1) * (1-x)**(b-1) / beta_fn(a, b)


# ── Dirichlet ─────────────────────────────────────────────────────────────────
def dirichlet_sample(alpha: np.ndarray, n: int = 1,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    return rng.dirichlet(alpha, size=n)


# ── Exponential ───────────────────────────────────────────────────────────────
def exponential_pdf(x: np.ndarray, lam: float) -> np.ndarray:
    return np.where(x >= 0, lam * np.exp(-lam * x), 0.0)


# ── Gamma ─────────────────────────────────────────────────────────────────────
def gamma_pdf(x: np.ndarray, shape: float, rate: float = 1) -> np.ndarray:
    from scipy.stats import gamma as gamma_dist
    return gamma_dist.pdf(x, a=shape, scale=1/rate)


# ── Softmax ───────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - np.max(z, axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ── Bayes' theorem ────────────────────────────────────────────────────────────
def bayes_update(prior: float, likelihood: float,
                 marginal: float) -> float:
    """P(H|E) = P(E|H)·P(H) / P(E)"""
    return likelihood * prior / marginal

def bayes_medical_test(sensitivity: float, specificity: float,
                       prevalence: float) -> dict:
    """Full Bayesian calculation for a diagnostic test."""
    p_pos_given_sick   = sensitivity
    p_pos_given_healthy = 1 - specificity
    p_pos = p_pos_given_sick*prevalence + p_pos_given_healthy*(1-prevalence)
    p_pos_given_negative = ((1-sensitivity)*prevalence /
                             ((1-sensitivity)*prevalence + specificity*(1-prevalence)))
    posterior = bayes_update(prevalence, p_pos_given_sick, p_pos)
    return {
        "prior": prevalence,
        "P(+|sick)": sensitivity,
        "P(+|healthy)": p_pos_given_healthy,
        "P(+)": p_pos,
        "posterior P(sick|+)": posterior,
        "posterior P(sick|-)": p_pos_given_negative,
    }


# ── Entropy & information ─────────────────────────────────────────────────────
def entropy(probs: np.ndarray, base: float = 2) -> float:
    """Shannon entropy in bits (base=2) or nats (base=e)."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if base == np.e:
        return float(-np.sum(p * np.log(p)))
    return float(-np.sum(p * np.log2(p)))

def kl_divergence(p: np.ndarray, q: np.ndarray,
                  base: float = np.e) -> float:
    """KL(P‖Q) — note: asymmetric."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = (p > 0) & (q > 0)
    ratio = np.where(mask, p / q, 1)
    log_fn = np.log if base == np.e else np.log2
    return float(np.sum(p[mask] * log_fn(ratio[mask])))

def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(P,Q) = -Σ p·log(q)"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    q = np.clip(q, 1e-15, None)
    return float(-np.sum(p * np.log(q)))

def mutual_information(joint: np.ndarray) -> float:
    """MI(X;Y) from joint distribution table."""
    joint = joint / joint.sum()
    px    = joint.sum(axis=1, keepdims=True)
    py    = joint.sum(axis=0, keepdims=True)
    indep = px * py
    mask  = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / indep[mask])))

def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JSD = 0.5·KL(P‖M) + 0.5·KL(Q‖M) where M = (P+Q)/2"""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ── Expectation / Variance ────────────────────────────────────────────────────
def expectation(values: np.ndarray, probs: np.ndarray) -> float:
    return float(np.dot(values, probs))

def variance(values: np.ndarray, probs: np.ndarray) -> float:
    mu = expectation(values, probs)
    return float(np.dot(probs, (values - mu)**2))


# ── CLT demonstration ────────────────────────────────────────────────────────
def clt_demo(population: np.ndarray, n: int = 30,
             n_samples: int = 1000, rng_seed: int = 42) -> np.ndarray:
    """Return sample means; should approximate Gaussian."""
    rng = np.random.default_rng(rng_seed)
    means = np.array([rng.choice(population, n).mean()
                      for _ in range(n_samples)])
    return means


# ── Conjugate prior helpers ───────────────────────────────────────────────────
CONJUGATE_TABLE = [
    ("Bernoulli likelihood", "Beta prior", "Beta posterior",
     "Beta(α+successes, β+failures)"),
    ("Categorical likelihood", "Dirichlet prior", "Dirichlet posterior",
     "Dirichlet(α + counts)"),
    ("Poisson likelihood", "Gamma prior", "Gamma posterior",
     "Gamma(α+Σxᵢ, β+n)"),
    ("Gaussian (known σ)", "Gaussian prior", "Gaussian posterior",
     "μ_post = (μ₀/σ₀²+Σx/σ²)/(1/σ₀²+n/σ²)"),
    ("Gaussian (known μ)", "Inv-Gamma prior", "Inv-Gamma posterior",
     "Updated shape & scale"),
]
