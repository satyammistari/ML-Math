"""core/optimization.py — Gradient Descent, Adam, Newton, schedulers (no display)."""

import numpy as np
from typing import Callable, Tuple, List, Dict, Optional


# ── History dataclass ─────────────────────────────────────────────────────────
class OptHistory:
    def __init__(self):
        self.losses:   List[float]      = []
        self.params:   List[np.ndarray] = []
        self.grads:    List[np.ndarray] = []

    def record(self, loss, param, grad=None):
        self.losses.append(float(loss))
        self.params.append(np.array(param))
        if grad is not None:
            self.grads.append(np.array(grad))


# ── Gradient Descent ──────────────────────────────────────────────────────────
def gradient_descent(f: Callable, grad_f: Callable,
                     x0: np.ndarray,
                     lr: float = 0.01,
                     n_steps: int = 200,
                     tol: float = 1e-7) -> Tuple[np.ndarray, OptHistory]:
    x = x0.copy().astype(float)
    h = OptHistory()
    for _ in range(n_steps):
        g  = grad_f(x)
        h.record(f(x), x, g)
        x  = x - lr * g
        if np.linalg.norm(g) < tol:
            break
    h.record(f(x), x)
    return x, h


# ── SGD with mini-batches ──────────────────────────────────────────────────────
def sgd(f_batch: Callable, grad_batch: Callable,
        x0: np.ndarray, data: np.ndarray,
        lr: float = 0.01, batch_size: int = 32,
        n_epochs: int = 10, rng_seed: int = 42) -> Tuple[np.ndarray, OptHistory]:
    rng  = np.random.default_rng(rng_seed)
    x    = x0.copy().astype(float)
    h    = OptHistory()
    n    = len(data)
    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = data[idx[start:start + batch_size]]
            g     = grad_batch(x, batch)
            x     = x - lr * g
        h.record(f_batch(x, data), x)
    return x, h


# ── Momentum ──────────────────────────────────────────────────────────────────
def momentum_gd(f: Callable, grad_f: Callable,
                x0: np.ndarray, lr: float = 0.01,
                beta: float = 0.9, n_steps: int = 200,
                tol: float = 1e-7) -> Tuple[np.ndarray, OptHistory]:
    x  = x0.copy().astype(float)
    v  = np.zeros_like(x)
    h  = OptHistory()
    for _ in range(n_steps):
        g  = grad_f(x)
        v  = beta * v + (1 - beta) * g
        x  = x - lr * v
        h.record(f(x), x, g)
        if np.linalg.norm(g) < tol:
            break
    return x, h


# ── Adam ──────────────────────────────────────────────────────────────────────
def adam(f: Callable, grad_f: Callable,
         x0: np.ndarray,
         lr: float = 0.001,
         beta1: float = 0.9,
         beta2: float = 0.999,
         eps: float = 1e-8,
         n_steps: int = 500,
         tol: float = 1e-7) -> Tuple[np.ndarray, OptHistory]:
    """
    Adam: Adaptive Moment Estimation.
    m̂ₜ = m/(1-β₁ᵗ)   v̂ₜ = v/(1-β₂ᵗ)
    θₜ = θₜ₋₁ - α·m̂ₜ/(√v̂ₜ + ε)
    """
    x  = x0.copy().astype(float)
    m  = np.zeros_like(x)   # 1st moment
    v  = np.zeros_like(x)   # 2nd moment
    h  = OptHistory()
    for t in range(1, n_steps + 1):
        g  = grad_f(x)
        m  = beta1 * m + (1 - beta1) * g
        v  = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x  = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        h.record(f(x), x, g)
        if np.linalg.norm(g) < tol:
            break
    return x, h


# ── Newton's method ───────────────────────────────────────────────────────────
def newtons_method(f: Callable, grad_f: Callable, hess_f: Callable,
                   x0: np.ndarray, n_steps: int = 50,
                   tol: float = 1e-9,
                   reg: float = 1e-6) -> Tuple[np.ndarray, OptHistory]:
    """
    xₜ₊₁ = xₜ - H⁻¹·∇f  (quadratic convergence near minimum)
    """
    x = x0.copy().astype(float)
    h = OptHistory()
    for _ in range(n_steps):
        g = grad_f(x)
        H = hess_f(x)
        H = H + reg * np.eye(len(x))   # regularise
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = g
        x = x - delta
        h.record(f(x), x, g)
        if np.linalg.norm(g) < tol:
            break
    return x, h


# ── Conjugate gradient ────────────────────────────────────────────────────────
def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-10,
                       max_iter: int = 1000) -> Tuple[np.ndarray, int]:
    """
    Solve Ax = b where A is SPD.
    Returns (solution, iterations).
    """
    x   = x0 if x0 is not None else np.zeros_like(b, dtype=float)
    r   = b - A @ x
    p   = r.copy()
    rs  = np.dot(r, r)
    for i in range(max_iter):
        Ap      = A @ p
        alpha   = rs / np.dot(p, Ap)
        x       = x + alpha * p
        r       = r - alpha * Ap
        rs_new  = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x, i + 1
        p  = r + (rs_new / rs) * p
        rs = rs_new
    return x, max_iter


# ── Learning rate schedulers ───────────────────────────────────────────────────
def lr_step_decay(lr0: float, step: int, decay: float = 0.1,
                  every: int = 30) -> float:
    return lr0 * (decay ** (step // every))

def lr_cosine(lr0: float, step: int, T: int,
              lr_min: float = 0.0) -> float:
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * step / T))

def lr_linear_warmup(lr_max: float, step: int, warmup: int,
                     T: int) -> float:
    if step < warmup:
        return lr_max * step / warmup
    return lr_max * max(0, (T - step) / (T - warmup))

def lr_exponential(lr0: float, step: int, gamma: float = 0.99) -> float:
    return lr0 * gamma ** step

def schedule_trajectory(scheduler: Callable, T: int, **kw) -> np.ndarray:
    return np.array([scheduler(step=t, **kw) for t in range(T)])


# ── Convexity check (numerical) ───────────────────────────────────────────────
def is_convex_check(f: Callable, x: np.ndarray,
                    y: np.ndarray, n_pts: int = 20) -> bool:
    """Check f(λx+(1-λ)y) ≤ λf(x)+(1-λ)f(y) for λ in (0,1)."""
    for lam in np.linspace(0.01, 0.99, n_pts):
        midpt = lam * x + (1 - lam) * y
        if f(midpt) > lam * f(x) + (1 - lam) * f(y) + 1e-8:
            return False
    return True
