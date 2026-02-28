"""core/calculus.py — Numerical and symbolic calculus (no display code)."""

import numpy as np
from typing import Callable, List, Tuple, Optional


# ── Numerical differentiation ──────────────────────────────────────────────────
def numerical_gradient(f: Callable, x: np.ndarray,
                       eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar-valued f at point x."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x.flat)):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        grad.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def numerical_jacobian(f: Callable, x: np.ndarray,
                       eps: float = 1e-5) -> np.ndarray:
    """Jacobian of vector-valued f: R^n → R^m at x."""
    fx = np.atleast_1d(f(x))
    m  = len(fx)
    n  = len(x.flat)
    J  = np.zeros((m, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        J[:, i] = (np.atleast_1d(f(xp)) - np.atleast_1d(f(xm))) / (2 * eps)
    return J


def numerical_hessian(f: Callable, x: np.ndarray,
                      eps: float = 1e-4) -> np.ndarray:
    """Numerical Hessian via finite differences."""
    n = len(x.flat)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp.flat[i] += eps; xpp.flat[j] += eps
            xpm = x.copy(); xpm.flat[i] += eps; xpm.flat[j] -= eps
            xmp = x.copy(); xmp.flat[i] -= eps; xmp.flat[j] += eps
            xmm = x.copy(); xmm.flat[i] -= eps; xmm.flat[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return H


def gradient_check(f: Callable, grad_f: Callable,
                   x: np.ndarray, eps: float = 1e-5) -> Tuple[float, bool]:
    """
    Compare analytical gradient to numerical gradient.
    Returns (relative_error, passed).
    """
    ag = np.asarray(grad_f(x)).flatten()
    ng = numerical_gradient(f, x, eps).flatten()
    num   = np.linalg.norm(ag - ng)
    denom = np.linalg.norm(ag) + np.linalg.norm(ng) + 1e-12
    rel   = num / denom
    return float(rel), rel < 1e-4


# ── Partial derivatives via sympy ──────────────────────────────────────────────
def symbolic_gradient(expr_str: str, var_names: List[str]):
    """
    Compute symbolic gradient using sympy.
    Returns list of (var, derivative expression) pairs.
    """
    try:
        import sympy as sp
        syms  = sp.symbols(" ".join(var_names))
        if not isinstance(syms, (list, tuple)):
            syms = (syms,)
        expr  = sp.sympify(expr_str)
        grads = [(str(s), sp.diff(expr, s)) for s in syms]
        return grads, expr
    except ImportError:
        return None, None


def symbolic_hessian(expr_str: str, var_names: List[str]):
    """Return symbolic Hessian matrix."""
    try:
        import sympy as sp
        syms = sp.symbols(" ".join(var_names))
        if not isinstance(syms, (list, tuple)):
            syms = (syms,)
        expr = sp.sympify(expr_str)
        H    = sp.hessian(expr, syms)
        return H
    except ImportError:
        return None


# ── Taylor series ─────────────────────────────────────────────────────────────
def taylor_approx(f: Callable, x0: float, x: np.ndarray,
                  order: int = 2) -> np.ndarray:
    """
    Taylor approximation of f around x0 up to given order.
    For 1D functions only.
    """
    eps = 1e-5
    f0  = f(x0)
    dx  = x - x0
    result = np.full_like(x, f0, dtype=float)
    if order >= 1:
        df  = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
        result += df * dx
    if order >= 2:
        d2f = (f(x0 + eps) - 2 * f0 + f(x0 - eps)) / eps ** 2
        result += 0.5 * d2f * dx ** 2
    if order >= 3:
        d3f = (f(x0+2*eps) - 2*f(x0+eps) + 2*f(x0-eps) - f(x0-2*eps)) / (2*eps**3)
        result += (1/6) * d3f * dx ** 3
    return result


# ── Chain rule ────────────────────────────────────────────────────────────────
def chain_rule_example() -> dict:
    """
    Demonstrate chain rule for h(x) = sin(x²) at x=2.
    h = f(g(x)) where g(x)=x², f(u)=sin(u)
    dh/dx = f'(g(x))·g'(x) = cos(x²)·2x
    """
    x   = 2.0
    g   = x ** 2         # g(x) = x²
    dg  = 2 * x          # g'(x) = 2x
    fg  = np.sin(g)      # f(u) = sin(u)
    dfg = np.cos(g)      # f'(u) = cos(u)
    dh  = dfg * dg       # chain rule
    numeric_dh = numerical_gradient(lambda v: np.sin(v[0]**2),
                                    np.array([x]))[0]
    return {"x": x, "g(x)": g, "g'(x)": dg,
            "f(g(x))": float(fg), "f'(g(x))": float(dfg),
            "dh/dx (analytic)": float(dh),
            "dh/dx (numeric)": float(numeric_dh),
            "error": abs(dh - numeric_dh)}


# ── Lagrange multipliers ───────────────────────────────────────────────────────
def lagrange_example() -> dict:
    """
    Maximise f(x,y) = xy subject to g(x,y) = x+y-1 = 0.
    Lagrangian L = xy - λ(x+y-1)
    ∂L/∂x = y - λ = 0  →  y = λ
    ∂L/∂y = x - λ = 0  →  x = λ
    g = 0  →  x+y = 1  →  x = y = 1/2
    max f = 1/4
    """
    x_opt = 0.5
    y_opt = 0.5
    lam   = 0.5
    f_opt = x_opt * y_opt
    return {"x*": x_opt, "y*": y_opt, "λ*": lam,
            "f(x*,y*)": f_opt, "constraint": x_opt + y_opt}


# ── Forward-mode automatic differentiation (dual numbers) ─────────────────────
class Dual:
    """Simple dual number for forward-mode AD."""
    __slots__ = ("val", "dot")

    def __init__(self, val: float, dot: float = 0.0):
        self.val = val
        self.dot = dot

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.dot + other.dot)
        return Dual(self.val + other, self.dot)
    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val,
                        self.dot * other.val + self.val * other.dot)
        return Dual(self.val * other, self.dot * other)
    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.dot - other.dot)
        return Dual(self.val - other, self.dot)

    def __rsub__(self, other):
        return Dual(other - self.val, -self.dot)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val / other.val,
                        (self.dot*other.val - self.val*other.dot)/other.val**2)
        return Dual(self.val / other, self.dot / other)

    def __pow__(self, n):
        return Dual(self.val ** n, n * self.val ** (n-1) * self.dot)

    def sin(self):
        return Dual(np.sin(self.val), np.cos(self.val) * self.dot)
    def cos(self):
        return Dual(np.cos(self.val), -np.sin(self.val) * self.dot)
    def exp(self):
        ev = np.exp(self.val)
        return Dual(ev, ev * self.dot)
    def log(self):
        return Dual(np.log(self.val), self.dot / self.val)

    def __repr__(self):
        return f"Dual(val={self.val:.6g}, dot={self.dot:.6g})"


def forward_diff(f, x: float) -> Tuple[float, float]:
    """Compute f(x) and f'(x) using forward-mode AD with dual numbers."""
    d = Dual(x, 1.0)
    result = f(d)
    return result.val, result.dot
