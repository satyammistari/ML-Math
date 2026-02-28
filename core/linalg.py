"""core/linalg.py — Pure linear-algebra computations (no display code)."""

import numpy as np
from typing import Tuple, List


# ── Vector operations ──────────────────────────────────────────────────────────
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def vector_norm(v: np.ndarray, p: float = 2) -> float:
    if p == 0:
        return float(np.sum(v != 0))
    if np.isinf(p):
        return float(np.max(np.abs(v)))
    return float(np.linalg.norm(v, ord=p))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project vector a onto vector b."""
    return (np.dot(a, b) / np.dot(b, b)) * b

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians."""
    c = cosine_similarity(a, b)
    return float(np.arccos(np.clip(c, -1, 1)))

def linear_combination(vectors: List[np.ndarray],
                       coeffs: List[float]) -> np.ndarray:
    return sum(c * v for c, v in zip(coeffs, vectors))


# ── Matrix basics ──────────────────────────────────────────────────────────────
def matmul_steps(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Return AB and a list of step strings showing each dot product."""
    C = A @ B
    steps = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            terms = " + ".join(f"{A[i,k]:.3g}×{B[k,j]:.3g}"
                               for k in range(A.shape[1]))
            steps.append(f"C[{i},{j}] = {terms} = {C[i,j]:.5g}")
    return C, steps

def matrix_rank(A: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(A))

def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(A, A.T, atol=tol))

def is_spd(A: np.ndarray) -> bool:
    """Check if A is symmetric positive definite."""
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


# ── Eigendecomposition ─────────────────────────────────────────────────────────
def eigen_decomp(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (eigenvalues, eigenvectors) sorted by descending |eigenvalue|."""
    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(-np.abs(vals))
    return vals[idx], vecs[:, idx]

def verify_eigen(A: np.ndarray, val: float,
                 vec: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.allclose(A @ vec, val * vec, atol=tol))

def characteristic_polynomial_coeffs(A: np.ndarray) -> np.ndarray:
    """Coefficients of det(A - λI) = 0 for small matrices."""
    n = A.shape[0]
    # Use numpy's eigvals to get roots, then reconstruct poly
    vals = np.linalg.eigvals(A)
    return np.poly(vals).real


# ── SVD ────────────────────────────────────────────────────────────────────────
def svd_full(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    return U, s, Vt

def svd_truncated(A: np.ndarray, k: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]

def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    U, s, Vt = svd_truncated(A, k)
    return (U * s) @ Vt

def explained_variance_ratio(singular_values: np.ndarray) -> np.ndarray:
    s2 = singular_values ** 2
    return s2 / s2.sum()

def nuclear_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="nuc"))


# ── PCA ────────────────────────────────────────────────────────────────────────
def pca(X: np.ndarray, n_components: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD on centred data.
    Returns: (Z, components, eigenvalues, explained_var_ratio)
    """
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    Z          = Xc @ components.T
    ev         = (s ** 2) / (len(X) - 1)
    evr        = ev / ev.sum()
    return Z, components, ev[:n_components], evr[:n_components]


# ── Gram-Schmidt ───────────────────────────────────────────────────────────────
def gram_schmidt(V: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Orthonormalise columns of V using Gram-Schmidt.
    Returns (Q, steps) where Q is orthonormal and steps describes each step.
    """
    n, m = V.shape
    Q = np.zeros_like(V, dtype=float)
    steps = []
    for k in range(m):
        u = V[:, k].astype(float)
        for j in range(k):
            proj = np.dot(Q[:, j], u) * Q[:, j]
            u    = u - proj
            steps.append(f"Step {k+1}: subtract proj onto q{j+1}: {proj.round(4)}")
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            steps.append(f"Step {k+1}: vector {k+1} is linearly dependent — skipped")
            continue
        Q[:, k] = u / norm
        steps.append(f"Step {k+1}: normalise → q{k+1} = {Q[:,k].round(4)}")
    return Q, steps


# ── Norms ─────────────────────────────────────────────────────────────────────
def all_norms(v: np.ndarray) -> dict:
    return {
        "L0": vector_norm(v, 0),
        "L1": vector_norm(v, 1),
        "L2": vector_norm(v, 2),
        "Linf": vector_norm(v, np.inf),
        "Frobenius (if matrix)": float(np.linalg.norm(v.reshape(-1), 2)),
    }


# ── Orthogonality checks ───────────────────────────────────────────────────────
def is_orthonormal(Q: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol))


# ── Condition number ───────────────────────────────────────────────────────────
def condition_number(A: np.ndarray, ord=None) -> float:
    return float(np.linalg.cond(A, p=ord))


# ── Determinant & inverse ──────────────────────────────────────────────────────
def safe_inverse(A: np.ndarray) -> Tuple[np.ndarray, bool]:
    try:
        return np.linalg.inv(A), True
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A), False
