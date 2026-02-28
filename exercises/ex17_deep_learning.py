"""Exercise Set 17: Deep Learning"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title; self.difficulty = difficulty
        self.description = description; self.hint = hint
        self.starter_code = starter_code; self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="Implement Batch Normalization Forward Pass",
        difficulty="Beginner",
        description="""
  Implement Batch Normalization forward pass from scratch.

  Given a batch x of shape (N, D):
    1. Compute μ = mean over batch (axis 0)
    2. Compute σ² = variance (axis 0)
    3. Normalize: x_hat = (x - μ) / √(σ² + ε)
    4. Scale & shift: y = γ * x_hat + β

  Verify: output has mean ≈ 0, std ≈ 1 (before γ,β scaling)
  Test with N=100, D=4, γ=ones, β=zeros.
""",
        hint="""
  mu = x.mean(axis=0)          # shape: (D,)
  var = x.var(axis=0)          # shape: (D,)
  x_hat = (x - mu) / sqrt(var + eps)  # shape: (N,D)
  y = gamma * x_hat + beta
  Check: x_hat.mean(axis=0) ≈ 0, x_hat.std(axis=0) ≈ 1
""",
        starter_code="""
import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    # TODO
    pass

np.random.seed(42)
x = np.random.randn(100, 4) * 5 + 3
gamma = np.ones(4); beta = np.zeros(4)
y, x_hat = batch_norm(x, gamma, beta)
print(f"x: mean={x.mean():.3f}, std={x.std():.3f}")
print(f"x_hat: mean={x_hat.mean(axis=0).round(3)}, std={x_hat.std(axis=0).round(3)}")
print(f"y: mean={y.mean():.3f}, std={y.std():.3f}")
""",
        solution_code="""
import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_hat + beta
    return y, x_hat

np.random.seed(42)
x = np.random.randn(100, 4) * 5 + 3
gamma = np.ones(4); beta = np.zeros(4)
y, x_hat = batch_norm(x, gamma, beta)
print(f"Before BN: mean={x.mean(axis=0).round(2)}, std={x.std(axis=0).round(2)}")
print(f"After BN:  mean={x_hat.mean(axis=0).round(4)}, std={x_hat.std(axis=0).round(4)}")

# Non-trivial gamma/beta
gamma2 = np.array([2.0, 0.5, 3.0, 1.0])
beta2  = np.array([1.0, -1.0, 0.0, 5.0])
y2, _ = batch_norm(x, gamma2, beta2)
print(f"With γ={gamma2}, β={beta2}:")
print(f"Output mean ≈ β: {y2.mean(axis=0).round(4)}")
print(f"Output std  ≈ γ: {y2.std(axis=0).round(4)}")
"""
    ),

    Exercise(
        title="Scaled Dot-Product Self-Attention",
        difficulty="Intermediate",
        description="""
  Implement scaled dot-product self-attention from scratch.

  Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V

  Given: X ∈ ℝ^(n×d),  Wq, Wk, Wv ∈ ℝ^(d×d_k)
  Compute:
    Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
    scores = Q @ K.T / sqrt(d_k)   [n×n]
    weights = softmax(scores, dim=-1)
    output = weights @ V            [n×d_v]

  Also implement causal (masked) attention for autoregressive decoding.
  Mask: upper triangle = -inf before softmax.
""",
        hint="""
  def softmax(x, axis=-1):
      e = np.exp(x - x.max(axis=axis, keepdims=True))
      return e / e.sum(axis=axis, keepdims=True)
  
  Causal mask: mask = np.triu(np.full((n,n), -inf), k=1)
  scores_masked = scores + mask  → upper triangle = -inf → softmax = 0
""",
        starter_code="""
import numpy as np

def self_attention(X, Wq, Wk, Wv, causal=False):
    # TODO: scaled dot-product attention
    pass

np.random.seed(0)
n, d, dk = 5, 8, 4
X = np.random.randn(n, d)
Wq = np.random.randn(d, dk)*0.1
Wk = np.random.randn(d, dk)*0.1
Wv = np.random.randn(d, dk)*0.1

out, weights = self_attention(X, Wq, Wk, Wv)
print(f"Output shape: {out.shape}")
print(f"Weights (rows sum to 1): {weights.sum(axis=1).round(4)}")

out_c, weights_c = self_attention(X, Wq, Wk, Wv, causal=True)
print(f"Causal weights upper triangle sum: {np.triu(weights_c, 1).sum():.6f}  (should be 0)")
""",
        solution_code="""
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, Wq, Wk, Wv, causal=False):
    n, d = X.shape
    dk = Wk.shape[1]
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    scores = Q @ K.T / np.sqrt(dk)  # (n, n)
    if causal:
        mask = np.triu(np.full((n,n), -np.inf), k=1)
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights

np.random.seed(0)
n, d, dk = 5, 8, 4
X = np.random.randn(n, d)
Wq = np.random.randn(d,dk)*0.1; Wk = np.random.randn(d,dk)*0.1; Wv = np.random.randn(d,dk)*0.1

out, w = self_attention(X, Wq, Wk, Wv)
print(f"Output shape: {out.shape}")
print(f"Attention weights:\\n{w.round(3)}")
print(f"Rows sum to 1: {np.allclose(w.sum(axis=1), 1)}")

out_c, wc = self_attention(X, Wq, Wk, Wv, causal=True)
print(f"\\nCausal attention weights:")
print(wc.round(3))
print(f"Upper triangle zeros (no future peeking): {np.allclose(np.triu(wc,1), 0)}")
"""
    ),

    Exercise(
        title="Mini Transformer Block",
        difficulty="Advanced",
        description="""
  Implement a complete Transformer encoder block:
    1. Multi-head self-attention (h=2 heads, d_k=d/h)
    2. Add + LayerNorm
    3. FFN: Linear(d,4d) → ReLU → Linear(4d,d)
    4. Add + LayerNorm

  Input: X ∈ ℝ^(n×d) with n=5, d=8

  Verify that:
    - Output shape = Input shape (n×d)
    - LayerNorm output has mean≈0, std≈1 per token
  
  This is the core computation of BERT, GPT, and all modern LLMs.
""",
        hint="""
  Multi-head: split d into h heads of dk=d//h each
    head_i processes X @ Wqi, X @ Wki, X @ Wvi
    concat outputs, project through Wo
  LayerNorm: normalize across features (axis=-1)
  FFN: Linear(d→4d) with ReLU, then Linear(4d→d)
  Residual: output = LayerNorm(sublayer(x) + x)
""",
        starter_code="""
import numpy as np

def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mu) / (std + eps)

def transformer_block(X, h=2):
    n, d = X.shape; dk = d // h
    # TODO: 1. Multi-head attention
    # TODO: 2. Add+LayerNorm
    # TODO: 3. FFN
    # TODO: 4. Add+LayerNorm
    pass

np.random.seed(42)
X = np.random.randn(5, 8)
out = transformer_block(X)
print(f"Output shape: {out.shape}  (should be {X.shape})")
print(f"Output LN mean: {out.mean(axis=-1).round(4)}  (≈0)")
print(f"Output LN std:  {out.std(axis=-1).round(4)}  (≈1)")
""",
        solution_code="""
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mu) / (std + eps)

def multi_head_attn(X, h, d):
    n = len(X); dk = d // h; heads = []
    for i in range(h):
        np.random.seed(i)
        Wq = np.random.randn(d, dk)*0.1
        Wk = np.random.randn(d, dk)*0.1
        Wv = np.random.randn(d, dk)*0.1
        Q = X@Wq; K = X@Wk; V = X@Wv
        A = softmax(Q@K.T/np.sqrt(dk), axis=-1)
        heads.append(A@V)
    concat = np.concatenate(heads, axis=-1)  # (n, d)
    Wo = np.random.randn(d, d)*0.1
    return concat @ Wo

def transformer_block(X, h=2):
    n, d = X.shape
    # Self-attention + residual + LN
    attn_out = multi_head_attn(X, h, d)
    x1 = layer_norm(X + attn_out)
    # FFN + residual + LN
    np.random.seed(99)
    W1 = np.random.randn(d, 4*d)*0.1; b1 = np.zeros(4*d)
    W2 = np.random.randn(4*d, d)*0.1; b2 = np.zeros(d)
    ffn = np.maximum(0, x1@W1+b1) @ W2 + b2  # ReLU FFN
    x2 = layer_norm(x1 + ffn)
    return x2

np.random.seed(42)
X = np.random.randn(5, 8)
out = transformer_block(X)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {out.shape}")
print(f"Per-token mean: {out.mean(axis=-1).round(4)}")
print(f"Per-token std:  {out.std(axis=-1).round(4)}")
print(f"Shape preserved: {out.shape == X.shape} ✓")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║   EXERCISES — Block 17: Deep Learning            ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m\n")
        for i, ex in enumerate(exercises, 1):
            dc = "\033[92m" if ex.difficulty=="Beginner" else "\033[93m" if ex.difficulty=="Intermediate" else "\033[91m"
            print(f"  {i}. {dc}[{ex.difficulty}]\033[0m {ex.title}")
        print("\n  \033[90m[0] Back\033[0m")
        choice = input("\n\033[96mSelect: \033[0m").strip()
        if choice == "0": break
        try:
            ex = exercises[int(choice)-1]; _run_exercise(ex)
        except (ValueError, IndexError): pass


def _run_exercise(ex):
    dc = "\033[92m" if ex.difficulty=="Beginner" else "\033[93m" if ex.difficulty=="Intermediate" else "\033[91m"
    print(f"\n\033[95m━━━ {ex.title} ━━━\033[0m")
    print(f"  {dc}{ex.difficulty}\033[0m\n{ex.description}")
    while True:
        cmd = input("\n  [h]int [c]ode [r]un [s]olution [b]ack: ").strip().lower()
        if cmd=='b': break
        elif cmd=='h': print(f"\033[93mHINT\033[0m\n{ex.hint}")
        elif cmd=='c': print(f"\033[94mSTARTER\033[0m\n{ex.starter_code}")
        elif cmd=='s': print(f"\033[92mSOLUTION\033[0m\n{ex.solution_code}")
        elif cmd=='r':
            try: exec(compile(ex.solution_code,"<sol>","exec"),{})
            except Exception as e: print(f"\033[91m{e}\033[0m")
