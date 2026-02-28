"""
blocks/b08_backprop.py
Block 8: Backpropagation and Automatic Differentiation
Topics: Computation Graphs, Forward Pass, Backward Pass, Vanishing/Exploding
        Gradients, Layer-wise Derivations, Autodiff, Gradient Checking.
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
# TOPIC 1 — Computation Graphs
# ══════════════════════════════════════════════════════════════════════════════
def topic_computation_graphs():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Computation Graphs")
    section_header("COMPUTATION GRAPHS")
    print()

    section_header("1. THEORY")
    print(white("""
  A COMPUTATION GRAPH is a directed acyclic graph (DAG) where:
    • NODES represent operations (add, multiply, exp, relu, …) or variables
    • EDGES represent tensors (data flowing between operations)
    • Direction FORWARD: inputs → outputs (compute values)
    • Direction BACKWARD: outputs → inputs (compute gradients)

  Every computation in a neural network can be expressed as a computation graph.
  This representation makes automatic differentiation mechanical and efficient.

  FORWARD PASS: traverse the graph left-to-right, computing each node's value
  from its inputs. Cache intermediate values for use in the backward pass.

  BACKWARD PASS: traverse right-to-left, using the chain rule at each node:
      ∂L/∂x = ∂L/∂z · ∂z/∂x   (upstream gradient × local gradient)

  COMPUTATIONAL BENEFITS:
    • Each node only needs to know its LOCAL gradient (∂output/∂input)
    • Global gradients are assembled via chain rule multiplication
    • Memory: O(n) nodes, O(n) memory for cached values
    • Time: O(n) forward + O(n) backward ≈ 2-3× forward cost

  PyTorch builds dynamic graphs at runtime (define-by-run).
  TensorFlow 1.x used static graphs defined before execution.
  Modern frameworks blend both approaches.
"""))
    _pause()

    section_header("2. EXAMPLE: z = (x + y) * w")
    x_val, y_val, w_val = 3.0, 2.0, 4.0
    a_val = x_val + y_val        # a = x + y
    z_val = a_val * w_val        # z = a * w
    # Upstream gradient = dL/dz = 1 (we're computing dz/deverything)
    dLdz  = 1.0
    dLda  = dLdz * w_val         # ∂L/∂a = ∂L/∂z · ∂z/∂a = 1 · w = 4
    dLdw  = dLdz * a_val         # ∂L/∂w = ∂L/∂z · ∂z/∂w = 1 · a = 5
    dLdx  = dLda * 1.0           # ∂L/∂x = ∂L/∂a · ∂a/∂x = 4 · 1 = 4
    dLdy  = dLda * 1.0           # ∂L/∂y = ∂L/∂a · ∂a/∂y = 4 · 1 = 4

    print(bold_cyan(f"  Input values:  x={x_val}, y={y_val}, w={w_val}"))
    print()
    print(cyan("  COMPUTATION GRAPH:"))
    print()
    print(cyan("    x ──┐"))
    print(cyan("        ├──[+]── a ──[*]── z"))
    print(cyan("    y ──┘            │"))
    print(cyan("                     w"))
    print()
    print(green(f"  Forward pass:"))
    print(green(f"    a = x + y = {x_val} + {y_val} = {a_val}"))
    print(green(f"    z = a * w = {a_val} * {w_val} = {z_val}"))
    print()
    print(yellow(f"  Backward pass (∂L/∂z = {dLdz}):"))
    print(yellow(f"    ∂L/∂w = ∂L/∂z · ∂z/∂w = {dLdz} · a = {dLdz} · {a_val} = {dLdw}"))
    print(yellow(f"    ∂L/∂a = ∂L/∂z · ∂z/∂a = {dLdz} · w = {dLdz} · {w_val} = {dLda}"))
    print(yellow(f"    ∂L/∂x = ∂L/∂a · ∂a/∂x = {dLda} · 1 = {dLdx}"))
    print(yellow(f"    ∂L/∂y = ∂L/∂a · ∂a/∂y = {dLda} · 1 = {dLdy}"))
    print()
    _pause()

    section_header("3. ASCII COMPUTATION GRAPH WITH GRADIENT FLOW")
    print(cyan("                          ┌──────────────────────────────────┐"))
    print(cyan("  FORWARD VALUES          │  BACKWARD GRADIENTS              │"))
    print(cyan("  ─────────────           │  ─────────────────               │"))
    print(cyan("                          └──────────────────────────────────┘"))
    print()
    print(f"  x={x_val}  ────────────────────────────────────────► {yellow(f'∂L/∂x = {dLdx:.1f}')}")
    print(f"          ↘")
    print(f"           [+] → a={a_val} → [*] → z={z_val}    {red('← ∂L/∂z = 1.0')}")
    print(f"          ↗            ↑")
    print(f"  y={y_val}  ──────────────────  {yellow(f'∂L/∂y = {dLdy:.1f}')}")
    print(f"                       │")
    print(f"  w={w_val}  ──────────────────► {yellow(f'∂L/∂w = {dLdw:.1f}')}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Manual Computation Graph with Backprop", """
import numpy as np

# Forward pass
x, y, w = 3.0, 2.0, 4.0
a        = x + y          # a = 5.0
z        = a * w          # z = 20.0

# Backward pass (chain rule at each node)
dL_dz = 1.0               # upstream gradient
dL_da = dL_dz * w         # ∂(a*w)/∂a = w
dL_dw = dL_dz * a         # ∂(a*w)/∂w = a
dL_dx = dL_da * 1.0       # ∂(x+y)/∂x = 1
dL_dy = dL_da * 1.0       # ∂(x+y)/∂y = 1

print(f"z = {z}")
print(f"∂L/∂x = {dL_dx}, ∂L/∂y = {dL_dy}, ∂L/∂w = {dL_dw}")

# Verify with PyTorch (if available)
try:
    import torch
    x_t = torch.tensor(3.0, requires_grad=True)
    y_t = torch.tensor(2.0, requires_grad=True)
    w_t = torch.tensor(4.0, requires_grad=True)
    z_t = (x_t + y_t) * w_t
    z_t.backward()
    print(f"PyTorch: ∂L/∂x={x_t.grad}, ∂L/∂y={y_t.grad}, ∂L/∂w={w_t.grad}")
except ImportError:
    print("PyTorch not available (manual result verified above)")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Each node stores its local gradient; global gradient = product along path",
        "Chain rule is the ONLY calculus rule backprop uses",
        "Dynamic graphs (PyTorch): rebuilt each forward pass — flexible for control flow",
        "Static graphs (TF 1.x): defined once, then executed — easier to optimise",
        "Memory bottleneck: must cache all intermediate values for backward pass",
        "Gradient checkpointing: recompute activations to save memory (speed/memory tradeoff)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# EXTRA TOPIC — Backprop Intuition & Chain Rule
# ══════════════════════════════════════════════════════════════════════════════
def topic_backprop_intuition():
        clear()
        breadcrumb("mlmath", "Backpropagation", "Backprop Intuition")
        section_header("BACKPROP INTUITION — CHAIN RULE IN ACTION")
        print()

        section_header("1. CORE IDEA")
        print(white("""
    Backpropagation answers: "How does changing each weight affect the loss?"
    Mathematically it is just the chain rule applied backwards through a
    computation graph.

        Forward:  x → [Layer1] → h₁ → [Layer2] → h₂ → [Loss] → L
        Backward: ∂L/∂w₁ = ∂L/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂w₁

    Frameworks (PyTorch, JAX, TF) build this graph automatically and run the
    backward pass for you, but conceptually it's just repeated chain rule.
"""))
        _pause()

        section_header("2. TOY EXAMPLE — MANUAL VS AUTOGRAD")
        print(white("""
    Consider a tiny 2-layer scalar network:

            x → h = x·w₁ → ŷ = h·w₂ → L = (ŷ − 10)²

    We can work out gradients by hand using the chain rule:
        dL/dŷ = 2(ŷ − 10)
        dL/dw₂ = dL/dŷ · ∂ŷ/∂w₂ = 2(ŷ − 10) · h
        dL/dw₁ = dL/dŷ · ∂ŷ/∂h · ∂h/∂w₁ = 2(ŷ − 10) · w₂ · x
"""))

        code_block("Manual vs PyTorch backprop", """
import torch

x  = torch.tensor([2.0])
w1 = torch.tensor([3.0], requires_grad=True)
w2 = torch.tensor([4.0], requires_grad=True)

# Forward
h      = x * w1           # = 6
y_pred = h * w2           # = 24
loss   = (y_pred - 10)**2 # = 196

# Backward
loss.backward()
print(w1.grad, w2.grad)   # dL/dw1 = 224, dL/dw2 = 168
""")
        _pause()

        section_header("3. VANISHING GRADIENTS")
        print(white("""
    With sigmoid activations, each layer multiplies the gradient by σ'(z).
    Since 0 < σ'(z) ≤ 0.25, a deep stack of sigmoids shrinks gradients:

            after 10 layers:  (0.25)¹⁰ ≈ 10⁻⁶

    Early layers receive almost zero signal and barely learn. Modern
    deep nets mitigate this with ReLU activations, residual connections,
    normalisation, and careful weight initialisation.
"""))
        print()
        topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Forward Pass
# ══════════════════════════════════════════════════════════════════════════════
def topic_forward_pass():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Forward Pass")
    section_header("FORWARD PASS — STEP BY STEP")
    print()

    section_header("1. THEORY")
    print(white("""
  The forward pass computes the network's prediction from input to output,
  caching all intermediate values needed for the backward pass.

  For a 2-layer network: input x → Linear → ReLU → Linear → output → Loss

  At each layer:
    1. Compute pre-activation: z = Wx + b
    2. Apply activation:       a = activation(z)
    3. Cache z and a for backward pass

  MSE LOSS: L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
  For a single example: L = (y - ŷ)²
"""))
    _pause()

    section_header("2. FULL WORKED EXAMPLE — 2-Layer Network")
    # Architecture: 2→2→1, ReLU, MSE loss
    x  = np.array([1.0, 2.0])
    W1 = np.array([[0.1, 0.2],
                   [0.3, 0.4]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[0.5, 0.6]])
    b2 = np.array([0.0])
    y  = np.array([1.0])    # target

    # Layer 1
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)   # ReLU

    # Layer 2
    z2 = W2 @ a1 + b2
    yhat = z2  # no activation on output (regression)

    # Loss
    loss = float(np.mean((y - yhat) ** 2))

    print(bold_cyan(f"  Input x = {x}"))
    print(bold_cyan(f"  W1 = {W1.tolist()}"))
    print(bold_cyan(f"  b1 = {b1},  W2 = {W2.tolist()},  b2 = {b2}"))
    print(bold_cyan(f"  Target y = {y}"))
    print()
    print(cyan(f"  ── Layer 1 (Linear + ReLU) ──────────────────────────"))
    print(f"  z1 = W1 @ x + b1")
    print(f"     = [{W1[0,0]}×{x[0]} + {W1[0,1]}×{x[1]}, {W1[1,0]}×{x[0]} + {W1[1,1]}×{x[1]}]")
    print(f"     = {z1}")
    print(f"  a1 = ReLU(z1) = max(0, {z1}) = {a1}")
    print()
    print(cyan(f"  ── Layer 2 (Linear) ────────────────────────────────"))
    print(f"  z2 = W2 @ a1 + b2")
    print(f"     = [{W2[0,0]}×{a1[0]:.2f} + {W2[0,1]}×{a1[1]:.2f}] + {b2}")
    print(f"     = {z2}")
    print(f"  ŷ  = {yhat}")
    print()
    print(cyan(f"  ── MSE Loss ────────────────────────────────────────"))
    print(f"  L = (y - ŷ)² = ({y[0]} - {yhat[0]:.4f})² = {loss:.6f}")
    print()
    print(green(f"  Cached for backprop: z1={z1}, a1={a1}, z2={z2}"))
    print()
    _pause()

    section_header("3. PYTHON CODE")
    code_block("2-Layer Network Forward Pass", """
import numpy as np

# Network parameters
x  = np.array([1.0, 2.0])
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.0, 0.0])
W2 = np.array([[0.5, 0.6]])
b2 = np.array([0.0])
y  = np.array([1.0])

def relu(z):   return np.maximum(0, z)
def mse(y, yhat): return np.mean((y - yhat)**2)

# Forward pass
z1   = W1 @ x + b1
a1   = relu(z1)
z2   = W2 @ a1 + b2
yhat = z2
loss = mse(y, yhat)

print(f"z1   = {z1}")
print(f"a1   = {a1}")
print(f"z2   = {z2}")
print(f"ŷ    = {yhat}")
print(f"Loss = {loss:.6f}")

# Cache dict (needed for backprop)
cache = {'z1': z1, 'a1': a1, 'z2': z2}
""")
    _pause()

    section_header("4. KEY INSIGHTS")
    for ins in [
        "Cache ALL intermediate values (z, a at each layer) — needed for backward pass",
        "Matrix multiply W@x is the fundamental operation in every linear layer",
        "For regression: output = z (no activation); for classification: softmax(z)",
        "Loss function choice determines ultimate gradient shape",
        "Batch processing: stack [x₁,…,x_n] into rows of X, compute X @ Wᵀ",
        "Numerical precision: float32 is standard; float16 for large models",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Backward Pass
# ══════════════════════════════════════════════════════════════════════════════
def topic_backward_pass():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Backward Pass")
    section_header("BACKWARD PASS — CHAIN RULE AT EVERY NODE")
    print()

    section_header("1. THEORY")
    print(white("""
  The backward pass efficiently computes ∂L/∂θ for all parameters θ using the
  chain rule. It traverses the computation graph in reverse (output → input).

  KEY FORMULAS per layer type:
  • Linear:    z = Wx + b
      ∂L/∂W = δ · xᵀ   (outer product of upstream gradient and layer input)
      ∂L/∂b = δ         (upstream gradient directly)
      ∂L/∂x = Wᵀ · δ   (propagate gradient to previous layer)
    where δ = ∂L/∂z is the upstream gradient.

  • ReLU:      a = max(0, z)
      ∂L/∂z = δ_next ⊙ 1(z > 0)   (mask: pass gradient only where z was positive)

  • MSE Loss:  L = (y - ŷ)²
      ∂L/∂ŷ = 2(ŷ - y)  / n     (n = batch size)
"""))
    _pause()

    section_header("2. FULL BACKWARD PASS — Same Network as Forward Pass")
    x  = np.array([1.0, 2.0])
    W1 = np.array([[0.1, 0.2],
                   [0.3, 0.4]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[0.5, 0.6]])
    b2 = np.array([0.0])
    y  = np.array([1.0])

    # Forward
    z1   = W1 @ x + b1
    a1   = np.maximum(0, z1)
    z2   = W2 @ a1 + b2
    yhat = z2
    loss = float(np.mean((y - yhat)**2))

    # Backward
    # Step 1: Loss gradient w.r.t. output
    dL_dyhat = 2 * (yhat - y)           # shape (1,)

    # Step 2: Layer 2 gradients
    dL_dW2 = np.outer(dL_dyhat, a1)     # shape (1,2)
    dL_db2 = dL_dyhat                   # shape (1,)
    dL_da1 = W2.T @ dL_dyhat           # shape (2,), gradient into a1

    # Step 3: ReLU backward
    dL_dz1 = dL_da1 * (z1 > 0).astype(float)  # mask

    # Step 4: Layer 1 gradients
    dL_dW1 = np.outer(dL_dz1, x)       # shape (2,2)
    dL_db1 = dL_dz1                     # shape (2,)

    print(bold_cyan(f"  Forward:  z1={z1}, a1={a1}, ŷ={yhat[0]:.4f}, L={loss:.6f}"))
    print()
    print(cyan("  ── Backward through MSE Loss ──────────────────────────────"))
    print(f"  ∂L/∂ŷ = 2(ŷ-y) = 2×({yhat[0]:.4f}-{y[0]}) = {dL_dyhat[0]:.6f}")
    print()
    print(cyan("  ── Backward through Layer 2 (Linear) ─────────────────────"))
    print(f"  δ₂ = ∂L/∂z2 = {dL_dyhat}")
    print(f"  ∂L/∂W2 = δ₂ · a1ᵀ = {dL_dW2}")
    print(f"  ∂L/∂b2 = δ₂ = {dL_db2}")
    print(f"  ∂L/∂a1 = W2ᵀ · δ₂ = {dL_da1}")
    print()
    print(cyan("  ── Backward through ReLU ───────────────────────────────────"))
    print(f"  z1 = {z1}  →  z1>0: {(z1>0).tolist()}")
    print(f"  ∂L/∂z1 = ∂L/∂a1 ⊙ 1(z1>0) = {dL_da1} ⊙ {(z1>0).astype(int)} = {dL_dz1}")
    print()
    print(cyan("  ── Backward through Layer 1 (Linear) ─────────────────────"))
    print(f"  δ₁ = ∂L/∂z1 = {dL_dz1}")
    print(f"  ∂L/∂W1 = δ₁ · xᵀ = {dL_dW1}")
    print(f"  ∂L/∂b1 = δ₁ = {dL_db1}")
    print()
    _pause()

    section_header("3. ASCII GRADIENT FLOW")
    print(cyan("  BACKWARD GRADIENT FLOW (right to left):"))
    print()
    print(f"  x={x}  ◄────────────────────────────────── ∂L/∂x = {W1.T @ dL_dz1}")
    print(f"              ↖ W1ᵀδ₁")
    print(f"           [W1,b1] ← ∂L/∂W1={np.round(dL_dW1,4)}, ∂L/∂b1={np.round(dL_db1,4)}")
    print(f"              ↓z1={z1}")
    print(f"           [ReLU] ← ∂L/∂z1={np.round(dL_dz1,4)}")
    print(f"              ↓a1={a1}")
    print(f"           [W2,b2] ← ∂L/∂W2={np.round(dL_dW2,4)}, ∂L/∂b2={np.round(dL_db2,4)}")
    print(f"              ↓z2={np.round(z2,4)}")
    print(f"           [MSE] ← ∂L/∂ŷ={np.round(dL_dyhat,4)}")
    print(f"              Loss = {loss:.6f}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Full Forward + Backward Pass with Gradient Updates", """
import numpy as np

x  = np.array([1.0, 2.0])
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.zeros(2); W2 = np.array([[0.5, 0.6]]); b2 = np.zeros(1)
y  = np.array([1.0]); lr = 0.01

for step in range(5):
    # Forward
    z1 = W1 @ x + b1;  a1 = np.maximum(0, z1)
    z2 = W2 @ a1 + b2; yhat = z2
    loss = float(np.mean((y - yhat)**2))

    # Backward
    dL_dyhat = 2*(yhat - y)
    dL_dW2   = np.outer(dL_dyhat, a1)
    dL_db2   = dL_dyhat
    dL_da1   = W2.T @ dL_dyhat
    dL_dz1   = dL_da1 * (z1 > 0)
    dL_dW1   = np.outer(dL_dz1, x)
    dL_db1   = dL_dz1

    # Gradient descent update
    W2 -= lr * dL_dW2;  b2 -= lr * dL_db2
    W1 -= lr * dL_dW1;  b1 -= lr * dL_db1

    print(f"Step {step+1}: loss={loss:.6f}, yhat={yhat[0]:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "∂L/∂W = δ · xᵀ — outer product, not matrix multiply",
        "∂L/∂x = Wᵀ · δ — this propagates gradient to the previous layer",
        "ReLU backward: pass gradient through if z>0, block (zero) if z≤0",
        "Batch: dL/dW = (1/n) X_batchᵀ @ delta — average over batch",
        "Gradient flows backward; weights are updated with gradient descent",
        "The whole algorithm is just repeated chain rule — nothing mysterious!",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Vanishing Gradients
# ══════════════════════════════════════════════════════════════════════════════
def topic_vanishing_gradients():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Vanishing Gradients")
    section_header("VANISHING GRADIENTS")
    print()

    section_header("1. THEORY")
    print(white("""
  The vanishing gradient problem occurs when gradients become exponentially small
  as they propagate back through many layers.  This prevents early layers from
  learning meaningful features.

  MATHEMATICAL PROOF (sigmoid):
  sigmoid σ(x) = 1/(1+e⁻ˣ)
  σ'(x) = σ(x)(1-σ(x)),   max at x=0: σ'(0) = 0.25

  For a deep network with L sigmoid layers, the gradient contribution at layer l:
      ∂L/∂W_l ∝ ∏ᵢ₌ₗ₊₁ᴸ σ'(zᵢ)

  Since σ' ≤ 0.25, after L layers the gradient shrinks by at most (0.25)ᴸ.
  With L=10 layers: (0.25)¹⁰ ≈ 10⁻⁶  — effectively zero!

  WHY ReLU FIXES THIS:
  ReLU'(x) = 1 if x>0, else 0.  When x>0, the gradient passes through unchanged.
  No squashing — gradients don't shrink (though they can die for x≤0).

  SOLUTIONS TO VANISHING GRADIENTS:
  1. ReLU activations (and variants: Leaky ReLU, GELU)
  2. Batch Normalisation: re-centres and re-scales activations each layer
  3. Residual connections (ResNets): shortcut paths bypass layers
  4. Weight initialisation: He (ReLU), Xavier/Glorot (sigmoid/tanh)
  5. LSTM/GRU gates in RNNs: learned gates control gradient flow
"""))
    _pause()

    section_header("2. GRADIENT MAGNITUDE vs LAYER DEPTH")
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)
    def relu_grad(x): return (x > 0).astype(float)

    n_layers = 20
    sig_grads  = np.ones(n_layers)
    relu_grads = np.ones(n_layers)
    rng = np.random.default_rng(0)
    for l in range(1, n_layers):
        z_s = rng.normal(0, 1, 10)
        z_r = rng.normal(0, 1, 10)
        sig_grads[l]  = sig_grads[l-1] * np.mean(sigmoid_grad(z_s))
        relu_grads[l] = relu_grads[l-1] * np.mean(relu_grad(z_r))

    print(bold_cyan(f"  Gradient magnitude propagating backward through {n_layers} layers:"))
    print()
    print(f"  {'Layer':<8} {'Sigmoid grad':<18} {'ReLU grad':<18} {'Sigmoid bar'}")
    print(grey("  " + "─" * 70))
    for l in range(0, n_layers, 2):
        sig_bar = "█" * max(1, int(np.log10(max(sig_grads[l], 1e-12) + 1) * 8))
        relu_bar = "█" * max(1, min(30, int(relu_grads[l] * 10)))
        print(f"  {l:<8} {sig_grads[l]:<18.2e} {relu_grads[l]:<18.4f} {red(sig_bar)} {green(relu_bar)}")
    print()
    print(red(f"  Sigmoid: gradient at layer 0 = {sig_grads[n_layers-1]:.2e}  (vanished!)"))
    print(green(f"  ReLU:    gradient at layer 0 = {relu_grads[n_layers-1]:.4f}  (survives)"))
    print()
    _pause()

    section_header("3. PLOTEXT — Gradient Magnitude vs Depth")
    try:
        loss_curve_plot(sig_grads,  title="Sigmoid gradient magnitude vs layer depth")
    except Exception:
        print(grey("  plotext unavailable — see ASCII above"))
    try:
        loss_curve_plot(relu_grads, title="ReLU gradient magnitude vs layer depth")
    except Exception:
        pass
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Vanishing Gradient Demonstration", """
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

# Simulate gradient flow through L sigmoid layers
L = 20
grad = 1.0
print(f"Layer 0: grad = {grad:.6f}")

for l in range(1, L+1):
    # z drawn from unit normal (typical initialisation)
    z = np.random.randn()
    grad *= sigmoid_prime(z)
    if l % 4 == 0:
        print(f"Layer {l:2d}: grad = {grad:.2e}  {'(vanished!)' if grad < 1e-5 else ''}")

# ReLU: gradient stays ~1 (positive region)
grad_relu = 1.0
for l in range(1, L+1):
    z = np.random.randn()
    grad_relu *= 1.0 if z > 0 else 0.0

print(f"\\nReLU (active path) gradient after {L} layers: {grad_relu:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Sigmoid max gradient = 0.25 → shrinks by ≥4× per layer",
        "After 10 sigmoid layers: gradient ≈ (0.25)^10 ≈ 10^{-6}",
        "ReLU gradient = 1 for positive inputs — no squashing in active region",
        "Residual connections provide gradient highways: ∂L/∂x reaches '1 + ...'",
        "BatchNorm re-centres activations, preventing saturation",
        "LSTM forget gate ≈ 1 during long-range dependencies — preserves gradients",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Exploding Gradients
# ══════════════════════════════════════════════════════════════════════════════
def topic_exploding_gradients():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Exploding Gradients")
    section_header("EXPLODING GRADIENTS AND GRADIENT CLIPPING")
    print()

    section_header("1. THEORY")
    print(white("""
  Exploding gradients are the opposite problem: gradients grow exponentially
  as they propagate back, eventually causing NaN losses and divergence.

  WHEN DOES IT HAPPEN?
  • RNNs with many time steps — gradient is product of Jacobians over all steps
  • Deep networks with weights initialised too large
  • When spectral radius of weight matrix > 1

  MATHEMATICAL CONDITION:
  If weight matrix W has spectral radius ρ(W) = max eigenvalue of W:
    ρ(W) > 1 → gradients can explode
    ρ(W) < 1 → gradients can vanish
    ρ(W) ≈ 1 → stable (initialisation goal!)

  GRADIENT CLIPPING (by global norm):
  Given gradient vector g, clip if ‖g‖ > threshold τ:
      g_clipped = g × τ / ‖g‖  if ‖g‖ > τ
      g_clipped = g             otherwise

  This preserves gradient DIRECTION while limiting its MAGNITUDE.

  ELEMENT-WISE CLIPPING: clip each element independently.
  Simpler but changes gradient direction.  Norm-clipping is preferred.

  In practice, gradient clipping is nearly universal in RNN/LSTM training,
  and common in transformer training (max_grad_norm = 1.0 in many papers).
"""))
    _pause()

    section_header("2. GRADIENT CLIPPING WORKED EXAMPLE")
    rng = np.random.default_rng(42)
    grad = rng.normal(0, 5.0, size=10)   # large gradient vector
    threshold = 1.0

    grad_norm = np.linalg.norm(grad)
    if grad_norm > threshold:
        grad_clipped = grad * threshold / grad_norm
    else:
        grad_clipped = grad.copy()
    clipped_norm = np.linalg.norm(grad_clipped)

    print(bold_cyan(f"  Original gradient norm: ‖g‖ = {grad_norm:.4f}"))
    print(bold_cyan(f"  Threshold τ = {threshold}"))
    print(bold_cyan(f"  Clipped gradient norm:  ‖g_clip‖ = {clipped_norm:.4f}"))
    print()
    print(f"  Original: {np.round(grad[:5], 3)} ...")
    print(f"  Clipped:  {np.round(grad_clipped[:5], 3)} ...")
    print()
    print(green("  Direction preserved: cos(g, g_clip) = " +
                f"{np.dot(grad, grad_clipped)/(grad_norm*clipped_norm):.6f} (≈ 1.0)"))
    print()

    section_header("3. ASCII — Effect of Gradient Clipping")
    print(cyan("  Gradient components before and after clipping:"))
    for i, (orig, clipped) in enumerate(zip(grad, grad_clipped)):
        bar_orig = "█" * int(min(abs(orig) * 2, 30))
        bar_clip = "█" * int(min(abs(clipped) * 2, 30))
        sign = "+" if orig >= 0 else "-"
        print(f"  g[{i}]  {yellow(sign+bar_orig):<35}  → clipped: {green(sign+bar_clip)}")
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Gradient Clipping by Global Norm", """
import numpy as np

def clip_grad_norm(gradients, max_norm):
    '''
    Clip gradients by global L2 norm.
    gradients: list of numpy arrays (one per parameter)
    max_norm:  float, maximum allowed norm
    Returns scaled gradients if norm > max_norm, else originals.
    '''
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef  = max_norm / (total_norm + 1e-8)
    if clip_coef < 1.0:
        return [g * clip_coef for g in gradients], total_norm
    return gradients, total_norm

# Simulate large gradients
rng = np.random.default_rng(0)
grads = [rng.normal(0, 5, (10, 10)), rng.normal(0, 5, (10,))]

clipped, norm = clip_grad_norm(grads, max_norm=1.0)
print(f"Original norm:   {norm:.4f}")
print(f"Clipped g[0] norm: {np.linalg.norm(clipped[0]):.4f}")

# In PyTorch: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Spectral radius > 1 → gradients can explode in deep/recurrent networks",
        "Gradient clipping is a standard fix — used in nearly all RNN training",
        "Clip by norm (not element-wise) to preserve gradient direction",
        "typical max_grad_norm = 1.0 or 5.0 in practice",
        "Gradient explosion causes NaN losses — check with torch.isnan(loss)",
        "Better weight initialisation (orthogonal) partially prevents explosion",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Backprop Through Layers
# ══════════════════════════════════════════════════════════════════════════════
def topic_backprop_layers():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Layer-wise Gradient Derivations")
    section_header("BACKPROP THROUGH SPECIFIC LAYERS")
    print()

    section_header("1. LINEAR LAYER: z = Wx + b")
    print(formula("  Forward:   z = Wx + b"))
    print(formula("  ∂L/∂W = δ · xᵀ     (δ = ∂L/∂z, upstream gradient)"))
    print(formula("  ∂L/∂b = δ"))
    print(formula("  ∂L/∂x = Wᵀ · δ"))
    print()
    print(white("  DERIVATION: L is a scalar, z is a vector (n_out,), x is (n_in,)"))
    print(white("  L = loss(...z...) so ∂L/∂zⱼ = δⱼ"))
    print(white("  ∂zⱼ/∂Wⱼᵢ = xᵢ,  so ∂L/∂Wⱼᵢ = δⱼ xᵢ  →  ∂L/∂W = δ·xᵀ"))
    print(white("  ∂zⱼ/∂xᵢ = Wⱼᵢ,  so ∂L/∂xᵢ = Σⱼ δⱼ Wⱼᵢ = (Wᵀδ)ᵢ"))
    print()
    _pause()

    section_header("2. ReLU: a = max(0, z)")
    print(formula("  Forward:   aᵢ = max(0, zᵢ)"))
    print(formula("  ∂L/∂zᵢ = δᵢ · 1(zᵢ > 0)   (element-wise mask)"))
    print(formula("  ∂L/∂z = δ ⊙ (z > 0)"))
    print()
    z_ex = np.array([-1.0, 2.0, -0.5, 3.0])
    delta_ex = np.array([0.5, 0.8, -0.3, 1.2])
    grad_z = delta_ex * (z_ex > 0)
    print(bold_cyan(f"  Example:  z = {z_ex}"))
    print(bold_cyan(f"            δ = {delta_ex}"))
    print(bold_cyan(f"  ∂L/∂z = {delta_ex} ⊙ {(z_ex>0).astype(int)} = {grad_z}"))
    print(grey("  Note: z[0] and z[2] are negative → their gradients are blocked"))
    print()
    _pause()

    section_header("3. SOFTMAX + CROSS-ENTROPY (Combined — Elegant Result)")
    print(formula("  Softmax:       ŷᵢ = eˢⁱ / Σⱼ eˢʲ"))
    print(formula("  Cross-entropy: L = -Σᵢ yᵢ log ŷᵢ"))
    print(formula("  Combined:      ∂L/∂sᵢ = ŷᵢ - yᵢ   ← VERY CLEAN!"))
    print()
    print(white("  DERIVATION: Let p = softmax(s), y = one-hot label."))
    print(white("  ∂L/∂sⱼ = Σᵢ (∂L/∂pᵢ)(∂pᵢ/∂sⱼ)"))
    print(white("  ∂L/∂pᵢ = -yᵢ/pᵢ"))
    print(white("  ∂pᵢ/∂sⱼ = pᵢ(1-pᵢ) if i=j, else -pᵢpⱼ"))
    print(white("  Substituting and simplifying: ∂L/∂sⱼ = pⱼ - yⱼ"))
    print()
    s_ex = np.array([2.0, 1.0, 0.1])
    y_ex = np.array([1.0, 0.0, 0.0])
    exp_s = np.exp(s_ex - s_ex.max())
    p_ex = exp_s / exp_s.sum()
    grad_s = p_ex - y_ex
    print(bold_cyan(f"  s = {s_ex},  y = {y_ex}"))
    print(bold_cyan(f"  ŷ = softmax(s) = {np.round(p_ex, 4)}"))
    print(bold_cyan(f"  ∂L/∂s = ŷ - y = {np.round(grad_s, 4)}"))
    print()
    _pause()

    section_header("4. BATCH NORMALISATION — All 8 Intermediate Gradients")
    print(formula("  Forward:   μ = mean(x)           (step 1)"))
    print(formula("             d = x - μ              (step 2)"))
    print(formula("             σ² = mean(d²)           (step 3)"))
    print(formula("             v = σ² + ε              (step 4)"))
    print(formula("             s = 1/√v                (step 5)"))
    print(formula("             x̂ = d · s               (step 6)"))
    print(formula("             y = γ·x̂ + β             (step 7)  (scale/shift)"))
    print()
    print(white("  BACKWARD (all 8 gradients via chain rule):"))
    print(formula("  ∂L/∂γ = Σ ∂L/∂yᵢ · x̂ᵢ"))
    print(formula("  ∂L/∂β = Σ ∂L/∂yᵢ"))
    print(formula("  ∂L/∂x̂ = ∂L/∂y · γ"))
    print(formula("  ∂L/∂s = Σ (∂L/∂x̂ᵢ · dᵢ)"))
    print(formula("  ∂L/∂v = ∂L/∂s · (-½) · v^{-3/2}"))
    print(formula("  ∂L/∂σ² = ∂L/∂v"))
    print(formula("  ∂L/∂d = ∂L/∂x̂ · s + ∂L/∂σ² · 2d/n"))
    print(formula("  ∂L/∂x = ∂L/∂d - mean(∂L/∂d) - x̂·mean(∂L/∂d·x̂)"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Backprop Through Linear, ReLU, Softmax+CE, BatchNorm", """
import numpy as np

# --- Linear layer ---
def linear_backward(delta, cache):
    x, W = cache
    dW = np.outer(delta, x)
    db = delta.copy()
    dx = W.T @ delta
    return dx, dW, db

# --- ReLU backward ---
def relu_backward(delta, cache):
    z = cache
    return delta * (z > 0)

# --- Softmax + CrossEntropy combined gradient ---
def softmax_ce_backward(logits, y_onehot):
    '''Returns ∂L/∂logits = softmax(logits) - y_onehot   '''
    e = np.exp(logits - logits.max())
    yhat = e / e.sum()
    return yhat - y_onehot

# --- BatchNorm backward (simplified 1D) ---
def batchnorm_backward(delta, cache):
    x_hat, gamma, sigma2, eps = cache
    n = len(delta)
    dx_hat   = delta * gamma
    dsigma2  = np.sum(dx_hat * x_hat * (-0.5) * (sigma2+eps)**(-1.5))
    dmu      = np.sum(dx_hat * -(sigma2+eps)**(-0.5)) + dsigma2 * np.mean(-2*(x_hat))
    dx       = dx_hat / np.sqrt(sigma2+eps) + dsigma2 * 2*(x_hat)/n + dmu/n
    dgamma   = np.sum(delta * x_hat)
    dbeta    = np.sum(delta)
    return dx, dgamma, dbeta

# Test softmax+CE gradient
logits = np.array([2.0, 1.0, 0.1])
y      = np.array([1.0, 0.0, 0.0])
grad   = softmax_ce_backward(logits, y)
print(f"Softmax+CE gradient: {np.round(grad, 4)}")   # ŷ - y
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Linear backward: ∂L/∂W = δ·xᵀ — all you need to remember",
        "∂L/∂x = Wᵀδ — propagates error back to the previous layer",
        "Softmax + CE combined gradient is elegantly ŷ - y — implement together!",
        "BatchNorm backward has 8 intermediate steps but derives mechanically",
        "γ and β in BatchNorm are learnable — they restore representational power",
        "LayerNorm (used in Transformers) has same structure but normalises over features",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Automatic Differentiation
# ══════════════════════════════════════════════════════════════════════════════
def topic_autodiff():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Automatic Differentiation")
    section_header("AUTOMATIC DIFFERENTIATION")
    print()

    section_header("1. THEORY")
    print(white("""
  AUTOMATIC DIFFERENTIATION (autodiff) computes exact derivatives of programs
  mechanically — it is not symbolic differentiation (too slow) and not numerical
  differentiation (finite differences, approximate).

  TWO MODES:

  FORWARD MODE (Dual Numbers):
  Augment every number with a derivative: x̃ = (x, ẋ) where ẋ = dx/dt.
  Apply each operation to both parts simultaneously:
      (a,ȧ) + (b,ḃ) = (a+b, ȧ+ḃ)
      (a,ȧ) × (b,ḃ) = (ab, aḃ+ḃa)
      sin(a,ȧ) = (sin a, cos(a)·ȧ)
  One forward pass computes the function AND its derivative w.r.t. ONE input.
  Good for: f: ℝ → ℝⁿ (few inputs, many outputs)

  REVERSE MODE (Wengert Tape / Backpropagation):
  Execute forward pass, record all operations on a "tape".
  Replay backward to compute ∂L/∂xᵢ for ALL inputs in one pass.
  One backward pass computes ALL n gradients simultaneously.
  Good for: f: ℝⁿ → ℝ (many inputs, one output) — this is ML training!

  COMPARISON:
  Forward mode: O(n) passes to get all n partial derivatives
  Reverse mode: O(1) backward pass to get all n partial derivatives
  For neural nets with n ~ millions of parameters, reverse mode wins decisively.
"""))
    _pause()

    section_header("2. DUAL NUMBERS — Forward Mode Example")
    print(formula("  Dual number: x̃ = x + ε·dx,  where ε² = 0"))
    print()
    print(white("  Compute f(x) = x² + sin(x) and f'(x) at x=2.0:"))
    print()
    x_val = 2.0; dx_val = 1.0  # seed: dx/dx = 1
    # Step 1: x̃ = (2.0, 1.0)
    # Step 2: x̃² = (x², 2x·dx) = (4.0, 4.0)
    x2_val = x_val**2;  x2_dot = 2*x_val*dx_val
    # Step 3: sin(x̃) = (sin x, cos(x)·dx) = (0.9093, 0.4161)
    sinx_val = np.sin(x_val); sinx_dot = np.cos(x_val)*dx_val
    # Step 4: sum = (x² + sin x, 2x·dx + cos(x)·dx)
    f_val = x2_val + sinx_val; f_dot = x2_dot + sinx_dot

    print(bold_cyan(f"  x̃ = ({x_val}, {dx_val})  ← value and seed derivative"))
    print(bold_cyan(f"  x̃² = ({x2_val:.4f}, {x2_dot:.4f})  ← d/dx[x²] = 2x = {2*x_val}"))
    print(bold_cyan(f"  sin(x̃) = ({sinx_val:.4f}, {sinx_dot:.4f})  ← d/dx[sin x] = cos x = {np.cos(x_val):.4f}"))
    print(bold_cyan(f"  f(x̃) = ({f_val:.4f}, {f_dot:.4f})  ← f and f'"))
    print()
    # Verify analytically
    f_prime_analytic = 2*x_val + np.cos(x_val)
    print(green(f"  Analytical f'(2) = 2×2 + cos(2) = {f_prime_analytic:.4f}"))
    print(green(f"  Forward mode:      {f_dot:.4f}  ← exact match!"))
    print()
    _pause()

    section_header("3. COMPARISON TABLE")
    print()
    table(
        ["Property",   "Forward Mode (dual)", "Reverse Mode (backprop)"],
        [
            ["Direction",     "Input → Output",       "Output → Input"],
            ["Tape",          "Not needed",            "Record all ops (Wengert tape)"],
            ["Cost per param","1 forward pass",        "1 forward + 1 backward"],
            ["All gradients", "n forward passes",      "1 backward pass"],
            ["Best for",      "f: ℝ → ℝⁿ",            "f: ℝⁿ → ℝ (ML training)"],
            ["Memory",        "O(1) extra",            "O(n) for tape"],
            ["Framework",     "JAX jvp()",             "PyTorch .backward()"],
        ]
    )
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Dual Numbers (Forward Mode) and Wengert Tape (Reverse Mode)", """
import numpy as np

# ── Forward mode with dual numbers ──────────────────────────────────────────
class Dual:
    def __init__(self, val, dot=0.0):
        self.val = val;  self.dot = dot
    def __add__(self, o):   return Dual(self.val+o.val, self.dot+o.dot)
    def __mul__(self, o):   return Dual(self.val*o.val, self.val*o.dot+o.val*self.dot)
    def __repr__(self):     return f"Dual({self.val:.4f}, d={self.dot:.4f})"

def sin_dual(x: Dual):  return Dual(np.sin(x.val), np.cos(x.val)*x.dot)
def sqr_dual(x: Dual):  return x * x

# f(x) = x² + sin(x) at x=2
x = Dual(2.0, 1.0)    # derivative seed dx/dx = 1
result = sqr_dual(x) + sin_dual(x)
print(f"f(2) = {result.val:.4f}")
print(f"f'(2) = {result.dot:.4f}  (analytic: {2*2 + np.cos(2):.4f})")

# ── Mini reverse mode (Wengert tape) ─────────────────────────────────────────
# PyTorch does this automatically:
try:
    import torch
    x = torch.tensor(2.0, requires_grad=True)
    f = x**2 + torch.sin(x)
    f.backward()
    print(f"PyTorch f'(2) = {x.grad.item():.4f}")
except ImportError:
    print("PyTorch not available — dual numbers result above is exact")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Forward mode: augment computation with derivatives — O(1) memory",
        "Reverse mode: record tape, replay backward — computes all gradients in 1 pass",
        "For n params → n outputs: forward mode is O(n); reverse is O(1)",
        "Neural nets: n = millions, 1 output (loss) → reverse mode is essential",
        "JAX supports both: jvp (forward) and vjp (reverse), composable",
        "Higher-order derivatives: run autodiff on autodiff (Hessians, etc.)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Gradient Checking
# ══════════════════════════════════════════════════════════════════════════════
def topic_gradient_checking():
    clear()
    breadcrumb("mlmath", "Backpropagation", "Gradient Checking")
    section_header("GRADIENT CHECKING")
    print()

    section_header("1. THEORY")
    print(white("""
  GRADIENT CHECKING verifies analytical gradients by comparing with numerical
  finite differences.  This is an invaluable debugging tool when implementing
  backprop from scratch.

  CENTRAL FINITE DIFFERENCE (most accurate):
      ∂f/∂θᵢ ≈ [f(θ + ε·eᵢ) - f(θ - ε·eᵢ)] / (2ε)

  Error is O(ε²): with ε=1e-5 the approximation is accurate to ~1e-10.
  Forward difference f(θ+ε·eᵢ) - f(θ))/ε has O(ε) error — much worse.

  RELATIVE ERROR:
      rel_err = ‖∇_analytic - ∇_numerical‖ / (‖∇_analytic‖ + ‖∇_numerical‖)

  Thresholds:
    • rel_err < 1e-7: excellent (implement is correct)
    • rel_err 1e-5 to 1e-7: acceptable (might be fine)
    • rel_err > 1e-3: likely a bug in your backprop

  PRACTICAL NOTES:
  • Check a random subset of parameters (checking all is expensive)
  • Use double precision (float64) for gradient checking
  • Disable dropout and batch normalisation during gradient check (stochastic)
  • L2 regularisation: don't forget to include in both analytical and numerical
  • After verifying, switch back to float32 for training efficiency
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Central diff:   ∂f/∂θᵢ ≈ [f(θ+εeᵢ) - f(θ-εeᵢ)] / 2ε"))
    print(formula("  Relative error: ‖grad_A - grad_N‖ / (‖grad_A‖ + ‖grad_N‖)"))
    print(formula("  ε = 1e-5 gives O(1e-10) approximation error"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Check a 2-Layer Network")
    rng = np.random.default_rng(1)
    x  = rng.normal(0, 1, 4)
    y  = rng.normal(0, 1, 1)
    W1 = rng.normal(0, 0.1, (3, 4))
    b1 = rng.zeros(3)
    W2 = rng.normal(0, 0.1, (1, 3))
    b2 = rng.zeros(1)

    def forward(x, W1, b1, W2, b2):
        z1 = W1 @ x + b1; a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2; yhat = z2
        loss = float(np.mean((y - yhat)**2))
        return loss, z1, a1, z2, yhat

    # Analytical gradient of W1
    loss, z1, a1, z2, yhat = forward(x, W1, b1, W2, b2)
    dL_dyhat = 2*(yhat - y); dL_da1 = W2.T @ dL_dyhat
    dL_dz1   = dL_da1 * (z1 > 0); dL_dW1 = np.outer(dL_dz1, x)

    # Numerical gradient of W1 (central differences)
    eps = 1e-5
    dL_dW1_num = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus   = W1.copy(); W1_plus[i,j] += eps
            W1_minus  = W1.copy(); W1_minus[i,j] -= eps
            loss_plus, *_ = forward(x, W1_plus, b1, W2, b2)
            loss_minus,*_ = forward(x, W1_minus,b1, W2, b2)
            dL_dW1_num[i,j] = (loss_plus - loss_minus) / (2*eps)

    rel_err = np.linalg.norm(dL_dW1 - dL_dW1_num) / (np.linalg.norm(dL_dW1) + np.linalg.norm(dL_dW1_num) + 1e-12)

    print(bold_cyan(f"  Analytical ∂L/∂W1 (first row):  {np.round(dL_dW1[0], 8)}"))
    print(bold_cyan(f"  Numerical  ∂L/∂W1 (first row):  {np.round(dL_dW1_num[0], 8)}"))
    print(bold_cyan(f"  Max absolute diff: {np.max(np.abs(dL_dW1 - dL_dW1_num)):.2e}"))
    print(bold_cyan(f"  Relative error:    {rel_err:.2e}"))
    print()
    if rel_err < 1e-7:
        print(green(f"  ✓ Gradient check PASSED (rel_err={rel_err:.2e} < 1e-7)"))
    elif rel_err < 1e-3:
        print(yellow(f"  ~ Gradient check ACCEPTABLE (rel_err={rel_err:.2e})"))
    else:
        print(red(f"  ✗ Gradient check FAILED (rel_err={rel_err:.2e} > 1e-3)"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Gradient Checker for Any Function", """
import numpy as np

def gradient_check(fn, params, epsilon=1e-5):
    '''
    Numerically verify gradients for any scalar-output function.
    fn:     callable(params) -> (loss, grads)
    params: list of numpy arrays
    Returns: max relative error across all parameters.
    '''
    loss, analytic_grads = fn(params)
    max_rel_err = 0.0

    for p_idx, (param, grad) in enumerate(zip(params, analytic_grads)):
        flat_p = param.flatten()
        for i in range(min(len(flat_p), 20)):  # check first 20 for speed
            # +ε
            flat_plus = flat_p.copy(); flat_plus[i] += epsilon
            params[p_idx] = flat_plus.reshape(param.shape)
            loss_plus, _ = fn(params)

            # -ε
            flat_minus = flat_p.copy(); flat_minus[i] -= epsilon
            params[p_idx] = flat_minus.reshape(param.shape)
            loss_minus, _ = fn(params)

            # restore
            params[p_idx] = flat_p.reshape(param.shape)

            num_grad = (loss_plus - loss_minus) / (2 * epsilon)
            ana_grad = grad.flatten()[i]

            denom = abs(ana_grad) + abs(num_grad) + 1e-12
            err   = abs(ana_grad - num_grad) / denom
            max_rel_err = max(max_rel_err, err)

    return max_rel_err

print("Use this checker after implementing any new layer's backward pass.")
print("If max_rel_err < 1e-7: implementation is correct.")
print("If max_rel_err > 1e-3: there is a bug in backprop.")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Gradient checking is the gold standard for verifying backprop implementations",
        "Central differences (2ε denominator) are far more accurate than forward differences",
        "Relative error < 1e-7 means correct; > 1e-3 means there is a bug",
        "Common bugs: forgot factor of 2, wrong transpose, omitted bias gradient",
        "Disable stochastic layers (dropout, BN) — they must be deterministic for checking",
        "Use float64 for gradient checks to reduce floating-point error",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Computation Graphs",         topic_computation_graphs),
        ("Forward Pass",               topic_forward_pass),
        ("Backward Pass",              topic_backward_pass),
        ("Backprop Intuition",         topic_backprop_intuition),
        ("Vanishing Gradients",        topic_vanishing_gradients),
        ("Exploding Gradients",        topic_exploding_gradients),
        ("Backprop Through Layers",    topic_backprop_layers),
        ("Automatic Differentiation",  topic_autodiff),
        ("Gradient Checking",          topic_gradient_checking),
    ]
    block_menu("b08", "Backpropagation", topics)
