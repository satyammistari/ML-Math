"""
blocks/b09_activation_functions.py
Block 9: Activation Functions for Neural Networks
Topics: Sigmoid, Tanh, ReLU, Leaky Variants, GELU, Swish, Softmax, Comparison.
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


# ──────────────────────────────────────────────────────────────────────────────
# Shared activation functions used across topics
# ──────────────────────────────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

def _tanh(x):
    return np.tanh(x)

def _relu(x):
    return np.maximum(0.0, x)

def _leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def _elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(np.clip(x, -100, 0)) - 1.0))

def _gelu(x):
    return x * _sigmoid(1.702 * x)

def _swish(x):
    return x * _sigmoid(x)

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — Sigmoid
# ══════════════════════════════════════════════════════════════════════════════
def topic_sigmoid():
    clear()
    breadcrumb("mlmath", "Activation Functions", "Sigmoid")
    section_header("SIGMOID ACTIVATION")
    print()

    section_header("1. THEORY")
    print(white("""
  The sigmoid (logistic) function maps any real value to (0, 1):
      σ(x) = 1 / (1 + e⁻ˣ)

  DERIVATIVE (elegant factored form):
      σ'(x) = σ(x) · (1 - σ(x))
  Maximum derivative at x=0: σ'(0) = 0.5 × 0.5 = 0.25.
  For |x| > 5: σ(x) ≈ 0 or 1, and σ'(x) ≈ 0  — the SATURATION region.

  SATURATION PROBLEM:
  When x is large (positive or negative), the gradient σ'(x) → 0.
  Gradients propagating back through saturated sigmoid neurons are killed.
  This leads to vanishing gradients in deep networks.

  NOT ZERO-CENTRED:
  Sigmoid outputs are always in (0,1), never negative.  For a hidden layer,
  this means the gradient of weights W has a fixed sign based on the input.
  This can slow convergence (zig-zagging in weight space).
  Tanh fixes this by being zero-centred.

  USE CASES:
  • Output layer for binary classification (probability in [0,1])
  • Logistic regression
  • Gating mechanisms (LSTM, attention) — controlled interpolation
  • Rarely used as hidden layer activation in deep nets (ReLU is preferred)
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  σ(x) = 1 / (1 + e⁻ˣ) = eˣ / (1 + eˣ)"))
    print(formula("  σ'(x) = σ(x)(1 - σ(x))"))
    print(formula("  σ(0) = 0.5,   σ(∞) = 1,   σ(-∞) = 0"))
    print(formula("  σ(-x) = 1 - σ(x)  (antisymmetry)"))
    print(formula("  Max gradient: σ'(0) = 0.25"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    xs = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    print(bold_cyan(f"  {'x':>6}  {'σ(x)':>10}  {'σ′(x)':>10}  {'Saturated?'}"))
    print(grey("  " + "─"*50))
    for x in xs:
        s = _sigmoid(x); sp = s*(1-s)
        sat = red("YES (vanishing!)") if abs(x) >= 4 else green("no")
        print(f"  {x:>6.1f}  {s:>10.6f}  {sp:>10.6f}  {sat}")
    print()
    _pause()

    section_header("4. ASCII — Sigmoid Curve")
    xs_plot = np.linspace(-6, 6, 50)
    ys_plot  = _sigmoid(xs_plot)
    yp_plot  = ys_plot * (1 - ys_plot)
    print(cyan("  σ(x) = blue  |  σ'(x) = yellow"))
    print()
    for x, s, sp in zip(xs_plot[::5], ys_plot[::5], yp_plot[::5]):
        bar_s  = "█" * int(s  * 20)
        bar_sp = "░" * int(sp * 60)
        print(f"  x={x:5.1f}  {cyan(bar_s):<25}  {yellow(bar_sp)}")
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Sigmoid and Its Derivative", """
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1.0 - s)

x = np.linspace(-6, 6, 200)
print(f"sigma(0)  = {sigmoid(0):.4f}")       # 0.5
print(f"sigma(5)  = {sigmoid(5):.4f}")       # ≈ 0.9933
print(f"sigma(-5) = {sigmoid(-5):.4f}")      # ≈ 0.0067
print(f"sigma'(0) = {sigmoid_prime(0):.4f}") # 0.25 (max)
print(f"sigma'(5) = {sigmoid_prime(5):.6f}") # ≈ 6.6e-3 (nearly zero!)
print(f"sigma'(10)= {sigmoid_prime(10):.2e}")# ≈ 4.5e-5 (vanished)

# Not zero-centred: mean output
print(f"Mean sigmoid output over [-6,6]: {np.mean(sigmoid(x)):.4f}")  # > 0.5
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Sigmoid saturates for |x|>5 — gradient essentially zero there",
        "Max gradient 0.25 means each sigmoid layer shrinks gradient by ≥4×",
        "Not zero-centred (always > 0) — can cause weight gradient zig-zagging",
        "Use at OUTPUT layer for binary classification, not hidden layers",
        "Logistic regression uses sigmoid: P(y=1|x) = σ(wᵀx + b)",
        "LSTM/GRU use sigmoid for gates: multiplies between 0 (forget) and 1 (keep)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Tanh
# ══════════════════════════════════════════════════════════════════════════════
def topic_tanh():
    clear()
    breadcrumb("mlmath", "Activation Functions", "Tanh")
    section_header("TANH ACTIVATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Tanh (hyperbolic tangent) maps ℝ → (-1, 1):
      tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ) = 2σ(2x) - 1

  DERIVATIVE:
      tanh'(x) = 1 - tanh²(x)
  Maximum at x=0: tanh'(0) = 1 — 4× stronger gradient than sigmoid!

  ZERO-CENTRED ADVANTAGE:
  Unlike sigmoid, tanh outputs are centred at 0 (range (-1,1)).
  This makes gradient updates more symmetric and can speed convergence.
  Specifically, gradients for different examples don't all have the same sign.

  STILL SATURATES:
  For |x| > 3, tanh(x) ≈ ±1 and tanh'(x) ≈ 0.
  The vanishing gradient problem still exists, just less severe than sigmoid.

  RELATIONSHIP TO SIGMOID:
  tanh(x) = 2 · σ(2x) - 1
  They can represent the same functions — tanh is just a rescaled, shifted sigmoid.

  USE CASES:
  • RNN/LSTM hidden states (zero-centred helps in sequence modelling)
  • Any task requiring outputs in (-1, 1)
  • Not preferred for very deep feedforward nets (ReLU dominates there)
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)"))
    print(formula("  tanh'(x) = 1 - tanh²(x)"))
    print(formula("  tanh(0) = 0,   tanh(∞) = 1,   tanh(-∞) = -1"))
    print(formula("  tanh(x) = 2σ(2x) - 1  (relation to sigmoid)"))
    print(formula("  Max gradient: tanh'(0) = 1  (vs 0.25 for sigmoid)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    xs = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    print(bold_cyan(f"  {'x':>6}  {'tanh(x)':>10}  {'tanh′(x)':>10}  {'Compare sigmoid′'}"))
    print(grey("  " + "─"*55))
    for x in xs:
        t = _tanh(x); tp = 1.0 - t**2
        sp = _sigmoid(x) * (1 - _sigmoid(x))
        ratio = tp / (sp + 1e-12)
        print(f"  {x:>6.1f}  {t:>10.6f}  {tp:>10.6f}  {grey(f'≈ {ratio:.1f}× sig grad')}")
    print()
    _pause()

    section_header("4. ASCII — Tanh vs Sigmoid")
    xs_plot = np.linspace(-5, 5, 40)
    for x in xs_plot[::4]:
        t = _tanh(x); s = _sigmoid(x)
        # centre at index 10 (x=0)
        t_pos = int((t + 1) / 2 * 20)  # map [-1,1] to [0,20]
        s_pos = int(s * 20)             # map [0,1] to [0,20]
        line = [" "] * 25
        if 0 <= t_pos < 25: line[t_pos] = "T"
        if 0 <= s_pos < 25: line[s_pos] = "S"
        print(f"  x={x:5.1f}  |{''.join(line)}|  tanh={t:.3f}  sig={s:.3f}")
    print(grey("  T = tanh  (range -1 to 1)"))
    print(grey("  S = sigmoid  (range 0 to 1)"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Tanh and Its Derivative", """
import numpy as np

def tanh(x):   return np.tanh(x)

def tanh_prime(x):
    t = np.tanh(x)
    return 1.0 - t**2

x = np.linspace(-5, 5, 100)
print(f"tanh(0)    = {tanh(0):.4f}")          # 0.0
print(f"tanh(3)    = {tanh(3):.4f}")          # ≈ 0.995
print(f"tanh'(0)   = {tanh_prime(0):.4f}")    # 1.0 (max)
print(f"tanh'(3)   = {tanh_prime(3):.4f}")    # ≈ 0.01 (nearly zero)

# Zero-centred check
print(f"Mean tanh over [-5,5]: {np.mean(tanh(x)):.6f}")   # ≈ 0 (zero-centred)
print(f"Mean sigmoid over [-5,5]: {np.mean(1/(1+np.exp(-x))):.4f}")  # > 0

# Relation to sigmoid
sig = lambda z: 1/(1+np.exp(-z))
diff = np.max(np.abs(tanh(x) - (2*sig(2*x) - 1)))
print(f"Max |tanh(x) - 2σ(2x) - 1| = {diff:.2e}")  # ≈ 0 (same function)
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "tanh is zero-centred: outputs in (-1,1), unlike sigmoid [0,1]",
        "tanh max gradient = 1 vs sigmoid max gradient = 0.25",
        "Still saturates for |x|>3 — vanishing gradients in very deep nets",
        "Mathematically: tanh(x) = 2σ(2x) - 1 — a rescaled sigmoid",
        "Used in LSTM cells for hidden state updates (zero-centred helps)",
        "For deep feedforward nets, ReLU and variants generally outperform tanh",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — ReLU
# ══════════════════════════════════════════════════════════════════════════════
def topic_relu():
    clear()
    breadcrumb("mlmath", "Activation Functions", "ReLU")
    section_header("RELU (RECTIFIED LINEAR UNIT)")
    print()

    section_header("1. THEORY")
    print(white("""
  ReLU is the most widely used hidden-layer activation:
      ReLU(x) = max(0, x)

  DERIVATIVE:
      ReLU'(x) = 1 if x > 0, else 0

  NON-DIFFERENTIABLE at x=0 (subgradient = 0 or 1 by convention, usually 0).

  WHY RELU IS SO EFFECTIVE:
  • No gradient saturation for positive inputs → no vanishing gradients
  • Computationally cheap: just a threshold operation
  • Sparse activations: negative inputs produce exactly 0 → only a fraction
    activate → efficient, regularising, brain-inspired
  • Enables very deep networks (without residual connections or BN,
    sigmoid/tanh struggle beyond ~5 layers; ReLU works to 100+)

  DYING RELU PROBLEM:
  If a neuron receives only negative inputs (e.g. bad initialisation or
  large negative bias), it will ALWAYS output 0. Gradient is 0, so the
  neuron never updates — it "dies" and stays dead.
  Solution: Leaky ReLU, proper initialisation (He init), lower learning rates.

  SPARSE ACTIVATION BENEFIT:
  On average, ~50% of neurons are active for normally distributed inputs.
  This sparsity is computationally efficient and can improve generalisation
  by forcing the network to use compact representations.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ReLU(x) = max(0, x)"))
    print(formula("  ReLU'(x) = 1  if x > 0"))
    print(formula("  ReLU'(x) = 0  if x ≤ 0  (dead neuron region)"))
    print(formula("  He init:  W ~ N(0, √(2/n_in))   optimal for ReLU"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Dying ReLU")
    rng = np.random.default_rng(42)
    n_neurons = 10
    # Simulate a layer with a large negative bias (dead neuron scenario)
    x_input = rng.normal(0, 1, (50, 4))
    W = rng.normal(0, 0.1, (n_neurons, 4))
    b = -5.0 * np.ones(n_neurons)   # very negative bias → all inputs negative
    preact = x_input @ W.T + b
    act = np.maximum(0, preact)
    active_frac = np.mean(act > 0, axis=0)

    print(bold_cyan(f"  Scenario: bias = -5.0 (very negative)"))
    print(bold_cyan(f"  n_neurons = {n_neurons},  n_samples = 50"))
    print()
    print(f"  {'Neuron':<8} {'Active fraction':<20} {'Status':<15} {'Bar'}")
    print(grey("  " + "─"*60))
    for i, frac in enumerate(active_frac):
        status = green("alive") if frac > 0.01 else red("DEAD")
        bar = "█" * int(frac * 20)
        print(f"  {i:<8} {frac:<20.4f} {status:<15} {cyan(bar)}")
    print()
    n_dead = int(np.sum(active_frac < 0.01))
    print(bold_cyan(f"  Dead neurons (activation rate < 1%): {n_dead}/{n_neurons}"))
    print(green("  Fix: use Leaky ReLU or initialise bias to 0"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("ReLU, Its Derivative, and He Initialisation", """
import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def relu_prime(x):
    return (x > 0).astype(float)

# He initialisation (optimal for ReLU)
def he_init(fan_in, fan_out):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

x = np.linspace(-3, 3, 100)
print(f"relu(2.0)  = {relu(2.0):.1f}")       # 2.0
print(f"relu(-2.0) = {relu(-2.0):.1f}")      # 0.0
print(f"relu'(2.0) = {relu_prime(2.0):.1f}") # 1.0
print(f"relu'(-2.0)= {relu_prime(-2.0):.1f}")# 0.0

# Sparse activation: fraction of neurons active
inputs = np.random.randn(1000, 256)
W = he_init(256, 512)
preact = inputs @ W
active = np.mean(relu(preact) > 0)
print(f"Active neuron fraction (normal inputs): {active:.2%}")  # ≈ 50%

# Dying ReLU with bad bias
W2 = he_init(256, 512); b2 = -5.0*np.ones(512)
preact2 = inputs @ W2 + b2
active2 = np.mean(relu(preact2) > 0)
print(f"Active with b=-5 (dying): {active2:.2%}")  # ≈ 0%
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "ReLU'(x) = 1 for x>0, no saturation — gradient flows freely",
        "Dying ReLU: negative pre-activations → dead neuron, never updates",
        "He initialisation: std = √(2/n_in) — compensates for 50% of neurons being dead",
        "ReLU is non-differentiable at 0, but this rarely matters in practice",
        "Sparse activations (~50% active) can improve generalisation",
        "For most hidden layers in feedforward nets, ReLU or GELU are the default choice",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Leaky Variants
# ══════════════════════════════════════════════════════════════════════════════
def topic_leaky_variants():
    clear()
    breadcrumb("mlmath", "Activation Functions", "Leaky ReLU Variants")
    section_header("LEAKY RELU AND VARIANTS")
    print()

    section_header("1. THEORY")
    print(white("""
  All these variants address the DYING ReLU problem by allowing a small
  gradient to flow for negative inputs.

  LEAKY ReLU:
      f(x) = x        if x ≥ 0
      f(x) = αx       if x < 0   (α typically 0.01)
  Always has a non-zero gradient (α for x<0) → neurons can recover.

  PReLU (Parametric ReLU) — He et al. 2015:
      f(x) = x   if x ≥ 0
      f(x) = αx  if x < 0   where α is LEARNED per channel
  Learns the optimal slope for negative inputs.

  ELU (Exponential Linear Unit) — Clevert et al. 2015:
      f(x) = x              if x ≥ 0
      f(x) = α(eˣ - 1)     if x < 0   (α typically 1.0)
  Smooth everywhere, mean activations closer to 0, speeds convergence.
  More expensive than ReLU (exp computation).

  SELU (Scaled ELU) — Klambauer et al. 2017:
      f(x) = λx              if x ≥ 0
      f(x) = λα(eˣ - 1)     if x < 0
  With specific λ ≈ 1.0507 and α ≈ 1.6733, SELU is SELF-NORMALISING:
  if inputs have mean 0, variance 1, then outputs will too — essentially
  free batch normalisation! Works well with lecun_normal initialisation.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Leaky ReLU: f(x) = x if x≥0,  else αx  (α=0.01)"))
    print(formula("  PReLU:      same, α learned  (per-channel)"))
    print(formula("  ELU:        f(x) = x if x≥0,  else α(eˣ-1)"))
    print(formula("  SELU:       f(x) = λ·ELU(x, α)  λ=1.0507, α=1.6733"))
    print(formula("  SELU self-norm: E[output]=0, Var[output]=1 given E[input]=0, Var=1"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    xs = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])
    selu_lambda = 1.0507009873554805
    selu_alpha  = 1.6732631921736576

    print(bold_cyan(f"  {'x':>6}  {'Leaky(0.01)':>14}  {'ELU(1.0)':>10}  {'SELU':>10}"))
    print(grey("  " + "─"*55))
    for x in xs:
        lk  = _leaky_relu(x, 0.01)
        el  = _elu(x, 1.0)
        se  = selu_lambda * (_elu(x, selu_alpha))
        print(f"  {x:>6.2f}  {lk:>14.6f}  {el:>10.6f}  {se:>10.6f}")
    print()
    _pause()

    section_header("4. ASCII — Comparison of Variants at x<0")
    xs_neg = np.linspace(-4, 0, 20)
    print(cyan("  Negative region behavior (x from -4 to 0):"))
    for x in xs_neg[::4]:
        lk = _leaky_relu(x, 0.01)
        el = _elu(x, 1.0)
        # normalized bars for display
        bar_lk = "░" * max(0, int(abs(lk) * 2))
        bar_el = "░" * max(0, int(abs(el) * 2))
        print(f"  x={x:5.2f}  Leaky: {yellow(bar_lk):<12}({lk:.3f})  ELU: {cyan(bar_el):<12}({el:.3f})")
    print(grey("  ELU has smoother negative continuation (exponential curve)"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Leaky ReLU, ELU, SELU", """
import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(np.clip(x, -100, 0)) - 1.0))

def selu(x):
    LAMBDA = 1.0507009873554805
    ALPHA  = 1.6732631921736576
    return LAMBDA * elu(x, alpha=ALPHA)

x = np.random.randn(10000)
print(f"Leaky ReLU:  mean={leaky_relu(x).mean():.4f}, var={leaky_relu(x).var():.4f}")
print(f"ELU:         mean={elu(x).mean():.4f}, var={elu(x).var():.4f}")
print(f"SELU:        mean={selu(x).mean():.4f}, var={selu(x).var():.4f}")
# SELU should be close to mean≈0, var≈1 for standard-normal inputs
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Leaky ReLU: α=0.01 ensures gradient never fully dies — simple fix",
        "PReLU learns α per channel — more expressive, slight overfit risk",
        "ELU: smooth for x<0, mean-shifts output toward 0 (self-normalising tendency)",
        "SELU: exact self-normalisation with tuned λ, α — free of batch norm",
        "SELU requires LeCun normal init and no other normalisation to work properly",
        "In practice: Leaky ReLU or GELU work well; SELU valuable for fully-connected nets",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — GELU
# ══════════════════════════════════════════════════════════════════════════════
def topic_gelu():
    clear()
    breadcrumb("mlmath", "Activation Functions", "GELU")
    section_header("GELU (GAUSSIAN ERROR LINEAR UNIT)")
    print()

    section_header("1. THEORY")
    print(white("""
  GELU (Hendrycks & Gimpel, 2016) is defined as:
      GELU(x) = x · Φ(x)
  where Φ(x) is the CDF of the standard Gaussian (cumulative of N(0,1)).

  INTERPRETATION:
  GELU stochastically gates the input: at each value x, the input is
  multiplied by the probability that a Gaussian random variable ≤ x.
  This is a smooth approximation to dropping inputs randomly with probability
  proportional to how negative they are (a smooth version of dropout).

  APPROXIMATION:
      GELU(x) ≈ x · σ(1.702 x)  (sigmoid approximation, fast in practice)

  Or the tanh approximation:
      GELU(x) ≈ 0.5x [1 + tanh(√(2/π) · (x + 0.044715 x³))]

  PROPERTIES:
  • Not monotonic: has a slight dip below 0 for small negative x
  • Smooth everywhere (unlike ReLU)
  • Not zero-centred (like ReLU)
  • Empirically outperforms ReLU on NLP tasks

  USAGE:
  • GPT-2, GPT-3, BERT, T5 — virtually all modern Transformers use GELU
  • Vision Transformers (ViT) also use GELU
  • Becoming the de facto standard for Transformer architectures
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  GELU(x) = x · Φ(x)  where Φ = Standard Normal CDF"))
    print(formula("  Fast approx: GELU(x) ≈ x · σ(1.702x)"))
    print(formula("  Tanh approx: 0.5x[1+tanh(√(2/π)·(x+0.044715x³))]"))
    print(formula("  GELU'(x) = Φ(x) + x · φ(x)  where φ = N(0,1) PDF"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    from scipy.special import ndtr   # Gaussian CDF
    xs = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    print(bold_cyan(f"  {'x':>6}  {'GELU exact':>12}  {'GELU approx':>14}  {'Difference'}"))
    print(grey("  " + "─"*55))
    for x in xs:
        gelu_exact = x * ndtr(x)
        gelu_approx = _gelu(x)
        diff = abs(gelu_exact - gelu_approx)
        print(f"  {x:>6.1f}  {gelu_exact:>12.6f}  {gelu_approx:>14.6f}  {diff:.2e}")
    print()
    _pause()

    section_header("4. ASCII — GELU vs ReLU")
    xs_plot = np.linspace(-4, 4, 40)
    print(cyan("  GELU(x) = cyan,  ReLU(x) = yellow"))
    print()
    for x in xs_plot[::4]:
        g = _gelu(x); r = _relu(x)
        bar_g = "█" * int(max(g, 0) * 5)
        bar_r = "░" * int(r * 5)
        dip = red("← below x-axis") if g < 0 else ""
        print(f"  x={x:5.1f}  {cyan(bar_g):<15} G={g:+.3f}  {yellow(bar_r):<12} R={r:.3f} {dip}")
    print(grey("  Note: GELU dips slightly below 0 near x ≈ -0.2"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("GELU Implementation and Comparison", """
import numpy as np
from scipy.special import ndtr  # Gaussian CDF

def gelu_exact(x):
    return x * ndtr(x)

def gelu_approx_sigmoid(x):
    return x / (1.0 + np.exp(-1.702 * x))

def gelu_approx_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-4, 4, 100)

# Compare accuracy
diff_sig  = np.max(np.abs(gelu_exact(x) - gelu_approx_sigmoid(x)))
diff_tanh = np.max(np.abs(gelu_exact(x) - gelu_approx_tanh(x)))
print(f"Max error sigmoid approx: {diff_sig:.6f}")
print(f"Max error tanh approx:    {diff_tanh:.6f}")

# Properties
print(f"GELU(-0.2) = {gelu_exact(-0.2):.4f}  (slight dip below 0)")
print(f"GELU(0)    = {gelu_exact(0):.4f}")
print(f"GELU(1)    = {gelu_exact(1):.4f}")
print("Used in: BERT, GPT-2, GPT-3, T5, ViT — backbone of modern Transformers")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "GELU = x·Φ(x): stochastic gate — probability proportional to input magnitude",
        "Smooth everywhere, unlike ReLU — may help gradient flow",
        "Slight dip below 0 for small negative x — non-monotonic",
        "Fast sigmoid approximation x·σ(1.702x) used in most frameworks",
        "Standard activation in GPT, BERT, T5, ViT and virtually all modern Transformers",
        "Outperforms ReLU on NLP benchmarks; similar to ReLU+ on vision",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Swish / SiLU
# ══════════════════════════════════════════════════════════════════════════════
def topic_swish():
    clear()
    breadcrumb("mlmath", "Activation Functions", "Swish / SiLU")
    section_header("SWISH AND SILU")
    print()

    section_header("1. THEORY")
    print(white("""
  Swish (Ramachandran et al., Google Brain, 2017):
      Swish(x) = x · σ(x) = x / (1 + e⁻ˣ)

  SiLU (Sigmoid Linear Unit) is the SAME FUNCTION by different name.
  The terms are used interchangeably in the literature.

  SELF-GATED:
  The function gates its own input with a sigmoid:
      Swish(x) = x · [gate(x)]  where gate = σ(x)
  For large positive x: gate → 1, Swish(x) → x  (acts like identity)
  For large negative x: gate → 0, Swish(x) → 0  (gates out negative inputs)
  Near x=0: smoothly interpolates — no abrupt zero as in ReLU

  PROPERTIES:
  • Smooth everywhere (infinitely differentiable)
  • Non-monotonic: has a local minimum near x ≈ -1.28
  • Not zero-centred
  • Unbounded above (like ReLU), bounded below by ~-0.28

  PARAMETERISED SWISH: Swish_β(x) = x · σ(β·x)
  • β=0: Swish = x/2 (linear)
  • β=1: standard Swish
  • β→∞: Swish → ReLU

  PERFORMANCE:
  Swish/SiLU consistently matches or slightly outperforms ReLU on image
  classification (EfficientNet uses Swish), NLP, and reinforcement learning.
  It is the default activation in many EfficientNet and MobileNetV3 variants.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Swish(x) = x · σ(x)"))
    print(formula("  SiLU(x) = x · σ(x)   ← same function"))
    print(formula("  Swish'(x) = σ(x) + x · σ(x)(1 - σ(x))"))
    print(formula("            = σ(x) · (1 + x(1-σ(x)))"))
    print(formula("  Swish_β(x) = x · σ(βx),  β=1 is standard"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    xs = np.array([-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0])
    print(bold_cyan(f"  {'x':>6}  {'Swish(x)':>12}  {'ReLU(x)':>10}  {'Diff':>10}"))
    print(grey("  " + "─"*50))
    for x in xs:
        sw = _swish(x); rl = _relu(x)
        diff = sw - rl
        print(f"  {x:>6.1f}  {sw:>12.6f}  {rl:>10.4f}  {diff:>+10.6f}")
    print()
    print(green("  Swish < 0 for x ≈ -1.28 (minimum) — unlike ReLU which is always ≥ 0"))
    print()
    _pause()

    section_header("4. ASCII — Swish Curve")
    xs_plot = np.linspace(-5, 5, 40)
    print(cyan("  Swish(x) = x·σ(x)"))
    print()
    for x in xs_plot[::4]:
        sw = _swish(x)
        bar_len = int(max(sw, 0) * 4)
        neg = red("↙") if sw < 0 else ""
        bar = "█" * bar_len
        print(f"  x={x:5.1f}  {cyan(bar):<20}  Swish={sw:+.4f} {neg}")
    print(grey("  Note: dips below 0 near x=-1.3"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Swish / SiLU Implementation", """
import numpy as np

def swish(x, beta=1.0):
    sig = 1.0 / (1.0 + np.exp(-beta * np.clip(x, -500, 500)))
    return x * sig

def swish_prime(x):
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return sig + x * sig * (1.0 - sig)

# Properties
print(f"Swish(0)    = {swish(0):.4f}")         # 0.0
print(f"Swish(1)    = {swish(1):.4f}")         # ≈ 0.7311
print(f"Swish(-1)   = {swish(-1):.4f}")        # ≈ -0.2689
print(f"Swish'(0)   = {swish_prime(0):.4f}")   # 0.5
print(f"Swish'(1)   = {swish_prime(1):.4f}")

# Minimum point
xs = np.linspace(-3, 0, 1000)
min_idx = np.argmin(swish(xs))
print(f"Minimum at x = {xs[min_idx]:.3f},  Swish = {swish(xs[min_idx]):.4f}")

# In PyTorch: torch.nn.SiLU() (same as Swish(x), beta=1)
print("PyTorch: torch.nn.functional.silu(x)")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Swish = SiLU = x·σ(x) — two names for the exact same function",
        "Self-gated: negative activation region smoothed out (not abruptly zero)",
        "Non-monotonic: minimum at x ≈ -1.28, value ≈ -0.28",
        "Used in EfficientNet, MobileNetV3 — consistently beats ReLU on image tasks",
        "Parameterised Swish_β: β→∞ approaches ReLU, β=0 is linear",
        "Computational cost: one sigmoid multiply, ~same as GELU approximation",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Softmax
# ══════════════════════════════════════════════════════════════════════════════
def topic_softmax():
    clear()
    breadcrumb("mlmath", "Activation Functions", "Softmax")
    section_header("SOFTMAX ACTIVATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Softmax converts a vector of real-valued logits into a probability distribution:
      softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ

  OUTPUT PROPERTIES:
  • Each output is in (0, 1)
  • Outputs sum to exactly 1  (probability distribution!)
  • Preserves ordering: larger input → larger output
  • Invariant to additive constant: softmax(x + c) = softmax(x)  ← stability trick!

  NUMERICAL STABILITY:
  Computing eˣⁱ directly overflows for large x.
  The trick: subtract max(x) before exponentiating:
      eˣⁱ⁻ᶜ / Σⱼ eˣʲ⁻ᶜ  where c = max(x)
  This is algebraically identical but numerically safe.

  TEMPERATURE SCALING:
  softmax(x/T)ᵢ = eˣⁱ/ᵀ / Σⱼ eˣʲ/ᵀ
  • T → 0: approaches one-hot (argmax)  — "sharp" distribution
  • T → ∞: approaches uniform           — "soft" / uncertain distribution
  • T = 1: standard softmax
  Used in knowledge distillation, language model sampling, attention.

  GRADIENT (combined with cross-entropy):
  If L = cross-entropy loss and p = softmax(logits):
      ∂L/∂logitsᵢ = pᵢ - yᵢ   ← elegantly simple
  This is why softmax + cross-entropy are always implemented together.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ"))
    print(formula("  Stable:      softmax(x-max(x))ᵢ  (numerically safe)"))
    print(formula("  Temperature: softmax(x/T)  T→0 = argmax, T→∞ = uniform"))
    print(formula("  Jacobian:    ∂pᵢ/∂xⱼ = pᵢ(δᵢⱼ - pⱼ)  (dense matrix)"))
    print(formula("  CE gradient: ∂L/∂xᵢ = pᵢ - yᵢ  (combined)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Temperature Scaling")
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    print(bold_cyan(f"  Logits: {logits}"))
    print()
    print(bold_cyan(f"  {'Temperature':<14} {'P(class 0)':>12} {'P(class 1)':>12} {'P(class 2)':>12} {'P(class 3)':>12}"))
    print(grey("  " + "─" * 65))
    for T in [0.1, 0.5, 1.0, 2.0, 5.0, 100.0]:
        p = _softmax(logits / T)
        print(f"  T={T:<12.1f} {p[0]:>12.6f} {p[1]:>12.6f} {p[2]:>12.6f} {p[3]:>12.6f}")
    print()
    print(green("  T→0:   concentrated on class 0 (highest logit)"))
    print(green("  T→∞:   uniform distribution [0.25, 0.25, 0.25, 0.25]"))
    print()
    _pause()

    section_header("4. ASCII — Softmax Output Distribution")
    p_default = _softmax(logits)
    print(cyan("  softmax([2.0, 1.0, 0.1, -1.0]):"))
    for i, pi in enumerate(p_default):
        bar = "█" * int(pi * 50)
        print(f"  Class {i}: {cyan(bar):<55} {pi:.4f}")
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("Numerically Stable Softmax and Temperature", """
import numpy as np

def softmax(x, temperature=1.0):
    '''Numerically stable softmax with temperature.'''
    x_scaled = x / temperature
    x_shift  = x_scaled - x_scaled.max()   # subtract max for stability
    exp_x    = np.exp(x_shift)
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.1, -1.0])

# Standard softmax
p = softmax(logits)
print(f"softmax: {np.round(p, 4)}")
print(f"Sum: {p.sum():.6f}")   # should be exactly 1

# Temperature effects
for T in [0.1, 1.0, 10.0]:
    p_T = softmax(logits, T)
    print(f"T={T:4.1f}: {np.round(p_T, 4)}")

# Numerical stability demo
big_logits = np.array([1000.0, 999.0, 998.0])
print(f"Stable softmax of [1000,999,998]: {np.round(softmax(big_logits), 4)}")
# Without the max-subtraction trick, np.exp(1000) = inf!
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Always subtract max(x) before exp for numerical stability",
        "Cross-entropy + softmax gradient simplifies to ŷ - y (implement together)",
        "Temperature T<1: sharper, more confident; T>1: softer, more uncertain",
        "Softmax Jacobian is dense (n×n) — never compute it explicitly",
        "For multi-label: sigmoid per class is MORE appropriate than softmax",
        "log-softmax = log_sum_exp trick for stable log probabilities",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Comparison
# ══════════════════════════════════════════════════════════════════════════════
def topic_comparison():
    clear()
    breadcrumb("mlmath", "Activation Functions", "Comparison")
    section_header("ACTIVATION FUNCTION COMPARISON")
    print()

    section_header("1. THEORY SUMMARY")
    print(white("""
  Choosing the right activation function depends on:
  1. ARCHITECTURE: hidden layer vs output layer
  2. DEPTH: deep nets need non-saturating activations
  3. TASK: regression (linear), binary class (sigmoid), multi-class (softmax)
  4. FRAMEWORK: modern transformers use GELU; CNNs often use ReLU/Swish

  GENERAL RULES:
  • Hidden layers: ReLU (default), GELU (transformers), Swish (efficient nets)
  • Binary classification output: Sigmoid
  • Multi-class classification output: Softmax
  • Regression output: Linear (no activation)
  • Avoid Sigmoid/Tanh in hidden layers of deep nets (saturation)
"""))
    _pause()

    section_header("2. PROPERTY TABLE")
    print()
    table(
        ["Function",      "Range",    "Zero-centred", "Monotonic", "Saturates", "C∞ smooth", "Common use"],
        [
            ["Sigmoid",     "(0,1)",    "No",           "Yes",       "Yes",       "Yes",       "Binary out"],
            ["Tanh",        "(-1,1)",   "Yes",          "Yes",       "Yes",       "Yes",       "RNN hidden"],
            ["ReLU",        "[0,∞)",    "No",           "Yes",       "No",        "No",        "CNN/MLP hidden"],
            ["Leaky ReLU",  "(-∞,∞)",  "No",           "Yes",       "No",        "No",        "Better ReLU"],
            ["ELU",         "(-α,∞)",   "Near 0",       "Yes",       "Partial",   "Yes",       "Deep nets"],
            ["SELU",        "(-λα,∞)",  "Yes",          "Yes",       "Partial",   "Yes",       "Self-norm MLP"],
            ["GELU",        "(-0.17,∞)","No",           "No",        "No",        "Yes",       "Transformers"],
            ["Swish/SiLU",  "(-0.28,∞)","No",           "No",        "No",        "Yes",       "EfficientNet"],
            ["Softmax",     "(0,1)ⁿ",   "No",           "N/A",       "No",        "Yes",       "Multi-class out"],
        ]
    )
    print()
    _pause()

    section_header("3. PLOTEXT — All Activations from -5 to 5")
    try:
        xs = np.linspace(-5, 5, 80)
        funcs = [
            ("Sigmoid", _sigmoid(xs)),
            ("Tanh",    _tanh(xs)),
            ("ReLU",    _relu(xs)),
            ("GELU",    _gelu(xs)),
            ("Swish",   _swish(xs)),
        ]
        for name, ys in funcs:
            loss_curve_plot(ys, title=f"{name}(x) for x ∈ [-5, 5]")
    except Exception:
        print(grey("  plotext unavailable — see ASCII comparison below"))
    print()
    _pause()

    section_header("4. ASCII — All Activations at Key Points")
    xs_key = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(bold_cyan(f"  {'Function':<12} {'x=-3':>8} {'x=-1':>8} {'x=0':>8} {'x=1':>8} {'x=3':>8}"))
    print(grey("  " + "─"*55))
    functions = [
        ("Sigmoid",    _sigmoid),
        ("Tanh",       _tanh),
        ("ReLU",       _relu),
        ("Leaky(0.01)",lambda x: _leaky_relu(x, 0.01)),
        ("ELU(1)",     lambda x: _elu(x, 1.0)),
        ("GELU",       _gelu),
        ("Swish",      _swish),
    ]
    for name, fn in functions:
        vals = [fn(x) for x in xs_key]
        vals_str = "".join(f"{v:>8.3f}" for v in vals)
        print(f"  {name:<12} {vals_str}")
    print()
    _pause()

    section_header("5. PLOTEXT — All Derivatives")
    try:
        xs = np.linspace(-4, 4, 80)
        eps = 1e-5
        derivs = [
            ("Sigmoid'",  (_sigmoid(xs+eps) - _sigmoid(xs-eps)) / (2*eps)),
            ("Tanh'",     (_tanh(xs+eps)    - _tanh(xs-eps))    / (2*eps)),
            ("ReLU'",     (xs > 0).astype(float)),
            ("GELU'",     (_gelu(xs+eps)    - _gelu(xs-eps))    / (2*eps)),
            ("Swish'",    (_swish(xs+eps)   - _swish(xs-eps))   / (2*eps)),
        ]
        for name, dy in derivs:
            loss_curve_plot(dy, title=f"{name} for x ∈ [-4, 4]")
    except Exception:
        print(grey("  plotext unavailable"))
    print()
    _pause()

    section_header("6. RECOMMENDATIONS")
    print()
    recs = [
        ("Task",  "Recommendation",  "Reason"),
        ["Hidden (CNN/MLP)",    "ReLU",          "Fast, no saturation, standard baseline"],
        ["Hidden (Transformer)","GELU",           "Used in GPT, BERT, T5 — smoother than ReLU"],
        ["Hidden (EfficientNet)","Swish/SiLU",    "Consistently +0.1-0.3% over ReLU on ImageNet"],
        ["Hidden (deep MLP)",  "SELU",            "Self-normalising, no BN needed"],
        ["Binary output",      "Sigmoid",         "Outputs probability ∈ (0,1)"],
        ["Multi-class output", "Softmax",         "Outputs sum-to-1 probability vector"],
        ["Regression output",  "None (linear)",   "Unbounded predictions"],
        ["RNN/LSTM hidden",    "Tanh",            "Zero-centred, historical convention"],
    ]
    table(recs[0], recs[1:])
    print()
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Comparing All Activations", """
import numpy as np

activations = {
    'sigmoid':  lambda x: 1/(1+np.exp(-x)),
    'tanh':     np.tanh,
    'relu':     lambda x: np.maximum(0, x),
    'leaky':    lambda x: np.where(x>=0, x, 0.01*x),
    'elu':      lambda x: np.where(x>=0, x, np.exp(np.clip(x,-100,0))-1),
    'gelu':     lambda x: x / (1+np.exp(-1.702*x)),
    'swish':    lambda x: x / (1+np.exp(-x)),
}

x = np.linspace(-5, 5, 100)

for name, fn in activations.items():
    y = fn(x)
    print(f"{name:8s}: range=[{y.min():.3f}, {y.max():.3f}], "
          f"mean={y.mean():+.3f}, active={np.mean(y>0):.1%}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    for ins in [
        "No single best activation — depends on architecture, depth, and task",
        "ReLU: fast and effective baseline for most hidden-layer applications",
        "GELU: standard for Transformers — smooth approximation to ReLU+dropout",
        "Sigmoid/Tanh: avoid in deep hidden layers; use at output or in gates",
        "Swish outperforms ReLU on ImageNet by small margin; used in EfficientNet",
        "Softmax + log(softmax) numerically: always use log-sum-exp trick",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Sigmoid",             topic_sigmoid),
        ("Tanh",                topic_tanh),
        ("ReLU",                topic_relu),
        ("Leaky ReLU Variants", topic_leaky_variants),
        ("GELU",                topic_gelu),
        ("Swish / SiLU",        topic_swish),
        ("Softmax",             topic_softmax),
        ("Comparison",          topic_comparison),
    ]
    block_menu("b09", "Activation Functions", topics)
