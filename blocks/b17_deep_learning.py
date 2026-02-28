"""
Block 17: Deep Learning Mathematics
Covers: Universal Approximation, Weight Init, BatchNorm, Attention, Transformers
"""
import numpy as np
import math


def run():
    topics = [
        ("Universal Approximation Theorem", universal_approximation),
        ("Weight Initialization",           weight_initialization),
        ("Batch Normalization",             batch_normalization),
        ("Layer Normalization",             layer_normalization),
        ("Dropout Mathematics",             dropout_math),
        ("Self-Attention Mechanism",        self_attention),
        ("Multi-Head Attention",            multi_head_attention),
        ("Transformer Architecture",        transformer_arch),
        ("Positional Encoding",             positional_encoding),
        ("Residual Connections",            residual_connections),
        ("Loss Functions Zoo",              loss_functions),
    ]
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║      BLOCK 17 — DEEP LEARNING MATHEMATICS        ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m")
        print("\033[90mmmlmath > Block 17 > Deep Learning\033[0m\n")
        for i, (name, _) in enumerate(topics, 1):
            print(f"  \033[93m{i:2d}.\033[0m {name}")
        print("\n  \033[90m[0] Back to main menu\033[0m")
        choice = input("\n\033[96mSelect topic: \033[0m").strip()
        if choice == "0":
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(topics):
                topics[idx][1]()
        except (ValueError, IndexError):
            print("\033[91mInvalid choice.\033[0m")


# ─────────────────────────────────────────────────────────────
def universal_approximation():
    print("\n\033[95m━━━ Universal Approximation Theorem ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Universal Approximation\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
A feedforward neural network with at least ONE hidden layer containing
sufficiently many neurons can approximate ANY continuous function on a
compact subset of ℝⁿ to arbitrary precision.

This theorem (Cybenko 1989, Hornik 1991) is why neural nets are called
"universal function approximators." The catch: it says NOTHING about
how to find the weights; it only guarantees existence.

In ML practice this means: if your network can't fit the training data,
the problem is optimization (finding weights), not capacity. A wider/
deeper network always has enough representational power.

Depth vs Width: Wide shallow networks CAN approximate functions, but
deep networks do so exponentially more efficiently in terms of neurons.
A function needing O(2^n) neurons in a 2-layer network might need only
O(n²) neurons with depth.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Theorem (Cybenko): Let σ be any continuous sigmoidal function.
  For any f ∈ C([0,1]ⁿ) and ε > 0, there exist N, aᵢ, bᵢ ∈ ℝ, wᵢ ∈ ℝⁿ:

    F(x) = Σᵢ₌₁ᴺ aᵢ · σ(wᵢᵀx + bᵢ)   such that   |F(x) − f(x)| < ε

  Deep network output (L layers):
    h⁽⁰⁾ = x
    h⁽ˡ⁾ = σ(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
    ŷ   = W⁽ᴸ⁾ h⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾

  Depth efficiency: depth-L net can represent functions requiring
  O(2ⁿ) neurons in depth-2. Depth multiplies representational power.
""")

    print("\033[93mSTEP-BY-STEP EXAMPLE\033[0m")
    print("""
  Approximating f(x) = sin(x) on [0, 2π] with 1 hidden layer:

  [1] Choose N = 10 neurons with sigmoid activations
  [2] For each neuron i: compute σ(wᵢx + bᵢ) — a shifted/scaled S-curve
  [3] Linear combination: F(x) = Σ aᵢ σ(wᵢx + bᵢ)
  [4] Adjust aᵢ, wᵢ, bᵢ via gradient descent to minimize ∫(F-f)²dx
  [5] As N → ∞, max|F(x) - sin(x)| → 0
""")

    # ASCII visualization
    print("\033[93mASCII VISUALIZATION — Approximating sin(x)\033[0m")
    width, height = 60, 12
    x_vals = [i * 2 * math.pi / (width - 1) for i in range(width)]
    y_true = [math.sin(x) for x in x_vals]
    # Rough approximation with 3 sigmoids
    def sigmoid(z): return 1 / (1 + math.exp(-z))
    y_approx = [2.0 * sigmoid(4 * x - 3) - 2.0 * sigmoid(4 * x - 9) +
                0.5 * sigmoid(2 * x - 5) - 0.5 for x in x_vals]

    print("  \033[90mTrue sin(x) ─── Approx ···\033[0m")
    grid = [[' '] * width for _ in range(height)]
    for i, (yt, ya) in enumerate(zip(y_true, y_approx)):
        row_t = int((1 - yt) / 2 * (height - 1))
        row_a = int((1 - ya) / 2 * (height - 1))
        if 0 <= row_t < height:
            grid[row_t][i] = '\033[96m─\033[0m'
        if 0 <= row_a < height:
            grid[row_a][i] = '\033[93m·\033[0m'
    for row in grid:
        print("  " + "".join(row))
    print()

    # plotext
    try:
        import plotext as plt
        plt.clf()
        x_p = list(np.linspace(0, 2 * math.pi, 100))
        y_p = list(np.sin(x_p))
        plt.plot(x_p, y_p, label="sin(x)")
        plt.title("Universal Approximation Target: sin(x)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for terminal plots: pip install plotext]\033[0m")

    # matplotlib
    try:
        import matplotlib.pyplot as plt2
        x_m = np.linspace(0, 2 * math.pi, 200)
        def _net_approx(x, n_neurons=20):
            np.random.seed(42)
            w = np.random.randn(n_neurons) * 3
            b_arr = np.random.randn(n_neurons)
            a = np.random.randn(n_neurons) * 0.5
            return sum(a[i] / (1 + np.exp(-(w[i] * x + b_arr[i]))) for i in range(n_neurons))
        fig, axes = plt2.subplots(1, 2, figsize=(12, 4))
        for n, ax in zip([5, 20], axes):
            y_hat = np.array([_net_approx(xi, n) for xi in x_m])
            ax.plot(x_m, np.sin(x_m), 'b-', label='True sin(x)', lw=2)
            ax.plot(x_m, y_hat, 'r--', label=f'{n}-neuron net', lw=2)
            ax.legend(); ax.set_title(f'N={n} Neurons')
            ax.set_xlabel('x')
        plt2.suptitle('Universal Approximation of sin(x)')
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for rich plots: pip install matplotlib]\033[0m")

    print("\033[93mCODE SNIPPET\033[0m")
    print("""\033[94m
  import numpy as np

  def sigmoid(z):
      return 1 / (1 + np.exp(-z))

  class ShallowNet:
      def __init__(self, n_neurons=50, seed=42):
          np.random.seed(seed)
          self.W = np.random.randn(n_neurons)
          self.b = np.random.randn(n_neurons)
          self.a = np.random.randn(n_neurons)

      def forward(self, x):
          # Each neuron: sigmoid activation at shifted x
          h = sigmoid(np.outer(x, self.W) + self.b)  # (N, n_neurons)
          return h @ self.a  # linear combination

  x = np.linspace(0, 2*np.pi, 200)
  net = ShallowNet(n_neurons=50)
  y_pred = net.forward(x)
  # Train with gradient descent to approximate sin(x)
\033[0m""")

    print("\033[93mEXERCISE\033[0m")
    print("  Build a 2-layer network (5 hidden neurons) to approximate f(x) = x²")
    print("  on [-2, 2] using gradient descent. Report final MSE.")
    ans = input("  [h]int  [s]olution  [enter] to skip: ").strip().lower()
    if ans == 'h':
        print("  Hint: MSE = mean((f(x) - ŷ)²), gradient = -2(f-ŷ)·∂ŷ/∂W")
    elif ans == 's':
        print("""\033[94m
  import numpy as np
  x = np.linspace(-2, 2, 100); y = x**2
  np.random.seed(0)
  W1 = np.random.randn(5)*0.1; b1 = np.zeros(5)
  W2 = np.random.randn(5)*0.1; b2 = 0.0
  lr = 0.01
  for _ in range(5000):
      h = np.maximum(0, np.outer(x, W1) + b1)  # ReLU
      yhat = h @ W2 + b2
      err = yhat - y
      dW2 = h.T @ err / len(x)
      db2 = err.mean()
      dh  = np.outer(err, W2) * (h > 0)
      dW1 = np.outer(x, dh.mean(axis=0))  # simplified
      W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1[0]*len(x)
  print(f"MSE = {np.mean(err**2):.4f}")
\033[0m""")

    print("\033[93mKEY INSIGHTS\033[0m")
    print("""  • Width gives capacity; depth gives efficiency — prefer deep and narrow
  • UAT says EXISTENCE; gradient descent must FIND those weights
  • Activation functions MUST be non-polynomial for universality
  • Real power: deep networks learn hierarchical representations
  • Double descent: more capacity can REDUCE test error (modern insight)
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def weight_initialization():
    print("\n\033[95m━━━ Weight Initialization ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Weight Initialization\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Weight initialization is one of the most practically impactful but
underappreciated aspects of deep learning. If weights start too large,
activations explode; too small and they vanish — both prevent learning.

The goal is to keep the variance of activations and gradients roughly
constant across layers. Xavier/Glorot (2010) derived the optimal
variance for tanh networks; He/Kaiming (2015) corrected it for ReLU.

Good initialization doesn't guarantee fast training but BAD
initialization almost certainly causes slow or failed training.
Training a 10-layer network with all-same weights is impossible due
to symmetry — all neurons learn the same thing (symmetry breaking).
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Problem: for layer l with nᵢₙ inputs and nₒᵤₜ outputs,
  if W ~ N(0,1), Var(output) = nᵢₙ · Var(W) → grows with nᵢₙ

  Xavier/Glorot (for tanh/sigmoid):
    Var(W) = 2 / (nᵢₙ + nₒᵤₜ)
    W ~ Uniform(-√(6/(nᵢₙ+nₒᵤₜ)), +√(6/(nᵢₙ+nₒᵤₜ)))

  He/Kaiming (for ReLU — factor 2 because ReLU kills half):
    Var(W) = 2 / nᵢₙ
    W ~ N(0, √(2/nᵢₙ))

  Derivation for He init:
    Var(ReLU(z)) = ½ Var(z)   [ReLU zeros half the inputs]
    To keep Var(hˡ) = Var(hˡ⁻¹): nᵢₙ · Var(W) · ½ = 1
    ⟹ Var(W) = 2/nᵢₙ
""")

    print("\033[93mSTEP-BY-STEP VARIANCE ANALYSIS\033[0m")
    np.random.seed(42)
    n_layers = 10
    n_units = 256
    x_input = np.random.randn(1000, n_units)

    methods = {"Random N(0,1)": lambda ni, no: np.random.randn(ni, no),
               "Xavier":        lambda ni, no: np.random.randn(ni, no) * np.sqrt(2.0 / (ni + no)),
               "He":            lambda ni, no: np.random.randn(ni, no) * np.sqrt(2.0 / ni)}

    print(f"  Network: {n_layers} layers, {n_units} units each, ReLU activation")
    print(f"  {'Method':<18} {'L1 var':>10} {'L5 var':>10} {'L10 var':>10}")
    print("  " + "─" * 52)
    for name, init_fn in methods.items():
        h = x_input.copy()
        vars_log = []
        for _ in range(n_layers):
            W = init_fn(n_units, n_units)
            h = np.maximum(0, h @ W)
            vars_log.append(np.var(h))
        print(f"  {name:<18} {vars_log[0]:>10.4f} {vars_log[4]:>10.4f} {vars_log[9]:>10.4f}")
    print()

    # plotext
    try:
        import plotext as plt
        plt.clf()
        for name, init_fn in methods.items():
            h = np.random.randn(1000, n_units)
            v = []
            for _ in range(n_layers):
                W = init_fn(n_units, n_units)
                h = np.maximum(0, h @ W)
                v.append(float(np.var(h)))
            plt.plot(list(range(1, n_layers + 1)), v, label=name)
        plt.title("Activation Variance by Layer")
        plt.xlabel("Layer"); plt.ylabel("Variance")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for terminal plots]\033[0m")

    # matplotlib
    try:
        import matplotlib.pyplot as plt2
        fig, axes = plt2.subplots(1, 3, figsize=(14, 4))
        for ax, (name, init_fn) in zip(axes, methods.items()):
            h = np.random.randn(1000, n_units)
            v = []
            for _ in range(n_layers):
                W = init_fn(n_units, n_units)
                h = np.maximum(0, h @ W)
                v.append(float(np.var(h)))
            ax.plot(range(1, n_layers + 1), v, marker='o')
            ax.set_title(name)
            ax.set_xlabel("Layer"); ax.set_ylabel("Var(activations)")
        plt2.suptitle("Weight Initialization: Activation Variance Across Layers")
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for plots]\033[0m")

    print("\033[93mKEY INSIGHTS\033[0m")
    print("""  • All-zeros init → all neurons identical → symmetry, no learning
  • He init designed specifically for ReLU; Xavier for tanh/sigmoid
  • With BN, initialization matters less (but still affects speed)
  • Orthogonal init preserves gradient norms exactly for linear nets
  • LoRA/fine-tuning: start from pretrained, init new weights to zero
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def batch_normalization():
    print("\n\033[95m━━━ Batch Normalization ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Batch Normalization\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Batch Normalization (Ioffe & Szegedy 2015) normalizes the inputs to
each layer using the statistics of the current mini-batch, then
rescales with learnable parameters γ and β.

It fixes "internal covariate shift" — the constant change in the
distribution of layer inputs during training. More importantly, it:
(1) allows higher learning rates, (2) acts as regularizer, (3) makes
training dramatically more stable.

The backward pass through BN is non-trivial because μ and σ are
functions of the batch, creating complex gradient dependencies.
""")

    print("\033[93mFORMULAS — Forward Pass\033[0m")
    print("""
  Given mini-batch B = {x₁,...,xₘ}:

  μ_B  = (1/m) Σᵢ xᵢ                     [batch mean]
  σ²_B = (1/m) Σᵢ (xᵢ − μ_B)²            [batch variance]
  x̂ᵢ  = (xᵢ − μ_B) / √(σ²_B + ε)        [normalize]
  yᵢ  = γ x̂ᵢ + β                          [scale & shift]

  γ, β are learned per-feature. ε ≈ 1e-5 prevents division by zero.

  Forward Pass (backward shown in code):
  ∂L/∂γ = Σᵢ ∂L/∂yᵢ · x̂ᵢ
  ∂L/∂β = Σᵢ ∂L/∂yᵢ
""")

    print("\033[93mNUMERICAL EXAMPLE\033[0m")
    np.random.seed(0)
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    gamma, beta = 2.0, 1.0
    mu = x.mean(); sigma2 = x.var(); eps = 1e-5
    x_hat = (x - mu) / np.sqrt(sigma2 + eps)
    y_out = gamma * x_hat + beta
    print(f"  Input x:      {x}")
    print(f"  μ = {mu:.2f},  σ² = {sigma2:.2f}")
    print(f"  Normalized:   {np.round(x_hat, 3)}")
    print(f"  After BN(γ={gamma},β={beta}): {np.round(y_out, 3)}")

    # plotext — before/after distribution
    try:
        import plotext as plt
        np.random.seed(7)
        raw = np.random.randn(500) * 5 + 10
        mu_b = raw.mean(); std_b = raw.std()
        normed = (raw - mu_b) / (std_b + 1e-5)
        plt.clf()
        vals_r = list(np.sort(raw)); vals_n = list(np.sort(normed))
        plt.plot(vals_r, [np.random.uniform(0, 1) for _ in vals_r], label="Before BN", marker="dot")
        plt.plot(vals_n, [np.random.uniform(1, 2) for _ in vals_n], label="After BN", marker="dot")
        plt.title("Distribution Before vs After Batch Normalization")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for terminal plots]\033[0m")

    # matplotlib
    try:
        import matplotlib.pyplot as plt2
        np.random.seed(7)
        fig, axes = plt2.subplots(1, 2, figsize=(10, 4))
        raw = np.random.randn(500) * 5 + 10
        normed = (raw - raw.mean()) / (raw.std() + 1e-5)
        axes[0].hist(raw, bins=30, color='steelblue', alpha=0.7)
        axes[0].set_title("Before BatchNorm"); axes[0].set_xlabel("value")
        axes[1].hist(normed, bins=30, color='tomato', alpha=0.7)
        axes[1].set_title("After BatchNorm"); axes[1].set_xlabel("value")
        plt2.suptitle("Batch Normalization Effect on Distribution")
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for plots]\033[0m")

    print("\033[93mKEY INSIGHTS\033[0m")
    print("""  • BN uses BATCH statistics at train time; running stats at test time
  • Layer Norm (used in Transformers) normalizes across features, not batch
  • BN is redundant after certain modern improvements (like careful init + LR)
  • Batch size matters: very small batches → noisy statistics → worse performance
  • BN effectively prevents activations from saturating sigmoid/tanh
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def layer_normalization():
    print("\n\033[95m━━━ Layer Normalization ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Layer Normalization\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Layer Normalization (Ba et al. 2016) normalizes across the FEATURE
dimension for each sample independently, unlike BatchNorm which
normalizes across the BATCH dimension.

This makes LayerNorm independent of batch size — crucial for:
• Transformers (variable-length sequences)
• Recurrent networks (each timestep normalized differently)
• Online learning or batch size = 1

LayerNorm is the default normalization in BERT, GPT, T5, and all
modern transformer architectures.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  For a single sample x ∈ ℝᵈ:

  μ  = (1/d) Σⱼ xⱼ                  [mean over features]
  σ² = (1/d) Σⱼ (xⱼ − μ)²           [variance over features]
  x̂  = (x − μ) / √(σ² + ε)          [normalize]
  y  = γ ⊙ x̂ + β                    [per-feature scale & shift]

  BatchNorm:  μ across samples (batch axis)
  LayerNorm:  μ across features (feature axis)
  GroupNorm:  μ across feature groups
  InstanceNorm: μ per spatial location (style transfer)
""")

    print("\033[93mCOMPARISON TABLE\033[0m")
    rows = [
        ("Normalization", "Axis",          "Batch dep?", "Use case"),
        ("BatchNorm",     "batch",          "Yes",        "CNNs"),
        ("LayerNorm",     "features",       "No",         "Transformers"),
        ("InstanceNorm",  "spatial",        "No",         "Style transfer"),
        ("GroupNorm",     "feature groups", "No",         "Small batches"),
    ]
    print("  " + "─" * 65)
    for r in rows:
        print(f"  {r[0]:<18} {r[1]:<18} {r[2]:<12} {r[3]}")
    print("  " + "─" * 65)

    np.random.seed(0)
    x = np.random.randn(4)  # one sample, 4 features
    mu = x.mean(); sigma = x.std()
    x_hat = (x - mu) / (sigma + 1e-5)
    print(f"\n  Sample x:     {np.round(x, 3)}")
    print(f"  After LayerNorm: {np.round(x_hat, 3)}")
    print(f"  Mean≈0: {abs(x_hat.mean()) < 1e-4}, Std≈1: {abs(x_hat.std() - 1) < 0.01}\n")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def dropout_math():
    print("\n\033[95m━━━ Dropout Mathematics ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Dropout\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Dropout (Srivastava et al. 2014) randomly sets neurons to zero during
training with probability p (retain probability = 1-p). This prevents
co-adaptation of neurons and forces the network to learn redundant
representations.

Mathematically, each forward pass trains a different sub-network.
At test time, neurons are all active but scaled by (1-p) to maintain
the same expected activation.

Inverted dropout (the standard implementation) scales by 1/(1-p) at
TRAIN time, so test time needs NO scaling — cleaner code.

Dropout connects to: Bayesian approximation (Gal & Ghahramani 2016),
ensemble methods, and information bottleneck theory.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Standard dropout (train):
    mask mᵢ ~ Bernoulli(1 − p)
    h̃ᵢ = mᵢ · hᵢ
    E[h̃] = (1−p)·E[h]   ← need to scale at test time

  Inverted dropout (standard practice):
    mask mᵢ ~ Bernoulli(1 − p)
    h̃ᵢ = mᵢ · hᵢ / (1 − p)
    E[h̃] = E[h]         ← no test-time scaling needed

  Effective ensemble: with p=0.5, 2ⁿ different sub-networks
  Expected gradient for weight w connecting i → j:
    E[∂L/∂w] = (1−p) · ∂L_full/∂w
""")

    np.random.seed(42)
    h = np.array([0.5, 0.8, 0.3, 0.9, 0.6])
    p_drop = 0.5
    mask = np.random.binomial(1, 1 - p_drop, size=h.shape)
    h_dropped = mask * h / (1 - p_drop)
    print(f"\033[93mNUMERICAL\033[0m")
    print(f"  h (original):  {h}")
    print(f"  mask (p=0.5):  {mask}")
    print(f"  h̃ (inverted):  {np.round(h_dropped, 3)}")
    print(f"  E[h] ≈ {h.mean():.3f},  E[h̃] ≈ {h_dropped.mean():.3f}")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • p=0.1–0.5 for dense layers; p=0 for conv layers (usually)
  • Dropout disabled at inference (model.eval() in PyTorch)
  • MC Dropout: keep dropout at test time for uncertainty estimates
  • Modern alternatives: DropPath, DropBlock for spatial regularization
  • Transformers use attention dropout + residual dropout
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def self_attention():
    print("\n\033[95m━━━ Self-Attention Mechanism ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Self-Attention\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Self-attention (Vaswani et al. 2017) allows each position in a sequence
to attend to all other positions in one step — O(1) path length between
any two tokens vs O(n) for RNNs.

Every token is projected into 3 vectors:
  Q (query): "what am I looking for?"
  K (key):   "what do I contain?"
  V (value): "what do I pass forward if attended?"

Attention scores are computed as softmax(QKᵀ/√d_k)·V.
The √d_k scaling prevents the dot products from growing too large in
high dimensions, which would push softmax into saturation.

This mechanism is the foundation of BERT, GPT, T5, ViT and all modern
large models.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Input: X ∈ ℝⁿˣᵈ (n tokens, d features each)

  Q = X·Wᵩ,  K = X·Wₖ,  V = X·Wᵥ           [project to d_k, d_v dims]

  Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V

  Output shape: n × d_v

  Softmax: softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

  Why √d_k? If q,k ~ N(0,1), then q·k ~ N(0, d_k)
  Dividing by √d_k makes variance = 1 → stable softmax gradients
""")

    print("\033[93mSTEP-BY-STEP WITH REAL NUMBERS (3 tokens, d=4)\033[0m")
    np.random.seed(0)
    d = 4; n = 3
    X = np.random.randn(n, d)
    Wq = np.random.randn(d, d) * 0.1
    Wk = np.random.randn(d, d) * 0.1
    Wv = np.random.randn(d, d) * 0.1
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    scores = Q @ K.T / np.sqrt(d)
    def softmax_fn(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    A = softmax_fn(scores)
    out = A @ V
    print(f"  Q shape: {Q.shape},  K shape: {K.shape},  V shape: {V.shape}")
    print(f"\n  Attention scores (QKᵀ/√d):\n{np.round(scores, 3)}")
    print(f"\n  Attention weights (softmax):\n{np.round(A, 3)}")
    print(f"\n  Output:\n{np.round(out, 3)}")

    # matplotlib heatmap
    try:
        import matplotlib.pyplot as plt2
        fig, axes = plt2.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(scores, cmap='Blues', aspect='auto')
        axes[0].set_title("Attention Scores (raw)")
        axes[0].set_xlabel("Key position"); axes[0].set_ylabel("Query position")
        plt2.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(A, cmap='Oranges', aspect='auto')
        axes[1].set_title("Attention Weights (softmax)")
        axes[1].set_xlabel("Key position"); axes[1].set_ylabel("Query position")
        plt2.colorbar(im1, ax=axes[1])
        plt2.suptitle("Self-Attention (3 tokens, d=4)")
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for attention heatmap]\033[0m")

    print("\033[93mKEY INSIGHTS\033[0m")
    print("""  • Self-attention is O(n²·d) — quadratic in sequence length
  • Masked attention (decoder): prevent attending to future tokens
  • √d_k matters: without it, softmax saturates and gradients vanish
  • Attention weights are interpretable: show which tokens attend to which
  • Flash Attention: reorders computation to avoid O(n²) memory
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def multi_head_attention():
    print("\n\033[95m━━━ Multi-Head Attention ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Multi-Head Attention\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Multi-head attention runs h independent attention "heads" in parallel,
each with its own Q,K,V projection matrices. This lets the model
simultaneously attend to information from different representation
subspaces.

Think of it as: head 1 might attend to syntactic relationships, head 2
to semantic similarity, head 3 to positional patterns — all learned
automatically from data.

The outputs are concatenated and projected back to d_model dimensions.
With h heads of dimension d_k = d_model/h, the total compute is the
same as single-head attention.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  headᵢ = Attention(X·Wᵩᵢ, X·Wₖᵢ, X·Wᵥᵢ)
  MultiHead(X) = Concat(head₁,...,headₕ) · W_O

  Parameter count:
    Each head: Wᵩᵢ ∈ ℝᵈˣᵈᵏ, Wₖᵢ ∈ ℝᵈˣᵈᵏ, Wᵥᵢ ∈ ℝᵈˣᵈᵛ
    W_O: ∈ ℝʰᵈᵛˣᵈ
    Total: 3·d·d_k·h + h·d_v·d = 4·d²  [same as single head if d_k=d/h]

  GPT-3: d_model=12288, h=96, d_k=128
""")

    np.random.seed(1)
    d_model, h, n = 8, 2, 4
    d_k = d_model // h
    X = np.random.randn(n, d_model)
    heads = []
    for _ in range(h):
        Wq = np.random.randn(d_model, d_k) * 0.1
        Wk = np.random.randn(d_model, d_k) * 0.1
        Wv = np.random.randn(d_model, d_k) * 0.1
        Q = X @ Wq; K = X @ Wk; V = X @ Wv
        scores = Q @ K.T / np.sqrt(d_k)
        def softmax_fn(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        A = softmax_fn(scores)
        heads.append(A @ V)
    multi = np.concatenate(heads, axis=-1)
    Wo = np.random.randn(d_model, d_model) * 0.1
    out = multi @ Wo
    print(f"\n  Input X: {X.shape}")
    print(f"  Each head output: {heads[0].shape}")
    print(f"  Concatenated: {multi.shape} → Final: {out.shape}")

    print("\n\033[93mASCII: Multi-Head Attention Architecture\033[0m")
    print("""
   Input X ─────────────────────────────────────────
         │               │               │
      Head 1          Head 2   ...    Head h
    [Wq1,Wk1,Wv1]  [Wq2,Wk2,Wv2]  [Wqh,Wkh,Wvh]
         │               │               │
    Attn(Q1,K1,V1)  Attn(Q2,K2,V2)  Attn(Qh,Kh,Vh)
         │               │               │
         └───────────────┴───── concat ──┘
                                 │
                            × W_output
                                 │
                           Output (n×d_model)
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def transformer_arch():
    print("\n\033[95m━━━ Transformer Architecture ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Transformer\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
The Transformer (Vaswani et al. 2017) replaced RNNs for sequence
modeling using only attention and feedforward layers.

Encoder block: processes input in parallel (used in BERT)
Decoder block: auto-regressive generation (used in GPT)
Full Transformer: encoder + decoder (used in T5, original MT)

Key innovations: multi-head attention, positional encoding,
residual connections around every sub-layer, layer normalization.
""")

    print("\033[93mARCHITECTURE ASCII\033[0m")
    print("""
  ENCODER BLOCK                    DECODER BLOCK
  ─────────────────                ────────────────
  Input Embedding + PE             Output Embedding + PE
       │                                   │
  ┌────┴─────────┐               ┌─────────┴──────────┐
  │ Multi-Head   │               │ Masked Multi-Head  │
  │  Attention   │               │    Attention       │
  └────┬─────────┘               └─────────┬──────────┘
  Add & LayerNorm                     Add & LayerNorm
       │                                   │
  ┌────┴─────────┐               ┌─────────┴──────────┐
  │  Feed       │               │  Cross-Attention   │
  │  Forward    │               │  (enc K,V; dec Q)  │
  └────┬─────────┘               └─────────┬──────────┘
  Add & LayerNorm                     Add & LayerNorm
       │                                   │
  (repeated N times)             ┌─────────┴──────────┐
       │                         │  Feed Forward      │
  Encoder Output ─────────────→  └─────────┬──────────┘
                                       Add & LayerNorm
                                            │
                                      Linear + Softmax
                                            │
                                       Output tokens
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Feed Forward Network (FFN):
    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    d_ff = 4 · d_model  (standard expansion ratio)

  Full Encoder Layer:
    z = LayerNorm(x + MultiHead(x))    [attention sub-layer]
    y = LayerNorm(z + FFN(z))          [ffn sub-layer]

  Parameter count (1 layer, d_model=512, h=8, d_ff=2048):
    Attention: 4·512² = 1,048,576
    FFN:       2·512·2048 = 2,097,152
    LayerNorms: 2·2·512 = 2,048
    Total/layer ≈ 3.1M params

  GPT-3: 96 layers × ~37M/layer = 175B total parameters
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def positional_encoding():
    print("\n\033[95m━━━ Positional Encoding ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Positional Encoding\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Self-attention is PERMUTATION INVARIANT — if you shuffle the tokens,
you get the same output (just shuffled). This means the model can't
tell position 1 from position 100.

Positional encoding injects positional information by adding a
position-dependent vector to each token embedding.

Sinusoidal PE (Vaswani et al.): uses sin/cos of different frequencies.
Properties: unique for each position, bounded in [-1,1], allows the
model to learn relative positions via linear transformations.

RoPE (Rotary Position Embedding, Su et al. 2021): Rotates the Q,K
vectors by position-dependent angle. Used in LLaMA, PaLM, GPT-NeoX.
Better extrapolation to longer sequences.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Sinusoidal PE for position pos, dimension i:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

  Key property: PE(pos + k) can be written as linear transform of PE(pos)
  because: sin(a+b) = sin(a)cos(b) + cos(a)sin(b)

  RoPE: rotate query/key by angle pos · θᵢ where θᵢ = 10000^(-2i/d):
    Rot(x, pos) = x · e^(i·pos·θ)
""")

    d_model = 16; max_len = 20
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            pe[pos, i]   = math.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i+1] = math.cos(pos / (10000 ** (i / d_model)))

    print("\n\033[93msinusoidal PE (first 20 positions, d=16)\033[0m")
    print("  pos │ " + "  ".join(f"d{i:<2}" for i in range(0, d_model, 2)))
    print("  " + "─" * 70)
    for pos in range(min(6, max_len)):
        vals = "  ".join(f"{pe[pos, i]:+.2f}" for i in range(0, d_model, 2))
        print(f"  {pos:3d} │ {vals}")

    # matplotlib heatmap
    try:
        import matplotlib.pyplot as plt2
        fig, ax = plt2.subplots(figsize=(12, 5))
        im = ax.imshow(pe.T, aspect='auto', cmap='RdBu')
        ax.set_xlabel("Position"); ax.set_ylabel("Dimension")
        ax.set_title("Sinusoidal Positional Encoding Heatmap")
        plt2.colorbar(im, ax=ax)
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for PE heatmap]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def residual_connections():
    print("\n\033[95m━━━ Residual Connections ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Residual Connections\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
Residual (skip) connections (He et al. 2015) add the input of a layer
directly to its output: y = F(x) + x

This creates a "gradient highway" — gradients can flow directly from
loss to early layers without passing through any non-linearities.

Without residuals: in a 100-layer network, gradients must pass through
100 jacobians, and their product easily vanishes or explodes.

With residuals: ∂L/∂x = ∂L/∂y · (∂F/∂x + I)
The identity I ensures gradients are at LEAST as large as ∂L/∂y.
This is why ResNets can train 1000+ layer networks.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  Residual block:  y = F(x, {Wᵢ}) + x
  Gradient:        ∂L/∂x = ∂L/∂y · (∂F/∂x + I)

  Pre-LN (modern): LayerNorm → Sublayer → (+x)
  Post-LN (original): Sublayer → (+x) → LayerNorm

  Gradient flow through L residual blocks:
    ∂L/∂x = ∂L/∂xₗ · Π_{i=k}^{l} (I + ∂F_i/∂xᵢ)
    Even if each ∂F/∂x = 0, product = I (identity) ← no vanishing!
""")

    print("\033[93mASCII DIAGRAM\033[0m")
    print("""
  without residual:          with residual:
  ──Input──→[Layer]──→       ──Input──┬──→[Layer]──→─┐
                                      │               ↓
                                      └──────────────(+)──→Output
                             gradient flows both paths!
""")

    # plotext: gradient magnitude comparison
    try:
        import plotext as plt
        plt.clf()
        n_layers = 30
        grad_plain = [0.95 ** i for i in range(n_layers)]
        grad_residual = [max(0.9 ** i * 0.5 + 0.5, grad_plain[i]) for i in range(n_layers)]
        plt.plot(list(range(n_layers)), grad_plain, label="No Residual")
        plt.plot(list(range(n_layers)), grad_residual, label="With Residual")
        plt.title("Gradient Magnitude vs Layer Depth")
        plt.xlabel("Layer"); plt.ylabel("|gradient|")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for gradient plot]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • Residuals allow training networks 10-1000× deeper than without
  • Pre-LN (norm first) is now preferred for stability
  • Dense connections (DenseNet): each layer gets ALL previous outputs
  • Highway networks: learned gating instead of fixed identity
  • In Transformers: both attention and FFN sub-layers use residuals
""")
    input("\033[90m[Enter to continue]\033[0m")


# ─────────────────────────────────────────────────────────────
def loss_functions():
    print("\n\033[95m━━━ Loss Functions Zoo ━━━\033[0m")
    print("\033[90mmmlmath > Block 17 > Loss Functions\033[0m\n")

    print("\033[1mTHEORY\033[0m")
    print("""
The loss function defines WHAT the model is optimizing — wrong choice
leads to wrong behavior even with a perfect architecture.

Binary Cross-Entropy (BCE): for binary classification
Categorical Cross-Entropy (CCE): for multi-class
MSE: for regression (sensitive to outliers)
MAE (L1): for regression (robust to outliers)
Huber: smooth version, best of L1 and L2
KL Divergence: for distribution matching (VAEs, distillation)
Focal Loss: for imbalanced classes (object detection)
Contrastive: for metric learning (pairs/triplets)
""")

    losses = [
        ("BCE",        "−[y·log(p) + (1-y)·log(1-p)]",       "Binary classification"),
        ("CCE",        "−Σᵢ yᵢ log(pᵢ)",                      "Multi-class"),
        ("MSE",        "(1/n) Σ(y−ŷ)²",                      "Regression"),
        ("MAE",        "(1/n) Σ|y−ŷ|",                       "Robust regression"),
        ("Huber",      "½(y−ŷ)² if |e|<δ, else δ|e|−½δ²",   "Robust regression"),
        ("KL Div",     "Σ P(x)log(P(x)/Q(x))",               "Distribution matching"),
        ("Focal",      "−α(1-p)^γ log(p)",                    "Class imbalance"),
        ("Hinge",      "max(0, 1 − y·ŷ)",                    "SVM / margin-based"),
        ("Triplet",    "max(0, d(a,p) − d(a,n) + margin)",   "Metric learning"),
        ("NLL",        "−log P(y|x,θ)",                       "Probabilistic models"),
    ]
    print("\033[93m  Loss Function Table\033[0m")
    print("  " + "─" * 78)
    print(f"  {'Name':<12} {'Formula':<42} {'Use Case'}")
    print("  " + "─" * 78)
    for row in losses:
        print(f"  {row[0]:<12} {row[1]:<42} {row[2]}")
    print("  " + "─" * 78)

    # Numerical examples
    print("\n\033[93mNUMERICAL EXAMPLES\033[0m")
    y_true = np.array([1, 0, 1, 1, 0], dtype=float)
    y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.3])
    bce = -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"  y_true: {y_true},  y_pred: {y_pred}")
    print(f"  BCE = {bce:.4f},  MSE = {mse:.4f},  MAE = {mae:.4f}")

    # plotext
    try:
        import plotext as plt
        plt.clf()
        x = list(np.linspace(-3, 3, 100))
        mse_l = [xi ** 2 for xi in x]
        mae_l = [abs(xi) for xi in x]
        huber_l = [0.5 * xi**2 if abs(xi) < 1 else abs(xi) - 0.5 for xi in x]
        plt.plot(x, mse_l, label="MSE")
        plt.plot(x, mae_l, label="MAE")
        plt.plot(x, huber_l, label="Huber")
        plt.title("Regression Loss Functions vs Residual")
        plt.xlabel("y - ŷ (residual)"); plt.ylabel("Loss")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for loss plot]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • MSE gradient grows linearly with error — outliers dominate
  • MAE gradient is always ±1 — slow near zero, robust far out
  • BCE = NLL of Bernoulli dist; CCE = NLL of Categorical — probabilistic view
  • Focal loss down-weights easy examples, focuses on hard ones
  • Loss choice implies an assumed noise model: MSE←Gaussian, MAE←Laplacian
""")
    input("\033[90m[Enter to continue]\033[0m")
