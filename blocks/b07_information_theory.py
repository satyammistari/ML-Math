"""
blocks/b07_information_theory.py
Block 7: Information Theory for Machine Learning
Topics: Entropy, KL Divergence, Cross-Entropy, Mutual Information,
        JS Divergence, Channel Capacity, MDL, Information Bottleneck.
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
# TOPIC 1 — Shannon Entropy
# ══════════════════════════════════════════════════════════════════════════════
def topic_entropy():
    clear()
    breadcrumb("mlmath", "Information Theory", "Shannon Entropy")
    section_header("SHANNON ENTROPY")
    print()

    section_header("1. THEORY")
    print(white("""
  Shannon entropy H(X) measures the average UNCERTAINTY (or information content)
  of a random variable X:

      H(X) = -Σₓ p(x) log p(x)

  UNITS depend on the log base:
    log₂ → bits (binary, natural for coin flips)
    logₑ → nats (natural log, convenient for calculus)
    log₁₀ → dits / Hartleys

  INTUITION: a fair coin (p=0.5) has H = 1 bit — maximum surprise per flip.
  A biased coin (p=0.99) has H ≈ 0.08 bits — almost no surprise, we know the outcome.

  MAXIMUM ENTROPY PRINCIPLE:
  Among all distributions over k outcomes, the UNIFORM distribution maximises
  entropy: H_max = log k.  This is why uniform priors appear "maximally ignorant".
  Any other distribution encodes some structure (reduces uncertainty).

  ENTROPY AS COMPRESSION:
  Shannon proved that the optimal lossless compression code for source X requires
  on average H(X) bits per symbol — no code can do better (Source Coding Theorem).
  This links entropy directly to the limits of data compression.

  In machine learning, cross-entropy loss for classification is exactly the
  negative log-likelihood under the predicted distribution, which equals
  H(y_true, y_pred) = H(y_true) + KL(y_true || y_pred).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Entropy (bits):  H(X) = -Σₓ p(x) log₂ p(x)"))
    print(formula("  Entropy (nats):  H(X) = -Σₓ p(x) ln p(x)"))
    print(formula("  Maximum:         H_max = log k  (uniform over k outcomes)"))
    print(formula("  Bernoulli:       H(p) = -p log p - (1-p) log(1-p)"))
    print(formula("  Joint:           H(X,Y) = -Σ p(x,y) log p(x,y)"))
    print(formula("  Conditional:     H(X|Y) = H(X,Y) - H(Y)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Coin Flips and Loaded Die")
    def h_bits(probs):
        probs = np.array(probs)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    distributions = [
        ("Fair coin",       [0.5, 0.5]),
        ("Loaded coin p=.9",[0.9, 0.1]),
        ("Certain (p=1)",   [1.0]),
        ("Fair 6-die",      [1/6]*6),
        ("Loaded die",      [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
    ]
    print(bold_cyan(f"  {'Distribution':<22} {'Entropy (bits)':<18} {'Max possible':<15} {'Efficiency'}"))
    print(grey("  " + "─"*70))
    for name, probs in distributions:
        h = h_bits(probs)
        h_max = np.log2(len(probs))
        eff = h / h_max if h_max > 0 else 1.0
        bar = "█" * int(eff * 15)
        print(f"  {name:<22} {h:<18.4f} {h_max:<15.4f} {cyan(bar)}")
    print()
    _pause()

    section_header("4. ASCII BAR — Distribution Shapes")
    print(cyan("  Fair coin:           ██████████  ██████████"))
    print(cyan("  Loaded coin p=0.9:   ██████████████████  ██"))
    print(cyan("  Fair die:            ██  ██  ██  ██  ██  ██"))
    print(cyan("  Loaded die:          ██████████  ████  ██  ██  █  █"))
    print(grey("  Flatter distribution → higher entropy"))
    print()
    _pause()

    section_header("5. PLOTEXT — Bernoulli Entropy vs p")
    try:
        ps = np.linspace(0.001, 0.999, 80)
        hs = -(ps * np.log2(ps) + (1 - ps) * np.log2(1 - ps))
        loss_curve_plot(hs, title="Bernoulli entropy H(p) vs p ∈ [0,1]  (peak at p=0.5)")
    except Exception:
        print(grey("  plotext unavailable"))
        ps = np.linspace(0.05, 0.95, 15)
        for p in ps:
            h = -(p*np.log2(p) + (1-p)*np.log2(1-p))
            bar = "█" * int(h * 20)
            print(f"  p={p:.2f}  {cyan(bar)}  H={h:.4f}")
    print()
    _pause()

    section_header("6. PYTHON CODE")
    code_block("Shannon Entropy", """
import numpy as np

def entropy(probs, base=2):
    probs = np.array(probs)
    probs = probs[probs > 0]
    log_fn = np.log2 if base == 2 else np.log
    return -np.sum(probs * log_fn(probs))

# Fair coin: H = 1 bit
print(f"H(fair coin)       = {entropy([0.5, 0.5]):.4f} bits")

# Loaded coin
print(f"H(loaded p=0.9)    = {entropy([0.9, 0.1]):.4f} bits")

# Fair 6-sided die: H = log2(6)
print(f"H(fair die)        = {entropy([1/6]*6):.4f} bits = log2(6)={np.log2(6):.4f}")

# Bernoulli entropy function
ps = np.linspace(0.01, 0.99, 99)
hs = [entropy([p, 1-p]) for p in ps]
print(f"Max entropy at p={ps[np.argmax(hs)]:.2f}: H={max(hs):.4f}")
""")
    _pause()

    section_header("7. KEY INSIGHTS")
    for ins in [
        "H(X) = 0 when outcome is certain; H(X) = log k when uniform over k outcomes",
        "Entropy quantifies average information = average surprise per observation",
        "Shannon's source coding theorem: H(X) bits/symbol is compression limit",
        "Cross-entropy loss in ML = H(y_true) + KL(y_true || y_pred)",
        "Higher entropy = less predictable = harder to classify",
        "Maximum entropy principle: uniform distribution when you know only the support",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — KL Divergence
# ══════════════════════════════════════════════════════════════════════════════
def topic_kl_divergence():
    clear()
    breadcrumb("mlmath", "Information Theory", "KL Divergence")
    section_header("KL DIVERGENCE (RELATIVE ENTROPY)")
    print()

    section_header("1. THEORY")
    print(white("""
  KL divergence (Kullback-Leibler) measures how much distribution P differs
  from a reference distribution Q:

      KL(P ‖ Q) = Σₓ P(x) log[P(x)/Q(x)]

  INTERPRETATION: average number of extra bits needed to encode samples
  from P using a code optimised for Q instead of P.

  NON-NEGATIVITY (Gibbs' inequality):
  KL(P‖Q) ≥ 0, with equality iff P = Q.
  PROOF: By Jensen's inequality applied to the convex function -log:
    -KL(P‖Q) = Σ P(x) log[Q(x)/P(x)] ≤ log Σ P(x)·[Q(x)/P(x)] = log 1 = 0
  Therefore KL(P‖Q) ≥ 0.  □

  ASYMMETRY: KL(P‖Q) ≠ KL(Q‖P) in general.
  • KL(P‖Q): "forward" KL — penalises missing mass (when Q(x)=0 but P(x)>0)
             → mode-seeking (variational inference uses this)
  • KL(Q‖P): "reverse" KL — penalises extra mass
             → mean-seeking (moment matching)

  CONNECTION TO MLE: maximising likelihood is equivalent to minimising
  KL(P_data ‖ P_θ), the KL from the empirical distribution to the model.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  KL(P‖Q) = Σₓ P(x) log[P(x)/Q(x)]   (discrete)"))
    print(formula("  KL(P‖Q) = ∫ p(x) log[p(x)/q(x)] dx  (continuous)"))
    print(formula("  KL(P‖Q) = H(P,Q) - H(P)             (cross-entropy - entropy)"))
    print(formula("  KL ≥ 0, equality ⟺ P = Q            (Gibbs' inequality)"))
    print(formula("  KL(P‖Q) ≠ KL(Q‖P)                   (asymmetric!)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Asymmetry of KL")
    def kl_discrete(P, Q):
        P, Q = np.array(P, dtype=float), np.array(Q, dtype=float)
        mask = P > 0
        return np.sum(P[mask] * np.log(P[mask] / Q[mask]))

    P = np.array([0.5, 0.3, 0.2])
    Q = np.array([0.1, 0.6, 0.3])
    kl_pq = kl_discrete(P, Q)
    kl_qp = kl_discrete(Q, P)

    print(bold_cyan(f"  P = {P}"))
    print(bold_cyan(f"  Q = {Q}"))
    print(bold_cyan(f"  KL(P ‖ Q) = {kl_pq:.4f} nats"))
    print(bold_cyan(f"  KL(Q ‖ P) = {kl_qp:.4f} nats"))
    print(bold_cyan(f"  Difference: {abs(kl_pq - kl_qp):.4f}  ← asymmetry!"))
    print()

    # Gaussian KL
    mu1, s1 = 0.0, 1.0
    mu2, s2 = 1.0, 2.0
    kl_gauss = np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5
    kl_gauss_rev = np.log(s1/s2) + (s2**2 + (mu2-mu1)**2)/(2*s1**2) - 0.5
    print(bold_cyan(f"  Gaussian: P=N({mu1},{s1}²), Q=N({mu2},{s2}²)"))
    print(bold_cyan(f"  KL(P‖Q) = {kl_gauss:.4f},  KL(Q‖P) = {kl_gauss_rev:.4f}"))
    print()
    _pause()

    section_header("4. ASCII — KL as Extra Encoding Cost")
    print(cyan("  P = true data distribution  |  Q = our model"))
    print(cyan("  KL(P‖Q) = extra bits per symbol when coding P with Q-optimised code"))
    print()
    Ps = [0.5, 0.3, 0.2]
    Qs = [0.1, 0.6, 0.3]
    h_p = -np.sum(np.array(Ps) * np.log2(np.array(Ps)))
    cross_ent = -np.sum(np.array(Ps) * np.log2(np.array(Qs)))
    kl_bits = cross_ent - h_p
    print(f"  Optimal     (P-code):  H(P) = {h_p:.4f} bits")
    print(f"  Suboptimal  (Q-code):  H(P,Q) = {cross_ent:.4f} bits")
    print(f"  Extra cost:            KL = {kl_bits:.4f} bits  (waste!)")
    print()

    bar_opt  = "█" * int(h_p * 10)
    bar_cross = "█" * int(cross_ent * 10)
    print(f"  Optimal code:   {green(bar_opt)}")
    print(f"  Suboptimal:     {yellow(bar_cross)}")
    print(f"  {red('░'*int(kl_bits*10))}  ← KL penalty")
    print()
    _pause()

    section_header("5. PLOTEXT — Forward and Reverse KL")
    try:
        xs = np.linspace(-5, 5, 80)
        kl_fwd = []  # KL(P||Q_shifted) as shift varies
        kl_rev = []
        mu_shifts = np.linspace(-3, 3, 80)
        p_std = 1.0
        q_std = 2.0
        true_mu = 0.0
        for mu_q in mu_shifts:
            kf = np.log(q_std/p_std) + (p_std**2 + (true_mu-mu_q)**2)/(2*q_std**2) - 0.5
            kr = np.log(p_std/q_std) + (q_std**2 + (mu_q-true_mu)**2)/(2*p_std**2) - 0.5
            kl_fwd.append(kf)
            kl_rev.append(kr)
        loss_curve_plot(np.array(kl_fwd), title="KL(P||Q_shifted) vs Q mean  (P=N(0,1), Q=N(μ,4))")
    except Exception:
        print(grey("  plotext unavailable"))
    print()
    _pause()

    section_header("6. PYTHON CODE")
    code_block("KL Divergence — Discrete and Gaussian", """
import numpy as np
from scipy.stats import entropy as scipy_entropy

# --- Discrete KL ---
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.1, 0.6, 0.3])                    # base e (nats) by default
kl_pq = scipy_entropy(P, Q)                       # KL(P||Q)
kl_qp = scipy_entropy(Q, P)                       # KL(Q||P)
print(f"KL(P||Q) = {kl_pq:.4f},  KL(Q||P) = {kl_qp:.4f}")

# --- Gaussian KL: KL(N(mu1,s1²) || N(mu2,s2²)) ---
def kl_gaussian(mu1, s1, mu2, s2):
    return np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5

print(f"KL(N(0,1)||N(1,2)) = {kl_gaussian(0,1,1,2):.4f}")
print(f"KL(N(1,2)||N(0,1)) = {kl_gaussian(1,2,0,1):.4f}  (asymmetric!)")

# --- Connection to MLE: min KL(P_data||P_theta) = max log-likelihood ---
print("Minimising KL(data||model) is equivalent to maximising log-likelihood")
""")
    _pause()

    section_header("7. KEY INSIGHTS")
    for ins in [
        "KL(P‖Q) = 0 iff P=Q; always non-negative (Gibbs' inequality)",
        "KL is NOT a distance — asymmetric and doesn't satisfy triangle inequality",
        "MLE ≡ minimise KL(empirical distribution ‖ model)",
        "VAE ELBO = reconstruction loss + KL(posterior ‖ prior)",
        "Forward KL (mode-seeking) vs reverse KL (mean-seeking) gives different fits",
        "Avoid KL when Q(x)=0 but P(x)>0: infinite divergence (undefined code)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Cross-Entropy
# ══════════════════════════════════════════════════════════════════════════════
def topic_cross_entropy():
    clear()
    breadcrumb("mlmath", "Information Theory", "Cross-Entropy")
    section_header("CROSS-ENTROPY")
    print()

    section_header("1. THEORY")
    print(white("""
  Cross-entropy H(P, Q) between distributions P (true) and Q (predicted):

      H(P, Q) = -Σₓ P(x) log Q(x)

  DECOMPOSITION:
      H(P, Q) = H(P) + KL(P ‖ Q)

  Since H(P) is fixed (doesn't depend on Q), minimising cross-entropy
  is equivalent to minimising KL(P ‖ Q) — which means making Q as close
  to P as possible.

  CONNECTION TO NEGATIVE LOG-LIKELIHOOD:
  For classification with one-hot targets (y is a one-hot vector):
      H(y, ŷ) = -Σᵢ yᵢ log ŷᵢ = -log ŷ_{y} = negative log-likelihood

  So cross-entropy loss IS negative log-likelihood.  This is why we use it!

  BINARY CROSS-ENTROPY:
      BCE(y, ŷ) = -y log ŷ - (1-y) log(1-ŷ)
  Used for binary classification, sigmoid output.

  CATEGORICAL CROSS-ENTROPY:
      CE(y, ŷ) = -Σᵢ yᵢ log ŷᵢ
  Used for multi-class, softmax output.

  NUMERICAL STABILITY: never compute log(ŷ) where ŷ can be 0.
  Use log_softmax for stability: log(softmax) vs log(exp(x)/Σexp(x)).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Cross-entropy:  H(P,Q) = -Σ P(x) log Q(x)"))
    print(formula("  Decomposition:  H(P,Q) = H(P) + KL(P‖Q)"))
    print(formula("  Binary CE:      BCE = -y log ŷ - (1-y) log(1-ŷ)"))
    print(formula("  Categorical CE: CE = -Σᵢ yᵢ log ŷᵢ  (one-hot y)"))
    print(formula("  BCE gradient:   ∂BCE/∂ŷ = -(y/ŷ) + (1-y)/(1-ŷ)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE")
    y_true = np.array([0, 0, 1, 0])       # one-hot: class index 2
    y_pred_good = np.array([0.05, 0.05, 0.85, 0.05])
    y_pred_bad  = np.array([0.25, 0.25, 0.25, 0.25])
    y_pred_wrong = np.array([0.85, 0.05, 0.05, 0.05])

    def cross_ent(y, yhat):
        yhat = np.clip(yhat, 1e-12, 1.0)
        return -np.sum(y * np.log(yhat))

    def h_entropy(p):
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p))

    ce_good  = cross_ent(y_true, y_pred_good)
    ce_bad   = cross_ent(y_true, y_pred_bad)
    ce_wrong = cross_ent(y_true, y_pred_wrong)
    h_true   = h_entropy(y_true + 1e-12)

    print(bold_cyan(f"  True label: {y_true}  (class 2)"))
    print()
    print(bold_cyan(f"  Good pred  {y_pred_good}:   CE = {ce_good:.4f}"))
    print(bold_cyan(f"  Uniform    {y_pred_bad}:   CE = {ce_bad:.4f}"))
    print(bold_cyan(f"  Wrong pred {y_pred_wrong}:   CE = {ce_wrong:.4f}"))
    print()
    print(green("  Lower CE = better prediction. Perfect = minimum."))
    print(green(f"  CE = H(P) + KL(P||Q). With one-hot P, H(P)≈0, so CE ≈ KL."))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Cross-Entropy Loss in NumPy and PyTorch", """
import numpy as np

def cross_entropy_loss(y_true_idx, logits):
    '''Numerically stable cross-entropy from raw logits.'''
    # log-sum-exp trick for stability
    logits = np.array(logits, dtype=float)
    log_sum_exp = np.log(np.sum(np.exp(logits - logits.max()))) + logits.max()
    log_softmax = logits - log_sum_exp
    return -log_softmax[y_true_idx]

# Example: 3-class logits, true class = 1
logits = [2.0, 3.5, 0.5]
loss = cross_entropy_loss(y_true_idx=1, logits=logits)
print(f"CE loss: {loss:.4f}")   # low because class 1 has highest logit

# Binary Cross-Entropy
y = np.array([1, 0, 1, 1, 0])
y_hat = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
bce = -np.mean(y * np.log(y_hat + 1e-12) + (1-y) * np.log(1 - y_hat + 1e-12))
print(f"Binary CE loss: {bce:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Minimising cross-entropy = minimising KL(data || model) = MLE",
        "CE ≥ H(P) always — equality when Q = P (perfect model)",
        "Binary CE gradient w.r.t. sigmoid output is simply ŷ - y (elegant!)",
        "Use log-softmax not log(softmax) for numerical stability",
        "Label smoothing: replace one-hot with (1-ε)·one-hot + ε/K to prevent overconfidence",
        "Focal loss: -(1-ŷ)^γ log ŷ — down-weights easy examples, focuses on hard ones",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Mutual Information
# ══════════════════════════════════════════════════════════════════════════════
def topic_mutual_information():
    clear()
    breadcrumb("mlmath", "Information Theory", "Mutual Information")
    section_header("MUTUAL INFORMATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Mutual Information I(X;Y) measures how much knowing Y tells us about X:

      I(X;Y) = H(X) - H(X|Y)
             = H(Y) - H(Y|X)
             = H(X) + H(Y) - H(X,Y)
             = KL(P(X,Y) ‖ P(X)P(Y))    ← KL from joint to product of marginals

  INTERPRETATION: I(X;Y) = 0 iff X and Y are independent (joint = product).
  I(X;Y) > 0 means knowing Y reduces uncertainty about X.

  FEATURE SELECTION: select features with high I(feature; target).
  This is model-free (no linearity assumption) — captures any relationship.

  DATA PROCESSING INEQUALITY (DPI):
  If X → Y → Z is a Markov chain, then I(X;Z) ≤ I(X;Y).
  Processing (adding deterministic transformations) can only REDUCE mutual info.
  Deep networks: I(X;T_layer) decreases as layers get deeper (information
  is compressed). This motivated the Information Bottleneck view of deep learning.

  NORMALISED MUTUAL INFORMATION (NMI):
      NMI = I(X;Y) / √(H(X)·H(Y))  ∈ [0,1]
  Used to compare clustering quality.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)"))
    print(formula("  I(X;Y) = Σₓ Σᵧ p(x,y) log[p(x,y)/(p(x)p(y))]"))
    print(formula("  I(X;Y) = KL(P(X,Y) ‖ P(X)P(Y)) ≥ 0"))
    print(formula("  Chain:  H(X,Y) = H(X) + H(Y|X)"))
    print(formula("  DPI:    X→Y→Z ⟹ I(X;Z) ≤ I(X;Y)"))
    print(formula("  NMI:    I(X;Y) / √[H(X)·H(Y)]  ∈ [0,1]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Discrete I(X;Y)")
    # Joint distribution P(X,Y)
    joint = np.array([[0.2, 0.1],
                      [0.1, 0.4],
                      [0.1, 0.1]])   # 3 X values, 2 Y values
    px   = joint.sum(axis=1)
    py   = joint.sum(axis=0)
    hx   = -np.sum(px * np.log2(px + 1e-12))
    hy   = -np.sum(py * np.log2(py + 1e-12))
    hxy  = -np.sum(joint * np.log2(joint + 1e-12))
    mi   = hx + hy - hxy
    nmi  = mi / np.sqrt(hx * hy)
    hx_y = hxy - hy   # H(X|Y)

    print(bold_cyan(f"  Joint P(X,Y):  {joint}"))
    print(bold_cyan(f"  H(X)   = {hx:.4f} bits"))
    print(bold_cyan(f"  H(Y)   = {hy:.4f} bits"))
    print(bold_cyan(f"  H(X,Y) = {hxy:.4f} bits"))
    print(bold_cyan(f"  I(X;Y) = H(X)+H(Y)-H(X,Y) = {mi:.4f} bits"))
    print(bold_cyan(f"  H(X|Y) = H(X,Y)-H(Y) = {hx_y:.4f} bits"))
    print(bold_cyan(f"  NMI    = {nmi:.4f}  (0=independent, 1=fully dependent)"))
    print()

    # Independence check
    independent = np.outer(px, py)
    joint_close = np.allclose(joint, independent, atol=0.05)
    print(green(f"  Are X,Y independent? {'Yes (approx)' if joint_close else 'No'}"))
    print(green(f"  Knowing Y reduces uncertainty about X by {mi:.4f} bits"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Mutual Information — Discrete and sklearn", """
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris

# --- Discrete mutual information ---
def mutual_info_discrete(joint):
    px  = joint.sum(axis=1, keepdims=True)
    py  = joint.sum(axis=0, keepdims=True)
    eps = 1e-12
    return np.sum(joint * np.log2((joint + eps) / (px * py + eps)))

joint = np.array([[0.2, 0.1], [0.1, 0.4], [0.1, 0.1]])
print(f"I(X;Y) = {mutual_info_discrete(joint):.4f} bits")

# --- Continuous MI with sklearn for feature selection ---
X, y = load_iris(return_X_y=True)
mi_scores = mutual_info_classif(X, y, random_state=0)
for i, score in enumerate(mi_scores):
    print(f"  Feature {i}: MI = {score:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "I(X;Y) = 0 ↔ X and Y independent; always ≥ 0",
        "MI captures ANY statistical dependency, not just linear (unlike correlation)",
        "Data Processing Inequality: transformations can only reduce MI",
        "Information Bottleneck: tradeoff between I(X;T) compression and I(T;Y) relevance",
        "NMI ∈ [0,1] is useful for comparing clustering across different k values",
        "sklearn.mutual_info_classif uses k-nearest-neighbours entropy estimator",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Jensen-Shannon Divergence
# ══════════════════════════════════════════════════════════════════════════════
def topic_js_divergence():
    clear()
    breadcrumb("mlmath", "Information Theory", "Jensen-Shannon Divergence")
    section_header("JENSEN-SHANNON DIVERGENCE")
    print()

    section_header("1. THEORY")
    print(white("""
  The Jensen-Shannon Divergence (JSD) is a SYMMETRIC, SMOOTHED version of KL:

      JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),   M = ½(P+Q)

  PROPERTIES:
  • Symmetric: JSD(P‖Q) = JSD(Q‖P)  ← unlike KL
  • Bounded:   0 ≤ JSD ≤ log 2  (in nats, or ≤ 1 in bits)
  • Well-defined even when supports don't fully overlap (unlike KL)
  • √JSD is a proper metric (satisfies triangle inequality!)

  GENERALIZATION: JS divergence can be generalised to k distributions:
      JSD(P₁,…,Pₖ) = H(Σᵢ wᵢPᵢ) - Σᵢ wᵢH(Pᵢ)

  GANs: The original GAN objective (Goodfellow 2014) is equivalent to
  minimising JSD(P_data ‖ P_generator) when the discriminator is optimal.
  This connection explains the training instability — JSD is zero or
  log 2 when supports don't overlap, giving zero gradients (mode collapse,
  vanishing gradients). Wasserstein GAN fixes this by using Earth Mover distance.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  M = ½(P+Q)  ← mixture distribution"))
    print(formula("  JSD(P‖Q) = ½KL(P‖M) + ½KL(Q‖M)"))
    print(formula("  JSD = H(M) - ½H(P) - ½H(Q)"))
    print(formula("  0 ≤ JSD ≤ log 2 (nats),  0 ≤ JSD ≤ 1 (bits)"))
    print(formula("  d_JS = √JSD  is a metric"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — GAN Connection")
    def kl_div(P, Q, eps=1e-12):
        P, Q = np.maximum(P, eps), np.maximum(Q, eps)
        return np.sum(P * np.log(P / Q))

    def jsd(P, Q):
        M = 0.5 * (P + Q)
        return 0.5 * kl_div(P, M) + 0.5 * kl_div(Q, M)

    P = np.array([0.5, 0.3, 0.2])
    Q = np.array([0.1, 0.6, 0.3])
    j = jsd(P, Q)
    kl_pq = kl_div(P, Q)
    kl_qp = kl_div(Q, P)

    print(bold_cyan(f"  P = {P}"))
    print(bold_cyan(f"  Q = {Q}"))
    print(bold_cyan(f"  KL(P‖Q) = {kl_pq:.4f} nats  (asymmetric)"))
    print(bold_cyan(f"  KL(Q‖P) = {kl_qp:.4f} nats  (asymmetric)"))
    print(bold_cyan(f"  JSD(P,Q) = {j:.4f} nats  (symmetric, bounded [0,{np.log(2):.4f}])"))
    print(bold_cyan(f"  √JSD     = {np.sqrt(j):.4f}  (metric distance)"))

    # Non-overlapping support
    P2 = np.array([1.0, 0.0, 0.0])
    Q2 = np.array([0.0, 0.0, 1.0])
    j_nooverlap = jsd(P2, Q2)
    print()
    print(bold_cyan(f"  Non-overlapping: P=[1,0,0], Q=[0,0,1]"))
    print(bold_cyan(f"  JSD = {j_nooverlap:.4f} = log(2) = {np.log(2):.4f}  (maximum)"))
    print(green("  KL would be ∞ here — JSD handles this gracefully"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Jensen-Shannon Divergence", """
import numpy as np
from scipy.spatial.distance import jensenshannon

P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.1, 0.6, 0.3])

# scipy gives sqrt(JSD) (the metric form)
js_metric = jensenshannon(P, Q)   # √JSD
js_div    = js_metric ** 2        # JSD

print(f"JSD(P,Q) = {js_div:.4f} nats")
print(f"√JSD     = {js_metric:.4f}  (metric)")

# Manual implementation
def jsd_manual(P, Q):
    M = 0.5 * (P + Q)
    kl = lambda p, q: np.sum(p * np.log(p / np.maximum(q, 1e-12)))
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)

print(f"Manual JSD = {jsd_manual(P, Q):.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "JSD is symmetric (unlike KL) and bounded [0, log 2]",
        "√JSD is a true metric — satisfies triangle inequality",
        "Original GAN minimises JSD(P_data || P_gen) when discriminator is optimal",
        "JSD = 0 when P=Q, JSD = log 2 when supports fully disjoint",
        "Non-overlapping supports: KL → ∞, JSD → log 2 (graceful handling)",
        "Wasserstein distance is preferred over JSD for GAN training stability",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Channel Capacity
# ══════════════════════════════════════════════════════════════════════════════
def topic_channel_capacity():
    clear()
    breadcrumb("mlmath", "Information Theory", "Channel Capacity")
    section_header("CHANNEL CAPACITY")
    print()

    section_header("1. THEORY")
    print(white("""
  A COMMUNICATION CHANNEL takes input X and produces output Y with noise.
  The CAPACITY C is the maximum mutual information over all input distributions:

      C = max_{P(X)} I(X;Y)   [bits per channel use]

  BINARY SYMMETRIC CHANNEL (BSC):
  Each bit is flipped independently with probability ε.
  P(Y=0|X=0) = P(Y=1|X=1) = 1-ε  (transmitted correctly)
  P(Y=1|X=0) = P(Y=0|X=1) = ε    (flipped with probability ε)

  Capacity of BSC:  C = 1 - H(ε)  where H(ε) is the binary entropy.
  • ε=0 (perfect channel):  C = 1 bit/use
  • ε=0.5 (pure noise):     C = 0 bits/use  (50% flip = coin flip)
  • ε=1 (inverts):          C = 1 bit/use  (just flip the output!)

  SHANNON'S NOISY-CHANNEL CODING THEOREM:
  For any rate R < C, there exists an error-correcting code that transmits
  at rate R with arbitrarily small error probability.  For R > C, reliable
  transmission is impossible.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Channel capacity:  C = max_{P(X)} I(X;Y)"))
    print(formula("  BSC capacity:      C = 1 - H(ε) = 1 + ε log ε + (1-ε)log(1-ε)"))
    print(formula("  AWGN capacity:     C = ½ log₂(1 + SNR)  (Shannon-Hartley)"))
    print(formula("  Noisy-channel thm: R < C ⟹ reliable transmission possible"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — BSC Capacity vs Error Rate")
    def bsc_capacity(eps):
        if eps <= 0 or eps >= 1:
            return 1.0
        h = -(eps * np.log2(eps) + (1-eps) * np.log2(1-eps))
        return 1.0 - h

    error_rates = np.linspace(0, 1, 21)
    capacities  = [bsc_capacity(e) for e in error_rates]

    print(bold_cyan(f"  {'Error rate ε':<16} {'Capacity C (bits)':<20} {'Bar'}"))
    print(grey("  " + "─"*50))
    for eps, cap in zip(error_rates, capacities):
        bar = "█" * int(cap * 20)
        print(f"  ε={eps:.2f}          {cap:<20.4f} {cyan(bar)}")
    print()
    _pause()

    section_header("4. PLOTEXT — Capacity vs Error Rate")
    try:
        eps_arr = np.linspace(0.001, 0.999, 80)
        cap_arr = np.array([bsc_capacity(e) for e in eps_arr])
        loss_curve_plot(cap_arr, title="BSC Capacity C = 1 - H(ε) vs error rate ε ∈ [0,1]")
    except Exception:
        print(grey("  plotext unavailable"))
    print()
    _pause()

    section_header("5. PYTHON CODE")
    code_block("BSC and AWGN Channel Capacity", """
import numpy as np

def binary_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

def bsc_capacity(error_rate):
    return 1.0 - binary_entropy(error_rate)

def awgn_capacity(snr_linear):
    return 0.5 * np.log2(1 + snr_linear)

# BSC capacity
for eps in [0.0, 0.1, 0.25, 0.5]:
    print(f"BSC(ε={eps}): C = {bsc_capacity(eps):.4f} bits/use")

# AWGN capacity (Shannon-Hartley)
for snr_db in [0, 10, 20, 30]:
    snr = 10 ** (snr_db / 10)
    print(f"AWGN SNR={snr_db}dB: C = {awgn_capacity(snr):.2f} bits/use")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "BSC capacity C=1-H(ε): maximum at ε=0 and ε=1, zero at ε=0.5 (pure noise)",
        "Shannon's theorem guarantees reliable communication below capacity",
        "Above capacity, no code can achieve arbitrarily small error rate",
        "AWGN capacity C = ½log₂(1+SNR): doubling SNR adds ½ bit",
        "Deep learning can be seen as learning a good code for the task",
        "Information bottleneck formalises the compression-relevance tradeoff",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — Minimum Description Length
# ══════════════════════════════════════════════════════════════════════════════
def topic_mdl():
    clear()
    breadcrumb("mlmath", "Information Theory", "Minimum Description Length")
    section_header("MINIMUM DESCRIPTION LENGTH (MDL)")
    print()

    section_header("1. THEORY")
    print(white("""
  MDL is a principle for model selection based on lossless compression.
  The best model is the one that allows the most compressed description of the data.

  TWO-PART MDL:
      MDL = L(model) + L(data | model)
  where L denotes description length (in bits).
  A complex model has large L(model) but small L(data|model).
  A simple model has small L(model) but large L(data|model).
  MDL balances this tradeoff automatically.

  CONNECTION TO BIC:
  Under certain regularity conditions, MDL ≈ BIC (Bayesian Information Criterion):
      BIC = -2 ℓ̂ + k log n
  where k = number of parameters, n = sample size, ℓ̂ = max log-likelihood.

  CONNECTION TO BAYESIAN EVIDENCE:
  MDL is closely related to the marginal likelihood (model evidence):
      P(data|model) = ∫ P(data|θ, model) P(θ|model) dθ
  Maximising evidence automatically penalises complexity (Occam's razor).

  STOCHASTIC COMPLEXITY (Rissanen):
  A more refined MDL avoids the two-part split by using the universal code.
  The code length equals -log P(data|model) under this code.

  MDL, BIC, and MAP with Gaussian prior all implement Occam's razor —
  prefer simpler models unless the data strongly support complexity.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Two-part MDL:  L = L(θ) + L(data|θ)"))
    print(formula("  BIC:           -2 ℓ̂ + k log n  (approx MDL)"))
    print(formula("  AIC:           -2 ℓ̂ + 2k  (different penalty)"))
    print(formula("  Code length:   L(x) = -log₂ P(x)  (optimal for distribution P)"))
    print(formula("  Model evidence: P(D|M) = ∫ P(D|θ)P(θ|M)dθ"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — BIC for Model Selection")
    from scipy import stats as sc
    rng = np.random.default_rng(5)
    x   = rng.normal(0, 1, 50)

    # Model 1: Gaussian (k=2 params: mu, sigma)
    mu_hat = np.mean(x); sig_hat = np.std(x, ddof=0)
    ll1 = np.sum(sc.norm.logpdf(x, mu_hat, sig_hat))
    k1  = 2; n = len(x)
    bic1 = -2*ll1 + k1*np.log(n)
    aic1 = -2*ll1 + 2*k1

    # Model 2: Gaussian with mu=0 assumed (k=1 param: sigma)
    sig_hat2 = np.sqrt(np.mean(x**2))
    ll2 = np.sum(sc.norm.logpdf(x, 0, sig_hat2))
    k2  = 1
    bic2 = -2*ll2 + k2*np.log(n)
    aic2 = -2*ll2 + 2*k2

    print(bold_cyan(f"  Data: n=50 from N(0,1)"))
    print()
    table(
        ["Model", "k", "log-L", "BIC", "AIC"],
        [
            ["N(μ̂,σ̂²)", "2", f"{ll1:.2f}", f"{bic1:.2f}", f"{aic1:.2f}"],
            ["N(0,σ̂²)", "1", f"{ll2:.2f}", f"{bic2:.2f}", f"{aic2:.2f}"],
        ]
    )
    print()
    winner_bic = "N(0,σ̂²)" if bic2 < bic1 else "N(μ̂,σ̂²)"
    winner_aic = "N(0,σ̂²)" if aic2 < aic1 else "N(μ̂,σ̂²)"
    print(green(f"  BIC selects: {winner_bic}  (lower is better)"))
    print(green(f"  AIC selects: {winner_aic}"))
    print(green("  True model is N(0,1) — BIC correctly prefers the simpler model"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("BIC and AIC for Model Comparison", """
import numpy as np
from scipy import stats

def bic(log_likelihood, k_params, n_samples):
    return -2 * log_likelihood + k_params * np.log(n_samples)

def aic(log_likelihood, k_params):
    return -2 * log_likelihood + 2 * k_params

rng = np.random.default_rng(0)
data = rng.normal(2.0, 1.5, size=100)

# Fit Gaussian: MLE
mu, sigma = np.mean(data), np.std(data, ddof=0)
ll = np.sum(stats.norm.logpdf(data, mu, sigma))

bic_val = bic(ll, k_params=2, n_samples=100)
aic_val = aic(ll, k_params=2)
print(f"Gaussian fit: mu={mu:.2f}, sigma={sigma:.2f}")
print(f"Log-likelihood: {ll:.2f}")
print(f"BIC: {bic_val:.2f}")
print(f"AIC: {aic_val:.2f}")
print("Lower BIC/AIC = better model (up to sign)")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "MDL = Occam's razor: smallest description compressing data + model",
        "BIC penalises params by k·log(n) — stronger penalty for small n",
        "AIC penalises params by 2k — less conservative than BIC",
        "BIC ≈ -2 × log marginal likelihood (asymptotically)",
        "MDL and MAP with Gaussian prior both penalise model complexity",
        "Over-parameterised models (deep nets) need different regularisation analysis",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Information Bottleneck
# ══════════════════════════════════════════════════════════════════════════════
def topic_information_bottleneck():
    clear()
    breadcrumb("mlmath", "Information Theory", "Information Bottleneck")
    section_header("INFORMATION BOTTLENECK")
    print()

    section_header("1. THEORY")
    print(white("""
  The Information Bottleneck (IB) principle (Tishby et al., 1999) formalises
  a fundamental tradeoff: how much can we COMPRESS input X into representation T
  while PRESERVING relevant information about target Y?

  OBJECTIVE: find a stochastic encoder P(T|X) that minimises:
      min_{P(T|X)}  I(X;T) - β · I(T;Y)

  • I(X;T) = compression term: how much information about X is in T
  • I(T;Y) = relevance term: how much information about Y is in T
  • β ≥ 0 = trade-off parameter: β=0 → maximum compression, β=∞ → sufficient statistic

  THE IB CURVE: as β increases, we allow more I(X;T) to gain more I(T;Y).
  The IB curve traces the optimal tradeoff frontier.

  DEEP LEARNING CONNECTION (Tishby & Schwartz-Ziv, 2017):
  Each hidden layer can be viewed as an IB encoder.  The claim (controversial)
  is that training has two phases:
    1. Fitting (I(T;Y) increases)
    2. Compression (I(X;T) decreases — generalisation via forgetting)

  SUFFICIENT STATISTICS: T is a sufficient statistic for Y from X when
  I(X;Y) = I(T;Y) — T captures ALL the relevant information.

  DATA PROCESSING INEQUALITY: X → T → Y implies I(X;Y) ≥ I(T;Y), so
  processing can only destroy relevance, never create it.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  IB objective:  min I(X;T) - β·I(T;Y)"))
    print(formula("  I(X;T) ≥ I(X;Y) no — actually I(T;Y) ≤ I(X;Y)  (DPI)"))
    print(formula("  DPI:           X → T → Y  ⟹  I(T;Y) ≤ I(X;Y)"))
    print(formula("  Sufficient:    T sufficient ⟺ I(T;Y) = I(X;Y)"))
    print(formula("  IB self-cons.  P(T|X) ∝ exp[-β · KL(P(Y|X) ‖ P(Y|T))]"))
    print()
    _pause()

    section_header("3. IB TRADEOFF ILLUSTRATION")
    print(cyan("  Information Bottleneck curve: compression vs. relevance"))
    print()
    betas = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    I_XT  = np.log(1 + betas) * 2.0      # heuristic: compression grows with beta
    I_TY  = 1.2 * np.tanh(0.5 * betas)   # heuristic: relevance saturates
    print(bold_cyan(f"  {'β':<8} {'I(X;T) compression':<22} {'I(T;Y) relevance':<20} {'Bar'}"))
    print(grey("  " + "─" * 65))
    for b, ixt, ity in zip(betas, I_XT, I_TY):
        eff = ity / (I_TY.max() + 1e-6)
        bar = "█" * int(eff * 20)
        cbar = green(bar) if eff > 0.7 else (yellow(bar) if eff > 0.4 else red(bar))
        print(f"  β={b:<7.2f} I(X;T)={ixt:.3f}              I(T;Y)={ity:.4f}  {cbar}")
    print()
    print(grey("  β→0: maximum compression, irrelevant T"))
    print(grey("  β→∞: T = X (no compression), T is sufficient statistic"))
    print()
    _pause()

    section_header("4. PYTHON CODE")
    code_block("Information Bottleneck Concept", """
import numpy as np
from scipy.special import logsumexp

# Conceptual: IB tradeoff curve
# In practice, VIB (Variational Information Bottleneck) uses reparameterization

def ib_objective(beta, I_XT, I_TY):
    '''IB Lagrangian — minimise I(X;T) - beta * I(T;Y).'''
    return I_XT - beta * I_TY

# Suppose I(X;T) and I(T;Y) for different representations
betas = [0.1, 0.5, 1.0, 2.0, 5.0]

# Illustrative bottleneck values (would come from a trained encoder in practice)
for beta in betas:
    ixt = 2.0 * np.log(1 + beta)
    ity = 1.5 * np.tanh(0.3 * beta)
    obj = ib_objective(beta, ixt, ity)
    print(f"beta={beta:.1f}:  I(X;T)={ixt:.3f},  I(T;Y)={ity:.3f},  obj={obj:.3f}")

print("\\nVIB adds: min E[CE loss] + beta * KL(q(Z|X) || N(0,I))")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "IB formalises the compression-relevance tradeoff in representation learning",
        "β controls tradeoff: small β → compression, large β → preserve all information",
        "DPI: every layer can only reduce I(T;Y), not increase it beyond I(X;Y)",
        "Sufficient statistic: T that retains all information about Y given X",
        "VIB (Variational IB) implements IB in deep learning via reparameterisation",
        "Debate: does deep learning actually achieve the IB bound? Mixed evidence.",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Entropy",                  topic_entropy),
        ("KL Divergence",            topic_kl_divergence),
        ("Cross-Entropy",            topic_cross_entropy),
        ("Mutual Information",       topic_mutual_information),
        ("Jensen-Shannon Divergence",topic_js_divergence),
        ("Channel Capacity",         topic_channel_capacity),
        ("Minimum Description Length", topic_mdl),
        ("Information Bottleneck",   topic_information_bottleneck),
    ]
    block_menu("b07", "Information Theory", topics)
