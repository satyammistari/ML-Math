"""
Block 05 — Probability Theory
Axioms, Bayes, Distributions, Joint/Marginal/Conditional,
Expectation, CLT, Conjugate Priors
"""
import numpy as np
from math import factorial, exp, log, sqrt, pi, comb

from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold, white,
                       formula, value, section, emph, hint, red, bold_magenta)
from ui.widgets import box, section_header, breadcrumb, nav_bar, table, bar_chart, code_block, panel, pager, hr, print_sparkline
from ui.menu import topic_nav, clear, block_menu, mark_completed
from viz.ascii_plots import scatter, line_plot, multi_line


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Probability Axioms
# ─────────────────────────────────────────────────────────────────────────────
def topic_probability_axioms():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Axioms & Rules")
    section_header("PROBABILITY AXIOMS & FUNDAMENTAL RULES")

    section_header("1. THEORY")
    print(white("  Kolmogorov's axioms (1933) provide the mathematical foundation of probability:"))
    print(white("  1. Non-negativity:   P(A) ≥ 0  for every event A"))
    print(white("  2. Normalisation:    P(Ω) = 1  (the sample space has probability 1)"))
    print(white("  3. Countable additivity: for mutually exclusive A₁, A₂, ...:"))
    print(white("                        P(⋃ᵢ Aᵢ) = Σᵢ P(Aᵢ)"))
    print()
    print(white("  Everything else in probability theory follows from these three axioms."))
    print(white("  'Probability' can represent frequency (frequentist) or degree-of-belief (Bayesian)."))
    print()
    print(white("  Sum rule: P(A) = Σᵦ P(A, B)  (marginalise over B)"))
    print(white("  Product rule: P(A, B) = P(A|B) P(B) = P(B|A) P(A)"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Complement:       P(Aᶜ) = 1 − P(A)"))
    print(formula("  Addition rule:    P(A∪B) = P(A) + P(B) − P(A∩B)"))
    print(formula("  Product rule:     P(A∩B) = P(A|B)·P(B)"))
    print(formula("  Independence:     P(A∩B) = P(A)·P(B)  iff A ⊥ B"))
    print(formula("  Conditional:      P(A|B) = P(A∩B) / P(B)"))
    print(formula("  Total probability:P(B) = Σᵢ P(B|Aᵢ)·P(Aᵢ)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  Rolling two fair dice:"))
    print(white("  Sample space: Ω = {(i,j) : i,j ∈ 1..6}  |Ω| = 36"))
    print()

    die = range(1, 7)
    outcomes = [(i, j) for i in die for j in die]
    total = len(outcomes)

    A_sum7  = [(i,j) for i,j in outcomes if i+j == 7]
    A_sum10 = [(i,j) for i,j in outcomes if i+j >= 10]
    A_both_even = [(i,j) for i,j in outcomes if i%2==0 and j%2==0]
    A7_and_even = [(i,j) for i,j in A_sum7 if i%2==0 and j%2==0]

    p7    = len(A_sum7) / total
    p10p  = len(A_sum10) / total
    p_be  = len(A_both_even) / total
    p_7_given_be = len(A7_and_even) / len(A_both_even) if A_both_even else 0

    print(f"  P(sum=7)         = {len(A_sum7)}/{total} = {value(f'{p7:.4f}')}")
    print(f"  P(sum≥10)        = {len(A_sum10)}/{total} = {value(f'{p10p:.4f}')}")
    print(f"  P(both even)     = {len(A_both_even)}/{total} = {value(f'{p_be:.4f}')}")
    print(f"  P(sum=7|both even)= {len(A7_and_even)}/{len(A_both_even)} = {value(f'{p_7_given_be:.4f}')}")
    print()

    # Verify axioms
    all_events = [[(i,j)] for i,j in outcomes]
    total_check = sum(1/total for _ in outcomes)
    print(green(f"  ✓ Axiom check: Σ P({{ω}}) = {total_check:.4f} = 1"))
    print(green(f"  ✓ Addition:  P(A∪B) = P(A)+P(B)-P(A∩B) = {p7+p_be - len(A7_and_even)/total:.4f}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Probability of each sum when rolling 2 dice:"))
    print()
    sum_probs = {}
    for s in range(2, 13):
        count = sum(1 for i,j in outcomes if i+j == s)
        sum_probs[s] = count / total
    max_p = max(sum_probs.values())
    for s, p in sum_probs.items():
        bar_len = int(30 * p / max_p)
        col = green if abs(s-7) <= 1 else (yellow if abs(s-7) <= 2 else grey)
        print(f"  Sum {s:>2d}: {col('█' * bar_len):<35}  P = {value(f'{p:.4f}')}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import bar
        sums = list(sum_probs.keys()); probs = list(sum_probs.values())
        bar(probs, title="P(sum=k) for 2 dice", xlabel="Sum k", ylabel="Probability")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        sums_m = list(sum_probs.keys()); probs_m = list(sum_probs.values())
        plt.figure(figsize=(8, 4))
        colors_m = ['steelblue' if s != 7 else 'gold' for s in sums_m]
        plt.bar(sums_m, probs_m, color=colors_m, edgecolor='black', alpha=0.8)
        plt.xlabel("Sum of two dice"); plt.ylabel("Probability")
        plt.title("Probability Distribution: Sum of Two Dice")
        plt.xticks(sums_m); plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Probability Axioms & Rules", """\
import numpy as np
from itertools import product

# Sample space: two dice
outcomes = list(product(range(1,7), range(1,7)))
N = len(outcomes)   # 36

def P(event):
    \"\"\"Probability of an event (list of outcomes).\"\"\"
    return len(event) / N

# Events
A = [o for o in outcomes if sum(o) == 7]
B = [o for o in outcomes if o[0] > o[1]]
A_and_B = [o for o in outcomes if o in A and o in B]
A_or_B  = list(set(A) | set(B))

print(f"P(sum=7)     = {P(A):.4f}")
print(f"P(d1>d2)     = {P(B):.4f}")
print(f"P(A∩B)       = {P(A_and_B):.4f}")
print(f"P(A∪B)       = {P(A_or_B):.4f}")
print(f"P(A)+P(B)-P(A∩B) = {P(A)+P(B)-P(A_and_B):.4f}  (should equal P(A∪B))")
print(f"P(A|B)       = {P(A_and_B)/P(B):.4f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "All of probability follows from 3 axioms: non-negativity, normalisation, countable additivity.",
        "P(A|B) = P(A∩B)/P(B) — conditioning restricts the sample space to B.",
        "Independence P(A∩B)=P(A)P(B) ≠ mutual exclusivity P(A∩B)=0 — often confused.",
        "Law of total probability P(B) = ΣP(B|Aᵢ)P(Aᵢ) is the partition formula underlying Bayes.",
        "Frequentist: probability = limiting frequency. Bayesian: probability = degree of belief.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bayes' Theorem
# ─────────────────────────────────────────────────────────────────────────────
def topic_bayes_theorem():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Bayes' Theorem")
    section_header("BAYES' THEOREM")

    section_header("1. THEORY")
    print(white("  Bayes' theorem inverts conditional probabilities: given P(E|H), what is P(H|E)?"))
    print(white("  It is the mathematical foundation of rational belief updating under evidence."))
    print()
    print(white("  Components:"))
    print(white("  • Prior P(H):       our belief before seeing evidence"))
    print(white("  • Likelihood P(E|H): how likely evidence is given hypothesis"))
    print(white("  • Evidence P(E):    normalisation constant (often computed via total probability)"))
    print(white("  • Posterior P(H|E): updated belief after seeing evidence"))
    print()
    print(white("  Base rate fallacy: even accurate tests give surprisingly low posterior"))
    print(white("  probabilities when the prior P(H) is very small. A 99% accurate test for"))
    print(white("  a disease with 0.1% prevalence has positive predictive value of only ≈ 9%!"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Bayes:     P(H|E) = P(E|H)·P(H) / P(E)"))
    print(formula("  P(E):      P(E) = P(E|H)P(H) + P(E|Hᶜ)P(Hᶜ)"))
    print(formula("  Posterior ∝ Likelihood × Prior"))
    print(formula("  Odds form: P(H|E)/P(Hᶜ|E) = [P(E|H)/P(E|Hᶜ)] · [P(H)/P(Hᶜ)]"))
    print(formula("             Posterior odds = Bayes factor × Prior odds"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  Medical test example:"))
    sensitivity  = 0.99   # P(+|disease)
    specificity  = 0.95   # P(-|healthy)
    prevalence   = 0.001  # P(disease)

    p_pos_given_disease  = sensitivity
    p_pos_given_healthy  = 1 - specificity
    p_disease            = prevalence
    p_healthy            = 1 - prevalence

    p_positive = p_pos_given_disease * p_disease + p_pos_given_healthy * p_healthy
    p_disease_given_pos  = (p_pos_given_disease * p_disease) / p_positive
    p_disease_given_neg  = ((1-sensitivity) * p_disease) / (1 - p_positive)

    print()
    print(white(f"  Sensitivity (P(+|disease)):  {sensitivity}"))
    print(white(f"  Specificity (P(-|healthy)):  {specificity}"))
    print(white(f"  Prevalence  (P(disease)):    {prevalence}  (1 in 1000)"))
    print()
    print(white(f"  P(+ test):               {p_positive:.5f}"))
    print(red(   f"  P(disease | + test):     {p_disease_given_pos:.5f}  ({p_disease_given_pos*100:.1f}%)"))
    print(white( f"  P(disease | − test):     {p_disease_given_neg:.2e}"))
    print()
    print(yellow("  ⚠ Despite 99% sensitivity, only ≈1-2% of positive tests indicate disease!"))
    print(white("    This is the base rate fallacy — low prevalence dominates."))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Bayesian update: Prior → Likelihood → Posterior"))
    print()
    print(f"  {cyan('Prior')}  P(disease) = {prevalence}")
    print(f"  {'─'*50}")
    print(f"  P(disease) = {prevalence:.4f}  {cyan('█' * 1)}")
    print(f"  P(healthy) = {p_healthy:.4f}  {cyan('█' * 50)}")
    print()
    print(f"  {yellow('Evidence: positive test')}")
    print(f"  {'─'*50}")
    print(f"  P(+|disease)  = {sensitivity}  (likelihood)")
    print(f"  P(+|healthy)  = {p_pos_given_healthy:.2f}  (false positive rate)")
    print()
    print(f"  {green('Posterior')}  P(disease|+):")
    print(f"  {'─'*50}")
    bar_d = int(p_disease_given_pos * 50 / 0.1 + 1)
    bar_h = 50 - bar_d
    print(f"  P(disease|+) = {p_disease_given_pos:.4f}  {red('█' * bar_d)}")
    print(f"  P(healthy|+) = {1-p_disease_given_pos:.4f}  {green('█' * min(bar_h, 50))}")
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import bar
        priors_p = [prevalence, p_healthy]
        posts_p  = [p_disease_given_pos, 1-p_disease_given_pos]
        bar(priors_p, title="Prior: P(disease), P(healthy)", xlabel="State", ylabel="P")
        bar(posts_p,  title="Posterior: P(disease|+), P(healthy|+)", xlabel="State", ylabel="P")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        prevalences  = np.logspace(-4, -0.3, 100)
        ppp  = [(sensitivity * p) / (sensitivity * p + p_pos_given_healthy * (1-p)) for p in prevalences]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogx(prevalences, ppp, color='steelblue', lw=2)
        ax.axvline(prevalence, color='r', ls='--', label=f'prevalence={prevalence}')
        ax.axhline(p_disease_given_pos, color='g', ls='--', label=f'PPV={p_disease_given_pos:.3f}')
        ax.set_xlabel("Prevalence (log scale)"); ax.set_ylabel("P(disease | + test)")
        ax.set_title("Positive Predictive Value vs Prevalence"); ax.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Bayes' Theorem", """\
def bayes_update(prior, likelihood_pos, likelihood_neg):
    \"\"\"
    prior:           P(H)
    likelihood_pos:  P(E|H)   [sensitivity]
    likelihood_neg:  P(E|not H)  [false positive rate = 1-specificity]
    Returns:         P(H|E)   [positive predictive value]
    \"\"\"
    p_E = likelihood_pos * prior + likelihood_neg * (1 - prior)
    posterior = likelihood_pos * prior / p_E
    return posterior, p_E

# Medical test
ppv, p_pos = bayes_update(
    prior=0.001,           # 1 in 1000 have disease
    likelihood_pos=0.99,   # sensitivity
    likelihood_neg=0.05    # 1 - specificity
)
print(f"P(+test) = {p_pos:.5f}")
print(f"P(disease | + test) = {ppv:.4f}  ({ppv*100:.1f}%)")
print("Despite 99% sensitivity, only ~2% of positives are true!")

# Sequential Bayesian update (multiple tests)
p = 0.001
for test_result in [True, True]:   # two positive tests
    likelihood = 0.99 if test_result else 0.01
    fp_rate    = 0.05 if test_result else 0.95
    p, _ = bayes_update(p, likelihood, fp_rate)
    print(f"After {'positive' if test_result else 'negative'} test: P(disease) = {p:.4f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Posterior ∝ Likelihood × Prior — Bayes is literally just proportional probability.",
        "Base rate fallacy: ignoring low prior P(H) leads to severe overestimation of P(H|E).",
        "Positive predictive value (PPV) depends heavily on prevalence — a key medical statistics fact.",
        "Sequential Bayes: each posterior becomes the prior for the next observation — updating is iterative.",
        "Bayes factor = P(E|H)/P(E|Hᶜ) — measures how much the evidence favours H over Hᶜ.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Discrete Distributions
# ─────────────────────────────────────────────────────────────────────────────
def topic_distributions_discrete():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Discrete Distributions")
    section_header("DISCRETE PROBABILITY DISTRIBUTIONS")

    section_header("1. THEORY")
    print(white("  Discrete distributions assign probability masses to countable outcomes."))
    print(white("  They are characterised by their PMF (probability mass function) P(X=k)."))
    print()
    print(white("  Key distributions and ML uses:"))
    print(white("  • Bernoulli:   binary outcome, p parameter — binary classification label"))
    print(white("  • Binomial:    k successes in n trials — classification threshold analysis"))
    print(white("  • Poisson:     rare event count — NLP word counts, network packets"))
    print(white("  • Categorical: generalisation of Bernoulli — multi-class labels, softmax output"))
    print(white("  • Geometric:   waiting time to first success — survival models"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Bernoulli(p):    P(X=1)=p, P(X=0)=1-p   E=p  Var=p(1-p)"))
    print(formula("  Binomial(n,p):   P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ  E=np  Var=np(1-p)"))
    print(formula("  Poisson(λ):      P(X=k) = λᵏe⁻λ/k!           E=λ   Var=λ"))
    print(formula("  Geometric(p):    P(X=k) = (1-p)^(k-1)p        E=1/p Var=(1-p)/p²"))
    print(formula("  Categorical(p):  P(X=k) = pₖ,  Σpₖ=1          E=argmax pₖ"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    n, p = 10, 0.3
    lambda_val = 3.0

    print(white(f"  Binomial(n={n}, p={p}):"))
    binom_pmf = [comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1)]
    for k in range(n+1):
        bar_len = int(30 * binom_pmf[k] / max(binom_pmf))
        print(f"  k={k:>2}: {cyan('█' * bar_len):<35}  P={value(f'{binom_pmf[k]:.4f}')}")

    print()
    print(white(f"  Poisson(λ={lambda_val}):"))
    poisson_pmf = [lambda_val**k * exp(-lambda_val) / factorial(k) for k in range(11)]
    for k in range(11):
        bar_len = int(30 * poisson_pmf[k] / max(poisson_pmf))
        print(f"  k={k:>2}: {yellow('█' * bar_len):<35}  P={value(f'{poisson_pmf[k]:.4f}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Distribution properties comparison:"))
    rows_data = [
        ["Distribution",  "PMF",                   "Mean",  "Variance",    "ML Use"],
        ["Bernoulli(p)",  "p^x(1-p)^(1-x)",       "p",     "p(1-p)",      "Binary CLS"],
        ["Binomial(n,p)", "C(n,k)pᵏ(1-p)^(n-k)",  "np",    "np(1-p)",     "Count correct"],
        ["Poisson(λ)",    "λᵏe⁻λ/k!",              "λ",     "λ",           "NLP word count"],
        ["Geometric(p)",  "(1-p)^(k-1)p",          "1/p",   "(1-p)/p²",    "Survival"],
        ["Categorical",   "pₖ",                    "—",     "—",           "Multi-class"],
    ]
    col_w = [16, 22, 8, 12, 15]
    divider = "  +" + "+".join("-"*(w+2) for w in col_w) + "+"
    print(grey(divider))
    print("  | " + " | ".join(cyan(c).ljust(w+9) for c, w in zip(rows_data[0], col_w)) + " |")
    print(grey(divider))
    for row in rows_data[1:]:
        print("  | " + " | ".join(white(c).ljust(w+9) for c, w in zip(row, col_w)) + " |")
    print(grey(divider))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import bar, multi_loss
        binom_list = binom_pmf
        bar(binom_list, title=f"Binomial(n={n},p={p}) PMF", xlabel="k", ylabel="P(X=k)")
        bar(poisson_pmf, title=f"Poisson(λ={lambda_val}) PMF", xlabel="k", ylabel="P(X=k)")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ks = list(range(n+1))
        axes[0].bar(ks, binom_pmf, color='steelblue', edgecolor='black', alpha=0.8)
        axes[0].set_title(f"Binomial(n={n}, p={p})"); axes[0].set_xlabel("k"); axes[0].set_ylabel("P(X=k)")
        ks2 = list(range(11))
        axes[1].bar(ks2, poisson_pmf[:11], color='tomato', edgecolor='black', alpha=0.8)
        axes[1].set_title(f"Poisson(λ={lambda_val})"); axes[1].set_xlabel("k"); axes[1].set_ylabel("P(X=k)")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Discrete Distributions", """\
import numpy as np
from math import comb, factorial, exp

# Binomial PMF
def binomial_pmf(k, n, p):
    return comb(n, k) * p**k * (1-p)**(n-k)

# Poisson PMF
def poisson_pmf(k, lam):
    return lam**k * exp(-lam) / factorial(k)

# Geometric PMF
def geometric_pmf(k, p):
    return (1-p)**(k-1) * p

# Examples
n, p, lam = 10, 0.3, 3.0
print("Binomial P(X=3):", binomial_pmf(3, n, p))
print("Poisson  P(X=3):", poisson_pmf(3, lam))
print("Geometric P(X=3):", geometric_pmf(3, p))

# Sampling from distributions
rng = np.random.default_rng(42)
binom_samples  = rng.binomial(n, p, size=1000)
poisson_samples= rng.poisson(lam, size=1000)
print("Binomial  mean:", binom_samples.mean(), "expected:", n*p)
print("Poisson   mean:", poisson_samples.mean(), "expected:", lam)
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Poisson is the limit of Binomial(n,p) as n→∞, p→0 with np=λ fixed — models rare events.",
        "Geometric distribution is memoryless: P(X>s+t|X>s) = P(X>t) — exponential in discrete.",
        "Categorical with k=2 reduces to Bernoulli; Multinomial generalises Binomial to k categories.",
        "Softmax output in classification = Categorical parameters; cross-entropy = -log likelihood.",
        "MLE of Binomial p: p̂ = (successes)/(trials); MLE of Poisson λ: λ̂ = sample mean.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Continuous Distributions
# ─────────────────────────────────────────────────────────────────────────────
def topic_distributions_continuous():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Continuous Distributions")
    section_header("CONTINUOUS PROBABILITY DISTRIBUTIONS")

    section_header("1. THEORY")
    print(white("  Continuous distributions assign probability densities to real-valued outcomes."))
    print(white("  Unlike PMFs, PDF values can exceed 1 — P(a≤X≤b) = ∫ₐᵇ f(x)dx."))
    print()
    print(white("  Key distributions and ML uses:"))
    print(white("  • Gaussian N(μ,σ²): most common, CLT limit, prior in Gaussian processes"))
    print(white("  • Beta(α,β):         distribution over [0,1], prior for Bernoulli/Binomial p"))
    print(white("  • Gamma(α,β):        positive values, prior for Poisson rate λ"))
    print(white("  • Laplace(μ,b):      heavier tails than Gaussian, corresponds to L1 prior/LASSO"))
    print(white("  • Exponential(λ):    memoryless waiting times, scale parameter machines"))
    print(white("  • Uniform(a,b):      maximum entropy on bounded interval, initial weight prior"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Gaussian:    f(x) = exp(−(x−μ)²/2σ²) / (σ√2π)    E=μ   Var=σ²"))
    print(formula("  Beta(α,β):   f(x) = xᵅ⁻¹(1−x)ᵝ⁻¹/B(α,β)  [0,1]   E=α/(α+β)"))
    print(formula("  Gamma(α,β):  f(x) = βᵅxᵅ⁻¹e⁻ᵝˣ/Γ(α)       [0,∞)   E=α/β"))
    print(formula("  Laplace(μ,b):f(x) = exp(−|x−μ|/b)/(2b)             E=μ   Var=2b²"))
    print(formula("  Multivariate Gaussian: f(x) = exp(−½(x−μ)ᵀΣ⁻¹(x−μ)) / √((2π)ⁿ|Σ|)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    rng = np.random.default_rng(42)
    xs = np.linspace(-4, 4, 200)

    def gaussian_pdf(x, mu=0, sigma=1):
        return np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

    def laplace_pdf(x, mu=0, b=1):
        return np.exp(-np.abs(x-mu)/b) / (2*b)

    print(white("  Gaussian N(0,1) vs Laplace(0,1):"))
    for x0 in [-2, -1, 0, 1, 2]:
        g = gaussian_pdf(x0); l = laplace_pdf(x0)
        print(f"  x={x0:+d}: Gaussian={value(f'{g:.4f}')},  Laplace={value(f'{l:.4f}')},  ratio={value(f'{l/g:.3f}')}")
    print()
    print(white("  Multivariate Gaussian N(μ, Σ):"))
    mu_mv  = np.array([0., 0.])
    Sigma  = np.array([[1., 0.5], [0.5, 2.]])
    Sigma_inv = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    def mvn_pdf(x, mu, S_inv, det_S):
        n   = len(x)
        diff = x - mu
        return np.exp(-0.5 * diff @ S_inv @ diff) / np.sqrt((2*np.pi)**n * det_S)
    for pt in [np.array([0.,0.]), np.array([1.,1.]), np.array([2.,2.])]:
        print(f"  f({pt}) = {value(f'{mvn_pdf(pt, mu_mv, Sigma_inv, det_Sigma):.5f}')}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  PDF shapes (normalised to width 30):"))
    print()
    for label, pdf_vals in [
        ("Gaussian N(0,1)", gaussian_pdf(xs)),
        ("Laplace (0,1)",   laplace_pdf(xs)),
    ]:
        max_v = max(pdf_vals)
        print(f"  {white(label)}:")
        idx_step = 10
        row = "  "
        for v in pdf_vals[::idx_step]:
            bar_h = int(12 * v / max_v)
            row += "█" if bar_h > 6 else ("▄" if bar_h > 3 else ("▁" if bar_h > 0 else " "))
        print(cyan(row) if "Gaussian" in label else yellow(row))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_distribution
        samples_g = rng.normal(0, 1, 2000).tolist()
        samples_l = rng.laplace(0, 1, 2000).tolist()
        multi_distribution([samples_g, samples_l], labels=["Gaussian N(0,1)", "Laplace(0,1)"],
                           title="Gaussian vs Laplace tails", n_bins=40)
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        xs_m = np.linspace(-4, 4, 300)
        for mu, sigma, col in [(0, 1, 'steelblue'), (2, 0.5, 'tomato'), (0, 2, 'gold')]:
            axes[0].plot(xs_m, gaussian_pdf(xs_m, mu, sigma),
                         label=f'N({mu},{sigma}²)', color=col)
        axes[0].set_title("Gaussian PDFs"); axes[0].legend()
        beta_params = [(0.5,0.5,'steelblue'), (2,2,'tomato'), (2,5,'gold')]
        xs_b = np.linspace(0.001, 0.999, 300)
        for a, b, col in beta_params:
            from scipy.stats import beta as beta_dist
            axes[1].plot(xs_b, beta_dist.pdf(xs_b, a, b), label=f'Beta({a},{b})', color=col)
        axes[1].set_title("Beta PDFs"); axes[1].legend()
        axes[2].plot(xs_m, gaussian_pdf(xs_m), label='Gaussian', color='steelblue')
        axes[2].plot(xs_m, laplace_pdf(xs_m),  label='Laplace',  color='tomato')
        axes[2].set_title("Gaussian vs Laplace (tails)"); axes[2].legend()
        plt.tight_layout(); plt.show()
    except (ImportError, Exception) as e:
        print(grey(f"  matplotlib: {e}"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Continuous Distributions", """\
import numpy as np
from scipy import stats

# Gaussian
g = stats.norm(loc=0, scale=1)
print("N(0,1) PDF at 0:", g.pdf(0))         # 0.3989
print("N(0,1) CDF at 0:", g.cdf(0))         # 0.5

# Beta distribution — prior for probabilities
beta = stats.beta(a=2, b=5)
print("Beta(2,5) mean:", beta.mean())        # 2/7 ≈ 0.286
print("Beta(2,5) PDF at 0.5:", beta.pdf(0.5))

# Multivariate Gaussian sampling
mu    = np.array([1., 2.])
Sigma = np.array([[2., 1.], [1., 3.]])
rng   = np.random.default_rng(42)
L     = np.linalg.cholesky(Sigma)
samples = mu + L @ rng.standard_normal((2, 1000))  # reparameterisation
print("Sample mean:", samples.mean(axis=1))  # ≈ [1., 2.]
print("Sample cov:\\n", np.cov(samples))      # ≈ Sigma

# KL divergence D_KL(P||Q) for two Gaussians
mu1, s1 = 0., 1.; mu2, s2 = 1., 2.
kl = np.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5
print(f"KL(N({mu1},{s1}²)||N({mu2},{s2}²)) = {kl:.4f}")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Gaussian is the maximum-entropy distribution with fixed mean and variance — 'least informative'.",
        "Laplace PDF ∝ exp(-|x-μ|/b) corresponds to L1 regularisation (LASSO) as a prior.",
        "Beta(1,1) = Uniform — the non-informative prior for a probability parameter.",
        "Multivariate Gaussian: Σ diagonal ↔ independent features; full Σ captures correlations.",
        "All exponential family members share a conjugate prior structure with the Bayesian update.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Joint / Marginal / Conditional
# ─────────────────────────────────────────────────────────────────────────────
def topic_joint_marginal_conditional():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Joint/Marginal/Conditional")
    section_header("JOINT, MARGINAL & CONDITIONAL DISTRIBUTIONS")

    section_header("1. THEORY")
    print(white("  A joint distribution P(X,Y) specifies probabilities for all combinations of"))
    print(white("  values of random variables X and Y simultaneously."))
    print()
    print(white("  Marginalisation: to get P(X), sum (or integrate) over all values of Y."))
    print(white("  Intuition: collapse the 2D table into a 1D row total."))
    print()
    print(white("  Conditional distribution P(Y|X=x): restrict the joint to rows where X=x,"))
    print(white("  then normalise so probabilities sum to 1. It answers: given X=x, what is"))
    print(white("  the distribution of Y?"))
    print()
    print(white("  Independence: X ⊥ Y iff P(X,Y) = P(X)·P(Y) — joint factorises as product."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Joint (discrete):   P(X=x, Y=y)  — full table"))
    print(formula("  Marginal:           P(X=x) = Σᵧ P(X=x, Y=y)"))
    print(formula("  Conditional:        P(Y=y|X=x) = P(X=x,Y=y) / P(X=x)"))
    print(formula("  Independence:       P(X,Y) = P(X)·P(Y)"))
    print(formula("  Chain rule:         P(X,Y,Z) = P(X|Y,Z)·P(Y|Z)·P(Z)"))
    print(formula("  Continuous margin:  p(x) = ∫ p(x,y) dy"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    # 3×3 joint distribution table
    joint = np.array([[0.10, 0.08, 0.04],
                      [0.12, 0.20, 0.06],
                      [0.06, 0.14, 0.20]])
    X_vals = ["X=0", "X=1", "X=2"]
    Y_vals = ["Y=0", "Y=1", "Y=2"]

    print(white("  Joint distribution P(X,Y):"))
    print()
    print("  " + " " * 8 + "  ".join(f"{yv:>8}" for yv in Y_vals))
    for i, xv in enumerate(X_vals):
        row_str = "  " + f"{xv:>8}  "
        for j in range(3):
            row_str += value(f"{joint[i,j]:>8.3f}") + "  "
        print(row_str)
    print()

    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)
    print(white("  Marginal P(X):   " + "  ".join(f"{p:.3f}" for p in marginal_x)))
    print(white("  Marginal P(Y):   " + "  ".join(f"{p:.3f}" for p in marginal_y)))
    print()

    x_given = 1  # condition on X=1
    conditional_Y_given_X1 = joint[x_given, :] / marginal_x[x_given]
    print(white(f"  Conditional P(Y | X=1):  {np.round(conditional_Y_given_X1, 4)}"))
    print(white(f"  Sum = {conditional_Y_given_X1.sum():.4f} (should be 1)"))
    print()

    # Independence check
    independent_approx = np.outer(marginal_x, marginal_y)
    indep_err = np.max(np.abs(joint - independent_approx))
    print(white(f"  Independence check: max|P(X,Y) − P(X)P(Y)| = {indep_err:.4f}"))
    if indep_err > 0.01:
        print(red("  → X and Y are NOT independent"))
    else:
        print(green("  → X and Y are approximately independent"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Joint distribution visualised:"))
    print()
    print("  " + " " * 5 + "".join(f"  Y={j}  " for j in range(3)))
    for i in range(3):
        row = f"  X={i}  "
        for j in range(3):
            size = int(joint[i, j] * 80)
            row += ("██" if size >= 12 else ("▓▓" if size >= 8 else ("░░" if size >= 4 else "  ")))
            row += "  "
        row += f"  Σ={value(f'{marginal_x[i]:.3f}')}"
        print(row)
    print()
    marginal_row = "  Σ    "
    for j in range(3):
        marginal_row += value(f"  {marginal_y[j]:.3f}  ")
    print(marginal_row)
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import bar
        bar(marginal_x.tolist(), title="Marginal P(X)", xlabel="X value", ylabel="P(X=x)")
        bar(marginal_y.tolist(), title="Marginal P(Y)", xlabel="Y value", ylabel="P(Y=y)")
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(joint, cmap='Blues', vmin=0)
        axes[0].set_title("Joint P(X,Y)"); fig.colorbar(im0, ax=axes[0])
        axes[0].set_xlabel("Y"); axes[0].set_ylabel("X")
        axes[1].bar(range(3), marginal_x, color='steelblue')
        axes[1].set_title("Marginal P(X)"); axes[1].set_xticks(range(3))
        axes[2].bar(range(3), conditional_Y_given_X1, color='tomato')
        axes[2].set_title("P(Y | X=1)"); axes[2].set_xticks(range(3))
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Joint / Marginal / Conditional", """\
import numpy as np

# Joint distribution table
P_XY = np.array([[0.10, 0.08, 0.04],
                 [0.12, 0.20, 0.06],
                 [0.06, 0.14, 0.20]])

# Marginals (sum over other variable)
P_X = P_XY.sum(axis=1)   # sum over columns (Y axis)
P_Y = P_XY.sum(axis=0)   # sum over rows (X axis)

print("P(X):", P_X)
print("P(Y):", P_Y)
print("Sum:", P_X.sum(), P_Y.sum())

# Conditional P(Y | X=x)
def conditional(P_XY, x):
    return P_XY[x, :] / P_XY[x, :].sum()

for x in range(3):
    print(f"P(Y|X={x}):", np.round(conditional(P_XY, x), 4))

# Independence test
P_X_outer_P_Y = np.outer(P_X, P_Y)
print("\\nMax independence violation:", np.max(np.abs(P_XY - P_X_outer_P_Y)))
print("Independent:", np.allclose(P_XY, P_X_outer_P_Y, atol=0.01))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Marginalisation = summing out a variable; integrating out uncertainty about latent variables.",
        "P(X|Y=y) ≠ P(X) unless X⊥Y — observing Y changes our beliefs about X when they're dependent.",
        "Chain rule P(X₁,...,Xₙ) = P(X₁)P(X₂|X₁)...P(Xₙ|X₁,...,Xₙ₋₁) — any factorisation order.",
        "Bayes' theorem is just a rearrangement of the definition of conditional probability.",
        "In Bayesian inference, latent variable z is marginalised: P(x) = ∫ P(x|z)P(z)dz.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Expectation & Variance
# ─────────────────────────────────────────────────────────────────────────────
def topic_expectation_variance():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Expectation & Variance")
    section_header("EXPECTATION & VARIANCE")

    section_header("1. THEORY")
    print(white("  The expectation E[X] = Σ xᵢpᵢ (discrete) = ∫ x f(x) dx (continuous) is the"))
    print(white("  probability-weighted average — the 'centre of mass' of the distribution."))
    print()
    print(white("  Variance Var[X] = E[(X-E[X])²] = E[X²] − (E[X])² measures spread."))
    print(white("  Standard deviation = √Var[X] is in the same units as X."))
    print()
    print(white("  Key properties:"))
    print(white("  • Linearity:   E[aX+b] = aE[X]+b;  Var[aX+b] = a²Var[X]"))
    print(white("  • E[X+Y] = E[X]+E[Y] always (linearity)"))
    print(white("  • Var[X+Y] = Var[X]+Var[Y]+2Cov[X,Y]"))
    print(white("  • If X⊥Y:   Var[X+Y] = Var[X]+Var[Y]"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Discrete E[X]:   Σᵢ xᵢ P(X=xᵢ)"))
    print(formula("  Continuous E[X]: ∫ x f(x) dx"))
    print(formula("  Variance:        Var[X] = E[X²] − (E[X])²"))
    print(formula("  Covariance:      Cov[X,Y] = E[XY] − E[X]E[Y]"))
    print(formula("  Correlation:     ρ = Cov[X,Y] / (√Var[X] · √Var[Y])  ∈ [−1,1]"))
    print(formula("  Law of total E:  E[X] = E_Y[E[X|Y]]"))
    print(formula("  Law of total V:  Var[X] = E[Var[X|Y]] + Var[E[X|Y]]"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    n, p = 10, 0.3
    binom_pmf = {k: comb(n, k) * p**k * (1-p)**(n-k) for k in range(n+1)}

    E_X     = sum(k * binom_pmf[k] for k in range(n+1))
    E_X2    = sum(k**2 * binom_pmf[k] for k in range(n+1))
    Var_X   = E_X2 - E_X**2
    Std_X   = sqrt(Var_X)

    print(white(f"  Binomial(n={n}, p={p}):"))
    print(white(f"  E[X]    = {value(f'{E_X:.4f}')}  (theoretical: np = {n*p:.4f})"))
    print(white(f"  E[X²]   = {value(f'{E_X2:.4f}')}"))
    print(white(f"  Var[X]  = {value(f'{Var_X:.4f}')}  (theoretical: np(1-p) = {n*p*(1-p):.4f})"))
    print(white(f"  Std[X]  = {value(f'{Std_X:.4f}')}"))
    print()

    # Covariance example
    rng = np.random.default_rng(42)
    X = rng.standard_normal(1000)
    Y = 0.7 * X + 0.3 * rng.standard_normal(1000)
    cov_xy = np.mean(X*Y) - np.mean(X)*np.mean(Y)
    rho    = cov_xy / (np.std(X) * np.std(Y))
    print(white(f"  Y = 0.7X + 0.3ε:"))
    print(white(f"  Cov[X,Y] = {value(f'{cov_xy:.4f}')}"))
    print(white(f"  Corr[X,Y] = {value(f'{rho:.4f}')}  (theoretical ≈ 0.7/√(0.49+0.09)={0.7/sqrt(0.49+0.09):.3f})"))
    print()

    # Law of total expectation
    print(white("  Law of total expectation demo:"))
    print(white("  Let X|Y=0 ~ N(0,1),  X|Y=1 ~ N(3,1),  P(Y=0)=0.6, P(Y=1)=0.4"))
    E_X_given_Y0 = 0.0; E_X_given_Y1 = 3.0
    P_Y0 = 0.6; P_Y1 = 0.4
    E_X_total = E_X_given_Y0 * P_Y0 + E_X_given_Y1 * P_Y1
    print(white(f"  E[X] = E[X|Y=0]P(Y=0) + E[X|Y=1]P(Y=1) = {E_X_given_Y0}×{P_Y0} + {E_X_given_Y1}×{P_Y1} = {E_X_total}"))
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Binomial(10,0.3) with mean and ±std marked:"))
    binom_vals = list(binom_pmf.values())
    max_v = max(binom_vals)
    for k in range(n+1):
        p_k = binom_pmf[k]; bar_len = int(30 * p_k / max_v)
        if abs(k - E_X) < Std_X:
            col = green
        elif abs(k - E_X) < 2 * Std_X:
            col = yellow
        else:
            col = grey
        mark = "← μ" if abs(k - E_X) < 0.5 else ""
        print(f"  k={k:>2}: {col('█' * bar_len + '░' * (30 - bar_len)):<40}  {value(f'{p_k:.3f}')}  {cyan(mark)}")
    print(grey(f"\n  Green = within 1σ ({E_X:.1f}±{Std_X:.1f}),  Yellow = within 2σ"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import histogram
        samples = rng.binomial(n, p, size=2000).tolist()
        histogram(samples, title=f"Binomial({n},{p}) samples — E[X]={E_X:.1f}, Std={Std_X:.2f}",
                  xlabel="k", bins=n+1)
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        ks = list(range(n+1))
        colors_m = ['limegreen' if abs(k - E_X) < Std_X else
                    ('gold' if abs(k - E_X) < 2*Std_X else 'steelblue') for k in ks]
        plt.bar(ks, binom_vals, color=colors_m, edgecolor='black', alpha=0.8)
        plt.axvline(E_X, color='red', lw=2, ls='--', label=f'E[X]={E_X:.1f}')
        plt.axvspan(E_X-Std_X, E_X+Std_X, alpha=0.2, color='green', label=f'±σ={Std_X:.2f}')
        plt.xlabel("k"); plt.ylabel("P(X=k)")
        plt.title(f"Binomial({n},{p}) — Mean and Variance"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Expectation & Variance", """\
import numpy as np
from math import comb

def binomial_moments(n, p):
    pmf = {k: comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)}
    EX   = sum(k * pmf[k] for k in pmf)
    EX2  = sum(k**2 * pmf[k] for k in pmf)
    VarX = EX2 - EX**2
    return EX, VarX, pmf

EX, VarX, pmf = binomial_moments(10, 0.3)
print(f"E[X] = {EX:.4f}  (np = {10*0.3})")
print(f"Var[X] = {VarX:.4f}  (np(1-p) = {10*0.3*0.7})")

# Covariance matrix
rng = np.random.default_rng(42)
X = rng.standard_normal((3, 500))
X[1] = 0.5*X[0] + 0.5*rng.standard_normal(500)
X[2] = -0.3*X[0] + rng.standard_normal(500)
cov_mat = np.cov(X)
print("Covariance matrix:\\n", np.round(cov_mat, 3))

# Law of total variance
Y = rng.binomial(1, 0.4, 500)
X_sim = np.where(Y==0, rng.normal(0,1,500), rng.normal(3,1,500))
print("\\nE[X]          =", X_sim.mean())
print("Law of total E =", 0 * 0.6 + 3 * 0.4)
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Linearity of expectation holds always — even for dependent variables (unlike variance).",
        "Var[X+Y] = Var[X] + Var[Y] only if X⊥Y; in general add 2Cov[X,Y].",
        "Correlation ρ∈[-1,1] measures linear dependence; ρ=0 does not imply independence.",
        "Jensen's inequality: E[f(X)] ≥ f(E[X]) for convex f — foundation of EM algorithm proof.",
        "Law of total variance: Var[X] = E[Var[X|Y]] + Var[E[X|Y]] — within + between group variance.",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Central Limit Theorem
# ─────────────────────────────────────────────────────────────────────────────
def topic_clt():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Central Limit Theorem")
    section_header("CENTRAL LIMIT THEOREM")

    section_header("1. THEORY")
    print(white("  The Central Limit Theorem (CLT) is one of the most profound results in"))
    print(white("  probability. It states that the sum of n i.i.d. random variables, regardless"))
    print(white("  of their original distribution, converges in distribution to a Gaussian as n→∞."))
    print()
    print(white("  Formal statement: let X₁,...,Xₙ be i.i.d. with mean μ and variance σ²."))
    print(white("  Then (X̄ₙ − μ)/(σ/√n) → N(0,1) in distribution."))
    print()
    print(white("  Why it matters: statistical inference (t-tests, confidence intervals), sampling,"))
    print(white("  and ML theory all rely on CLT. The ubiquity of Gaussian distributions in nature"))
    print(white("  is partly explained by it — many quantities are sums of many small effects."))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  CLT:    (X̄ₙ − μ) / (σ/√n) → N(0,1)  as n → ∞"))
    print(formula("  X̄ₙ ~ N(μ, σ²/n)  approximately"))
    print(formula("  Speed:  O(1/√n) Berry-Esseen: |F_n(x) − Φ(x)| ≤  C·E[|X|³]/(σ³√n)"))
    print(formula("  Works for: continuous, discrete, bounded, heavy-tailed distributions"))
    print(formula("  Does NOT work for: infinite variance (Cauchy), dependent data"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    rng = np.random.default_rng(42)
    ns = [1, 5, 30, 100]
    M  = 5000  # number of samples of X̄ₙ

    distributions = {
        "Uniform(0,1)": (lambda n: rng.uniform(0, 1, (M, n)), 0.5, 1/12),
        "Exp(λ=1)":     (lambda n: rng.exponential(1, (M, n)), 1.0, 1.0),
        "Bernoulli(0.3)":(lambda n: rng.binomial(1, 0.3, (M, n)), 0.3, 0.21),
    }

    print(white("  Sample mean distribution (5000 samples of X̄ₙ):"))
    print()
    for dist_name, (sampler, mu, var) in distributions.items():
        print(white(f"  {dist_name}:"))
        for n in ns:
            samples = sampler(n)
            means   = samples.mean(axis=1)
            std_err = np.sqrt(var / n)
            obs_mean = means.mean(); obs_std = means.std()
            norm_stat = (means - mu) / std_err
            # Check normality: how well does N(0,1) fit?
            z_skew = np.mean(norm_stat**3)
            print(f"    n={n:>3}:  E[X̄]={obs_mean:.3f}(≈{mu:.3f}),  Std={obs_std:.4f}(≈{std_err:.4f}),  skew={z_skew:+.3f}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  CLT demo: Exponential(λ=1) sample means for n=1,5,30,100"))
    print()
    for n in ns:
        samples = rng.exponential(1, (3000, n)).mean(axis=1)
        mu_s, std_s = samples.mean(), samples.std()
        bins = np.linspace(0, 3, 16)
        hist, _ = np.histogram(samples, bins=bins)
        max_h = max(hist)
        print(f"  n={n:>3} (μ≈{mu_s:.2f}, σ≈{std_s:.3f}):")
        bar_str = ""
        for h in hist:
            bar_h = int(8 * h / max_h)
            bar_str += "█" * bar_h + " "
        colorf = green if n >= 30 else (yellow if n >= 5 else red)
        print(f"  {colorf(bar_str)}")
    print(grey("\n  Green = good Gaussian approximation (n≥30)"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_distribution
        all_means = []
        labels_clt = []
        for n in [1, 5, 30, 100]:
            means = rng.exponential(1, (2000, n)).mean(axis=1)
            # Standardise
            std_err = np.sqrt(1.0/n)
            z = ((means - 1.0) / std_err).tolist()
            all_means.append(z)
            labels_clt.append(f"n={n}")
        multi_distribution(all_means, labels=labels_clt,
                           title="CLT: Exp(1) standardised sample means",
                           n_bins=30)
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        xs_m = np.linspace(-4, 4, 200)
        from scipy.stats import norm as sp_norm
        for ax, n in zip(axes.flat, [1, 5, 30, 100]):
            means = rng.exponential(1, (5000, n)).mean(axis=1)
            std_err = np.sqrt(1.0 / n)
            z_scores = (means - 1.0) / std_err
            ax.hist(z_scores, bins=50, density=True, alpha=0.6, color='steelblue', label='Sample means')
            ax.plot(xs_m, sp_norm.pdf(xs_m), 'r--', lw=2, label='N(0,1)')
            ax.set_title(f"n = {n}"); ax.legend()
        fig.suptitle("CLT: Exp(1) sample means → N(0,1)")
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Central Limit Theorem", """\
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def clt_demo(dist_sampler, mu, sigma, ns=[1,5,30,100], M=3000):
    \"\"\"Show CLT convergence for different n.\"\"\"
    for n in ns:
        samples = dist_sampler(M, n)
        means   = samples.mean(axis=1)
        z_scores = (means - mu) / (sigma / np.sqrt(n))
        print(f"n={n:>4}: E[Z]={z_scores.mean():.3f}, Std[Z]={z_scores.std():.3f}",
              "(should be ≈ 0, 1)")

# Exponential(1): mu=1, sigma=1
print("Exponential(1):")
clt_demo(lambda M,n: rng.exponential(1,(M,n)), mu=1, sigma=1)

# Bernoulli(0.3): mu=0.3, sigma=sqrt(0.21)
import math
print("\\nBernoulli(0.3):")
clt_demo(lambda M,n: rng.binomial(1,0.3,(M,n)), mu=0.3, sigma=math.sqrt(0.21))
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "CLT explains why Gaussians are everywhere: sums of many small effects converge to Gaussian.",
        "Convergence rate is O(1/√n); n=30 is 'rule of thumb' for Gaussian approximation to be decent.",
        "Heavy-tailed distributions (α-stable with α<2) violate CLT — extremes dominate.",
        "Bootstrap: resampling uses CLT to build CI without knowing the original distribution.",
        "CLT is the foundation of hypothesis testing: t-statistic = (X̄-μ₀)/(s/√n) → N(0,1).",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Conjugate Priors
# ─────────────────────────────────────────────────────────────────────────────
def topic_conjugate_priors():
    clear()
    breadcrumb("mlmath", "Probability Theory", "Conjugate Priors")
    section_header("CONJUGATE PRIORS")

    section_header("1. THEORY")
    print(white("  A prior P(θ) is conjugate to likelihood P(x|θ) if the posterior P(θ|x)"))
    print(white("  has the same functional form as the prior. This enables closed-form Bayesian"))
    print(white("  updates — no numerical integration needed."))
    print()
    print(white("  Why conjugate priors matter:"))
    print(white("  • Analytical posterior → efficient sequential updating"))
    print(white("  • Interpretable: prior parameters = 'pseudo-observations'"))
    print(white("  • Computationally free: just add counts or update parameters"))
    print()
    print(white("  Key pairs:"))
    print(white("  • Beta(α,β) prior + Binomial likelihood → Beta(α+heads, β+tails) posterior"))
    print(white("  • Dirichlet(α) + Categorical → Dirichlet(α + counts)"))
    print(white("  • Gaussian(μ₀,σ₀²) + Gaussian likelihood → Gaussian posterior"))
    print(white("  • Gamma(α,β) + Poisson → Gamma(α+Σx, β+n)"))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Beta-Bernoulli:   p ~ Beta(α,β),  x|p ~ Bern(p)"))
    print(formula("  Posterior:        p|x ~ Beta(α + #heads, β + #tails)"))
    print(formula("  Posterior mean:   E[p|x] = (α+heads)/(α+β+n)"))
    print(formula("  Dirichlet-Cat:    π ~ Dir(α),  x|π ~ Cat(π)  → π|x ~ Dir(α+counts)"))
    print(formula("  Gaussian-Gaussian: β ~ N(μ₀,σ²₀), y|β ~ N(Xβ,σ²) → N(μₙ,σ²ₙ)"))
    _pause()

    section_header("3. WORKED EXAMPLE")
    print(white("  Beta-Binomial conjugate update:"))
    print()

    # Initial prior
    alpha_0, beta_0 = 2, 5
    print(white(f"  Prior: Beta({alpha_0}, {beta_0})"))
    prior_mean = alpha_0 / (alpha_0 + beta_0)
    print(white(f"  Prior mean = {prior_mean:.4f}"))
    print()

    # Observe data: 7 heads, 3 tails
    heads, tails = 7, 3
    alpha_n = alpha_0 + heads
    beta_n  = beta_0 + tails
    posterior_mean = alpha_n / (alpha_n + beta_n)
    posterior_mode = (alpha_n - 1) / (alpha_n + beta_n - 2)
    posterior_var  = alpha_n * beta_n / ((alpha_n + beta_n)**2 * (alpha_n + beta_n + 1))

    print(white(f"  Data: {heads} heads, {tails} tails"))
    print(white(f"  Posterior: Beta({alpha_n}, {beta_n})"))
    print(white(f"  Posterior mean    = {posterior_mean:.4f}  (MLE would be {heads/(heads+tails):.4f})"))
    print(white(f"  Posterior mode    = {posterior_mode:.4f}"))
    print(white(f"  Posterior std     = {sqrt(posterior_var):.4f}"))
    print()
    print(white("  Sequential updates (same endpoint regardless of order):"))
    # Update one flip at a time
    a, b = alpha_0, beta_0
    for flip in ["H"]*heads + ["T"]*tails:
        a += (1 if flip == "H" else 0); b += (1 if flip == "T" else 0)
        if (a + b - alpha_0 - beta_0) in [1, 3, 7, 10]:
            print(f"  After {a+b-alpha_0-beta_0} flips: Beta({a},{b})  mean={a/(a+b):.4f}")
    _pause()

    section_header("4. ASCII VISUALIZATION")
    print(white("  Beta distribution: Prior Beta(2,5), Posterior Beta(9,8):"))
    print()
    xs_b = np.linspace(0.01, 0.99, 60)

    def beta_pdf_manual(x, a, b):
        from math import gamma
        return x**(a-1) * (1-x)**(b-1) * gamma(a+b) / (gamma(a) * gamma(b))

    prior_vals = [beta_pdf_manual(x, alpha_0, beta_0) for x in xs_b]
    post_vals  = [beta_pdf_manual(x, alpha_n, beta_n)  for x in xs_b]
    max_p = max(max(prior_vals), max(post_vals))

    print(cyan(f"  Prior Beta({alpha_0},{beta_0}):"))
    row = "  "
    for v in prior_vals[::3]:
        bar_h = int(8 * v / max_p)
        row += "▁▂▃▄▅▆▇█"[min(bar_h, 7)] if bar_h > 0 else " "
    print(cyan(row))
    print(green(f"\n  Posterior Beta({alpha_n},{beta_n}):"))
    row = "  "
    for v in post_vals[::3]:
        bar_h = int(8 * v / max_p)
        row += "▁▂▃▄▅▆▇█"[min(bar_h, 7)] if bar_h > 0 else " "
    print(green(row))
    print(grey("  0                   p=0.5                   1"))
    _pause()

    section_header("5. PLOTEXT")
    try:
        from viz.terminal_plots import multi_distribution
        rng = np.random.default_rng(99)
        prior_s    = rng.beta(alpha_0, beta_0, 3000).tolist()
        post_s     = rng.beta(alpha_n, beta_n, 3000).tolist()
        # likelihood: P(7H, 3T | p) ∝ p^7(1-p)^3 — approximate via rejection
        possible_p = rng.uniform(0, 1, 30000)
        weights    = possible_p**heads * (1 - possible_p)**tails
        weights   /= weights.sum()
        lik_s      = list(rng.choice(possible_p, size=3000, p=weights))
        multi_distribution([prior_s, lik_s, post_s],
                           labels=[f"Prior Beta({alpha_0},{beta_0})",
                                   "Likelihood p^7(1-p)^3",
                                   f"Posterior Beta({alpha_n},{beta_n})"],
                           title="Beta-Binomial Conjugate Update", n_bins=40)
    except Exception as e:
        print(grey(f"  plotext unavailable: {e}"))
    _pause()

    section_header("6. MATPLOTLIB")
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import beta as beta_dist
        xs_m = np.linspace(0.001, 0.999, 300)
        plt.figure(figsize=(8, 4))
        plt.plot(xs_m, beta_dist.pdf(xs_m, alpha_0, beta_0), 'b-', lw=2,
                 label=f'Prior Beta({alpha_0},{beta_0})', alpha=0.8)
        unnorm_lik = xs_m**heads * (1-xs_m)**tails
        unnorm_lik /= np.trapz(unnorm_lik, xs_m)
        plt.plot(xs_m, unnorm_lik, 'g--', lw=2, label=f'Likelihood p^{heads}(1-p)^{tails}', alpha=0.8)
        plt.plot(xs_m, beta_dist.pdf(xs_m, alpha_n, beta_n), 'r-', lw=2,
                 label=f'Posterior Beta({alpha_n},{beta_n})', alpha=0.8)
        plt.axvline(posterior_mean, color='red', ls=':', label=f'Post. mean={posterior_mean:.3f}')
        plt.xlabel("p"); plt.ylabel("density")
        plt.title("Beta-Binomial Conjugate Update"); plt.legend()
        plt.tight_layout(); plt.show()
    except ImportError:
        print(grey("  matplotlib not installed"))
    _pause()

    section_header("7. PYTHON CODE")
    code_block("Conjugate Prior Update", """\
import numpy as np

def beta_binomial_update(alpha0, beta0, heads, tails):
    \"\"\"Update Beta prior with Binomial data.\"\"\"
    alpha_n = alpha0 + heads
    beta_n  = beta0 + tails
    mean_n  = alpha_n / (alpha_n + beta_n)
    var_n   = alpha_n * beta_n / ((alpha_n+beta_n)**2 * (alpha_n+beta_n+1))
    return alpha_n, beta_n, mean_n, var_n

# Prior: Beta(2, 5) — skeptical about high p
a0, b0 = 2, 5
print(f"Prior: Beta({a0},{b0}), mean={a0/(a0+b0):.3f}")

# Observe 7 heads, 3 tails
a1, b1, mean1, var1 = beta_binomial_update(a0, b0, 7, 3)
print(f"After 7H, 3T: Beta({a1},{b1}), mean={mean1:.3f}, std={var1**0.5:.3f}")

# Sequential updating is equivalent to batch update
a, b = a0, b0
for outcome in [1,1,0,1,1,0,1,1,0,1]:   # same 7H3T in order
    a, b, _, _ = beta_binomial_update(a, b, outcome, 1-outcome)
print(f"Sequential: Beta({a},{b}), mean={a/(a+b):.3f}")   # same result!

# Conjugate table
print("\\nConjugate pairs:")
print("  Beta(α,β)    + Binomial  → Beta(α+h, β+t)")
print("  Dirichlet(α) + Categorical → Dirichlet(α + counts)")
print("  Gamma(α,β)   + Poisson   → Gamma(α+Σx, β+n)")
print("  Normal(μ₀,σ²₀)+ Normal  → Normal posterior")
""")
    _pause()

    section_header("8. KEY INSIGHTS")
    insights = [
        "Conjugate prior: posterior = same family, just add data to prior parameters.",
        "Beta(1,1)=Uniform is the non-informative conjugate prior for Bernoulli/Binomial.",
        "Posterior mean = weighted average of prior mean and MLE — prior is 'pseudo-data'.",
        "With n→∞ data, posterior concentrates on true parameter regardless of prior (consistency).",
        "Dirichlet-Categorical is the multivariate generalisation — used in NLP topic models (LDA).",
    ]
    for ins in insights:
        print(f"  {green('✦')}  {white(ins)}")
    topic_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Block runner
# ─────────────────────────────────────────────────────────────────────────────
def run():
    topics = [
        ("Probability Axioms",           topic_probability_axioms),
        ("Bayes' Theorem",               topic_bayes_theorem),
        ("Discrete Distributions",       topic_distributions_discrete),
        ("Continuous Distributions",     topic_distributions_continuous),
        ("Joint/Marginal/Conditional",   topic_joint_marginal_conditional),
        ("Expectation & Variance",       topic_expectation_variance),
        ("Central Limit Theorem",        topic_clt),
        ("Conjugate Priors",             topic_conjugate_priors),
    ]
    block_menu("b05", "Probability Theory", topics)
    mark_completed("b05")
