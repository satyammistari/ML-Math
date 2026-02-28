"""Exercise Set 05: Probability"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title
        self.difficulty = difficulty
        self.description = description
        self.hint = hint
        self.starter_code = starter_code
        self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="Bayes' Theorem -- Medical Test",
        difficulty="Beginner",
        description="""
  A disease affects 1% of the population.
  Test: sensitivity (TPR) = 99%, specificity (TNR) = 95%

  If a person tests POSITIVE, what is P(disease | positive)?

  Bayes: P(D|T+) = P(T+|D)*P(D) / P(T+)
  where  P(T+)  = P(T+|D)*P(D) + P(T+|~D)*P(~D)

  Also compare at multiple prevalence levels.
""",
        hint="""
  P(T+|~D) = 1 - specificity = 0.05  (false positive rate)
  P(T+) = sensitivity*prevalence + fpr*(1-prevalence)
  P(D|T+) = sensitivity*prevalence / P(T+)
""",
        starter_code="""
def bayes_medical(prevalence, sensitivity, specificity):
    # TODO: compute P(disease | positive test)
    fpr   = None  # false positive rate = 1 - specificity
    p_pos = None  # law of total probability
    return None   # Bayes numerator / denominator

for prev in [0.01, 0.10]:
    r = bayes_medical(prev, 0.99, 0.95)
    print(f"Prevalence {prev*100:.0f}%: P(D|T+) = {r:.4f}")
""",
        solution_code="""
def bayes_medical(prevalence, sensitivity, specificity):
    fpr   = 1 - specificity
    p_pos = sensitivity * prevalence + fpr * (1 - prevalence)
    return sensitivity * prevalence / p_pos

print("Posterior P(disease | positive test):")
for prev in [0.001, 0.01, 0.05, 0.10, 0.50]:
    r = bayes_medical(prev, 0.99, 0.95)
    print(f"  Prevalence {prev*100:5.1f}% -> {r:.4f} ({r*100:.1f}%)")
print()
print("Key insight: at low prevalence most positives are FALSE positives!")
"""
    ),

    Exercise(
        title="MLE for Gaussian Distribution",
        difficulty="Intermediate",
        description="""
  Implement MLE estimators for N(mu, sigma^2) from scratch:
    mu_MLE     = (1/n) * sum(x_i)
    sigma2_MLE = (1/n) * sum((x_i - mu_MLE)^2)   [biased]
    s2         = (1/(n-1)) * sum((x_i - mu_MLE)^2) [unbiased, Bessel]

  Tasks:
    1. Implement both (no np.std / np.var)
    2. Sample from N(5, 4), compare to truth
    3. Simulate 5000 experiments to confirm E[sigma2_MLE] = (n-1)/n * sigma^2
""",
        hint="""
  For n=10 samples: E[sigma2_MLE] ~ (9/10)*4 = 3.6  (biased down)
                    E[s2]         ~ 4.0              (unbiased)
  The bias exists because mu_MLE is estimated, not known.
""",
        starter_code="""
import numpy as np

def mle_gaussian(x):
    n = len(x)
    mu_hat          = None  # TODO
    sigma2_mle      = None  # TODO biased /n
    sigma2_unbiased = None  # TODO unbiased /(n-1)
    return mu_hat, sigma2_mle, sigma2_unbiased

np.random.seed(42)
x = np.random.normal(5, 2, 10)
mu, s2m, s2u = mle_gaussian(x)
print(f"MLE:      mu={mu:.4f}, sigma2={s2m}")
print(f"Unbiased: mu={mu:.4f}, s2={s2u}")
""",
        solution_code="""
import numpy as np

def mle_gaussian(x):
    n = len(x)
    mu_hat          = sum(x) / n
    sigma2_mle      = sum((xi - mu_hat)**2 for xi in x) / n
    sigma2_unbiased = sum((xi - mu_hat)**2 for xi in x) / (n - 1)
    return mu_hat, sigma2_mle, sigma2_unbiased

np.random.seed(42)
x = np.random.normal(5, 2, 10)
mu, s2m, s2u = mle_gaussian(x)
print(f"MLE:      mu={mu:.4f}, sigma2={s2m:.4f}")
print(f"Unbiased: mu={mu:.4f}, s2={s2u:.4f}")
print(f"True:     mu=5.0000, sigma2=4.0000")

np.random.seed(0)
n_exp, n_obs = 5000, 10
mle_vals, unb_vals = [], []
for _ in range(n_exp):
    xi = np.random.normal(5, 2, n_obs)
    _, s2m_, s2u_ = mle_gaussian(xi)
    mle_vals.append(s2m_)
    unb_vals.append(s2u_)
print(f"\\n{n_exp} experiments with n={n_obs}:")
print(f"  E[sigma2_MLE] = {np.mean(mle_vals):.4f}  (theory: {4*(n_obs-1)/n_obs:.4f})")
print(f"  E[s2_unbiased]= {np.mean(unb_vals):.4f}  (theory: 4.0000)")
"""
    ),

    Exercise(
        title="Monte Carlo Estimation of Pi",
        difficulty="Advanced",
        description="""
  Estimate pi using Monte Carlo sampling.

  1. Sample (x,y) uniform in [-1,1]^2, count inside unit circle
       pi ~= 4 * (# inside) / (# total)

  2. Show convergence O(1/sqrt(n)):
     test n = 100, 1000, 10000, 100000

  3. Variance study: 200 independent estimates at n=5000.
     Compare empirical std to theoretical sqrt(pi*(4-pi)/n).
""",
        hint="""
  inside = (x**2 + y**2) <= 1
  pi_est = 4 * inside.mean()
  Theoretical std = sqrt(pi*(4-pi)/n)  ~= 0.023 for n=5000
""",
        starter_code="""
import numpy as np

def estimate_pi(n, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    # TODO: return pi estimate
    pass

for n in [100, 1000, 10000, 100000]:
    est = estimate_pi(n)
    if est:
        print(f"n={n:>7}: pi~={est:.5f}, err={abs(est-3.14159):.5f}")
""",
        solution_code="""
import numpy as np

def estimate_pi(n, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    return 4 * ((x**2 + y**2) <= 1).mean()

print("Convergence:")
for n in [100, 1000, 10000, 100000]:
    est = estimate_pi(n)
    print(f"  n={n:>7}: pi~={est:.5f}, err={abs(est-np.pi):.5f}")

print("\\nVariance at n=5000 (200 runs):")
ests = [estimate_pi(5000, seed=i) for i in range(200)]
emp  = np.std(ests)
theo = np.sqrt(np.pi*(4-np.pi)/5000)
print(f"  Empirical std:   {emp:.5f}")
print(f"  Theoretical std: {theo:.5f}")
print(f"  Ratio: {emp/theo:.3f}  (should be ~1.0)")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m" + "\u2554" + "\u2550"*50 + "\u2557" + "\033[0m")
        print("\033[96m" + "\u2551" + "     EXERCISES -- Block 05: Probability           " + "\u2551" + "\033[0m")
        print("\033[96m" + "\u255a" + "\u2550"*50 + "\u255d" + "\033[0m\n")
        for i, ex in enumerate(exercises, 1):
            dc = ("\033[92m" if ex.difficulty == "Beginner" else
                  "\033[93m" if ex.difficulty == "Intermediate" else "\033[91m")
            print(f"  {i}. {dc}[{ex.difficulty}]\033[0m {ex.title}")
        print("\n  \033[90m[0] Back\033[0m")
        choice = input("\n\033[96mSelect exercise: \033[0m").strip()
        if choice == "0":
            break
        try:
            _run_exercise(exercises[int(choice) - 1])
        except (ValueError, IndexError):
            pass


def _run_exercise(ex):
    dc = ("\033[92m" if ex.difficulty == "Beginner" else
          "\033[93m" if ex.difficulty == "Intermediate" else "\033[91m")
    print(f"\n\033[95m--- {ex.title} ---\033[0m")
    print(f"  Difficulty: {dc}{ex.difficulty}\033[0m\n")
    print("\033[1mPROBLEM\033[0m")
    print(ex.description)
    while True:
        cmd = input("\n  [h]int  [c]ode  [r]un  [s]olution  [b]ack: ").strip().lower()
        if cmd == "b":
            break
        elif cmd == "h":
            print(f"\n\033[93mHINT\033[0m\n{ex.hint}")
        elif cmd == "c":
            print(f"\n\033[94mSTARTER CODE\033[0m\n{ex.starter_code}")
        elif cmd == "s":
            print(f"\n\033[92mSOLUTION\033[0m\n{ex.solution_code}")
        elif cmd == "r":
            print("\n\033[92mRunning...\033[0m")
            try:
                exec(compile(ex.solution_code, "<solution>", "exec"), {})
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")
