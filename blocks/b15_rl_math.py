"""
blocks/b15_rl_math.py
Block 15: Reinforcement Learning Math
Topics: MDP, Bellman, Dynamic Programming, Monte Carlo, Temporal Difference,
        Policy Gradient, GAE, Actor-Critic, DQN, Exploration.
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
from viz.ascii_plots import scatter
from viz.terminal_plots import distribution_plot, loss_curve_plot
from viz.matplotlib_plots import show_heatmap, plot_decision_boundary


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — MDP
# ══════════════════════════════════════════════════════════════════════════════
def topic_mdp():
    clear()
    breadcrumb("mlmath", "RL Math", "Markov Decision Process")
    section_header("MARKOV DECISION PROCESS (MDP)")
    print()

    section_header("1. THEORY")
    print(white("""
  A Markov Decision Process (MDP) formalises sequential decision-making under
  uncertainty. An MDP is defined by the tuple (S, A, P, R, γ):

  • S: state space (finite or continuous)
  • A: action space (finite or continuous)
  • P(s'|s,a): transition probability — probability of reaching s' from s via a
  • R(s,a): expected reward for taking action a in state s
  • γ ∈ [0,1): discount factor — how much to prefer immediate vs future rewards

  MARKOV PROPERTY: the future depends only on the current state, not the history:
      P(Sₜ₊₁ = s' | S₁,A₁,...,Sₜ,Aₜ) = P(Sₜ₊₁ = s' | Sₜ, Aₜ)

  POLICY: π(a|s) = probability of taking action a in state s. We seek an optimal
  policy π* that maximises the expected total discounted return.

  RETURN: the total discounted reward from time t:
      Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... = Σₖ₌₀^∞ γᵏ rₜ₊ₖ₊₁
  Discount γ<1 ensures geometric series converges and encodes time preference.

  VALUE FUNCTIONS:
  • State-value:   V^π(s) = E_π[Gₜ | Sₜ=s]   — expected return from state s
  • Action-value:  Q^π(s,a) = E_π[Gₜ | Sₜ=s, Aₜ=a]
  Relationship: V^π(s) = Σₐ π(a|s) Q^π(s,a)

  OPTIMALITY: V*(s) = max_π V^π(s). Any policy π* greedy w.r.t. V* is optimal.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Return:     Gₜ = Σₖ₌₀^∞ γᵏ rₜ₊ₖ₊₁"))
    print(formula("  V^π(s) = E[Gₜ | Sₜ=s, π]"))
    print(formula("  Q^π(s,a) = E[Gₜ | Sₜ=s, Aₜ=a, π]"))
    print(formula("  V^π(s) = Σₐ π(a|s) Q^π(s,a)"))
    print(formula("  Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Simple 3-State MDP")
    rng = np.random.default_rng(0)
    # States: 0=Start, 1=Good, 2=Terminal; Actions: go_left=0, go_right=1
    S, A, gamma = 3, 2, 0.9
    # P[s, a, s'] = transition prob
    P = np.zeros((S, A, S))
    P[0, 0, 0] = 1.0         # go left from start → stay
    P[0, 1, 1] = 0.7; P[0, 1, 0] = 0.3  # go right: 70% → Good
    P[1, 0, 0] = 0.4; P[1, 0, 1] = 0.6
    P[1, 1, 2] = 1.0         # from Good, go right → Terminal
    P[2, :, 2] = 1.0         # Terminal absorbs
    R = np.array([[-0.1, 0.2], [0.5, 1.0], [0.0, 0.0]])  # R[s,a]

    # Evaluate random policy
    pi = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    print(f"\n  {bold_cyan('3-State MDP: States=[Start, Good, Terminal], Actions=[Left, Right]')}\n")
    state_names = ["Start", "Good ", "Terminal"]
    action_names = ["Left", "Right"]
    print(f"  Transition probabilities P(s'|s,a):\n")
    for s in range(S):
        for a in range(A):
            probs = ", ".join([f"{state_names[s2]}:{P[s,a,s2]:.1f}" for s2 in range(S)])
            print(f"  P(·|{state_names[s]},{action_names[a]}) = [{probs}]  R={R[s,a]:+.1f}")
    print()
    print(f"  Discount γ = {gamma}")
    print(f"  Expected return from Start (random policy): discounted geometric series")
    for g in [0.5, 0.9, 0.99]:
        approx = R[0, :].mean() + g * (R[1, :].mean()) / (1 - g)
        print(f"  γ={g}: rough return ≈ {green(f'{approx:.3f}')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("MDP class", """
import numpy as np

class MDP:
    def __init__(self, n_states, n_actions, P, R, gamma=0.9):
        self.S, self.A = n_states, n_actions
        self.P = P          # P[s, a, s']
        self.R = R          # R[s, a]
        self.gamma = gamma

    def step(self, s, a, rng):
        '''Sample next state and reward.'''
        s_next = rng.choice(self.S, p=self.P[s, a])
        return s_next, self.R[s, a]

    def evaluate_policy(self, pi, n_episodes=1000, max_steps=200, rng=None):
        rng = rng or np.random.default_rng(0)
        returns = []
        for _ in range(n_episodes):
            s = rng.choice(self.S - 1)  # random non-terminal start
            G = 0.0; discount = 1.0
            for _ in range(max_steps):
                if s == self.S - 1: break
                a = rng.choice(self.A, p=pi[s])
                s, r = self.step(s, a, rng)
                G += discount * r; discount *= self.gamma
            returns.append(G)
        return np.mean(returns), np.std(returns)

P = np.zeros((3, 2, 3)); P[0,0,0]=1; P[0,1,1]=0.7; P[0,1,0]=0.3
P[1,0,0]=0.4; P[1,0,1]=0.6; P[1,1,2]=1; P[2,:,2]=1
R = np.array([[-0.1, 0.2],[0.5, 1.0],[0.0,0.0]])
mdp = MDP(3, 2, P, R, gamma=0.9)

# Compare policies
pi_rand  = np.array([[0.5,0.5],[0.5,0.5],[0.5,0.5]])
pi_greedy= np.array([[0.0,1.0],[0.0,1.0],[0.5,0.5]])
mu_r, _ = mdp.evaluate_policy(pi_rand)
mu_g, _ = mdp.evaluate_policy(pi_greedy)
print(f"Random policy return:  {mu_r:.4f}")
print(f"Greedy policy return:  {mu_g:.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "MDP = formal framework for sequential decision-making under uncertainty",
        "Markov property: optimal policy depends only on current state, not history",
        "Discount γ < 1: ensures convergence + encodes preference for immediate rewards",
        "V(s) and Q(s,a) are dual value functions; V(s) = Σₐ π(a|s) Q(s,a)",
        "Optimal policy can always be found; hard when S, A are large/continuous",
        "Extension to POMDPs: agent receives observations, not full state",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Bellman Equations
# ══════════════════════════════════════════════════════════════════════════════
def topic_bellman():
    clear()
    breadcrumb("mlmath", "RL Math", "Bellman Equations")
    section_header("BELLMAN EQUATIONS")
    print()

    section_header("1. THEORY")
    print(white("""
  The Bellman equations express the recursive structure of value functions:
  the value of a state is the immediate reward plus the (discounted) value
  of the next state.

  BELLMAN EXPECTATION (for policy π):
      V^π(s) = Σₐ π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
      Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σₐ' π(a'|s') Q^π(s',a')

  These are linear equations in V^π. In matrix form (for finite MDPs):
      V^π = R^π + γ P^π V^π   →   V^π = (I - γ P^π)⁻¹ R^π
  where R^π_s = Σₐ π(a|s) R(s,a) and P^π_{ss'} = Σₐ π(a|s) P(s'|s,a).

  BELLMAN OPTIMALITY:
      V*(s)   = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
      Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
  These nonlinear equations define the optimal value function V*.
  Key property: the Bellman optimality operator 𝒯* is a γ-contraction (Banach's
  fixed point theorem), so iterating 𝒯* converges to V* from any initialisation.

  ASCII BACKUP DIAGRAM shows how value propagates from successor states:

      V(s)  ←───  [a₁] ──── P → s₁' → r₁
                  [a₂] ──── P → s₂' → r₂
  The value of s backs up through actions and transitions.
"""))
    print()
    print(cyan("  Backup diagram for V(s):"))
    print(grey("  ") + cyan("s") + grey("──[a₁]──P(0.7)──") + green("s₁'") + grey(" r₁=+1"))
    print(grey("  │") + grey("       \\─P(0.3)──") + green("s₂'") + grey(" r₁=-1"))
    print(grey("  └──[a₂]──P(1.0)──") + green("s₃'") + grey(" r₂=+2"))
    print()
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Expectation: V^π(s) = Σₐπ(a|s)[R(s,a) + γΣ_{s'}P(s'|s,a)V^π(s')]"))
    print(formula("  Matrix form: V^π = (I − γP^π)⁻¹ R^π"))
    print(formula("  Optimality:  V*(s) = maxₐ[R(s,a) + γΣ_{s'}P(s'|s,a)V*(s')]"))
    print(formula("  Contraction: ‖𝒯*V - 𝒯*U‖∞ ≤ γ‖V - U‖∞  (→ fixed point)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Matrix Bellman Solution")
    # 4-state chain MDP with uniform random policy
    n_states = 4; gamma = 0.9
    P_pi = np.array([[0.9, 0.1, 0.0, 0.0],
                     [0.1, 0.8, 0.1, 0.0],
                     [0.0, 0.1, 0.8, 0.1],
                     [0.0, 0.0, 0.1, 0.9]])
    R_pi = np.array([-0.1, 0.2, 0.5, 1.0])

    V_exact = np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)

    print(f"\n  {bold_cyan('4-State Chain MDP: Exact Bellman Solution V^π = (I-γP^π)⁻¹R^π:')}\n")
    rows = [[str(s), f"{R_pi[s]:.2f}", f"{V_exact[s]:.4f}"] for s in range(n_states)]
    table(["State", "R^π(s)", "V^π(s)"], rows, [cyan, yellow, green])

    print(f"\n  {bold_cyan('Verify: check |V^π(s) - (R + γP^πV^π)(s)| < 1e-10:')}")
    residual = np.max(np.abs(V_exact - (R_pi + gamma * P_pi @ V_exact)))
    print(f"  Max residual = {green(f'{residual:.2e}')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("Bellman equations (matrix + iterative)", """
import numpy as np

def bellman_exact(P_pi, R_pi, gamma=0.9):
    '''Exact policy value via linear solve.'''
    return np.linalg.solve(np.eye(len(R_pi)) - gamma * P_pi, R_pi)

def bellman_iteration(P_pi, R_pi, V0=None, gamma=0.9, tol=1e-8, max_iter=1000):
    '''Iterative V ← R + γPV until convergence.'''
    V = np.zeros_like(R_pi) if V0 is None else V0.copy()
    for i in range(max_iter):
        V_new = R_pi + gamma * P_pi @ V
        if np.max(np.abs(V_new - V)) < tol: return V_new, i
        V = V_new
    return V, max_iter

# Bellman optimality (2-state, 2-action)
P = np.array([[[0.8,0.2],[0.2,0.8]],
              [[0.1,0.9],[0.9,0.1]]])
R = np.array([[0.0, 1.0], [0.5, -0.1]])

def bellman_optimality(P, R, gamma=0.9, tol=1e-8):
    S, A, _ = P.shape; V = np.zeros(S)
    for _ in range(10000):
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol: break
        V = V_new
    pi_star = Q.argmax(axis=1)
    return V, pi_star

V_star, pi = bellman_optimality(P, R)
print(f"V*(s0)={V_star[0]:.4f}, V*(s1)={V_star[1]:.4f}")
print(f"Optimal actions: {['left','right'][pi[0]]}, {['left','right'][pi[1]]}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Bellman equations decompose V(s) into immediate reward + discounted future value",
        "Linear Bellman expectation → solvable exactly as (I-γP)V = R (O(n³) for n states)",
        "Bellman optimality operator 𝒯* is γ-contraction → unique fixed point V*",
        "Value iteration converges geometrically at rate γ per sweep",
        "Backup diagram: value propagates bottom-up from leaves to root",
        "Foundation for Q-learning, TD, SARSA — all approximate Bellman equations",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Dynamic Programming
# ══════════════════════════════════════════════════════════════════════════════
def topic_dynamic_programming():
    clear()
    breadcrumb("mlmath", "RL Math", "Dynamic Programming")
    section_header("DYNAMIC PROGRAMMING (DP) IN RL")
    print()

    section_header("1. THEORY")
    print(white("""
  Dynamic Programming (DP) algorithms compute exact value functions by exploiting
  the MDP's recursive Bellman structure. They require full model (P, R) — the
  "planning" setting. Two main algorithms:

  VALUE ITERATION (VI): iterates the Bellman optimality operator until convergence:
      V_{k+1}(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V_k(s')]
  Contraction theorem guarantees ‖V_k - V*‖∞ ≤ γᵏ‖V₀ - V*‖∞.
  Implicit policy: π_k(s) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V_k(s')].

  POLICY ITERATION (PI): alternates policy evaluation and improvement:
  1. POLICY EVALUATION: solve (I - γP^π)V^π = R^π (exactly or iteratively)
  2. POLICY IMPROVEMENT: π'(s) = argmax_a Q^π(s,a) = argmax_a [R + γP·V^π]
  Monotone improvement theorem: V^{π'} ≥ V^π; converges in finite steps.

  COMPARISON:
  - VI: O(|S|²|A|) per sweep, many sweeps; no policy kept explicitly
  - PI: O(|S|³) for exact eval + O(|S|²|A|) improvement; fewer major iterations
  - Modified policy iteration: partial policy evaluation (k Bellman sweeps) bridges VI and PI

  GRIDWORLD: demonstrate on 4×4 grid with 4 actions (N,S,E,W), -1 reward per step,
  terminal state at (0,0) and (3,3).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  VI:  V_{k+1}(s) = maxₐ [R(s,a) + γ Σ_{s'} P(s'|s,a) Vₖ(s')]"))
    print(formula("  PI eval:  V^π = (I - γP^π)⁻¹ R^π"))
    print(formula("  PI improve:  π'(s) = argmaxₐ Q^π(s,a) = argmaxₐ [R(s,a) + γΣP·V^π]"))
    print(formula("  Error bound:  ‖Vₖ - V*‖∞ ≤ γᵏ / (1-γ) · ‖V₁ − V₀‖∞"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 4×4 GridWorld")
    grid_h, grid_w = 4, 4
    n_states  = grid_h * grid_w
    n_actions = 4  # 0=N, 1=S, 2=E, 3=W
    gamma = 0.9
    terminals = {0, 15}  # (0,0) and (3,3)

    def state_to_rc(s): return s // grid_w, s % grid_w
    def rc_to_state(r, c): return r * grid_w + c

    P = np.zeros((n_states, n_actions, n_states))
    R = np.full((n_states, n_actions), -1.0)
    for s in terminals:
        P[s, :, s] = 1.0; R[s, :] = 0.0

    moves = [(-1,0), (1,0), (0,1), (0,-1)]
    for s in range(n_states):
        if s in terminals: continue
        r, c = state_to_rc(s)
        for a, (dr, dc) in enumerate(moves):
            nr, nc = max(0, min(grid_h-1, r+dr)), max(0, min(grid_w-1, c+dc))
            s_next = rc_to_state(nr, nc)
            P[s, a, s_next] = 1.0

    # Value iteration
    V = np.zeros(n_states); pi_grid = np.zeros(n_states, dtype=int)
    errors = []
    for i in range(200):
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        V_new = Q.max(axis=1)
        errors.append(np.max(np.abs(V_new - V)))
        V = V_new; pi_grid = Q.argmax(axis=1)
        if errors[-1] < 1e-8: break

    print(f"\n  {bold_cyan('4×4 GridWorld: Value Iteration (converged in {k} sweeps)'.replace('{k}', str(len(errors))))}\n")
    print(f"  {'Value function V*(s):':}\n")
    dir_chars = ['↑', '↓', '→', '←']
    for r in range(grid_h):
        row_v, row_pi = "", ""
        for c in range(grid_w):
            s = rc_to_state(r, c)
            row_v  += f"{V[s]:+5.1f} "
            row_pi += f"  {dir_chars[pi_grid[s]] if s not in terminals else '●'}  "
        print(f"  {green(row_v)}")
    print(f"\n  {bold_cyan('Optimal policy:')}")
    for r in range(grid_h):
        row_pi = ""
        for c in range(grid_w):
            s = rc_to_state(r, c)
            row_pi += f"  {yellow(dir_chars[pi_grid[s]]) if s not in terminals else cyan('●')}  "
        print(f"  {row_pi}")
    print()
    print_sparkline(errors[:30], label="Max ΔV per sweep", color_fn=cyan)
    print()
    _pause()

    section_header("4. CODE")
    code_block("Value Iteration and Policy Iteration", """
import numpy as np

def value_iteration(P, R, gamma=0.9, tol=1e-8):
    S = R.shape[0]; V = np.zeros(S)
    for _ in range(10000):
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new
    return V, Q.argmax(axis=1)

def policy_iteration(P, R, gamma=0.9):
    S, A, _ = P.shape
    pi = np.zeros(S, dtype=int)
    for _ in range(1000):
        # Policy evaluation (exact)
        P_pi = P[np.arange(S), pi]
        R_pi = R[np.arange(S), pi]
        V = np.linalg.solve(np.eye(S) - gamma*P_pi, R_pi)
        # Policy improvement
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        pi_new = Q.argmax(axis=1)
        if np.all(pi_new == pi): return V, pi
        pi = pi_new
    return V, pi

# Compare on 3-state MDP
P3 = np.array([[[0.8,0.2,0.0],[0.2,0.8,0.0]],
               [[0.0,0.9,0.1],[0.1,0.0,0.9]],
               [[0.0,0.0,1.0],[0.0,0.0,1.0]]])
R3 = np.array([[-0.1, 0.2],[0.5, 1.0],[0.0, 0.0]])
V_vi, pi_vi = value_iteration(P3, R3)
V_pi, pi_pi = policy_iteration(P3, R3)
print(f"VI: V*={V_vi}, π*={pi_vi}")
print(f"PI: V*={V_pi}, π*={pi_pi}")
print(f"Match: {np.allclose(V_vi, V_pi, atol=1e-6)}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "DP requires full model (P, R) — tractable for small/medium MDPs only",
        "Value iteration: simple, no policy tracked; converges geometrically at rate γ",
        "Policy iteration: fewer iterations (each more expensive) — often faster in practice",
        "Contraction theorem guarantees VI converges from any initialisation",
        "GridWorld: elegant test bed — discounted -1/step reward drives agent to terminal",
        "DP is the backbone — Q-learning, TD, etc. are model-free approximations",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Monte Carlo Methods
# ══════════════════════════════════════════════════════════════════════════════
def topic_monte_carlo():
    clear()
    breadcrumb("mlmath", "RL Math", "Monte Carlo Methods")
    section_header("MONTE CARLO METHODS IN RL")
    print()

    section_header("1. THEORY")
    print(white("""
  Monte Carlo (MC) methods learn from complete episodes of experience, requiring
  no model of the environment (model-free). They compute empirical averages of
  returns to estimate value functions.

  FIRST-VISIT MC: for each episode, for the first time state s is visited:
      V(s) ← V(s) + (1/N(s)) · (G - V(s))   [incremental mean update]
  where N(s) is the visit count and G is the return from that first visit.

  EVERY-VISIT MC: update V(s) for every visit to s in the episode. Both
  converge to V^π(s) as n → ∞, but have different bias/variance tradeoffs.

  INCREMENTAL UPDATE with constant learning rate α (non-stationary):
      V(s) ← V(s) + α · (G - V(s))
  This tracks a running average with exponential weighting on recent returns.

  COMPARISON WITH DP:
  - MC does not need P(s'|s,a) — learns directly from interaction
  - MC updates only states actually visited in the episode
  - MC has higher variance but zero bias (uses actual returns, not estimates)
  - Bootstrap: DP/TD use estimated values; MC waits for full episode return

  EXPLORATION: standard MC explores only visited (s,a) pairs. ε-greedy
  ensures all state-action pairs are eventually explored (GLIE condition).
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Return:    G_t = r_{t+1} + γr_{t+2} + ... + γ^{T-t-1} r_T"))
    print(formula("  First-visit MC: V(s) ← V(s) + (1/N(s))(G - V(s))"))
    print(formula("  α-update: V(s) ← V(s) + α(G - V(s))  (α fixed)"))
    print(formula("  Q MC: Q(s,a) ← Q(s,a) + α(G - Q(s,a))"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — First-Visit MC on Blackjack")
    rng = np.random.default_rng(3)

    def draw_card():
        return min(rng.integers(1, 14), 10)

    def play_episode(V_approx, epsilon=0.3):
        player, dealer_up = draw_card() + draw_card(), draw_card()
        trajectory = []
        while player < 21:
            state = player  # simplified state
            action = 1 if (rng.random() < epsilon or V_approx.get(state, 0) < 0) else 0
            trajectory.append((state, action))
            if action == 0: break  # stick
            player += draw_card()
        dealer = dealer_up
        while dealer < 17: dealer += draw_card()
        if player > 21:
            reward = -1.0
        elif dealer > 21 or player > dealer:
            reward = 1.0
        elif player == dealer:
            reward = 0.0
        else:
            reward = -1.0
        return trajectory, reward

    V_mc = {}; N = {}
    n_episodes = 5000
    win_history = []
    for ep in range(n_episodes):
        traj, G = play_episode(V_mc)
        visited = set()
        for (s, a), _ in [(t, G) for t in traj]:
            if s not in visited:
                visited.add(s)
                N[s] = N.get(s, 0) + 1
                V_mc[s] = V_mc.get(s, 0) + (G - V_mc.get(s, 0)) / N[s]
        win_history.append(1 if G > 0 else 0)

    win_rate = sum(win_history[-500:]) / 500
    print(f"\n  {bold_cyan('Monte Carlo Value Estimation on Simplified Blackjack:')}\n")
    print(f"  Episodes: {n_episodes},  Win rate (last 500): {green(f'{win_rate:.3f}')}")
    print()
    print(f"  {'Player Total':<16} {'MC V(s)':<12} {'Visits'}")
    print(f"  {'─'*13}   {'─'*9}   {'─'*6}")
    for s in sorted(V_mc.keys()):
        if s >= 12:
            bar = green("▓" * max(0, int((V_mc[s]+1)*10))) + red("░" * max(0, int((-V_mc[s])*10)))
            print(f"  V({s:2d})  {bar}  {green(f'{V_mc[s]:+.3f}'):<18} {yellow(str(N.get(s,0)))}")
    print()
    print_sparkline([sum(win_history[i:i+100])/100 for i in range(0, n_episodes-100, 100)],
                    label="Win rate (100-ep):", color_fn=green)
    print()
    _pause()

    section_header("4. CODE")
    code_block("First-visit Monte Carlo", """
import numpy as np
from collections import defaultdict

def first_visit_mc(env_fn, policy, gamma=0.9, n_episodes=1000, alpha=None):
    '''
    env_fn() -> episode = list of (state, action, reward) triples
    policy(state) -> action
    alpha: if None, uses 1/N running mean; else constant step-size
    '''
    V = defaultdict(float); N = defaultdict(int)
    for _ in range(n_episodes):
        episode = env_fn(policy)
        visited = set()
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if s not in visited:
                visited.add(s)
                N[s] += 1
                lr = alpha if alpha else 1.0 / N[s]
                V[s] += lr * (G - V[s])
    return dict(V), dict(N)

def mc_control(env_fn, n_actions, gamma=0.9, n_episodes=5000,
               epsilon=0.1, alpha=0.01):
    '''ε-greedy MC control.'''
    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))

    for ep in range(n_episodes):
        eps = max(0.01, epsilon * (1 - ep/n_episodes))  # decay
        def policy(s):
            if np.random.random() < eps:
                return np.random.randint(n_actions)
            return Q[s].argmax()
        episode = env_fn(policy)
        visited = set()
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma*G + r
            if (s,a) not in visited:
                visited.add((s,a)); N[s][a] += 1
                Q[s][a] += alpha * (G - Q[s][a])
    return {s: Q[s].argmax() for s in Q}, dict(Q)

print("MC control: estimates Q*(s,a) without model, using complete episodes.")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "MC: model-free, learns from complete episodes — no bootstrap needed",
        "First-visit vs every-visit: both converge; first-visit has lower variance",
        "MC = zero bias, high variance (returns vary); TD = high bias, low variance",
        "MC cannot learn from incomplete episodes (e.g., continuous tasks)",
        "GLIE (greedy in the limit with infinite exploration) → convergence of MC control",
        "MC for policy improvement: on-policy (same π samples and updates) or off-policy (importance sampling)",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Temporal Difference
# ══════════════════════════════════════════════════════════════════════════════
def topic_temporal_difference():
    clear()
    breadcrumb("mlmath", "RL Math", "Temporal Difference Learning")
    section_header("TEMPORAL DIFFERENCE (TD) LEARNING")
    print()

    section_header("1. THEORY")
    print(white("""
  TD learning combines the sampling advantage of MC methods with the bootstrapping
  advantage of DP — learning online from incomplete episodes using estimated values.

  TD(0) UPDATE (one-step TD):
      δₜ = rₜ₊₁ + γV(Sₜ₊₁) − V(Sₜ)    [TD error]
      V(Sₜ) ← V(Sₜ) + α δₜ

  The TD error δₜ = actual reward + estimated future − current estimate.
  Positive δₜ: current estimate too low. Negative: too high.

  SARSA (on-policy TD control): updates Q using the behaviour policy's next action A':
      Q(s,a) ← Q(s,a) + α [r + γQ(s',a') − Q(s,a)]
  On-policy: the policy generating experience is the same as the policy being improved.

  Q-LEARNING (off-policy TD control): updates Q using the greedy next action:
      Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]
  Off-policy: learns optimal Q* regardless of the behaviour policy used for exploration.

  EXPECTED SARSA: Q ← Q + α[r + γ Σₐ π(a|s')Q(s',a) − Q(s,a)] — lower variance.

  TD(λ): generalises TD(0) and MC via eligibility traces.
      TD(0) = TD(λ=0) = one-step bootstrap
      MC    = TD(λ=1) = full episode return
      0<λ<1 = exponentially-weighted combination (n-step returns)
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  TD error: δₜ = rₜ₊₁ + γV(sₜ₊₁) - V(sₜ)"))
    print(formula("  TD(0):    V(s) ← V(s) + α·δₜ"))
    print(formula("  SARSA:    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]"))
    print(formula("  Q-learn:  Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Q-learning on GridWorld Chain")
    rng = np.random.default_rng(5)
    n_states, n_actions = 10, 2  # chain: left=0, right=1
    gamma, alpha, epsilon = 0.9, 0.1, 0.3

    def step_chain(s, a):
        if s == n_states - 1: return 0, 1.0  # terminal reward
        if a == 1: s_next = min(s + 1, n_states - 1)  # go right
        else: s_next = max(s - 1, 0)                   # go left
        return s_next, -0.01

    Q = np.zeros((n_states, n_actions))
    ep_rewards = []
    for ep in range(2000):
        s = rng.integers(0, n_states - 1)
        total_r = 0.0
        for _ in range(50):
            a = rng.integers(n_actions) if rng.random() < epsilon else Q[s].argmax()
            s_next, r = step_chain(s, a)
            total_r += r
            td_target = r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
            if s == n_states - 1: break
        ep_rewards.append(total_r)

    print(f"\n  {bold_cyan('Q-learning on 10-state Chain (2000 episodes):')}\n")
    print(f"  {'State':<8} {'Q(s,left)':<14} {'Q(s,right)':<14} {'Best action'}")
    print(f"  {'─'*6}   {'─'*11}   {'─'*11}   {'─'*11}")
    for s in range(n_states):
        best_a = "→ right" if Q[s, 1] > Q[s, 0] else "← left"
        print(f"  s={s:<4}  {yellow(f'{Q[s,0]:.4f}'):<20} {green(f'{Q[s,1]:.4f}'):<20} {cyan(best_a)}")
    print()
    ep_avg = [sum(ep_rewards[i:i+100])/100 for i in range(0, 2000-100, 100)]
    print_sparkline(ep_avg, label="Avg reward/ep (100-ep):", color_fn=green)
    print()
    _pause()

    section_header("4. CODE")
    code_block("TD(0), SARSA, Q-learning comparison", """
import numpy as np

def td0(env_step, n_states, gamma=0.9, alpha=0.1, n_episodes=1000, seed=0):
    rng = np.random.default_rng(seed); V = np.zeros(n_states)
    for _ in range(n_episodes):
        s = rng.integers(0, n_states - 1)
        for _ in range(200):
            a = rng.integers(2)
            s_next, r, done = env_step(s, a)
            V[s] += alpha * (r + gamma * V[s_next] - V[s])
            s = s_next
            if done: break
    return V

def sarsa(env_step, n_states, n_actions, gamma=0.9, alpha=0.1,
          n_episodes=1000, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((n_states, n_actions))
    for ep in range(n_episodes):
        s = rng.integers(0, n_states - 1)
        a = rng.integers(n_actions) if rng.random() < epsilon else Q[s].argmax()
        for _ in range(200):
            s2, r, done = env_step(s, a)
            a2 = rng.integers(n_actions) if rng.random() < epsilon else Q[s2].argmax()
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
            if done: break
    return Q

def qlearning(env_step, n_states, n_actions, gamma=0.9, alpha=0.1,
              n_episodes=1000, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((n_states, n_actions))
    for ep in range(n_episodes):
        s = rng.integers(0, n_states - 1)
        for _ in range(200):
            a = rng.integers(n_actions) if rng.random() < epsilon else Q[s].argmax()
            s2, r, done = env_step(s, a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            s = s2
            if done: break
    return Q

print("SARSA: on-policy (converges to π_ε-greedy's Q)")
print("Q-learning: off-policy (converges to Q* regardless of behaviour policy)")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "TD learns online from incomplete episodes — no need to wait for episode end",
        "TD error δ = r + γV(s') - V(s) is the prediction error — central to RL",
        "SARSA = on-policy (uses a'); Q-learning = off-policy (uses max a') → Q*",
        "TD(0) bootstraps from V(s'); MC uses full return G — TD is lower variance",
        "Q-learning can be unsafe near cliffs (maximises Q*, not Q^π_behaviour)",
        "Double Q-learning: use two Q-tables to reduce maximisation bias",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Policy Gradient
# ══════════════════════════════════════════════════════════════════════════════
def topic_policy_gradient():
    clear()
    breadcrumb("mlmath", "RL Math", "Policy Gradient")
    section_header("POLICY GRADIENT METHODS")
    print()

    section_header("1. THEORY")
    print(white("""
  Policy gradient methods directly optimise the policy π_θ(a|s) by computing
  the gradient of the expected return J(θ) w.r.t. parameters θ.

  OBJECTIVE: J(θ) = E_{τ~π_θ}[R(τ)] = E[Σₜ γᵗ rₜ₊₁]
  where τ = (s₀, a₀, r₁, s₁, a₁, ...) is a trajectory under π_θ.

  POLICY GRADIENT THEOREM (Sutton et al., 1999):
      ∇_θ J(θ) = E_{π_θ} [ Σₜ ∇_θ log π_θ(aₜ|sₜ) · Q^{π_θ}(sₜ, aₜ) ]
  We can estimate Q^π with the sample return Gₜ → REINFORCE algorithm.

  REINFORCE:
      θ ← θ + α · Gₜ · ∇_θ log π_θ(aₜ|sₜ)
  This is an unbiased estimator of ∇J, but has high variance.

  BASELINE SUBTRACTION: uses V^π(sₜ) as a baseline b(sₜ):
      θ ← θ + α · (Gₜ − b(sₜ)) · ∇_θ log π_θ(aₜ|sₜ)
  Subtracting b doesn't bias the gradient (since E[b·∇log π] = 0) but reduces
  variance. The advantage A(s,a) = Q(s,a) - V(s) is the optimal baseline,
  positive when action a is better than average.

  SCORE FUNCTION (log-derivative trick):
      ∇_θ π_θ(a|s) = π_θ(a|s) · ∇_θ log π_θ(a|s)
  This enables gradient estimation by sampling: we only need log π, not ∇π directly.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ∇J(θ) = E_π[Σₜ ∇log π_θ(aₜ|sₜ) · Qᵠ(sₜ,aₜ)]"))
    print(formula("  REINFORCE: θ ← θ + α Gₜ ∇log π_θ(aₜ|sₜ)"))
    print(formula("  With baseline: θ ← θ + α (Gₜ - b(sₜ)) ∇log π_θ(aₜ|sₜ)"))
    print(formula("  Advantage: A(s,a) = Q(s,a) - V(s)"))
    print(formula("  Score fn: ∇_θ π(a|s) = π(a|s) ∇log π(a|s)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — REINFORCE on Bandit")
    rng = np.random.default_rng(12)
    n_arms = 5
    true_rewards = np.array([0.2, 0.4, 0.7, 0.3, 0.9])  # arm 4 is best

    # Softmax policy
    theta = np.zeros(n_arms)
    alpha_pg = 0.1
    returns_history = []
    for ep in range(1000):
        # Softmax probabilities
        logits = theta - theta.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        a = rng.choice(n_arms, p=probs)
        r = true_rewards[a] + rng.normal(0, 0.2)
        # REINFORCE update: ∇log π = one-hot(a) - probs
        grad_log = np.eye(n_arms)[a] - probs
        theta += alpha_pg * r * grad_log
        returns_history.append(r)

    final_probs = np.exp(theta - theta.max()); final_probs /= final_probs.sum()
    print(f"\n  {bold_cyan('REINFORCE on 5-arm Bandit (1000 episodes):')}\n")
    print(f"  {'Arm':<6} {'True μ':<12} {'Final π(a)':<14} {'Best?'}")
    print(f"  {'─'*4}   {'─'*9}   {'─'*11}   {'─'*5}")
    for a in range(n_arms):
        bar = green("█" * int(final_probs[a] * 40))
        best = green("✓") if a == 4 else ""
        print(f"  a={a:<3}  {yellow(f'{true_rewards[a]:.2f}'):<18} "
              f"{bar} {green(f'{final_probs[a]:.3f}'):<16} {best}")

    avg_r = [sum(returns_history[i:i+100])/100 for i in range(0, 1000-100, 100)]
    print()
    print_sparkline(avg_r, label="Avg reward (100-ep):", color_fn=green)
    print()
    _pause()

    section_header("4. CODE")
    code_block("REINFORCE with baseline", """
import numpy as np

def reinforce(env_episode, n_states, n_actions, theta0=None,
              gamma=0.99, alpha_pi=0.01, alpha_v=0.01,
              n_episodes=5000, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((n_states, n_actions)) if theta0 is None else theta0
    V_hat = np.zeros(n_states)

    def policy(s):
        logits = theta[s] - theta[s].max()
        p = np.exp(logits); p /= p.sum()
        return p

    returns_log = []
    for ep in range(n_episodes):
        episode = env_episode(policy)   # [(s,a,r), ...]
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            advantage = G - V_hat[s]
            # Policy gradient
            p = policy(s)
            grad = -p.copy(); grad[a] += 1
            theta[s] += alpha_pi * advantage * grad
            # Value baseline update
            V_hat[s] += alpha_v * advantage
        returns_log.append(G)
    return theta, V_hat, returns_log

print("REINFORCE: unbiased gradient, high variance.")
print("Adding V(s) baseline: same gradient in expectation, much lower variance.")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Policy gradient theorem: ∇J = E[∇log π · Q] — log-derivative trick",
        "REINFORCE is unbiased but high variance (single trajectory return Gₜ)",
        "Baseline b(s) reduces variance without introducing bias — use V(s)",
        "Advantage A=Q-V captures how much better action a is than average",
        "Policy gradient handles continuous actions naturally (unlike tabular Q-learning)",
        "Score function estimator generalises beyond RL: VI, training generative models",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 7 — GAE
# ══════════════════════════════════════════════════════════════════════════════
def topic_advantage_gae():
    clear()
    breadcrumb("mlmath", "RL Math", "Generalised Advantage Estimation")
    section_header("GENERALISED ADVANTAGE ESTIMATION (GAE)")
    print()

    section_header("1. THEORY")
    print(white("""
  GAE (Schulman et al., 2016) provides a principled way to trade off bias and
  variance when estimating the advantage function A(s,a) = Q(s,a) - V(s).

  TD RESIDUAL: the one-step TD error is:
      δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)
  If V = V*, then δₜ is an unbiased estimate of A^π(sₜ, aₜ). In practice, V is
  approximate, introducing bias (but reducing variance vs MC).

  n-STEP ADVANTAGE ESTIMATES (for n=1,2,...):
      Â₁ = δₜ                                                      [high bias, low var]
      Â₂ = δₜ + γδₜ₊₁                                             [less bias]
      Â_MC = δₜ + γδₜ₊₁ + ... + γ^{T-t-1}δ_{T-1}                  [zero bias, high var]

  GAE(γ,λ): exponentially-weighted sum of n-step estimates:
      Â_t^GAE(γ,λ) = Σₗ₌₀^∞ (γλ)ˡ δₜ₊ₗ

  LAMBDA CONTROLS BIAS-VARIANCE TRADE-OFF:
  - λ=0: Â = δₜ = TD(0) advantage — maximum bias, minimum variance
  - λ=1: Â = Σ(γᵏ)δₜ₊ₖ = Monte Carlo advantage (if V=0) — zero bias, maximum variance
  - 0<λ<0.97 typical in PPO/A2C

  RECURSIVE COMPUTATION (efficient, one backward pass):
      Â_T = δ_T
      Â_t = δₜ + (γλ) Â_{t+1}
  This allows O(T) computation for the entire rollout.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)  [TD residual]"))
    print(formula("  GAE: Â_t = Σₗ₌₀^∞ (γλ)ˡ δₜ₊ₗ"))
    print(formula("  Recursive: Â_t = δₜ + γλ Â_{t+1}  [backward sweep]"))
    print(formula("  λ=0: Â=δₜ (TD), λ=1: Â=MC return − V(s) (zero bias)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — GAE on Sample Rollout")
    rng = np.random.default_rng(8)
    T = 20; gamma = 0.99

    rewards  = rng.normal(0.5, 1.0, T)
    V_values = rng.normal(2.0, 0.5, T + 1)  # V(s0..sT)

    td_errors = rewards + gamma * V_values[1:] - V_values[:-1]

    def compute_gae(td_errors, gamma, lam):
        T = len(td_errors)
        adv = np.zeros(T)
        adv[-1] = td_errors[-1]
        for t in range(T - 2, -1, -1):
            adv[t] = td_errors[t] + gamma * lam * adv[t + 1]
        return adv

    print(f"\n  {bold_cyan('GAE for different λ values (T=20 step rollout):')}\n")
    lambdas = [0.0, 0.5, 0.95, 1.0]
    gae_estimates = {l: compute_gae(td_errors, gamma, l) for l in lambdas}

    rows = []
    for t in range(0, T, 2):
        row = [str(t), f"{td_errors[t]:+.3f}"]
        for l in lambdas:
            row.append(f"{gae_estimates[l][t]:+.4f}")
        rows.append(row)
    table(["t", "δₜ", "GAE(λ=0)", "GAE(λ=0.5)", "GAE(λ=0.95)", "GAE(λ=1)"],
          rows, [grey, yellow, red, cyan, green, white])

    print(f"\n  {bold_cyan('Variance comparison:')}\n")
    for l in lambdas:
        adv = gae_estimates[l]
        var = adv.std()
        bar = yellow("█" * int(min(var * 8, 40)))
        print(f"  λ={l:.2f}: std={yellow(f'{var:.3f}')}  {bar}")
    print()

    try:
        import plotext as plt
        plt.clear_figure()
        for l in [0.0, 0.95, 1.0]:
            plt.plot(list(range(T)), gae_estimates[l].tolist(), label=f"λ={l}")
        plt.title("GAE Advantage Estimates vs λ")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for GAE comparison plot)"))
    _pause()

    section_header("4. CODE")
    code_block("GAE computation (as used in PPO/A2C)", """
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    '''
    rewards: (T,) array of rewards
    values:  (T+1,) array — V(s0)...V(sT) (last is bootstrap value or 0)
    dones:   (T,) bool — episode end flags
    '''
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # zero out if episode ended
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values[:-1]  # used as target for value function
    return advantages, returns

# Normalise advantages (standard in PPO)
def normalise(adv, eps=1e-8):
    return (adv - adv.mean()) / (adv.std() + eps)

rng = np.random.default_rng(0)
T = 128
rewards = np.clip(rng.normal(0.3, 1.0, T), -10, 10)
values  = rng.normal(2.0, 0.3, T+1)
dones   = np.zeros(T); dones[63] = 1; dones[127] = 1  # two episodes

adv, ret = compute_gae(rewards, values, dones, lam=0.95)
adv_norm = normalise(adv)
print(f"Advantage mean={adv.mean():.4f}, std={adv.std():.4f}")
print(f"After normalise: mean={adv_norm.mean():.4f}, std={adv_norm.std():.4f}")
print(f"Returns (value targets) mean={ret.mean():.4f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "GAE unifies TD (λ=0) and MC (λ=1) advantage estimation via exponential weighting",
        "Recursive formula Â_t = δₜ + γλÂ_{t+1} computes GAE in one O(T) backward pass",
        "λ=0.95 typical in PPO — slightly biased but much lower variance than MC",
        "Normalising advantages (mean=0, std=1) per batch stabilises policy gradient training",
        "GAE assumes V(s) estimates are accurate; V is trained simultaneously in A2C/PPO",
        "Done mask handles episode boundaries correctly in multi-episode rollouts",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 8 — Actor-Critic
# ══════════════════════════════════════════════════════════════════════════════
def topic_actor_critic():
    clear()
    breadcrumb("mlmath", "RL Math", "Actor-Critic / PPO")
    section_header("ACTOR-CRITIC AND PPO")
    print()

    section_header("1. THEORY")
    print(white("""
  Actor-Critic (AC) methods combine:
  - Actor: π_θ(a|s) — policy network that chooses actions
  - Critic: V_φ(s)  — value function network that estimates state values

  A2C (Advantage Actor-Critic):
      Actor loss:  L^π = -E [ Â_t · log π_θ(aₜ|sₜ) ]   (REINFORCE with advantage)
      Critic loss: L^V = E [ (Gₜ - V_φ(sₜ))² ]          (MSE to returns)
      Entropy bonus: L^H = -β E[Σₐ π(a|s) log π(a|s)]   (encourages exploration)
      Total: L = L^π + c_V · L^V - c_H · L^H

  PROXIMAL POLICY OPTIMISATION (PPO, Schulman et al., 2017):
  PPO prevents excessively large policy updates by clipping the probability ratio
  rₜ(θ) = π_θ(aₜ|sₜ) / π_{θ_old}(aₜ|sₜ):

      L^CLIP = E [ min( rₜ(θ) · Aₜ,  clip(rₜ(θ), 1-ε, 1+ε) · Aₜ ) ]

  When A > 0 (action was good): clip if rₜ > 1+ε (don't raise probability too much)
  When A < 0 (action was bad): clip if rₜ < 1-ε (don't lower probability too much)

  PPO is simpler to implement than TRPO (trust region) but similarly stable.
  Key insight: the min() operator ensures we never increase the objective beyond
  what the clipped version allows — a pessimistic bound on improvement.

  PPO ALGORITHM (simplified):
  1. Collect T timesteps of (s,a,r) under π_{θ_old}
  2. Compute advantages Â_t via GAE
  3. Perform K epochs of mini-batch gradient ascent on L^CLIP
  4. Update θ_old ← θ
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  A2C actor loss: L^π = -E[Â_t · log π_θ(aₜ|sₜ)]"))
    print(formula("  Critic loss: L^V = E[(Gₜ - V_φ(sₜ))²]"))
    print(formula("  PPO ratio: rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)"))
    print(formula("  L^CLIP = E[min(rₜ·Aₜ, clip(rₜ, 1-ε, 1+ε)·Aₜ)]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — PPO Clipping Analysis")
    rng = np.random.default_rng(22)
    eps_clip = 0.2
    ratios = np.linspace(0.0, 2.5, 100)
    A_pos, A_neg = 1.0, -1.0

    def ppo_obj(r, A, eps):
        clipped = np.clip(r, 1 - eps, 1 + eps) * A
        unclipped = r * A
        return np.minimum(unclipped, clipped)

    obj_pos = ppo_obj(ratios, A_pos, eps_clip)
    obj_neg = ppo_obj(ratios, A_neg, eps_clip)

    print(f"\n  {bold_cyan('PPO L^CLIP vs probability ratio r = π_new/π_old:')}\n")
    print(f"  {'r = π_new/π_old':<20} {'L^CLIP (A>0)':<18} {'L^CLIP (A<0)'}")
    print(f"  {'─'*17}   {'─'*13}   {'─'*13}")
    sample_idx = [0, 20, 40, 50, 60, 80, 99]
    for i in sample_idx:
        r = ratios[i]
        p = green(f"{obj_pos[i]:+.4f}") if obj_pos[i] >= 0 else red(f"{obj_pos[i]:+.4f}")
        n = green(f"{obj_neg[i]:+.4f}") if obj_neg[i] >= 0 else red(f"{obj_neg[i]:+.4f}")
        clip_mark = yellow(" ← clipped") if abs(r - 1) > eps_clip else ""
        print(f"  r={r:.2f}{' ':14} {p:<20} {n:<20} {clip_mark}")

    print(f"\n  {hint('Clipping prevents policy from moving too far from old policy.')}")
    print(f"  {hint('ε=0.2 means ratios outside [0.8, 1.2] are clipped.')}")
    print()
    _pause()

    section_header("4. CODE")
    code_block("PPO core update step (numpy)", """
import numpy as np

def ppo_loss(log_probs, log_probs_old, advantages, returns, values,
             eps_clip=0.2, c_value=0.5, c_entropy=0.01):
    '''
    log_probs:     log π_θ(aₜ|sₜ)  — current policy (shape T)
    log_probs_old: log π_old(aₜ|sₜ) — reference policy (shape T)
    advantages:    Â_t (normalised, shape T)
    returns:       Gₜ (shape T)
    values:        V_φ(sₜ) (shape T)
    '''
    # Probability ratio rₜ = exp(log π - log π_old)
    ratios = np.exp(log_probs - log_probs_old)

    # Clipped surrogate objective
    surr1 = ratios * advantages
    surr2 = np.clip(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    policy_loss = -np.mean(np.minimum(surr1, surr2))

    # Value function loss (MSE)
    value_loss = np.mean((returns - values)**2)

    # Entropy bonus (simple approximation: assume Gaussian or categorical)
    entropy = -np.mean(log_probs)

    total_loss = policy_loss + c_value * value_loss - c_entropy * entropy
    return total_loss, policy_loss, value_loss, entropy

# Example with random data
rng = np.random.default_rng(0)
T = 256
log_probs     = rng.normal(-1.5, 0.3, T)
log_probs_old = log_probs + rng.normal(0, 0.05, T)
advantages    = rng.normal(0, 1, T)
returns       = rng.normal(2.0, 0.5, T)
values        = rng.normal(2.0, 0.5, T)

loss, lp, lv, H = ppo_loss(log_probs, log_probs_old, advantages, returns, values)
print(f"Total Loss={loss:.4f}  Policy={lp:.4f}  Value={lv:.4f}  Entropy={H:.4f}")
ratios = np.exp(log_probs - log_probs_old)
print(f"Ratio range: [{ratios.min():.3f}, {ratios.max():.3f}]")
print(f"Clipped fraction: {(np.abs(ratios-1) > 0.2).mean():.3f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Actor: π_θ directs behaviour; Critic: V_φ evaluates states — online bootstrapping",
        "PPO clips probability ratio to (1±ε): prevents destructive large policy updates",
        "min() in L^CLIP is pessimistic — limits improvement in direction of advantage",
        "PPO = TRPO but simpler: no 2nd-order KL constraint, just clipping",
        "Entropy bonus prevents premature collapse to deterministic policy",
        "K epochs per batch with mini-batches — key computational efficiency of PPO",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 9 — DQN
# ══════════════════════════════════════════════════════════════════════════════
def topic_dqn():
    clear()
    breadcrumb("mlmath", "RL Math", "Deep Q-Network (DQN)")
    section_header("DEEP Q-NETWORK (DQN)")
    print()

    section_header("1. THEORY")
    print(white("""
  DQN (Mnih et al., 2015) scales Q-learning to large state spaces by approximating
  Q*(s,a) with a deep neural network Q(s,a; θ). Two key innovations:

  1. EXPERIENCE REPLAY: store transitions (s,a,r,s') in a replay buffer D of capacity N.
     - Sample random mini-batches at each update instead of learning online.
     - Breaks temporal correlations between consecutive transitions.
     - Each transition can be replayed multiple times — data efficiency.

  2. TARGET NETWORK: maintain a separate frozen network Q(s,a;θ⁻) for computing
     TD targets. Copy θ to θ⁻ every C steps:
         y = r + γ max_{a'} Q(s', a'; θ⁻)
     Without this, chasing a non-stationary target causes oscillations/divergence.

  LOSS: L(θ) = E_{(s,a,r,s')~D} [ (y - Q(s,a;θ))² ]
     where y = r + γ max_{a'} Q(s',a'; θ⁻)  (if s' not terminal)
          y = r                               (if terminal)

  ε-GREEDY DECAY: ε starts at 1.0 (fully random) and decays to εₘᵢₙ (e.g. 0.01)
  over the first N_explore frames. Sufficient exploration before exploitation.

  DOUBLE DQN (DDQN): the max over Q(s',a';θ⁻) introduces maximisation bias.
  DDQN: use online θ to select a*, use target θ⁻ to evaluate:
      y = r + γ Q(argmax_{a'} Q(s',a';θ), s'; θ⁻)

  PRIORITISED REPLAY: sample transitions proportional to |TD error|^α,
  importance-weighting to correct the distribution shift.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  DQN target: y = r + γ max_{a'} Q(s',a'; θ⁻)"))
    print(formula("  DQN loss:   L(θ) = E[(y - Q(s,a;θ))²]"))
    print(formula("  ε decay:    εₜ = max(εₘᵢₙ, ε₀ - t/N · (ε₀ - εₘᵢₙ))"))
    print(formula("  DDQN target: y = r + γ Q(argmax_{a'} Q(s',a';θ), s'; θ⁻)"))
    print(formula("  Priority: pᵢ = |δᵢ|^α + ε,  P(i) = pᵢ / Σⱼ pⱼ"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — DQN on CartPole (Tabular Simulation)")
    rng = np.random.default_rng(30)

    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.buf = []; self.cap = capacity
        def push(self, s, a, r, s_next, done):
            if len(self.buf) >= self.cap: self.buf.pop(0)
            self.buf.append((s, a, r, s_next, done))
        def sample(self, batch_size, rng):
            idx = rng.choice(len(self.buf), batch_size, replace=False)
            return [self.buf[i] for i in idx]
        def __len__(self): return len(self.buf)

    n_states, n_actions = 8, 2  # discretised CartPole
    Q_online = rng.normal(0, 0.1, (n_states, n_actions))
    Q_target = Q_online.copy()
    buf = ReplayBuffer(2000)
    alpha, gamma_dqn, eps = 0.05, 0.99, 1.0
    total_rewards, td_errors_hist = [], []
    target_update_freq = 50

    for ep in range(500):
        s = rng.integers(0, n_states)
        total_r = 0.0
        for step in range(30):
            a = rng.integers(n_actions) if rng.random() < eps else Q_online[s].argmax()
            r = rng.normal(0.3 if a == 0 else 0.5, 0.2)
            s_next = rng.integers(0, n_states)
            done = (step == 29)
            buf.push(s, a, r, s_next, done); total_r += r; s = s_next
            if len(buf) >= 64:
                batch = buf.sample(64, rng)
                for bs, ba, br, bsn, bd in batch:
                    y = br if bd else br + gamma_dqn * Q_target[bsn].max()
                    td_err = y - Q_online[bs, ba]
                    td_errors_hist.append(abs(td_err))
                    Q_online[bs, ba] += alpha * td_err
        if ep % target_update_freq == 0: Q_target = Q_online.copy()
        eps = max(0.05, eps - 0.003)
        total_rewards.append(total_r)

    print(f"\n  {bold_cyan('DQN on 8-state Tabular MDP (500 episodes):')}\n")
    print(f"  Final ε: {yellow(f'{eps:.3f}')}  (decayed from 1.0)")
    print(f"  Avg reward (last 100 ep): {green(f'{np.mean(total_rewards[-100:]):.4f}')}")
    print(f"  Mean TD error:            {yellow(f'{np.mean(td_errors_hist):.4f}')}")
    avg_rewards = [sum(total_rewards[i:i+50])/50 for i in range(0, 500-50, 50)]
    print()
    print_sparkline(avg_rewards, label="Avg reward (50-ep):", color_fn=green)
    print()
    _pause()

    section_header("4. CODE")
    code_block("DQN core (numpy/pseudo-code)", """
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buf = deque(maxlen=maxlen)
    def push(self, *transition): self.buf.append(transition)
    def sample(self, n, rng):
        idx = rng.choice(len(self.buf), n, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)

def dqn_update(Q_online, Q_target, batch, alpha=0.001, gamma=0.99, double=True):
    '''batch: list of (s, a, r, s', done)'''
    loss_total = 0.0
    for s, a, r, sn, done in batch:
        if double:
            a_star = Q_online[sn].argmax()          # select with online
            y = r if done else r + gamma * Q_target[sn, a_star]  # eval with target
        else:
            y = r if done else r + gamma * Q_target[sn].max()
        td_error = y - Q_online[s, a]
        Q_online[s, a] += alpha * td_error
        loss_total += td_error**2
    return loss_total / len(batch)

# Mini experiment
rng = np.random.default_rng(0); S, A = 16, 4
Qo = np.zeros((S, A)); Qt = Qo.copy(); buf = ReplayBuffer(5000)
for _ in range(5000):
    s = rng.integers(S); a = rng.integers(A)
    r = float(a == s % A); sn = rng.integers(S)
    buf.push(s, a, r, sn, False)
    if len(buf) >= 64:
        loss = dqn_update(Qo, Qt, buf.sample(64, rng))

print(f"DQN loss after 5000 steps: {loss:.6f}")
print(f"% correct greedy actions: {sum(Qo[s].argmax()==s%A for s in range(S))/S:.2f}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "Experience replay: random sampling breaks temporal correlations → more i.i.d.",
        "Target network: frozen θ⁻ stabilises TD targets — prevents oscillating updates",
        "Without target network: chasing non-stationary target → catastrophic forgetting",
        "ε-greedy annealing: exploration then exploitation — key DQN practical detail",
        "Double DQN: reduces overestimation bias of max Q(s',a') by separating select/eval",
        "Prioritised replay: focus updates on high-error transitions → faster convergence",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 10 — Exploration
# ══════════════════════════════════════════════════════════════════════════════
def topic_exploration():
    clear()
    breadcrumb("mlmath", "RL Math", "Exploration Strategies")
    section_header("EXPLORATION STRATEGIES")
    print()

    section_header("1. THEORY")
    print(white("""
  The exploration-exploitation dilemma: should the agent take the action it
  believes is best (exploit) or try other actions to gain information (explore)?

  ε-GREEDY: with probability ε take random action, otherwise take greedy action.
  Simple but inefficient — explores uniformly, ignores information about uncertainty.
  Regret: Cumulative regret R_T = Σ(μ* - μ_{aₜ}) = O(√(KT log T)) with ε=√(K/T).

  UPPER CONFIDENCE BOUND (UCB1): picks action maximising optimistic estimate:
      a_t = argmax_a [ Q̄(a) + c · √(ln t / N(a)) ]
  Bonus term √(ln t / N(a)) is large for under-explored arms.
  Theoretical regret: R_T = O(√(KT log T)) — matches lower bound up to log factor.

  THOMPSON SAMPLING (TS): Bayesian approach — maintain posterior over μ_a,
  sample θ_a ~ posterior, select a = argmax θ_a.
  For Bernoulli bandits: Beta(α_a, β_a) posterior, updated with successes/failures.
  Empirically often outperforms UCB; achieves O(√(KT log T)) regret.

  POSTERIOR SAMPLING (PS): generalisation of TS to full RL — sample a complete MDP
  from posterior, plan optimally, and execute for one episode (PSRL algorithm).

  INTRINSIC MOTIVATION: add bonus rewards for visiting novel states to encourage
  exploration:
  - Count-based: r+ = β / √n(s)  (pseudo-counts for large state spaces)
  - Curiosity: reward = prediction error of next state (ICM, RND, BYOL-Explore)
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ε-greedy: a = argmax Q̄(a) if rand>ε else uniform random"))
    print(formula("  UCB1: a = argmax[ Q̄(a) + c√(ln t / N(a)) ]"))
    print(formula("  Thompson: θ_a ~ Beta(α_a, β_a), a = argmax θ_a"))
    print(formula("  Regret R_T = Σₜ (μ* - μ_{aₜ}),  minimax lower bound: Ω(√KT)"))
    print(formula("  Count bonus: r+ = β/√N(s)  [count-based exploration]"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — Bandit Comparison")
    rng = np.random.default_rng(42)
    K, T = 5, 2000
    true_means = np.array([0.2, 0.5, 0.8, 0.3, 0.6])  # arm 2 optimal
    mu_star = true_means.max()

    def pull(a): return rng.normal(true_means[a], 0.5)

    # ε-greedy
    def run_epsilon_greedy(eps=0.1):
        Q, N = np.zeros(K), np.zeros(K)
        regret = []
        for _ in range(T):
            a = rng.integers(K) if rng.random() < eps else Q.argmax()
            r = pull(a); N[a] += 1; Q[a] += (r - Q[a]) / N[a]
            regret.append(mu_star - true_means[a])
        return np.cumsum(regret)

    # UCB1
    def run_ucb(c=1.0):
        Q, N = np.zeros(K), np.zeros(K)
        regret = []
        for t in range(1, T + 1):
            bonus = c * np.sqrt(np.log(t) / (N + 1e-9))
            a = (Q + bonus).argmax()
            r = pull(a); N[a] += 1; Q[a] += (r - Q[a]) / N[a]
            regret.append(mu_star - true_means[a])
        return np.cumsum(regret)

    # Thompson Sampling (Gaussian)
    def run_thompson():
        mu_est = np.zeros(K); prec = np.ones(K)
        regret = []
        for _ in range(T):
            samples = rng.normal(mu_est, 1.0 / np.sqrt(prec))
            a = samples.argmax()
            r = pull(a)
            prec[a] += 1; mu_est[a] += (r - mu_est[a]) / prec[a]
            regret.append(mu_star - true_means[a])
        return np.cumsum(regret)

    reg_eps = run_epsilon_greedy(); reg_ucb = run_ucb(); reg_ts = run_thompson()

    print(f"\n  {bold_cyan('5-Arm Bandit (T=2000). Cumulative regret comparison:')}\n")
    print(f"  {'Method':<22} {'Regret @200':<16} {'Regret @1000':<16} {'Regret @2000'}")
    print(f"  {'─'*20}   {'─'*12}   {'─'*12}   {'─'*12}")
    for name, reg in [("ε-greedy (ε=0.1)", reg_eps),
                       ("UCB1 (c=1.0)", reg_ucb),
                       ("Thompson Sampling", reg_ts)]:
        r200  = reg[199]; r1000 = reg[999]; r2000 = reg[1999]
        print(f"  {name:<22} {red(f'{r200:.2f}'):<20} {yellow(f'{r1000:.2f}'):<20} "
              f"{green(f'{r2000:.2f}')}")
    print()
    print(f"  {hint('Lower regret = better. Thompson and UCB grow as O(√KT logT).')}")
    print()

    try:
        import plotext as plt
        plt.clear_figure()
        for label, reg in [("eps-greedy", reg_eps), ("UCB1", reg_ucb), ("Thompson", reg_ts)]:
            plt.plot(reg[::20].tolist(), label=label)
        plt.title("Cumulative Regret vs Time")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for cumulative regret plot)"))
    _pause()

    section_header("4. CODE")
    code_block("UCB1, Thompson Sampling (numpy)", """
import numpy as np

class UCB1:
    def __init__(self, k, c=1.0):
        self.Q = np.zeros(k); self.N = np.zeros(k); self.c = c; self.t = 0

    def act(self):
        self.t += 1
        return (self.Q + self.c * np.sqrt(np.log(self.t)/(self.N+1e-9))).argmax()

    def update(self, a, r):
        self.N[a] += 1; self.Q[a] += (r - self.Q[a]) / self.N[a]


class ThompsonBeta:
    '''Thompson sampling for Bernoulli bandits using Beta posterior.'''
    def __init__(self, k):
        self.alpha = np.ones(k); self.beta = np.ones(k)

    def act(self, rng):
        return rng.beta(self.alpha, self.beta).argmax()

    def update(self, a, r):
        '''r ∈ {0, 1} success/failure'''
        self.alpha[a] += r; self.beta[a] += 1 - r


# Bernoulli bandit experiment
rng = np.random.default_rng(0); K, T = 5, 3000
true_p = np.array([0.2, 0.4, 0.7, 0.3, 0.6])

ucb = UCB1(K); ts = ThompsonBeta(K)
ucb_r, ts_r = 0.0, 0.0
for _ in range(T):
    a_u = ucb.act(); r_u = int(rng.random() < true_p[a_u])
    ucb.update(a_u, r_u); ucb_r += r_u
    a_t = ts.act(rng); r_t = int(rng.random() < true_p[a_t])
    ts.update(a_t, r_t); ts_r += r_t

print(f"UCB1 cumulative reward: {ucb_r}  (out of {T})")
print(f"Thompson cumulative reward: {ts_r}  (out of {T})")
print(f"Oracle (always arm 3): {int(T * true_p[2])}")
print(f"UCB arm counts: {ucb.N.astype(int)}")
""")
    _pause()

    section_header("5. KEY INSIGHTS")
    for ins in [
        "ε-greedy: simple, robust, but wasteful — explores uniformly ignoring information",
        "UCB: 'optimism in the face of uncertainty' — explores under-visited arms proactively",
        "Thompson Sampling: Bayesian — balances exploration naturally; often best in practice",
        "Minimax lower bound: any algorithm must have Ω(√KT) regret on stochastic bandits",
        "UCB1 achieves O(√KT logT) — optimal up to logarithmic factor",
        "Intrinsic motivation extends exploration to RL: curiosity/novelty bonuses + extrinsic reward",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# Block entry point
# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("Markov Decision Process",          topic_mdp),
        ("Bellman Equations",                topic_bellman),
        ("Dynamic Programming",              topic_dynamic_programming),
        ("Monte Carlo Methods",              topic_monte_carlo),
        ("Temporal Difference Learning",     topic_temporal_difference),
        ("Policy Gradient",                  topic_policy_gradient),
        ("Generalised Advantage Estimation", topic_advantage_gae),
        ("Actor-Critic / PPO",               topic_actor_critic),
        ("Deep Q-Network (DQN)",             topic_dqn),
        ("Exploration Strategies",           topic_exploration),
    ]
    block_menu("b15", "Reinforcement Learning Math", topics)
