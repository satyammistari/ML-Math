"""
blocks/b16_mdp_solver.py
Block 16: MDP Solver
Topics: GridWorld, Value Iteration, Policy Iteration, Q-learning,
        Multi-Armed Bandit, Exploration vs Exploitation.
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
from viz.matplotlib_plots import show_heatmap


def _pause(msg="  [Enter] to continue..."):
    input(grey(msg))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — shared GridWorld environment
# ══════════════════════════════════════════════════════════════════════════════

class GridWorld:
    """4×4 GridWorld: start=(3,0), goal=(0,3), walls=[(1,1),(2,2)], 4 actions."""
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # N S E W
    ACT_SYM = {0: '↑', 1: '↓', 2: '→', 3: '←'}

    def __init__(self, H=4, W=4, start=(3, 0), goal=(0, 3),
                 walls=None, gamma=0.9):
        self.H, self.W = H, W
        self.start = start
        self.goal  = goal
        self.walls = set(walls) if walls else {(1, 1), (2, 2)}
        self.gamma = gamma
        self.n_states  = H * W
        self.n_actions = 4
        self._build_model()

    def _rc(self, s):  return s // self.W, s % self.W
    def _s(self, r, c): return r * self.W + c

    def _build_model(self):
        S, A = self.n_states, self.n_actions
        self.P = np.zeros((S, A, S))
        self.R = np.full((S, A), -0.1)
        g_s = self._s(*self.goal)
        wall_states = {self._s(r, c) for r, c in self.walls}

        for s in range(S):
            r, c = self._rc(s)
            if (r, c) in self.walls:
                self.P[s, :, s] = 1.0
                self.R[s, :] = 0.0
                continue
            if s == g_s:
                self.P[s, :, s] = 1.0
                self.R[s, :] = 0.0
                continue
            for a, (dr, dc) in self.ACTIONS.items():
                nr = max(0, min(self.H - 1, r + dr))
                nc = max(0, min(self.W - 1, c + dc))
                ns = self._s(nr, nc)
                if (nr, nc) in self.walls:
                    ns = s  # bounce back
                self.P[s, a, ns] = 1.0
                self.R[s, a] = 1.0 if ns == g_s else -0.1

    def reset(self):
        self.agent = list(self.start)
        return self._s(*self.agent)

    def step(self, a):
        r, c = self.agent
        dr, dc = self.ACTIONS[a]
        nr = max(0, min(self.H - 1, r + dr))
        nc = max(0, min(self.W - 1, c + dc))
        if (nr, nc) in self.walls:
            nr, nc = r, c
        self.agent = [nr, nc]
        s_next = self._s(nr, nc)
        done = (nr, nc) == self.goal
        reward = 1.0 if done else -0.1
        return s_next, reward, done

    def get_reward(self, s, a):
        return self.R[s, a]

    def render_ascii(self, V=None, policy=None, agent_pos=None):
        lines = []
        g_s = self._s(*self.goal)
        a_s = self._s(*agent_pos) if agent_pos else self._s(*self.start)
        for r in range(self.H):
            row = "  "
            for c in range(self.W):
                s = self._s(r, c)
                if (r, c) in self.walls:
                    row += bold_cyan('█') + ' '
                elif (r, c) == self.goal:
                    row += green('G') + ' '
                elif agent_pos and (r, c) == tuple(agent_pos):
                    row += yellow('A') + ' '
                elif V is not None:
                    row += cyan(f'{V[s]:+.1f}')[:8] + ' '
                elif policy is not None:
                    sym = self.ACT_SYM.get(policy[s], '·')
                    row += yellow(sym) + ' '
                else:
                    row += grey('·') + ' '
            lines.append(row)
        return lines


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 1 — GridWorld
# ══════════════════════════════════════════════════════════════════════════════
def topic_gridworld():
    clear()
    breadcrumb("mlmath", "MDP Solver", "GridWorld")
    section_header("GRIDWORLD ENVIRONMENT")
    print()

    section_header("1. THEORY")
    print(white("""
  A GridWorld MDP is a discrete 2-D grid where an agent moves through cells.
  It is the canonical test-bed for dynamic programming and reinforcement learning
  algorithms because its transition dynamics are deterministic and fully known.

  STATE SPACE: every cell (r,c) in an H×W grid is a state. Here we use a 4×4
  grid giving 16 states, numbered s = r·W + c (row-major). Walls are blocked;
  the agent bounces back if it tries to enter a wall cell.

  ACTION SPACE: four cardinal directions — North (↑), South (↓), East (→),
  West (←). In stochastic variants, each action may slip to an adjacent direction
  with probability p_slip, but here transitions are deterministic.

  REWARD STRUCTURE: the agent receives +1.0 upon reaching the goal cell G,
  and −0.1 for every other step. This step penalty discourages wandering and
  makes the shortest path optimal.

  POLICY: a deterministic policy π: S → A assigns one action to each state.
  The agent follows the policy until it reaches the terminal goal state.
  The task is to find the optimal policy π* that maximises the expected
  discounted return G = Σₜ γᵗ rₜ with discount factor γ = 0.9.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  State index: s = r · W + c  (row-major)"))
    print(formula("  Transition:  P(s'|s,a) ∈ {0,1}  (deterministic)"))
    print(formula("  Reward:      R(s,a) = +1.0 if s'=G, else −0.1"))
    print(formula("  Return:      G_t = Σₖ₌₀^∞  γᵏ r_{t+k+1}"))
    print(formula("  4×4 grid:    16 states, 4 actions → 64 Q-values"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 8×8 GridWorld Display")
    gw = GridWorld(H=8, W=8, start=(7, 0), goal=(0, 7),
                   walls={(2,2),(2,3),(3,2),(3,3),(5,5),(4,4)})
    print(f"\n  {bold_cyan('8×8 GridWorld  (█=wall, G=goal, A=agent, ·=empty):')}\n")
    col_labels = "  " + "  ".join([cyan(str(c)) for c in range(8)])
    print(col_labels)
    for r in range(8):
        row = f"  {grey(str(r))} "
        for c in range(8):
            if (r, c) in gw.walls:
                row += bold_cyan('█') + ' '
            elif (r, c) == (0, 7):
                row += green('G') + ' '
            elif (r, c) == (7, 0):
                row += yellow('A') + ' '
            else:
                row += grey('·') + ' '
        print(row)
    print()
    total_free = 64 - len(gw.walls) - 1
    print(f"  States: {yellow('64')}  |  Walls: {red(str(len(gw.walls)))}  |"
          f"  Free cells: {green(str(total_free))}")
    print(f"  Actions: {cyan('N(↑) S(↓) E(→) W(←)')}")
    print(f"  Reward: {green('+1.0')} on reaching G,  {red('−0.1')} all other steps")
    print()

    # Show step() by simulating 5 manual moves
    gw4 = GridWorld()
    s = gw4.reset()
    print(f"  {bold_cyan('4×4 GridWorld episodes (manual N/E policy):')}")
    path = [list(gw4.start)]
    total_r = 0.0
    for a in [0, 0, 0, 2, 2, 2]:
        s, r, done = gw4.step(a)
        rc = gw4.agent.copy()
        path.append(rc)
        total_r += r
        sym = GridWorld.ACT_SYM[a]
        print(f"    Action {cyan(sym)}  → state s={s} at ({rc[0]},{rc[1]})  "
              f"reward={green(f'{r:+.1f}')}  done={green(str(done))}")
    print(f"  Total reward: {green(f'{total_r:.1f}')}")
    print()
    _pause()

    section_header("4. VISUALIZATION")
    print(f"  {bold_cyan('4×4 Grid with axes (· = empty):')}\n")
    for line in gw4.render_ascii():
        print(line)
    print()
    _pause()

    section_header("5. CODE")
    code_block("GridWorld class", """
import numpy as np

class GridWorld:
    ACTIONS = {0:(-1,0), 1:(1,0), 2:(0,1), 3:(0,-1)}  # N S E W
    def __init__(self, H=4, W=4, start=(3,0), goal=(0,3),
                 walls=None, gamma=0.9):
        self.H, self.W = H, W
        self.start, self.goal = start, goal
        self.walls = set(walls) if walls else {(1,1),(2,2)}
        self.gamma = gamma
        self.n_states, self.n_actions = H*W, 4
        self._build_model()

    def _s(self, r, c): return r * self.W + c
    def _rc(self, s):   return s // self.W, s % self.W

    def _build_model(self):
        S, A = self.n_states, self.n_actions
        self.P = np.zeros((S, A, S))
        self.R = np.full((S, A), -0.1)
        g_s = self._s(*self.goal)
        for s in range(S):
            r, c = self._rc(s)
            if (r,c) in self.walls or s == g_s:
                self.P[s, :, s] = 1.0; self.R[s, :] = 0.0; continue
            for a,(dr,dc) in self.ACTIONS.items():
                nr = max(0, min(self.H-1, r+dr))
                nc = max(0, min(self.W-1, c+dc))
                if (nr,nc) in self.walls: nr,nc = r,c
                ns = self._s(nr,nc)
                self.P[s,a,ns] = 1.0
                self.R[s,a] = 1.0 if ns==g_s else -0.1

    def reset(self): self.agent = list(self.start); return self._s(*self.agent)

    def step(self, a):
        r,c = self.agent; dr,dc = self.ACTIONS[a]
        nr=max(0,min(self.H-1,r+dr)); nc=max(0,min(self.W-1,c+dc))
        if (nr,nc) in self.walls: nr,nc = r,c
        self.agent=[nr,nc]; s=self._s(nr,nc)
        done=(nr,nc)==self.goal
        return s, (1.0 if done else -0.1), done

    def get_reward(self, s, a): return self.R[s, a]
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "GridWorld is the standard testbed for DP and RL — transparent dynamics",
        "State = cell index; walls absorb actions (bounce-back transitions)",
        "Step penalty −0.1 makes shortest path optimal under any γ < 1",
        "R(s,a)=+1 at goal encodes terminal reward; goal absorbs all future steps",
        "Stochastic slipping models real-world uncertainty — changes optimal policy",
        "Scaling to N×N: state space O(N²), transition matrix O(N²·A) — DP still tractable",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 2 — Value Iteration
# ══════════════════════════════════════════════════════════════════════════════
def topic_value_iteration():
    clear()
    breadcrumb("mlmath", "MDP Solver", "Value Iteration")
    section_header("VALUE ITERATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Value Iteration (VI) computes the optimal value function V* by repeatedly
  applying the Bellman optimality operator 𝒯*:

      V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ V_k(s')]

  The operator 𝒯* is a γ-contraction in the sup-norm:
      ‖𝒯*V - 𝒯*U‖_∞ ≤ γ ‖V - U‖_∞
  By Banach's fixed-point theorem, iteration from any V₀ converges to V* at
  geometric rate γ. Convergence criterion: ‖V_{k+1} - V_k‖_∞ < δ(1−γ)/2γ
  guarantees the policy derived from V_{k+1} is within δ of optimal.

  POLICY EXTRACTION: once V* is approximated, extract the greedy policy:
      π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a) + γ V*(s')]
  For deterministic transitions, this simplifies to argmax_a [R(s,a) + γ V*(s'(s,a))].

  Q-VALUE FORM: equivalently maintain Q(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
  then V(s) = max_a Q(s,a). This is also useful for policy display.

  COMPLEXITY per sweep: O(|S|² · |A|) — evaluating all (s,a) pairs with full
  transition distribution. For the 4×4 GridWorld: 16 × 4 = 64 Q-values updated
  per sweep; convergence typically in 20–50 sweeps for γ=0.9.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  VI update:  V_{k+1}(s) = maxₐ[R(s,a) + γ Σ_{s'}P(s'|s,a)Vₖ(s')]"))
    print(formula("  Contraction: ‖𝒯*V - 𝒯*U‖∞ ≤ γ ‖V - U‖∞"))
    print(formula("  Error:      ‖Vₖ - V*‖∞ ≤ γᵏ/(1-γ) · ‖V₁ − V₀‖∞"))
    print(formula("  Policy:     π*(s) = argmaxₐ Σ_{s'}P(s'|s,a)[R + γV*(s')]"))
    print(formula("  Q-values:   Q(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 4×4 GridWorld")
    gw = GridWorld()
    S, A, gamma = gw.n_states, gw.n_actions, gw.gamma
    V = np.zeros(S)
    deltas = []

    for sweep in range(200):
        Q = gw.R + gamma * np.einsum('ijk,k->ij', gw.P, V)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new
        if delta < 0.001:
            break

    pi_opt = (gw.R + gamma * np.einsum('ijk,k->ij', gw.P, V)).argmax(axis=1)
    n_sweeps = len(deltas)

    print(f"\n  {bold_cyan(f'Value Iteration converged in {n_sweeps} sweeps (Δ < 0.001, γ=0.9):')}\n")
    print(f"  {bold_cyan('V*(s) — optimal value grid:')}")
    for r in range(gw.H):
        row = "  "
        for c in range(gw.W):
            s = gw._s(r, c)
            if (r, c) in gw.walls:
                row += bold_cyan(' ███ ')
            elif (r, c) == gw.goal:
                row += green(f' {V[s]:+.2f}')
            else:
                v_str = f'{V[s]:+.2f}'
                row += (cyan if V[s] > 0 else yellow)(f' {v_str}')
        print(row)
    print()
    print(f"  {bold_cyan('Optimal policy π*(s):')}")
    for r in range(gw.H):
        row = "  "
        for c in range(gw.W):
            s = gw._s(r, c)
            if (r, c) in gw.walls:
                row += bold_cyan('█ ')
            elif (r, c) == gw.goal:
                row += green('G ')
            else:
                row += yellow(GridWorld.ACT_SYM[pi_opt[s]]) + ' '
        print(row)
    print()
    print_sparkline(deltas, label="Max ΔV per sweep:", color_fn=cyan)
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import plotext as plt
        plt.clear_figure()
        plt.plot([float(d) for d in deltas], label="Max |ΔV|")
        plt.hline(0.001, color="red")
        plt.title("Value Iteration Convergence (Max ΔV per Sweep)")
        plt.xlabel("Sweep"); plt.ylabel("Max |ΔV|")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for convergence plot)"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt2
        V_grid = V.reshape(gw.H, gw.W)
        show_heatmap(V_grid, title="V*(s) — Optimal Value Function")
    except ImportError:
        print(grey("  (install matplotlib for value heatmap)"))
    _pause()

    section_header("5. CODE")
    code_block("Value Iteration", """
import numpy as np

def value_iteration(P, R, gamma=0.9, tol=0.001, max_iter=500):
    '''
    P: (S, A, S) transition probabilities
    R: (S, A) reward matrix
    Returns: V* (S,), pi* (S,), list of max-deltas
    '''
    S = R.shape[0]; V = np.zeros(S); deltas = []
    for _ in range(max_iter):
        Q = R + gamma * np.einsum('ijk,k->ij', P, V)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        deltas.append(delta); V = V_new
        if delta < tol: break
    pi = Q.argmax(axis=1)
    return V, pi, deltas

# Run on 4x4 GridWorld
gw = GridWorld()  # defined above
V_star, pi_star, deltas = value_iteration(gw.P, gw.R, gamma=0.9)
print(f"Converged in {len(deltas)} sweeps")
print(f"V*(start={gw.start}) = {V_star[gw._s(*gw.start)]:.4f}")
print(f"V*(goal={gw.goal})  = {V_star[gw._s(*gw.goal)]:.4f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "VI sweeps apply 𝒯* repeatedly — contraction guarantees convergence to V*",
        "Convergence rate is geometric: error halves every ~1/log(1/γ) sweeps",
        "γ=0.9 converges in ~50 sweeps; γ=0.99 needs ~500 — discount matters!",
        "Optimal policy extracted in O(|S||A|) once V* is known",
        "VI does not track policy explicitly — implicit in argmax at each step",
        "Asynchronous VI: update one state at a time in any order — still converges",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 3 — Policy Iteration
# ══════════════════════════════════════════════════════════════════════════════
def topic_policy_iteration():
    clear()
    breadcrumb("mlmath", "MDP Solver", "Policy Iteration")
    section_header("POLICY ITERATION")
    print()

    section_header("1. THEORY")
    print(white("""
  Policy Iteration (PI) alternates between two steps until convergence:

  STEP 1 — POLICY EVALUATION: given a fixed policy π, compute V^π exactly by
  solving the linear system:
      V^π = R^π + γ P^π V^π
      ⟹  (I − γ P^π) V^π = R^π
  where R^π(s) = Σₐ π(a|s) R(s,a) and P^π(s,s') = Σₐ π(a|s) P(s'|s,a).
  Direct solve: O(|S|³) via LU decomposition. Iterative evaluation is also valid.

  STEP 2 — POLICY IMPROVEMENT: the policy improvement theorem guarantees that
  the greedy policy w.r.t. V^π is at least as good:
      π'(s) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
  If π' = π, we have reached the optimal policy π* and algorithm terminates.

  CONVERGENCE: since there are finitely many deterministic policies (|A|^|S| total),
  and each improvement step strictly improves (or leaves unchanged), PI terminates
  in at most |A|^|S| iterations — in practice, far fewer (often 2–10 for small MDPs).

  COMPARISON: PI converges in fewer major iterations than VI (each one is more
  expensive due to linear solve), while VI requires many cheap sweeps. For small
  MDPs, use PI; for very large S use approximate or truncated policy evaluation.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Eval: (I − γP^π)V^π = R^π  →  V^π = (I−γP^π)⁻¹ R^π"))
    print(formula("  Improve: π'(s) = argmaxₐ[R(s,a) + γ Σ_{s'}P(s'|s,a)V^π(s')]"))
    print(formula("  Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')"))
    print(formula("  Improvement thm: V^{π'}(s) ≥ V^π(s) for all s"))
    print(formula("  Complexity: O(|S|³) eval + O(|S|²|A|) improve per iteration"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 4×4 GridWorld, Comparison With VI")
    gw = GridWorld()
    S, A, gamma = gw.n_states, gw.n_actions, gw.gamma

    def policy_evaluation(pi_det, P, R, gamma=0.9):
        """Exact policy evaluation via linear solve."""
        P_pi = P[np.arange(S), pi_det]   # (S, S)
        R_pi = R[np.arange(S), pi_det]   # (S,)
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)
        return V

    def policy_iteration(P, R, gamma=0.9):
        pi = np.zeros(S, dtype=int)
        n_iters = 0
        for _ in range(200):
            V = policy_evaluation(pi, P, R, gamma)
            Q = R + gamma * np.einsum('ijk,k->ij', P, V)
            pi_new = Q.argmax(axis=1)
            n_iters += 1
            if np.all(pi_new == pi):
                return V, pi, n_iters
            pi = pi_new
        return V, pi, n_iters

    def value_iteration_timed(P, R, gamma=0.9, tol=0.001):
        V = np.zeros(S); n_sweeps = 0
        for _ in range(500):
            Q = R + gamma * np.einsum('ijk,k->ij', P, V)
            V_new = Q.max(axis=1); n_sweeps += 1
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new; break
            V = V_new
        return V, Q.argmax(axis=1), n_sweeps

    V_pi, pi_pi, n_pi = policy_iteration(gw.P, gw.R, gamma)
    V_vi, pi_vi, n_vi = value_iteration_timed(gw.P, gw.R, gamma)

    print(f"\n  {bold_cyan('Policy Iteration vs Value Iteration on 4×4 GridWorld:')}\n")
    table(
        ["Method", "Iterations", "V*(start)", "Policies match?"],
        [
            ["Policy Iteration", str(n_pi),
             f"{V_pi[gw._s(*gw.start)]:.4f}", "—"],
            ["Value Iteration", f"{n_vi} sweeps",
             f"{V_vi[gw._s(*gw.start)]:.4f}",
             green("Yes") if np.all(pi_pi == pi_vi) else red("No")],
        ],
        [cyan, yellow, green, white]
    )

    print(f"\n  {bold_cyan('Policy Iteration — final policy (arrows):')}")
    for r in range(gw.H):
        row = "  "
        for c in range(gw.W):
            s = gw._s(r, c)
            if (r, c) in gw.walls:
                row += bold_cyan('█ ')
            elif (r, c) == gw.goal:
                row += green('G ')
            else:
                row += yellow(GridWorld.ACT_SYM[pi_pi[s]]) + ' '
        print(row)
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt2
        V_grid = V_pi.reshape(gw.H, gw.W)
        show_heatmap(V_grid, title="Policy Iteration V^π*(s)")
    except ImportError:
        print(grey("  (install matplotlib for policy value heatmap)"))
    _pause()

    section_header("5. CODE")
    code_block("Policy Iteration (exact evaluation)", """
import numpy as np

def policy_evaluation_exact(pi, P, R, gamma=0.9):
    S = len(pi)
    P_pi = P[np.arange(S), pi]         # (S,S)  transition under pi
    R_pi = R[np.arange(S), pi]         # (S,)   reward under pi
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)

def policy_iteration(P, R, gamma=0.9):
    S, A, _ = P.shape
    pi = np.zeros(S, dtype=int)        # start with all-action-0 policy
    for _ in range(1000):
        V  = policy_evaluation_exact(pi, P, R, gamma)
        Q  = R + gamma * np.einsum('ijk,k->ij', P, V)
        pi_new = Q.argmax(axis=1)      # greedy improvement
        if np.all(pi_new == pi): break  # stable → optimal
        pi = pi_new
    return V, pi

# On 4x4 GridWorld
V, pi_star = policy_iteration(gw.P, gw.R)
print("V*(start):", V[gw._s(*gw.start)])
print("Optimal policy arrows:")
for r in range(4):
    print(' '.join(['↑↓→←'[pi_star[r*4+c]] for c in range(4)]))
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "PI converges in far fewer iterations than VI — often 2–10 for small MDPs",
        "Each PI step is expensive (O(S³) linear solve) but sweeps VI needs ~50 sweeps",
        "Policy improvement theorem: greedy update never makes policy worse",
        "Terminated when π' = π — this is exactly π* (fixed point of improvement)",
        "Modified PI: k Bellman sweeps for eval instead of exact solve — bridges VI and PI",
        "Parallel: both VI and PI converge to same V*, differ only in trajectory",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 4 — Q-Table (Q-learning)
# ══════════════════════════════════════════════════════════════════════════════
def topic_q_table():
    clear()
    breadcrumb("mlmath", "MDP Solver", "Q-Learning")
    section_header("Q-LEARNING ON 4×4 GRIDWORLD")
    print()

    section_header("1. THEORY")
    print(white("""
  Q-learning is a model-free, off-policy TD algorithm that learns the optimal
  action-value function Q*(s,a) directly from interaction with the environment,
  without requiring the transition model P(s'|s,a).

  UPDATE RULE:
      Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]

  The target r + γ max_{a'} Q(s',a') is the one-step lookahead using the optimal
  next action. This off-policy nature (bootstrapping with max, not the actual
  action taken) allows learning Q* while following an exploratory ε-greedy policy.

  ε-GREEDY EXPLORATION: at each step, take a random action with probability ε,
  and the greedy action argmax_a Q(s,a) with probability 1−ε. ε is annealed from
  1.0 (fully random) to 0.1 over the training episodes to ensure exploration then
  exploitation. The GLIE condition (ε_t → 0 but Σε_t = ∞) guarantees convergence.

  CONVERGENCE: Q-learning converges to Q* with probability 1 if every (s,a) pair
  is visited infinitely often and the learning rate satisfies Robbins-Monro
  conditions: Σ αₜ = ∞ and Σ αₜ² < ∞. In practice, a fixed small α with ε-greedy
  exploration works well for finite MDPs.

  Q-TABLE: for a 4×4 GridWorld with 4 actions, the Q-table has 16×4 = 64 entries.
  Each entry Q(s,a) estimates the expected return from taking action a in state s
  and then following the optimal policy thereafter.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Q-update: Q(s,a) ← Q(s,a) + α[r + γ maxₐ' Q(s',a') − Q(s,a)]"))
    print(formula("  TD error: δ = r + γ maxₐ' Q(s',a') − Q(s,a)"))
    print(formula("  ε-decay:  εₜ = max(εₘᵢₙ, ε₀ − t/N·(ε₀−εₘᵢₙ))"))
    print(formula("  Policy:   π(s) = argmaxₐ Q(s,a)  (greedy)"))
    print(formula("  Target:   r + γ max Q(s',·)  uses greedy future → off-policy"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 500 Training Episodes")
    gw = GridWorld()
    S, A, gamma = gw.n_states, gw.n_actions, gw.gamma
    rng = np.random.default_rng(7)
    Q = np.zeros((S, A))
    alpha = 0.1; eps_start = 1.0; eps_end = 0.1; n_episodes = 500
    cum_rewards = []

    for ep in range(n_episodes):
        s = gw.reset()
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - ep / (n_episodes * 0.8))
        ep_r = 0.0
        for _ in range(100):
            a = rng.integers(A) if rng.random() < eps else Q[s].argmax()
            s_next, r, done = gw.step(a)
            td_target = r + gamma * Q[s_next].max() * (1 - done)
            Q[s, a] += alpha * (td_target - Q[s, a])
            ep_r += r; s = s_next
            if done: break
        cum_rewards.append(ep_r)

    learned_pi = Q.argmax(axis=1)
    print(f"\n  {bold_cyan('Q-Table after 500 episodes (4×4 GridWorld):')}\n")
    act_names = ['↑ N', '↓ S', '→ E', '← W']
    hdr = ["s (r,c)"] + act_names + ["Greedy π"]
    rows_t = []
    for s in range(S):
        r, c = gw._rc(s)
        best = Q[s].argmax()
        row = [f"s{s:02d} ({r},{c})"]
        for a in range(A):
            row.append(f"{Q[s,a]:+.3f}")
        row.append(GridWorld.ACT_SYM[best])
        rows_t.append(row)
    table(hdr, rows_t, [grey, cyan, cyan, cyan, cyan, yellow])

    avg_r = [float(np.mean(cum_rewards[max(0,i-50):i+1])) for i in range(0,500,10)]
    print()
    print_sparkline(avg_r, label="Avg reward (50-ep window):", color_fn=green)
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import plotext as plt
        plt.clear_figure()
        smoothed = [float(np.mean(cum_rewards[max(0,i-20):i+1])) for i in range(n_episodes)]
        plt.plot(smoothed, label="Cumulative reward (smoothed)")
        plt.title("Q-learning: Episode Reward vs Episodes (4×4 GridWorld)")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for reward curve)"))
    _pause()

    section_header("5. CODE")
    code_block("Q-learning with ε-greedy decay", """
import numpy as np

def q_learning(gw, n_episodes=500, alpha=0.1, gamma=0.9,
               eps_start=1.0, eps_end=0.1, seed=7):
    rng = np.random.default_rng(seed)
    Q = np.zeros((gw.n_states, gw.n_actions))
    rewards = []
    for ep in range(n_episodes):
        s = gw.reset(); ep_r = 0.0
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - ep/(n_episodes*0.8))
        for _ in range(100):
            a = (rng.integers(gw.n_actions) if rng.random() < eps
                 else Q[s].argmax())
            s2, r, done = gw.step(a)
            Q[s,a] += alpha * (r + gamma*Q[s2].max()*(1-done) - Q[s,a])
            ep_r += r; s = s2
            if done: break
        rewards.append(ep_r)
    return Q, rewards

Q, rew = q_learning(gw)
pi_learned = Q.argmax(axis=1)
print(f"Avg reward last 100 eps: {np.mean(rew[-100:]):.3f}")
print("Learned policy:")
for r in range(4):
    print(' '.join(['↑↓→←'[pi_learned[r*4+c]] for c in range(4)]))
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Q-learning is off-policy: learns Q* while exploring with any behaviour policy",
        "ε-greedy annealing: explore broadly early, exploit increasingly as Q improves",
        "16×4 Q-table for 4×4 grid — model-free; no P or R needed explicitly",
        "Fixed α=0.1 works in practice; Robbins-Monro schedule needed for proofs",
        "Q* uniquely defines π*; the table stores expected returns for all (s,a) pairs",
        "DQN replaces the table with a neural network Q(s,a;θ) for large state spaces",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 5 — Multi-Armed Bandit
# ══════════════════════════════════════════════════════════════════════════════
def topic_bandit():
    clear()
    breadcrumb("mlmath", "MDP Solver", "Multi-Armed Bandit")
    section_header("MULTI-ARMED BANDIT")
    print()

    section_header("1. THEORY")
    print(white("""
  The K-armed bandit is the simplest reinforcement learning problem: at each
  round t the agent selects an arm a_t ∈ {1,...,K} and receives a stochastic
  reward r_t ~ N(μ_{a_t}, σ²). The goal is to maximise cumulative reward
  (equivalently, minimise cumulative regret R_T = Σₜ (μ* − μ_{a_t})).

  ε-GREEDY: with probability ε take a random arm; otherwise take arm with highest
  empirical mean Q̄(a) = (1/N(a)) Σ rewards from arm a.
  Decaying schedule εₜ = ε₀/t gives O(log T) regret asymptotically.

  UCB1 (UPPER CONFIDENCE BOUND): acts on the principle of optimism-in-face-of-
  uncertainty — always try the arm that could plausibly be best:
      a_t = argmax_a [ Q̄(a) + c · √(ln t / N(a)) ]
  The bonus √(ln t / N(a)) is large for under-sampled arms, driving exploration.
  UCB1 achieves O(√(KT log T)) worst-case regret — optimal up to log factors.

  THOMPSON SAMPLING (Bayesian): maintain a Beta(α_a, β_a) posterior on the
  success probability of each arm (for Bernoulli rewards). At each round:
  1. Sample θ_a ~ Beta(α_a, β_a) for each arm.
  2. Select a* = argmax θ_a.
  3. Update: if reward=1, α_{a*} += 1; else β_{a*} += 1.
  Thompson Sampling matches UCB1 in theory and often outperforms it empirically.

  REGRET LOWER BOUND: Lai and Robbins (1985) showed no algorithm can do better
  than Ω(Σ_{a≠a*} (μ*−μa)/KL(μa, μ*) · log T) — UCB and TS match this.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  Regret:  R_T = Σₜ₌₁ᵀ (μ* − μ_{aₜ})"))
    print(formula("  ε-greedy: a = Unif(K) w.p. ε, else argmax Q̄(a)"))
    print(formula("  UCB1:  a = argmaxₐ [Q̄(a) + c√(ln t / N(a))]"))
    print(formula("  Thompson: θₐ ~ Beta(αₐ, βₐ), a = argmax θₐ"))
    print(formula("  Beta update: αₐ += r, βₐ += 1−r  (Bernoulli reward)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — 1000 Rounds, 5 Arms")
    rng = np.random.default_rng(99)
    K, T = 5, 1000
    true_means = np.array([0.25, 0.55, 0.80, 0.40, 0.65])  # arm 2 is best
    mu_star = true_means.max()

    def pull_gaussian(a): return rng.normal(true_means[a], 0.4)
    def pull_bernoulli(a): return float(rng.random() < true_means[a])

    # ε-greedy
    def run_eps(eps_init=0.3):
        Q, N = np.zeros(K), np.zeros(K)
        regrets = []
        for t in range(1, T+1):
            eps = eps_init / np.sqrt(t)
            a = rng.integers(K) if rng.random() < eps else Q.argmax()
            r = pull_gaussian(a)
            N[a] += 1; Q[a] += (r - Q[a]) / N[a]
            regrets.append(mu_star - true_means[a])
        return np.cumsum(regrets), Q

    # UCB
    def run_ucb(c=2.0):
        Q, N = np.zeros(K), np.zeros(K)
        regrets = []
        for t in range(1, T+1):
            bonus = c * np.sqrt(np.log(t) / (N + 1e-9))
            a = (Q + bonus).argmax()
            r = pull_gaussian(a)
            N[a] += 1; Q[a] += (r - Q[a]) / N[a]
            regrets.append(mu_star - true_means[a])
        return np.cumsum(regrets), Q

    # Thompson Sampling (Gaussian via sufficient statistics)
    def run_thompson():
        mu_est = np.zeros(K); prec = np.ones(K)
        regrets = []
        for _ in range(T):
            samples = rng.normal(mu_est, 1.0 / np.sqrt(prec))
            a = samples.argmax()
            r = pull_gaussian(a)
            prec[a] += 1; mu_est[a] += (r - mu_est[a]) / prec[a]
            regrets.append(mu_star - true_means[a])
        return np.cumsum(regrets), mu_est

    reg_eps, Q_eps = run_eps()
    reg_ucb, Q_ucb = run_ucb()
    reg_ts,  Q_ts  = run_thompson()

    print(f"\n  {bold_cyan('5-Arm Gaussian Bandit — Cumulative Regret Comparison (T=1000):')}\n")
    table(
        ["Algorithm", "Regret @100", "Regret @500", "Regret @1000", "Avg Reward"],
        [
            ["ε-greedy (decay)", f"{reg_eps[99]:.2f}", f"{reg_eps[499]:.2f}",
             f"{reg_eps[999]:.2f}", f"{true_means.max() - reg_eps[999]/T:.3f}"],
            ["UCB (c=2)", f"{reg_ucb[99]:.2f}", f"{reg_ucb[499]:.2f}",
             f"{reg_ucb[999]:.2f}", f"{true_means.max() - reg_ucb[999]/T:.3f}"],
            ["Thompson Sampling", f"{reg_ts[99]:.2f}", f"{reg_ts[499]:.2f}",
             f"{reg_ts[999]:.2f}", f"{true_means.max() - reg_ts[999]/T:.3f}"],
        ],
        [cyan, yellow, yellow, green, white]
    )
    print()

    try:
        import plotext as plt
        plt.clear_figure()
        plt.plot(reg_eps.tolist(), label="eps-greedy")
        plt.plot(reg_ucb.tolist(), label="UCB c=2")
        plt.plot(reg_ts.tolist(),  label="Thompson")
        plt.title("Cumulative Regret vs Rounds — 5-Arm Bandit")
        plt.xlabel("Round"); plt.ylabel("Cumulative Regret")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for regret comparison plot)"))
    _pause()

    section_header("5. CODE")
    code_block("Multi-Armed Bandit: ε-greedy, UCB, Thompson Sampling", """
import numpy as np

class EpsilonGreedy:
    def __init__(self, k, eps=0.1):
        self.Q=np.zeros(k); self.N=np.zeros(k); self.eps=eps

    def act(self, rng):
        return rng.integers(len(self.Q)) if rng.random()<self.eps else self.Q.argmax()

    def update(self, a, r):
        self.N[a]+=1; self.Q[a]+=(r-self.Q[a])/self.N[a]

class UCB:
    def __init__(self, k, c=2.0):
        self.Q=np.zeros(k); self.N=np.zeros(k); self.c=c; self.t=0

    def act(self, rng=None):
        self.t+=1
        return (self.Q + self.c*np.sqrt(np.log(self.t)/(self.N+1e-9))).argmax()

    def update(self, a, r):
        self.N[a]+=1; self.Q[a]+=(r-self.Q[a])/self.N[a]

class ThompsonGaussian:
    def __init__(self, k):
        self.mu=np.zeros(k); self.prec=np.ones(k)

    def act(self, rng):
        return rng.normal(self.mu, 1/np.sqrt(self.prec)).argmax()

    def update(self, a, r):
        self.prec[a]+=1; self.mu[a]+=(r-self.mu[a])/self.prec[a]

# Simulation
rng=np.random.default_rng(0); K,T=5,1000
true_means=np.array([0.25,0.55,0.80,0.40,0.65])
agents=[EpsilonGreedy(K), UCB(K), ThompsonGaussian(K)]
for agent in agents:
    total=0
    for _ in range(T):
        a=agent.act(rng)
        r=rng.normal(true_means[a], 0.4)
        agent.update(a, r); total+=r
    print(f"{type(agent).__name__}: total={total:.1f}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "Bandit = MDP with single state — pure exploration-exploitation trade-off",
        "ε-greedy: simple and effective but wastes pulls on clearly sub-optimal arms",
        "UCB: 'optimism in face of uncertainty' — upper confidence bound drives exploration",
        "Thompson: Bayesian sampling; naturally adapts as posterior concentrates",
        "Regret O(√KT log T) is achievable — within log factor of lower bound Ω(√KT)",
        "In full RL: exploration is harder — no easy confidence bound on state-action values",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC 6 — Exploration vs Exploitation
# ══════════════════════════════════════════════════════════════════════════════
def topic_exploration_exploitation():
    clear()
    breadcrumb("mlmath", "MDP Solver", "Exploration vs Exploitation")
    section_header("EXPLORATION vs EXPLOITATION")
    print()

    section_header("1. THEORY")
    print(white("""
  The exploration-exploitation dilemma is the central challenge in reinforcement
  learning: an agent must balance gathering new information (exploration) against
  using current knowledge to maximise reward (exploitation).

  PURE EXPLOITATION: always pick argmax Q̄(a). Greedy algorithm can get stuck at
  suboptimal arms if initial estimates are unlucky — incurs linear regret.

  PURE EXPLORATION: pick uniformly at random. Wastes reward by not exploiting
  known good arms — also incurs linear regret.

  ε-GREEDY SCHEDULES:
  - Constant ε: Θ(T) regret (exploration never stops)
  - Decaying ε_t = c/t: O(log T) regret but requires knowing gap Δ = μ*−μ₂
  - Annealed ε_t = ε₀·decay^t: practical, hyperparameter-sensitive

  SOFTMAX / BOLTZMANN: P(a) ∝ exp(Q(a)/τ) where temperature τ controls spread.
  τ → ∞: uniform exploration; τ → 0: pure exploitation.
  Advantage over ε-greedy: exploration favours second-best, not uniform random.

  UCB CONFIDENCE BOUNDS: the bonus √(ln t / N(a)) captures uncertainty.
  As N(a) grows, bonus shrinks — arm eventually exploited once confident enough.
  The log(t) numerator keeps re-exploring to detect non-stationarity.

  POSTERIOR SAMPLING (Thompson): natural Bayesian solution — uncertainty
  automatically encoded in posterior width. As data accumulates, posteriors
  sharpen and exploitation dominates. Works without explicit confidence sets.
"""))
    _pause()

    section_header("2. KEY FORMULAS")
    print(formula("  ε-decay: εₜ = min(1, c·K / (Δ²·t))  (Δ = reward gap)"))
    print(formula("  Softmax: P(a) ∝ exp(Q̄(a)/τ)"))
    print(formula("  UCB:     a = argmaxₐ [Q̄(a) + √(2 ln t / N(a))]"))
    print(formula("  Regret:  R_T = Σₐ Δₐ · E[N_T(a)]"))
    print(formula("  LB (Lai-Robbins): lim inf R_T/ln T ≥ Σ Δₐ/KL(μₐ,μ*)"))
    print()
    _pause()

    section_header("3. WORKED EXAMPLE — ASCII Visualisation")
    rng = np.random.default_rng(45)
    K, T_show = 4, 200
    true_means = np.array([0.3, 0.7, 0.5, 0.2])

    Q = np.zeros(K); N = np.zeros(K)
    arm_counts_eps = np.zeros(K)

    for t in range(1, T_show + 1):
        eps = 0.3 / np.sqrt(t)
        a = rng.integers(K) if rng.random() < eps else Q.argmax()
        r = rng.normal(true_means[a], 0.3)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]
        arm_counts_eps[a] += 1

    Q_ucb = np.zeros(K); N_ucb = np.zeros(K)
    arm_counts_ucb = np.zeros(K)
    for t in range(1, T_show + 1):
        bonus = 2.0 * np.sqrt(np.log(t + 1) / (N_ucb + 1e-9))
        a = (Q_ucb + bonus).argmax()
        r = rng.normal(true_means[a], 0.3)
        N_ucb[a] += 1; Q_ucb[a] += (r - Q_ucb[a]) / N_ucb[a]
        arm_counts_ucb[a] += 1

    arm_names = ["A", "B★", "C", "D"]
    print(f"\n  {bold_cyan('Arm pull distribution after T=200 rounds:')}\n")
    print(f"  {'Arm':<6} {'True μ':<10} {'ε-greedy pulls':<20} {'UCB pulls'}")
    print(f"  {'─'*4}   {'─'*7}   {'─'*16}   {'─'*16}")
    for i in range(K):
        bar_e = green("█" * int(arm_counts_eps[i] / 3)) if i == 1 else cyan("█" * int(arm_counts_eps[i] / 3))
        bar_u = green("█" * int(arm_counts_ucb[i] / 3)) if i == 1 else cyan("█" * int(arm_counts_ucb[i] / 3))
        print(f"  {arm_names[i]:<6} {yellow(f'{true_means[i]:.2f}'):<18} "
              f"{bar_e} {grey(str(int(arm_counts_eps[i]))):<10} "
              f"{bar_u} {grey(str(int(arm_counts_ucb[i])))}")

    print(f"\n  {bold_cyan('Exploration-Exploitation Tradeoff Concept:')}\n")
    width = 50
    for eps_v in [0.0, 0.1, 0.3, 0.5, 1.0]:
        exploit = int((1 - eps_v) * width)
        explore = width - exploit
        bar = green("█" * exploit) + red("░" * explore)
        label = f"ε={eps_v:.1f}"
        print(f"  {label:<8} [{bar}]  "
              f"exploit:{green(f'{exploit*2}%'):<10} explore:{red(f'{explore*2}%')}")
    print()
    print(f"  {hint('UCB concentrates pulls on arm B★ (highest mean) while ε-greedy is more uniform.')}")
    print()
    _pause()

    section_header("4. VISUALIZATION")
    try:
        import plotext as plt
        eps_values = np.linspace(0, 1, 50)
        exploit_pct = (1 - eps_values) * 100
        explore_pct = eps_values * 100
        plt.clear_figure()
        plt.plot(eps_values.tolist(), exploit_pct.tolist(), label="Exploit %")
        plt.plot(eps_values.tolist(), explore_pct.tolist(), label="Explore %")
        plt.title("ε-Greedy: Exploration vs Exploitation Trade-off")
        plt.xlabel("ε"); plt.ylabel("Percentage")
        plt.show()
    except ImportError:
        print(grey("  (install plotext for trade-off plot)"))
    _pause()

    section_header("5. CODE")
    code_block("Exploration strategies comparison", """
import numpy as np

def softmax_policy(Q, tau=0.5):
    '''Boltzmann exploration.'''
    logits = Q / tau - Q.max() / tau
    p = np.exp(logits); return p / p.sum()

def ucb_bonus(Q, N, t, c=2.0):
    return Q + c * np.sqrt(np.log(t + 1) / (N + 1e-9))

# Visualise UCB bonus shrinking as arm is pulled more
t = 500; arm_pulls = np.array([10, 50, 100, 200, 400])
q_est = 0.6  # assume Q̄(a) = 0.6
print(f"{'N(a)':<8} {'UCB bonus':<14} {'UCB index'}")
for n in arm_pulls:
    bonus = 2.0 * np.sqrt(np.log(t) / n)
    print(f"{n:<8} {bonus:.4f}         {q_est + bonus:.4f}")

print()
# Show softmax temperature effect
Q_ex = np.array([0.5, 0.8, 0.3, 0.6])
for tau in [0.01, 0.1, 0.5, 2.0, 10.0]:
    p = softmax_policy(Q_ex, tau)
    print(f"τ={tau:<5}: {np.round(p, 3)}")
""")
    _pause()

    section_header("6. KEY INSIGHTS")
    for ins in [
        "No exploration → gets stuck; too much exploration → wastes reward",
        "Regret measures cost of not always picking best arm — formalises dilemma",
        "UCB: systematic exploration driven by information deficit, not randomness",
        "Thompson: posterior sampling is Bayesian exploration — elegant and effective",
        "Decaying ε: ensures GLIE (greedy in limit) — Q-learning convergence guarantee",
        "In deep RL: exploration is an open problem — count-based, RND, curiosity methods",
    ]:
        print(f"  {green('✦')}  {white(ins)}")
    print()
    topic_nav()


# ══════════════════════════════════════════════════════════════════════════════
def run():
    topics = [
        ("GridWorld Environment",       topic_gridworld),
        ("Value Iteration",             topic_value_iteration),
        ("Policy Iteration",            topic_policy_iteration),
        ("Q-Learning (Q-Table)",        topic_q_table),
        ("Multi-Armed Bandit",          topic_bandit),
        ("Exploration vs Exploitation", topic_exploration_exploitation),
    ]
    block_menu("b16", "MDP Solver", topics)
