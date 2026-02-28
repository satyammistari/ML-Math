"""Exercise Set 15: Reinforcement Learning"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title; self.difficulty = difficulty
        self.description = description; self.hint = hint
        self.starter_code = starter_code; self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="Value Iteration on GridWorld",
        difficulty="Beginner",
        description="""
  Implement value iteration on a 4×4 GridWorld:
    - States: (0,0) to (3,3), start=(0,0), goal=(3,3)
    - Actions: up/down/left/right
    - Reward: +1 at goal, -0.01 at all other steps
    - γ (gamma) = 0.99

  Bellman update:
    V(s) ← max_a Σ P(s'|s,a)[R(s,a,s') + γ·V(s')]
  (Deterministic: P=1 for the actual next state)

  Run until max|V_new - V_old| < 1e-6.
  Print value function as 4×4 grid.
""",
        hint="""
  State = (row, col). Encode as s = row*4 + col.
  Transitions (deterministic): new_r = max(0,min(3,r+dr))
  V = np.zeros(16); at each step, for each state compute Q(s,a) for 4 actions.
  V_new[s] = max Q(s,a). Stop when max(|V_new-V|) < tol.
""",
        starter_code="""
import numpy as np

GRID_SIZE = 4; GOAL = (3,3); GAMMA = 0.99; ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

def state(r,c): return r*GRID_SIZE + c
def reward(r,c): return 1.0 if (r,c)==GOAL else -0.01
def step(r,c,a):
    dr,dc = ACTIONS[a]
    return max(0,min(GRID_SIZE-1,r+dr)), max(0,min(GRID_SIZE-1,c+dc))

def value_iteration(gamma=GAMMA, tol=1e-6):
    V = np.zeros(GRID_SIZE*GRID_SIZE)
    # TODO: Bellman update loop
    pass

V, n_iter = value_iteration()
print(f"Converged in {n_iter} iterations")
print(V.reshape(GRID_SIZE,GRID_SIZE).round(3))
""",
        solution_code="""
import numpy as np

GRID_SIZE = 4; GOAL = (3,3); GAMMA = 0.99; ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

def state(r,c): return r*GRID_SIZE+c
def reward(r,c): return 1.0 if (r,c)==GOAL else -0.01
def step(r,c,a):
    dr,dc=ACTIONS[a]
    return max(0,min(GRID_SIZE-1,r+dr)), max(0,min(GRID_SIZE-1,c+dc))

def value_iteration(gamma=GAMMA, tol=1e-6, max_iter=1000):
    V = np.zeros(GRID_SIZE**2)
    for it in range(max_iter):
        V_new = np.zeros_like(V)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                s = state(r,c)
                if (r,c)==GOAL: V_new[s]=reward(r,c); continue
                q = []
                for a in range(4):
                    nr,nc = step(r,c,a)
                    q.append(reward(r,c)+gamma*V[state(nr,nc)])
                V_new[s] = max(q)
        if np.max(np.abs(V_new-V)) < tol:
            return V_new, it+1
        V = V_new
    return V, max_iter

V, n_iter = value_iteration()
print(f"Converged in {n_iter} iterations")
grid = V.reshape(GRID_SIZE,GRID_SIZE)
print("Value Function (row=0 is top):")
for row in range(GRID_SIZE):
    print("  "+" ".join(f"{grid[row,c]:+.3f}" for c in range(GRID_SIZE)))

# Extract greedy policy
print("\\nGreedy Policy (0=up,1=dn,2=lt,3=rt):")
NAMES=['↑','↓','←','→']
for r in range(GRID_SIZE):
    row_str = "  "
    for c in range(GRID_SIZE):
        if (r,c)==GOAL: row_str+=" G "; continue
        best_a = max(range(4), key=lambda a: V[r*GRID_SIZE+max(0,min(3,r+ACTIONS[a][0]))*GRID_SIZE+max(0,min(3,c+ACTIONS[a][1]))])
        row_str+=f" {NAMES[best_a]} "
    print(row_str)
"""
    ),

    Exercise(
        title="Q-Learning on GridWorld",
        difficulty="Intermediate",
        description="""
  Implement Q-learning (model-free) on the same 4×4 GridWorld.

  Q-learning update:
    Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]

  Use ε-greedy policy: with prob ε→random, else argmax Q(s,·)
  Decay ε from 1.0 to 0.01 over training.

  Train for 5000 episodes. Plot episode reward.
  Compare final policy to value iteration result.
""",
        hint="""
  Q = np.zeros((n_states, n_actions))
  For each episode: reset to (0,0), step until goal or max_steps=100
    a = random if rand<eps else Q[s].argmax()
    Q[s,a] += alpha * (r + gamma*Q[s_next].max() - Q[s,a])
  eps decay: eps = max(0.01, eps * 0.999)
""",
        starter_code="""
import numpy as np
np.random.seed(42)

GRID_SIZE=4; GOAL_S=15; GAMMA=0.99; ALPHA=0.1; ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]

def state(r,c): return r*GRID_SIZE+c
def step(r,c,a):
    dr,dc=ACTIONS[a]
    return max(0,min(GRID_SIZE-1,r+dr)), max(0,min(GRID_SIZE-1,c+dc))

def q_learning(n_episodes=5000, alpha=ALPHA, gamma=GAMMA, eps_start=1.0):
    Q = np.zeros((GRID_SIZE**2, 4))
    rewards = []
    # TODO: Q-learning loop
    return Q, rewards

Q, rewards = q_learning()
print(f"Max Q values:\\n{Q.max(axis=1).reshape(GRID_SIZE,GRID_SIZE).round(3)}")
""",
        solution_code="""
import numpy as np
np.random.seed(42)

GRID_SIZE=4; GOAL=(3,3); GAMMA=0.99; ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]

def rc(s): return s//GRID_SIZE, s%GRID_SIZE
def state(r,c): return r*GRID_SIZE+c
def step(r,c,a):
    dr,dc=ACTIONS[a]
    return max(0,min(GRID_SIZE-1,r+dr)), max(0,min(GRID_SIZE-1,c+dc))
def reward(r,c): return 1.0 if (r,c)==GOAL else -0.01

def q_learning(n_episodes=5000, alpha=0.1, gamma=GAMMA, eps_start=1.0):
    Q = np.zeros((GRID_SIZE**2, 4)); eps=eps_start; rewards=[]
    for ep in range(n_episodes):
        r,c,ep_r = 0,0,0
        for _ in range(200):
            s=state(r,c)
            if (r,c)==GOAL: break
            a = np.random.randint(4) if np.random.rand()<eps else Q[s].argmax()
            nr,nc = step(r,c,a)
            s2 = state(nr,nc)
            rew = reward(nr,nc)
            Q[s,a] += alpha*(rew+gamma*Q[s2].max()-Q[s,a])
            r,c,ep_r = nr,nc,ep_r+rew
        rewards.append(ep_r)
        eps = max(0.01, eps*0.9995)
    return Q, rewards

Q, rewards = q_learning()
print(f"Avg last 500 eps reward: {np.mean(rewards[-500:]):.4f}")
print(f"Q-value grid (max action):")
for row in range(GRID_SIZE):
    print("  "+" ".join(f"{Q[state(row,c)].max():+.3f}" for c in range(GRID_SIZE)))
NAMES=['↑','↓','←','→']
print("\\nGreedy policy:")
for row in range(GRID_SIZE):
    cells = "  "
    for c in range(GRID_SIZE):
        cells += (" G " if (row,c)==GOAL else f" {NAMES[Q[state(row,c)].argmax()]} ")
    print(cells)
"""
    ),

    Exercise(
        title="REINFORCE Policy Gradient",
        difficulty="Advanced",
        description="""
  Implement the REINFORCE policy gradient algorithm for a simple
  1D continuous action space problem.

  Environment: particle moving on [-5, 5], target = 0
    State: position x ∈ [-5, 5]
    Action: velocity ∈ [-1, 1] (Gaussian policy)
    Reward: -|x| (negative distance to target)
    Episode length: 20 steps

  Policy: π(a|s) = N(μ_θ(s), σ²)  where μ_θ(s) = w·s + b
  REINFORCE update:
    ∇_θ J = E[G_t · ∇log π(a_t|s_t)]
    G_t = Σ_{k=0}^{T-t} γᵏ r_{t+k}  (return from step t)
""",
        hint="""
  log_prob of Gaussian: -0.5*(a-mu)^2/sigma^2 - log(sigma) - 0.5*log(2pi)
  Gradient of log_prob wrt w: (a-mu)/sigma^2 * s (chain rule)
  G_t = sum(gamma^k * r_{t+k} for k in range(T-t))
  Baseline (reduce variance): subtract mean(G) from returns
  Update: w += lr * sum(G_t * d_log_pi/d_w)
""",
        starter_code="""
import numpy as np

class REINFORCEAgent:
    def __init__(self, lr=0.01, gamma=0.99, sigma=0.5):
        self.lr=lr; self.gamma=gamma; self.sigma=sigma
        self.w=0.0; self.b=0.0  # linear policy: mu = w*s + b

    def policy(self, s):
        # TODO: return mean action and sampled action
        pass

    def update(self, states, actions, rewards):
        # TODO: compute returns G_t and update w,b
        pass

agent = REINFORCEAgent()
ep_rewards = []
for ep in range(1000):
    s,states,actions,rewards = 5*np.random.randn(),[], [],[]
    for _ in range(20):
        mu,a = agent.policy(s)
        a = np.clip(a,-1,1)
        r = -abs(s)
        states.append(s); actions.append(a); rewards.append(r)
        s = np.clip(s+a, -5, 5)
    ep_rewards.append(sum(rewards))
    agent.update(states, actions, rewards)
print(f"Avg last 100 ep reward: {np.mean(ep_rewards[-100:]):.3f}")
""",
        solution_code="""
import numpy as np

class REINFORCEAgent:
    def __init__(self, lr=0.01, gamma=0.99, sigma=0.3):
        self.lr=lr; self.gamma=gamma; self.sigma=sigma
        self.w=0.0; self.b=0.0

    def policy(self, s):
        mu = np.clip(self.w*s+self.b, -1, 1)
        a = mu + np.random.randn()*self.sigma
        return mu, a

    def update(self, states, actions, rewards):
        T = len(rewards)
        G = np.zeros(T)
        G[-1] = rewards[-1]
        for t in range(T-2,-1,-1):
            G[t] = rewards[t] + self.gamma*G[t+1]
        G = (G - G.mean()) / (G.std() + 1e-8)  # baseline normalization
        for t in range(T):
            s,a = states[t],actions[t]
            mu = np.clip(self.w*s+self.b,-1,1)
            log_grad = (a-mu)/self.sigma**2  # d log_pi / d mu
            d_mu_dw = s; d_mu_db = 1
            self.w += self.lr * G[t] * log_grad * d_mu_dw
            self.b += self.lr * G[t] * log_grad * d_mu_db

np.random.seed(42)
agent = REINFORCEAgent()
ep_rewards = []
for ep in range(2000):
    s = 5*np.random.randn()
    states,actions,rewards = [],[],[]
    for _ in range(20):
        mu,a = agent.policy(s)
        a = np.clip(a,-1,1)
        states.append(s); actions.append(a); rewards.append(-abs(s))
        s = np.clip(s+a,-5,5)
    ep_rewards.append(sum(rewards))
    agent.update(states,actions,rewards)
    if ep in [100,500,1000,2000-1]:
        print(f"Ep {ep+1:4d}: avg_r={np.mean(ep_rewards[-50:]):.3f}, w={agent.w:.3f}, b={agent.b:.3f}")
print(f"\\nFinal w={agent.w:.4f}  (good policy: w<0, drives toward 0)")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║   EXERCISES — Block 15: Reinforcement Learning  ║\033[0m")
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
