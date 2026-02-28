"""Exercise Set 08: Backpropagation"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title; self.difficulty = difficulty
        self.description = description; self.hint = hint
        self.starter_code = starter_code; self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="Backprop Through a Linear Layer",
        difficulty="Beginner",
        description="""
  Implement forward and backward pass for a single linear layer:
    Forward:  y = W @ x + b
    Backward: given dL/dy, compute dL/dW, dL/db, dL/dx

  Formulas:
    dL/dW = dL/dy @ x.T  (for batch: dL/dy.T @ x)
    dL/db = sum(dL/dy, axis=0)
    dL/dx = W.T @ dL/dy

  Test with a batch of 3 samples, d_in=4, d_out=2.
  Verify using numerical gradient check.
""",
        hint="""
  For batch (N, d_in) → (N, d_out):
    W: (d_out, d_in)
    y = X @ W.T + b  (if using row vectors)
    dL/dW = dL_dy.T @ X  shape: (d_out, d_in)
    dL/db = dL_dy.sum(axis=0)
    dL/dX = dL_dy @ W  shape: (N, d_in)
""",
        starter_code="""
import numpy as np

class LinearLayer:
    def __init__(self, d_in, d_out):
        np.random.seed(0)
        self.W = np.random.randn(d_out, d_in) * 0.1
        self.b = np.zeros(d_out)
        self.x = None  # cache for backward

    def forward(self, x):
        self.x = x
        # TODO: y = W @ x.T + b (handle batch)
        pass

    def backward(self, dL_dy):
        # TODO: compute dL_dW, dL_db, dL_dx
        pass

layer = LinearLayer(4, 2)
X = np.random.randn(3, 4)
y = layer.forward(X)
dL_dy = np.ones_like(y)
dW, db, dX = layer.backward(dL_dy)
print(f"y shape: {y.shape}, dX shape: {dX.shape}, dW shape: {dW.shape}")
""",
        solution_code="""
import numpy as np

class LinearLayer:
    def __init__(self, d_in, d_out):
        np.random.seed(0)
        self.W = np.random.randn(d_out, d_in) * 0.1
        self.b = np.zeros(d_out)
        self.x = None

    def forward(self, x):  # x: (N, d_in)
        self.x = x
        return x @ self.W.T + self.b  # (N, d_out)

    def backward(self, dL_dy):  # dL_dy: (N, d_out)
        dL_dW = dL_dy.T @ self.x           # (d_out, d_in)
        dL_db = dL_dy.sum(axis=0)           # (d_out,)
        dL_dx = dL_dy @ self.W              # (N, d_in)
        return dL_dW, dL_db, dL_dx

# Test
np.random.seed(0)
layer = LinearLayer(4, 2)
X = np.random.randn(3, 4)
y = layer.forward(X)
dL_dy = np.ones_like(y)
dW, db, dX = layer.backward(dL_dy)
print(f"y:  {y.shape}")
print(f"dX: {dX.shape}")
print(f"dW: {dW.shape}")
print(f"db: {db.shape}")

# Numerical gradient check
def f(X_flat):
    X2 = X_flat.reshape(3, 4)
    return (X2 @ layer.W.T + layer.b).sum()

eps = 1e-5
X_flat = X.flatten()
num_grad = np.zeros_like(X_flat)
for i in range(len(X_flat)):
    xp, xm = X_flat.copy(), X_flat.copy()
    xp[i] += eps; xm[i] -= eps
    num_grad[i] = (f(xp) - f(xm)) / (2*eps)

err = np.max(np.abs(dX.flatten() - num_grad))
print(f"Gradient check error: {err:.2e}")
print("PASSED ✓" if err < 1e-5 else "FAILED ✗")
"""
    ),

    Exercise(
        title="Backprop Through Softmax + Cross-Entropy",
        difficulty="Intermediate",
        description="""
  Implement combined Softmax + Cross-Entropy loss and its gradient.

  Forward:
    logits → softmax → cross-entropy loss

  Key result (combined gradient — very clean!):
    dL/d(logits) = softmax(logits) - one_hot(y_true)

  Verify this elegant result numerically.
  Test on 3 classes, batch size 4.
""",
        hint="""
  softmax(z)_i = exp(z_i - max(z)) / sum(exp(z - max(z)))  [numerically stable]
  CE(p, y) = -sum(y_onehot * log(p))  (averaged over batch)
  Combined gradient: dL/dz = (p - y_onehot) / N  [where N = batch size]
  This is one of the most elegant results in deep learning!
""",
        starter_code="""
import numpy as np

def softmax(logits):
    # TODO: numerically stable softmax
    pass

def cross_entropy_loss(logits, y_true):
    # TODO: compute CE loss
    pass

def softmax_ce_grad(logits, y_true):
    # TODO: return dL/d(logits)
    pass

logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3],
                   [1.0, 1.0, 3.0],
                   [0.2, 0.8, 2.1]])
y_true = np.array([0, 1, 2, 1])

loss = cross_entropy_loss(logits, y_true)
grad = softmax_ce_grad(logits, y_true)
print(f"Loss: {loss:.4f}")
print(f"Grad shape: {grad.shape}")
""",
        solution_code="""
import numpy as np

def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    n = len(y_true)
    return -np.sum(np.log(probs[np.arange(n), y_true] + 1e-10)) / n

def softmax_ce_grad(logits, y_true):
    probs = softmax(logits)
    n = len(y_true)
    probs[np.arange(n), y_true] -= 1
    return probs / n  # dL/dz = (p - y_onehot) / N

logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3],
                   [1.0, 1.0, 3.0],
                   [0.2, 0.8, 2.1]])
y_true = np.array([0, 1, 2, 1])

loss = cross_entropy_loss(logits, y_true)
grad = softmax_ce_grad(logits, y_true)
print(f"Loss: {loss:.4f}")
print(f"Gradient dL/dlogits:"); print(grad.round(4))

# Numerical check
def f_scalar(logits_flat):
    return cross_entropy_loss(logits_flat.reshape(4,3), y_true)
eps = 1e-5
lf = logits.flatten()
num_g = np.zeros_like(lf)
for i in range(len(lf)):
    lp, lm = lf.copy(), lf.copy()
    lp[i]+=eps; lm[i]-=eps
    num_g[i] = (f_scalar(lp) - f_scalar(lm))/(2*eps)
err = np.max(np.abs(grad.flatten() - num_g))
print(f"Numerical check error: {err:.2e}")
print("PASSED ✓" if err < 1e-5 else "FAILED ✗")
"""
    ),

    Exercise(
        title="Implement Full 2-Layer Neural Net Backprop",
        difficulty="Advanced",
        description="""
  Implement a complete 2-layer neural network with backpropagation.

  Architecture: x → [Linear(d_in,H)] → ReLU → [Linear(H,d_out)] → Softmax

  Loss: Cross-entropy
  Optimizer: SGD with learning rate 0.01

  Train on XOR problem (4 points, 2 classes).
  Should reach ~0 training loss.

  Layers: d_in=2, H=8, d_out=2
  Gradient check all parameters before training.
""",
        hint="""
  Forward: z1=W1@X+b1, h=relu(z1), z2=W2@h+b2, probs=softmax(z2)
  Backward:
    dz2 = probs - onehot          # combined softmax+CE gradient
    dW2 = h.T @ dz2
    dh  = dz2 @ W2.T
    dz1 = dh * (z1 > 0)           # ReLU backward
    dW1 = X.T @ dz1
  Update: W -= lr * dW
""",
        starter_code="""
import numpy as np

np.random.seed(42)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

class TwoLayerNet:
    def __init__(self, d_in=2, H=8, d_out=2):
        # TODO: initialize weights (He init)
        pass

    def forward(self, X):
        # TODO: two linear layers, ReLU between
        pass

    def backward(self, X, y):
        # TODO: backprop, return gradients
        pass

    def train(self, X, y, lr=0.01, n_iter=1000):
        # TODO: training loop
        pass

net = TwoLayerNet()
net.train(X_xor, y_xor, lr=0.1, n_iter=2000)
""",
        solution_code="""
import numpy as np

np.random.seed(42)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

class TwoLayerNet:
    def __init__(self, d_in=2, H=8, d_out=2):
        self.W1 = np.random.randn(d_in, H) * np.sqrt(2/d_in)
        self.b1 = np.zeros(H)
        self.W2 = np.random.randn(H, d_out) * np.sqrt(2/H)
        self.b2 = np.zeros(d_out)

    def softmax(self, z):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h  = np.maximum(0, self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.p  = self.softmax(self.z2)
        return self.p

    def loss(self, X, y):
        p = self.forward(X)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-10))

    def backward(self, X, y, lr=0.01):
        n = len(y)
        p = self.p.copy()
        p[np.arange(n), y] -= 1
        dz2 = p / n
        dW2 = self.h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh  = dz2 @ self.W2.T
        dz1 = dh * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        self.W1 -= lr*dW1; self.b1 -= lr*db1

    def train(self, X, y, lr=0.1, n_iter=3000):
        for i in range(n_iter):
            self.forward(X)
            self.backward(X, y, lr)
            if i % 500 == 0:
                l = self.loss(X, y)
                preds = self.forward(X).argmax(axis=1)
                acc = (preds == y).mean()
                print(f"Step {i:4d}: loss={l:.4f}, acc={acc:.2f}")

net = TwoLayerNet()
net.train(X_xor, y_xor, lr=0.1, n_iter=3000)
preds = net.forward(X_xor).argmax(axis=1)
print(f"\\nXOR predictions: {preds}  (should be [0,1,1,0])")
print(f"Correct: {(preds == y_xor).all()}")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║     EXERCISES — Block 08: Backpropagation        ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m\n")
        for i, ex in enumerate(exercises, 1):
            dc = "\033[92m" if ex.difficulty=="Beginner" else "\033[93m" if ex.difficulty=="Intermediate" else "\033[91m"
            print(f"  {i}. {dc}[{ex.difficulty}]\033[0m {ex.title}")
        print("\n  \033[90m[0] Back\033[0m")
        choice = input("\n\033[96mSelect: \033[0m").strip()
        if choice == "0": break
        try:
            ex = exercises[int(choice)-1]
            _run_exercise(ex)
        except (ValueError, IndexError): pass


def _run_exercise(ex):
    dc = "\033[92m" if ex.difficulty=="Beginner" else "\033[93m" if ex.difficulty=="Intermediate" else "\033[91m"
    print(f"\n\033[95m━━━ {ex.title} ━━━\033[0m")
    print(f"  {dc}{ex.difficulty}\033[0m\n{ex.description}")
    while True:
        cmd = input("\n  [h]int [c]ode [r]un [s]olution [b]ack: ").strip().lower()
        if cmd=='b': break
        elif cmd=='h': print(f"\n\033[93mHINT\033[0m\n{ex.hint}")
        elif cmd=='c': print(f"\n\033[94mSTARTER\033[0m\n{ex.starter_code}")
        elif cmd=='s': print(f"\n\033[92mSOLUTION\033[0m\n{ex.solution_code}")
        elif cmd=='r':
            try: exec(compile(ex.solution_code,"<sol>","exec"),{})
            except Exception as e: print(f"\033[91m{e}\033[0m")
