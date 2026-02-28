"""Exercise Set 10: Supervised Learning"""
import numpy as np


class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code, test_cases=None):
        self.title = title; self.difficulty = difficulty
        self.description = description; self.hint = hint
        self.starter_code = starter_code; self.solution_code = solution_code
        self.test_cases = test_cases or []


exercises = [
    Exercise(
        title="Linear Regression Normal Equation",
        difficulty="Beginner",
        description="""
  Implement Linear Regression using the Normal Equation:
    w* = (XᵀX)⁻¹ Xᵀy

  Test on a dataset where y = 2x₁ + 3x₂ - 1 + noise.
  Should recover weights ≈ [2, 3] and bias ≈ -1.

  Also compute R² score:
    R² = 1 - SS_res/SS_tot
    SS_res = sum((y - ŷ)²)
    SS_tot = sum((y - ȳ)²)
""",
        hint="""
  Add a bias column of ones to X (augmented X: [1, x₁, x₂]).
  w = np.linalg.lstsq(X_aug, y, rcond=None)[0]
  Or: w = np.linalg.inv(X.T @ X) @ X.T @ y  (may be ill-conditioned)
  R² = 1 - np.sum((y-yhat)**2) / np.sum((y-y.mean())**2)
""",
        starter_code="""
import numpy as np

np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
y = 2*X[:,0] + 3*X[:,1] - 1 + 0.5*np.random.randn(n)

def normal_equation(X, y):
    # TODO: add bias, compute w via normal equation
    pass

def r_squared(y, yhat):
    # TODO
    pass

w = normal_equation(X, y)
print(f"Weights: {w}")  # should be near [−1, 2, 3]
""",
        solution_code="""
import numpy as np

np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
y = 2*X[:,0] + 3*X[:,1] - 1 + 0.5*np.random.randn(n)

def normal_equation(X, y):
    X_aug = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X_aug, y, rcond=None)[0]

def r_squared(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot

w = normal_equation(X, y)
X_aug = np.column_stack([np.ones(n), X])
yhat = X_aug @ w
r2 = r_squared(y, yhat)
print(f"Learned weights: bias={w[0]:.4f}, w1={w[1]:.4f}, w2={w[2]:.4f}")
print(f"True weights:    bias=-1.0000, w1=2.0000, w2=3.0000")
print(f"R² = {r2:.6f}")
print(f"MSE = {np.mean((y-yhat)**2):.4f}")
"""
    ),

    Exercise(
        title="Logistic Regression with Gradient Descent",
        difficulty="Intermediate",
        description="""
  Implement Logistic Regression (binary) from scratch using gradient descent.

  Model: σ(X @ w + b) where σ(z) = 1/(1+exp(-z))
  Loss:  BCE = -mean(y·log(p) + (1-y)·log(1-p))
  Gradient:
    ∂L/∂w = (1/n) Xᵀ(p - y)
    ∂L/∂b = mean(p - y)

  Train on linearly separable 2D data.
  Achieve > 95% accuracy. Show decision boundary in ASCII.
""",
        hint="""
  Numerically stable: p = sigmoid(z), z = X @ w + b
  Gradient: error = p - y, dw = X.T @ error / n, db = error.mean()
  Update: w -= lr * dw, b -= lr * db
  Decision boundary: where X @ w + b = 0 → x2 = -(w[0]*x1 + b) / w[1]
""",
        starter_code="""
import numpy as np
np.random.seed(0)

X0 = np.random.randn(50,2) + [-2,0]
X1 = np.random.randn(50,2) + [2,0]
X = np.vstack([X0,X1])
y = np.array([0]*50 + [1]*50)

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr; self.n_iter = n_iter
        self.w = None; self.b = 0

    def fit(self, X, y):
        # TODO
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

clf = LogisticRegression()
clf.fit(X, y)
preds = clf.predict(X)
print(f"Accuracy: {(preds==y).mean():.4f}")
""",
        solution_code="""
import numpy as np
np.random.seed(0)

X0 = np.random.randn(50,2) + [-2,0]
X1 = np.random.randn(50,2) + [2,0]
X = np.vstack([X0,X1])
y = np.array([0]*50 + [1]*50, dtype=float)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=500):
        self.lr, self.n_iter = lr, n_iter
        self.w = None; self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        losses = []
        for i in range(self.n_iter):
            p = sigmoid(X @ self.w + self.b)
            loss = -np.mean(y*np.log(p+1e-10)+(1-y)*np.log(1-p+1e-10))
            losses.append(loss)
            err = p - y
            self.w -= self.lr * X.T @ err / n
            self.b -= self.lr * err.mean()
            if i % 100 == 0:
                print(f"  step {i:4d}: loss={loss:.4f}")
        return losses

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

clf = LogisticRegression()
clf.fit(X, y)
preds = clf.predict(X)
print(f"\\nAccuracy: {(preds==y.astype(int)).mean():.4f}")
print(f"Weights: {clf.w.round(4)}, Bias: {clf.b:.4f}")
print(f"Decision: x₂ = {-clf.w[0]/clf.w[1]:.2f}*x₁ + {-clf.b/clf.w[1]:.2f}")
"""
    ),

    Exercise(
        title="Decision Tree from Scratch (Information Gain)",
        difficulty="Advanced",
        description="""
  Implement a binary decision tree classifier using information gain.

  Gini impurity: G = 1 - Σ pₖ²
  Entropy: H = -Σ pₖ log₂(pₖ)
  Information gain: IG = H(parent) - Σ (|child|/|parent|) * H(child)

  At each node: find feature + threshold that maximizes IG.
  Stop when: depth >= max_depth OR node is pure.

  Test on sklearn's make_classification or Iris dataset.
""",
        hint="""
  For each feature i and threshold t:
    left = X[X[:,i] <= t], right = X[X[:,i] > t]
    ig = entropy(y) - (len(L)/n)*entropy(y_L) - (len(R)/n)*entropy(y_R)
  Build recursively up to max_depth.
  Predict: traverse tree from root following threshold comparisons.
""",
        starter_code="""
import numpy as np

def entropy(y):
    # TODO: H = -Σ p log2(p)
    pass

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature; self.threshold = threshold
        self.left = left; self.right = right; self.value = value  # leaf label

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth; self.root = None

    def fit(self, X, y):
        # TODO: build tree recursively
        pass

    def predict(self, X):
        # TODO
        pass

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X, y = X[y<2], y[y<2]  # binary
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
dt = DecisionTree(max_depth=3); dt.fit(Xtr, ytr)
print(f"Accuracy: {(dt.predict(Xte)==yte).mean():.4f}")
""",
        solution_code="""
import numpy as np

def entropy(y):
    if len(y)==0: return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts/len(y)
    return -sum(p*np.log2(p+1e-10) for p in probs)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature; self.threshold=threshold
        self.left=left; self.right=right; self.value=value

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth=max_depth; self.root=None

    def _best_split(self, X, y):
        best_ig, best_f, best_t = -1, None, None
        h = entropy(y)
        n = len(y)
        for f in range(X.shape[1]):
            thresholds = np.unique(X[:,f])
            for t in thresholds:
                L, R = y[X[:,f]<=t], y[X[:,f]>t]
                if len(L)==0 or len(R)==0: continue
                ig = h - (len(L)/n)*entropy(L) - (len(R)/n)*entropy(R)
                if ig > best_ig: best_ig,best_f,best_t = ig,f,t
        return best_f, best_t

    def _build(self, X, y, depth):
        if depth==self.max_depth or len(np.unique(y))==1 or len(y)==0:
            return Node(value=np.bincount(y).argmax())
        f, t = self._best_split(X, y)
        if f is None: return Node(value=np.bincount(y).argmax())
        mask = X[:,f]<=t
        return Node(f, t, self._build(X[mask],y[mask],depth+1),
                         self._build(X[~mask],y[~mask],depth+1))

    def fit(self, X, y): self.root=self._build(X, y.astype(int), 0)

    def _predict_one(self, node, x):
        if node.value is not None: return node.value
        return self._predict_one(node.left if x[node.feature]<=node.threshold else node.right, x)

    def predict(self, X): return np.array([self._predict_one(self.root, x) for x in X])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X, y = X[y<2], y[y<2]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=0)
dt = DecisionTree(max_depth=4); dt.fit(Xtr,ytr)
preds = dt.predict(Xte)
print(f"Decision Tree Accuracy: {(preds==yte).mean():.4f}")

from sklearn.tree import DecisionTreeClassifier
sk_dt = DecisionTreeClassifier(max_depth=4); sk_dt.fit(Xtr,ytr)
print(f"sklearn Tree Accuracy:  {sk_dt.score(Xte,yte):.4f}")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║     EXERCISES — Block 10: Supervised Learning   ║\033[0m")
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
