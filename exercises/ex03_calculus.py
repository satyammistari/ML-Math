"""
Exercise Set 03: Calculus
"""
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
        title="Numerical Gradient Check",
        difficulty="Beginner",
        description="""
  Implement a numerical gradient checker using the centered difference formula:
    ∂f/∂xᵢ ≈ (f(x + εeᵢ) - f(x - εeᵢ)) / (2ε)

  Test it on f(x, y) = x² + 3xy + y³
  Analytical gradient: ∇f = [2x + 3y,  3x + 3y²]
  At (x=2, y=1): ∇f = [7, 9]

  Report relative error between numerical and analytical gradient.
""",
        hint="""
  For each dimension i:
    x_plus = x.copy(); x_plus[i] += eps
    x_minus = x.copy(); x_minus[i] -= eps
    grad[i] = (f(x_plus) - f(x_minus)) / (2*eps)
  Relative error = ||grad_num - grad_anal|| / max(||grad_num||, ||grad_anal||)
""",
        starter_code="""
import numpy as np

def f(x):
    return x[0]**2 + 3*x[0]*x[1] + x[1]**3

def analytical_grad(x):
    return np.array([2*x[0] + 3*x[1], 3*x[0] + 3*x[1]**2])

def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    # TODO: implement centered difference for each dimension
    pass

x = np.array([2.0, 1.0])
g_anal = analytical_grad(x)
g_num  = numerical_grad(f, x)
print(f"Analytical: {g_anal}")
print(f"Numerical:  {g_num}")
# TODO: compute and print relative error
""",
        solution_code="""
import numpy as np

def f(x):
    return x[0]**2 + 3*x[0]*x[1] + x[1]**3

def analytical_grad(x):
    return np.array([2*x[0] + 3*x[1], 3*x[0] + 3*x[1]**2])

def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_p = x.copy(); x_p[i] += eps
        x_m = x.copy(); x_m[i] -= eps
        grad[i] = (f(x_p) - f(x_m)) / (2 * eps)
    return grad

x = np.array([2.0, 1.0])
g_anal = analytical_grad(x)
g_num  = numerical_grad(f, x)
rel_err = np.linalg.norm(g_num - g_anal) / max(np.linalg.norm(g_num), np.linalg.norm(g_anal))
print(f"Analytical:    {g_anal}")
print(f"Numerical:     {g_num.round(6)}")
print(f"Relative error: {rel_err:.2e}")
assert rel_err < 1e-5, "Gradient check failed!"
print("Gradient check PASSED ✓")
"""
    ),

    Exercise(
        title="Jacobian of Softmax",
        difficulty="Intermediate",
        description="""
  Derive and implement the Jacobian of the softmax function.

  softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

  The Jacobian ∂softmax(z)/∂z is an n×n matrix where:
    J_{ij} = sᵢ(δᵢⱼ - sⱼ)  [sᵢ = softmax(z)ᵢ, δᵢⱼ = Kronecker delta]

  Verify using numerical Jacobian.
  Input: z = [1.0, 2.0, 3.0]
""",
        hint="""
  J = diag(s) - s @ s.T
  where s = softmax(z) is a column vector.
  diag(s) is the diagonal matrix with s on the diagonal.
  Verify: J @ ones ≈ 0 (rows sum to 0 for softmax Jacobian)
""",
        starter_code="""
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def softmax_jacobian(z):
    # TODO: implement analytically
    # J_{ij} = s_i * (delta_{ij} - s_j)
    pass

def numerical_jacobian(f, z, eps=1e-5):
    n = len(z)
    J = np.zeros((n, n))
    for j in range(n):
        z_p, z_m = z.copy(), z.copy()
        z_p[j] += eps; z_m[j] -= eps
        J[:, j] = (f(z_p) - f(z_m)) / (2 * eps)
    return J

z = np.array([1.0, 2.0, 3.0])
J_anal = softmax_jacobian(z)
J_num  = numerical_jacobian(softmax, z)
print("Analytical J:\n", J_anal.round(4))
print("Numerical J:\n", J_num.round(4))
""",
        solution_code="""
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def softmax_jacobian(z):
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)

def numerical_jacobian(f, z, eps=1e-5):
    n = len(z)
    J = np.zeros((n, n))
    for j in range(n):
        z_p, z_m = z.copy(), z.copy()
        z_p[j] += eps; z_m[j] -= eps
        J[:, j] = (f(z_p) - f(z_m)) / (2 * eps)
    return J

z = np.array([1.0, 2.0, 3.0])
J_anal = softmax_jacobian(z)
J_num  = numerical_jacobian(softmax, z)
print("Analytical J:\n", J_anal.round(4))
print("Numerical J:\n", J_num.round(4))
err = np.max(np.abs(J_anal - J_num))
print(f"Max error: {err:.2e}")
print(f"Rows sum to 0: {np.allclose(J_anal.sum(axis=1), 0)}")
print("Jacobian check PASSED ✓" if err < 1e-5 else "FAILED ✗")
"""
    ),

    Exercise(
        title="Implement Autodiff with Dual Numbers",
        difficulty="Advanced",
        description="""
  Implement forward-mode automatic differentiation using dual numbers.

  A dual number: a + bε where ε² = 0
  If f(a + bε) = f(a) + f'(a)·b·ε, then the ε component gives the derivative.

  Implement a DualNumber class with +, *, sin, exp, ** operations.
  Then use it to automatically compute exact derivatives (no finite differences).

  Test: f(x) = (x³ + 2x) · sin(x) at x = 1.5
  Verify against numerical derivative.
""",
        hint="""
  class DualNumber:
      def __init__(self, val, deriv=0): ...
      def __add__(self, other): return DualNumber(self.val+other.val, self.deriv+other.deriv)
      def __mul__(self, other): return DualNumber(v1*v2, v1*d2 + d2*v1)  # product rule
      def sin(self): return DualNumber(sin(self.val), cos(self.val)*self.deriv)
  f(DualNumber(x, 1)).deriv  ← gives f'(x) exactly
""",
        starter_code="""
import math
import numpy as np

class DualNumber:
    def __init__(self, val, deriv=0.0):
        self.val = val
        self.deriv = deriv

    def __add__(self, other):
        # TODO
        pass

    def __mul__(self, other):
        # TODO: use product rule
        pass

    def __pow__(self, n):
        # TODO: power rule
        pass

    def sin(self):
        # TODO: d/dx sin(x) = cos(x)
        pass

    def __repr__(self):
        return f"Dual({self.val:.4f} + {self.deriv:.4f}ε)"

def f(x):
    return (x**3 + DualNumber(2)*x) * x.sin()

x_val = 1.5
x_dual = DualNumber(x_val, 1.0)  # deriv=1 to compute df/dx
result = f(x_dual)
print(f"f({x_val}) = {result.val:.6f}")
print(f"f'({x_val}) = {result.deriv:.6f}")
""",
        solution_code="""
import math
import numpy as np

class DualNumber:
    def __init__(self, val, deriv=0.0):
        self.val = float(val)
        self.deriv = float(deriv)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = DualNumber(other)
        return DualNumber(self.val + other.val, self.deriv + other.deriv)

    def __radd__(self, other): return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = DualNumber(other)
        return DualNumber(self.val*other.val,
                          self.val*other.deriv + self.deriv*other.val)

    def __rmul__(self, other): return self.__mul__(other)

    def __pow__(self, n):
        return DualNumber(self.val**n, n * self.val**(n-1) * self.deriv)

    def sin(self):
        return DualNumber(math.sin(self.val), math.cos(self.val) * self.deriv)

    def __repr__(self):
        return f"Dual({self.val:.6f} + {self.deriv:.6f}ε)"

def f(x):
    return (x**3 + DualNumber(2)*x) * x.sin()

x_val = 1.5
x_dual = DualNumber(x_val, 1.0)
result = f(x_dual)
print(f"f({x_val})  = {result.val:.6f}")
print(f"f'({x_val}) = {result.deriv:.6f}")

# Verify with numerical derivative
eps = 1e-7
def f_scalar(x): return (x**3 + 2*x) * math.sin(x)
num_grad = (f_scalar(x_val+eps) - f_scalar(x_val-eps)) / (2*eps)
print(f"Numerical:  {num_grad:.6f}")
print(f"Error:      {abs(result.deriv - num_grad):.2e}")
print("Autodiff PASSED ✓" if abs(result.deriv - num_grad) < 1e-6 else "FAILED ✗")
"""
    ),
]


def run():
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║     EXERCISES — Block 03: Calculus               ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m\n")
        for i, ex in enumerate(exercises, 1):
            diff_color = "\033[92m" if ex.difficulty == "Beginner" else \
                         "\033[93m" if ex.difficulty == "Intermediate" else "\033[91m"
            print(f"  {i}. {diff_color}[{ex.difficulty}]\033[0m {ex.title}")
        print("\n  \033[90m[0] Back\033[0m")
        choice = input("\n\033[96mSelect exercise: \033[0m").strip()
        if choice == "0":
            break
        try:
            ex = exercises[int(choice) - 1]
            _run_exercise(ex)
        except (ValueError, IndexError):
            pass


def _run_exercise(ex):
    diff_color = "\033[92m" if ex.difficulty == "Beginner" else \
                 "\033[93m" if ex.difficulty == "Intermediate" else "\033[91m"
    print(f"\n\033[95m━━━ {ex.title} ━━━\033[0m")
    print(f"  Difficulty: {diff_color}{ex.difficulty}\033[0m\n")
    print("\033[1mPROBLEM\033[0m")
    print(ex.description)
    while True:
        cmd = input("\n  [h]int  [c]ode  [r]un  [s]olution  [b]ack: ").strip().lower()
        if cmd == 'b':
            break
        elif cmd == 'h':
            print(f"\n\033[93mHINT\033[0m\n{ex.hint}")
        elif cmd == 'c':
            print(f"\n\033[94mSTARTER CODE\033[0m\n{ex.starter_code}")
        elif cmd == 's':
            print(f"\n\033[92mSOLUTION\033[0m\n{ex.solution_code}")
        elif cmd == 'r':
            print("\n\033[92mRunning...\033[0m")
            try:
                exec(compile(ex.solution_code, "<solution>", "exec"), {})
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")
