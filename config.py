"""
config.py — Shared constants, color helpers, and configuration for mlmath.
"""

import os
import sys

# ── Terminal geometry ──────────────────────────────────────────────────────────
def get_term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80

TERM_WIDTH = get_term_width()

# ── Version / metadata ─────────────────────────────────────────────────────────
VERSION    = "1.0.0"
APP_NAME   = "mlmath"
APP_TITLE  = "MLMATH — Mathematics for Machine Learning"

# ── Unicode support detection ──────────────────────────────────────────────────
def _supports_unicode() -> bool:
    try:
        "▁▂▃▄▅▆▇█┌┬┐".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError, AttributeError):
        return False

UNICODE_OK = _supports_unicode()

# ── Sparkline chars ────────────────────────────────────────────────────────────
SPARKS    = "▁▂▃▄▅▆▇█" if UNICODE_OK else "_.-:=|#@"
BAR_FULL  = "█" if UNICODE_OK else "#"
BAR_EMPTY = "░" if UNICODE_OK else "."

# ── Box-drawing charset (fallback to ASCII) ────────────────────────────────────
if UNICODE_OK:
    BOX = dict(tl="┌", tm="┬", tr="┐", ml="├", mm="┼", mr="┤",
               bl="└", bm="┴", br="┘", h="─", v="│", arr_r="→",
               arr_l="←", arr_u="↑", arr_d="↓")
else:
    BOX = dict(tl="+", tm="+", tr="+", ml="+", mm="+", mr="+",
               bl="+", bm="+", br="+", h="-", v="|", arr_r=">",
               arr_l="<", arr_u="^", arr_d="v")

# ── ANSI colour codes ──────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
ITALIC  = "\033[3m"
UNDER   = "\033[4m"

FG = dict(
    black   = "\033[30m",
    red     = "\033[31m",
    green   = "\033[32m",
    yellow  = "\033[33m",
    blue    = "\033[34m",
    magenta = "\033[35m",
    cyan    = "\033[36m",
    white   = "\033[37m",
    grey    = "\033[90m",
    bred    = "\033[91m",
    bgreen  = "\033[92m",
    byellow = "\033[93m",
    bblue   = "\033[94m",
    bmagenta= "\033[95m",
    bcyan   = "\033[96m",
    bwhite  = "\033[97m",
)

BG = dict(
    black   = "\033[40m",
    red     = "\033[41m",
    green   = "\033[42m",
    yellow  = "\033[43m",
    blue    = "\033[44m",
    magenta = "\033[45m",
    cyan    = "\033[46m",
    white   = "\033[47m",
)

# Semantic aliases
C_HEADER  = FG["cyan"]   + BOLD
C_FORMULA = FG["yellow"]
C_SUCCESS = FG["green"]
C_ERROR   = FG["red"]
C_WARN    = FG["bred"]
C_BLOCK   = FG["magenta"] + BOLD
C_BODY    = FG["white"]
C_HINT    = FG["grey"]
C_CODE    = FG["blue"]
C_VALUE   = FG["byellow"]
C_SECTION = FG["cyan"]
C_EMPH    = BOLD + FG["bwhite"]

# ── Block registry ─────────────────────────────────────────────────────────────
BLOCKS = [
    ("b01", "Linear Algebra Fundamentals",         "Vectors, matrices, eigenvalues, SVD, PCA"),
    ("b02", "Matrix Decompositions",               "LU, QR, Cholesky, condition number"),
    ("b03", "Calculus & Differentiation",          "Gradients, Jacobian, Hessian, chain rule"),
    ("b04", "Optimisation",                        "GD, Adam, convexity, Newton's method"),
    ("b05", "Probability Theory",                  "Distributions, Bayes, CLT, conjugate priors"),
    ("b06", "Statistics",                          "MLE, MAP, hypothesis testing, confidence intervals"),
    ("b07", "Information Theory",                  "Entropy, KL, cross-entropy, mutual information"),
    ("b08", "Backpropagation",                     "Computation graphs, chain rule, vanishing grad"),
    ("b09", "Activation Functions",                "Sigmoid, ReLU, GELU, Softmax, comparison"),
    ("b10", "Supervised Learning",                 "Linear/logistic regression, SVM, trees, kNN"),
    ("b11", "Bias-Variance Trade-off",             "Decomposition, overfitting, regularisation"),
    ("b12", "Unsupervised Learning",               "K-Means, GMM, DBSCAN, t-SNE, autoencoders"),
    ("b13", "EM Algorithm",                        "ELBO, Jensen, E-step/M-step, HMM"),
    ("b14", "Probabilistic ML",                    "Gaussian processes, VI, MCMC, normalising flows"),
    ("b15", "Reinforcement Learning Math",         "MDP, Bellman, policy gradient, actor-critic"),
    ("b16", "MDP Solver",                          "GridWorld, value/policy iteration, Q-learning"),
    ("b17", "Deep Learning Theory",                "UAT, init, BatchNorm, attention, transformers"),
    ("b18", "NLP Mathematics",                     "n-grams, TF-IDF, Word2Vec, BPE, BERT"),
    ("b19", "Kernel Methods",                      "Mercer, RBF, Gram matrix, kernel SVM, GP"),
    ("b20", "Model Evaluation",                    "Confusion matrix, ROC, calibration, AIC/BIC"),
]

# ── Progress file location ─────────────────────────────────────────────────────
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), ".mlmath_progress.json")
