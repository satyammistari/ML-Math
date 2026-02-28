# mlmath — Mathematics for Machine Learning, Interactively

A terminal-based interactive learning tool covering the complete mathematics of machine learning,
inspired by Marc Peter Deisenroth's *"Mathematics for Machine Learning"* (Cambridge University Press, 2020).

---

## Features

- **20 curriculum blocks** — theory, formulas, derivations, worked examples, and code
- **3-layer visualizations** — ASCII (always), plotext terminal plots, matplotlib windows
- **Interactive exercises** — beginner / intermediate / advanced, with hints and solutions
- **Progress tracker** — marks blocks as completed, shown as a progress bar
- **Topic search** — find any block by keyword

---

## Quick Start
<img width="1595" height="862" alt="image" src="https://github.com/user-attachments/assets/ebe7855f-c78d-40c5-bbb8-187fb9853de1" />

<img width="1905" height="1033" alt="image" src="https://github.com/user-attachments/assets/206365c9-0214-4ede-a411-dc70606ee32a" />

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
cd mlmath
python main.py
```

---

## Requirements

| Package       | Purpose                             |
|---------------|-------------------------------------|
| numpy         | Core numerical computing            |
| scipy         | Special functions, statistics       |
| sympy         | Symbolic math, LaTeX rendering      |
| scikit-learn  | Reference implementations & data    |
| rich          | Terminal UI (tables, markdown)      |
| plotext       | Terminal-native plots (Layer 2)     |
| matplotlib    | GUI plots in separate window (L3)   |

---

## Curriculum Overview

| #  | Block                   | Topics                                              |
|----|-------------------------|-----------------------------------------------------|
| 01 | Linear Algebra          | Vectors, matrices, norms, SVD, PCA                  |
| 02 | Matrix Decompositions   | LU, QR, Cholesky, eigendecomposition                |
| 03 | Calculus & Autodiff     | Gradients, Jacobians, Taylor, dual numbers          |
| 04 | Optimization            | GD, Newton, Lagrange multipliers, convexity         |
| 05 | Probability Theory      | Distributions, Bayes, MLE, conjugate priors         |
| 06 | Statistics              | Estimators, CI, hypothesis tests, p-values          |
| 07 | Information Theory      | Entropy, KL divergence, mutual information          |
| 08 | Backpropagation         | Computational graphs, chain rule, autograd          |
| 09 | Activation Functions    | ReLU, Sigmoid, GELU, Swish, Mish                    |
| 10 | Supervised Learning     | Linear/logistic regression, SVM, decision trees    |
| 11 | Bias-Variance Tradeoff  | Decomposition, regularization, cross-validation     |
| 12 | Unsupervised Learning   | K-means, PCA, DBSCAN, autoencoders                  |
| 13 | EM Algorithm            | Latent variables, E/M steps, GMM                    |
| 14 | Probabilistic ML        | Bayesian inference, GPs, variational inference      |
| 15 | RL Mathematics          | MDPs, Bellman equations, Q-learning, policy gradient|
| 16 | MDP Solvers             | Value/policy iteration, dynamic programming         |
| 17 | Deep Learning Math      | Attention, batch norm, transformers, residuals      |
| 18 | NLP Mathematics         | TF-IDF, word2vec, BERT, BPE, BLEU                   |
| 19 | Kernel Methods          | Mercer's theorem, SVM dual, kernel PCA, GPs         |
| 20 | Model Evaluation        | ROC/AUC, calibration, AIC/BIC, stat tests           |

---

## Project Structure

```
mlmath/
├── main.py                # Entry point — run this
├── config.py              # Colors, themes, settings
├── requirements.txt
│
├── blocks/                # 20 curriculum blocks
│   ├── __init__.py        # BLOCKS registry
│   ├── b01_linear_algebra.py
│   ├── b02_matrix_decomp.py
│   │   ...
│   └── b20_model_evaluation.py
│
├── exercises/             # Interactive exercises
│   ├── __init__.py        # all_exercises dict
│   ├── ex01_linear_algebra.py
│   ├── ex03_calculus.py
│   ├── ex05_probability.py
│   ├── ex08_backprop.py
│   ├── ex10_supervised.py
│   ├── ex12_unsupervised.py
│   ├── ex15_rl.py
│   └── ex17_deep_learning.py
│
├── core/                  # Math utility library
│   ├── linalg.py
│   ├── calculus.py
│   ├── optimization.py
│   ├── probability.py
│   └── stats.py
│
├── ui/                    # Terminal UI components
│   ├── colors.py
│   ├── widgets.py
│   └── menu.py
│
└── viz/                   # 3-layer visualization
    ├── ascii_plots.py     # Layer 1: always available
    ├── terminal_plots.py  # Layer 2: plotext
    └── matplotlib_plots.py # Layer 3: GUI window
```

---

## Navigation

```
Main Menu
  [1] Browse Curriculum Blocks  →  select 1-20 → full topic menu → topics
  [2] Practice Exercises        →  pick exercise set → attempt / hint / run
  [3] Search Topics             →  keyword search across all 20 blocks
  [4] Mark Block Complete       →  update progress bar
  [q] Quit
```

Within each block:
- Topics listed as a numbered menu — pick any topic for theory + example + viz
- `[0]` goes back to the previous menu

Within each exercise:
- `[h]` show hint
- `[c]` show starter code
- `[s]` reveal full solution
- `[r]` run solution and print output
- `[b]` go back

---

## Adding a New Block

1. Create `blocks/b21_your_topic.py` with a `run()` function:
```python
def run():
    topics = [("Topic Name", topic_fn), ...]
    while True:
        # print menu
        # dispatch by number
```

2. Register in `blocks/__init__.py`:
```python
from blocks.b21_your_topic import run as b21
BLOCKS.append(("Your Topic", b21, "Short description"))
```

3. Add to `main.py`'s `BLOCK_META` list and `block_modules` list.

---

## Adding a New Exercise Set

1. Create `exercises/exXX_topic.py` following the `Exercise` class pattern:
```python
class Exercise:
    def __init__(self, title, difficulty, description, hint, starter_code, solution_code): ...

exercises = [Exercise(...), Exercise(...), Exercise(...)]  # 3 per set

def run(): ...
def _run_exercise(ex): ...
```

2. Register in `exercises/__init__.py`:
```python
from exercises.exXX_topic import run as exXX
all_exercises["XX"] = exXX
```

---

## Reference

- Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* (2020)
  https://mml-book.github.io/ (free PDF)
- Course website: https://mml-book.com

---

## License

MIT License. Educational use encouraged.
