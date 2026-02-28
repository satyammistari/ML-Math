"""
blocks/__init__.py — registry of all curriculum blocks
"""
from blocks.b01_linear_algebra       import run as b01
from blocks.b02_matrix_decomp        import run as b02
from blocks.b03_calculus             import run as b03
from blocks.b04_optimization         import run as b04
from blocks.b05_probability          import run as b05
from blocks.b06_statistics           import run as b06
from blocks.b07_information_theory   import run as b07
from blocks.b08_backprop             import run as b08
from blocks.b09_activation_functions import run as b09
from blocks.b10_supervised_learning  import run as b10
from blocks.b11_bias_variance        import run as b11
from blocks.b12_unsupervised_learning import run as b12
from blocks.b13_em_algorithm         import run as b13
from blocks.b14_probabilistic_ml     import run as b14
from blocks.b15_rl_math              import run as b15
from blocks.b16_mdp_solver           import run as b16
from blocks.b17_deep_learning        import run as b17
from blocks.b18_nlp_math             import run as b18
from blocks.b19_kernel_methods       import run as b19
from blocks.b20_model_evaluation     import run as b20

BLOCKS = [
    ("Linear Algebra",           b01,  "Vectors, matrices, eigenvalues, SVD, PCA"),
    ("Matrix Decompositions",    b02,  "LU, QR, Cholesky, condition number, Ax=b"),
    ("Calculus",                 b03,  "Gradients, Jacobian, Hessian, chain rule"),
    ("Optimization",             b04,  "GD, Adam, SGD, convexity, Newton's method"),
    ("Probability",              b05,  "Distributions, Bayes, CLT, conjugate priors"),
    ("Statistics",               b06,  "MLE, MAP, Bayesian inference, testing"),
    ("Information Theory",       b07,  "Entropy, KL, cross-entropy, mutual info"),
    ("Backpropagation",          b08,  "Computation graphs, chain rule, grad check"),
    ("Activation Functions",     b09,  "Sigmoid, ReLU, GELU, Softmax, comparisons"),
    ("Supervised Learning",      b10,  "LinReg, LogReg, SVM, trees, kNN"),
    ("Bias-Variance",            b11,  "Decomposition, regularization, CV, overfit"),
    ("Unsupervised Learning",    b12,  "K-means, GMM, DBSCAN, PCA, t-SNE"),
    ("EM Algorithm",             b13,  "ELBO, Jensen, E-step/M-step, convergence"),
    ("Probabilistic ML",         b14,  "GP, Bayesian regression, MCMC, VI"),
    ("Reinforcement Learning",   b15,  "MDP, Bellman, TD, policy gradient"),
    ("MDP Solver",               b16,  "GridWorld, value/policy iteration, Q-table"),
    ("Deep Learning",            b17,  "UAT, weight init, BN, attention, transformers"),
    ("NLP Mathematics",          b18,  "N-grams, TF-IDF, Word2Vec, BERT, BPE"),
    ("Kernel Methods",           b19,  "Mercer, RBF, kernel SVM, GP, kernel PCA"),
    ("Model Evaluation",         b20,  "Confusion, ROC, PR, calibration, AIC/BIC"),
]
