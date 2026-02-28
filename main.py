#!/usr/bin/env python3
"""
mlmath — Terminal-based ML Mathematics Learning Tool
Run: python main.py
"""
import sys
import os

# Ensure mlmath can find its own modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


ASCII_TITLE = r"""
\033[96m
  ███╗   ███╗██╗     ███╗   ███╗ █████╗ ████████╗██╗  ██╗
  ████╗ ████║██║     ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║
  ██╔████╔██║██║     ██╔████╔██║███████║   ██║   ███████║
  ██║╚██╔╝██║██║     ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║
  ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║
  ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
\033[0m"""

SUBTITLE = "\033[93m  Mathematics for Machine Learning — Interactive Terminal Course\033[0m"

BLOCK_META = [
    ("Linear Algebra",            "Vectors, matrices, spaces, SVD"),
    ("Matrix Decompositions",     "LU, QR, Cholesky, Eigendecomp"),
    ("Calculus & Autodiff",       "Gradients, Jacobians, Taylor"),
    ("Optimization",              "GD, Newton, Lagrange, convexity"),
    ("Probability Theory",        "Distributions, Bayes, MLE"),
    ("Statistics",                "Estimators, CI, hypothesis tests"),
    ("Information Theory",        "Entropy, KL divergence, MI"),
    ("Backpropagation",           "Computational graphs, chain rule"),
    ("Activation Functions",      "ReLU, Sigmoid, GELU, Swish"),
    ("Supervised Learning",       "Regression, classification, SVM"),
    ("Bias-Variance Tradeoff",    "Regularization, cross-validation"),
    ("Unsupervised Learning",     "K-means, PCA, DBSCAN"),
    ("EM Algorithm",              "Latent variables, GMM"),
    ("Probabilistic ML",          "Bayesian inference, GPs"),
    ("RL Mathematics",            "MDPs, Bellman, policy gradient"),
    ("MDP Solvers",               "Value/policy iteration, Q-learning"),
    ("Deep Learning Math",        "Attention, BN, transformers"),
    ("NLP Mathematics",           "Embeddings, LMs, BERT"),
    ("Kernel Methods",            "SVM, Mercer, kernel PCA, GPs"),
    ("Model Evaluation",          "ROC, calibration, stat tests"),
]

EXERCISE_MAP = {1: "ex01", 3: "ex03", 5: "ex05", 8: "ex08",
                10: "ex10", 12: "ex12", 15: "ex15", 17: "ex17"}

progress = {}  # block_num: bool  (loaded from progress file)
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), ".mlmath_progress")


def load_progress():
    global progress
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                for line in f: progress[int(line.strip())] = True
        except Exception:
            pass


def save_progress():
    with open(PROGRESS_FILE, "w") as f:
        for k in progress: f.write(f"{k}\n")


def progress_bar(completed, total=20, width=20):
    filled = int(width * completed / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * completed / total
    return f"\033[92m[{bar}]\033[0m {completed}/{total} ({pct:.0f}%)"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print(ASCII_TITLE.replace("\\033", "\033"))
    print(SUBTITLE)
    completed = len(progress)
    print(f"\n  Progress: {progress_bar(completed)}\n")


def print_main_menu():
    print("\033[96m  ╔══════════════════════════════╗\033[0m")
    print("\033[96m  ║         MAIN MENU            ║\033[0m")
    print("\033[96m  ╚══════════════════════════════╝\033[0m\n")
    print("  \033[93m[1]\033[0m  Browse Curriculum Blocks")
    print("  \033[93m[2]\033[0m  Practice Exercises")
    print("  \033[93m[3]\033[0m  Search Topics")
    print("  \033[93m[4]\033[0m  Mark Block Completed")
    print("  \033[93m[5]\033[0m  Reset Progress")
    print("  \033[93m[q]\033[0m  Quit\n")


def print_blocks_menu():
    print("\n\033[96m  ╔══════════════════════════════════════════════════════════════╗\033[0m")
    print("\033[96m  ║                    CURRICULUM BLOCKS                         ║\033[0m")
    print("\033[96m  ╚══════════════════════════════════════════════════════════════╝\033[0m\n")
    for i, (name, desc) in enumerate(BLOCK_META, 1):
        done = "\033[92m✓\033[0m" if i in progress else " "
        num = f"\033[93m{i:2d}\033[0m"
        print(f"  {done} {num}. \033[95m{name:<28}\033[0m  \033[90m{desc}\033[0m")
    print("\n  \033[90m[0] Back\033[0m\n")


def run_block(num):
    """Dynamically import and run a curriculum block."""
    module_name = f"blocks.b{num:02d}_{BLOCK_META[num-1][0].lower().replace(' ','_').replace('-','_').replace('/','_').replace('&','').replace(',','')}"
    # Map correct module names
    block_modules = [
        "b01_linear_algebra", "b02_matrix_decomp", "b03_calculus",
        "b04_optimization", "b05_probability", "b06_statistics",
        "b07_information_theory", "b08_backprop", "b09_activation_functions",
        "b10_supervised_learning", "b11_bias_variance", "b12_unsupervised_learning",
        "b13_em_algorithm", "b14_probabilistic_ml", "b15_rl_math",
        "b16_mdp_solver", "b17_deep_learning", "b18_nlp_math",
        "b19_kernel_methods", "b20_model_evaluation",
    ]
    mod_name = "blocks." + block_modules[num - 1]
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        name, desc = BLOCK_META[num - 1]
        print(f"\n\033[95m{'═'*60}\033[0m")
        print(f"\033[95m  Block {num:02d}: {name}\033[0m")
        print(f"\033[90m  {desc}\033[0m")
        print(f"\033[95m{'═'*60}\033[0m")
        mod.run()
        # Offer to mark complete
        mark = input("\n\033[92mMark this block as completed? [y/n]: \033[0m").strip().lower()
        if mark == 'y':
            progress[num] = True
            save_progress()
            print("\033[92m  ✓ Marked complete!\033[0m")
    except ModuleNotFoundError as e:
        print(f"\033[91m  Error loading block: {e}\033[0m")
    except KeyboardInterrupt:
        print("\n\033[90m  (interrupted)\033[0m")


def run_exercises_menu():
    print("\n\033[96m  ╔══════════════════════════════════════════════════╗\033[0m")
    print("\033[96m  ║               PRACTICE EXERCISES                 ║\033[0m")
    print("\033[96m  ╠══════════════════════════════════════════════════╣\033[0m")
    ex_list = [
        (1,  "Linear Algebra",     "ex01_linear_algebra"),
        (3,  "Calculus",           "ex03_calculus"),
        (5,  "Probability",        "ex05_probability"),
        (8,  "Backpropagation",    "ex08_backprop"),
        (10, "Supervised Learning","ex10_supervised"),
        (12, "Unsupervised",       "ex12_unsupervised"),
        (15, "Reinforcement Learning","ex15_rl"),
        (17, "Deep Learning",      "ex17_deep_learning"),
    ]
    for i, (blk, name, mod) in enumerate(ex_list, 1):
        print(f"  \033[93m[{i}]\033[0m Block {blk:02d}: \033[95m{name}\033[0m")
    print("\n  \033[90m[0] Back\033[0m\n")
    while True:
        choice = input("  \033[96mSelect exercise set: \033[0m").strip()
        if choice == '0':
            break
        try:
            blk, name, mod_name = ex_list[int(choice) - 1]
            import importlib
            mod = importlib.import_module("exercises." + mod_name)
            mod.run()
            break
        except (ValueError, IndexError):
            print("\033[91m  Invalid choice\033[0m")
        except ModuleNotFoundError as e:
            print(f"\033[91m  Exercise module not found: {e}\033[0m")
            break
        except KeyboardInterrupt:
            break


def search_topics():
    print("\n\033[96m  ╔══════════════════════════════════════════════════╗\033[0m")
    print("\033[96m  ║               SEARCH TOPICS                      ║\033[0m")
    print("\033[96m  ╚══════════════════════════════════════════════════╝\033[0m\n")
    query = input("  \033[93mSearch for: \033[0m").strip().lower()
    if not query:
        return
    results = []
    for i, (name, desc) in enumerate(BLOCK_META, 1):
        if query in name.lower() or query in desc.lower():
            results.append((i, name, desc))
    if results:
        print(f"\n  Found {len(results)} block(s) matching '\033[93m{query}\033[0m':\n")
        for num, name, desc in results:
            done = "\033[92m✓\033[0m" if num in progress else " "
            print(f"    {done} Block {num:02d}: \033[95m{name}\033[0m — \033[90m{desc}\033[0m")
        # Offer to jump to first result
        if len(results) == 1:
            go = input(f"\n  Open Block {results[0][0]}? [y/n]: ").strip().lower()
            if go == 'y':
                run_block(results[0][0])
    else:
        print(f"\n  \033[91mNo blocks found for '{query}'\033[0m")
    input("\n  \033[90m[Press Enter to continue]\033[0m")


def mark_complete_menu():
    num_str = input("  Enter block number to mark complete (1-20): ").strip()
    try:
        n = int(num_str)
        if 1 <= n <= 20:
            progress[n] = True
            save_progress()
            print(f"\033[92m  ✓ Block {n:02d} ({BLOCK_META[n-1][0]}) marked complete!\033[0m")
        else:
            print("\033[91m  Invalid block number\033[0m")
    except ValueError:
        print("\033[91m  Invalid input\033[0m")
    input("\n  \033[90m[Enter to continue]\033[0m")


def reset_progress():
    confirm = input("  \033[91mReset all progress? [yes/no]: \033[0m").strip().lower()
    if confirm == 'yes':
        progress.clear()
        save_progress()
        print("\033[92m  Progress reset!\033[0m")
    input("\n  \033[90m[Enter to continue]\033[0m")


def blocks_browser():
    while True:
        clear()
        print_header()
        print_blocks_menu()
        choice = input("  \033[96mEnter block number (or 0 to go back): \033[0m").strip()
        if choice == '0':
            break
        try:
            n = int(choice)
            if 1 <= n <= 20:
                run_block(n)
            else:
                print("\033[91m  Enter 1–20\033[0m")
                input()
        except ValueError:
            pass
        except KeyboardInterrupt:
            break


def main():
    load_progress()
    print("\033[?25h", end="")  # show cursor
    try:
        while True:
            clear()
            print_header()
            print_main_menu()
            try:
                choice = input("  \033[96mChoose: \033[0m").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n\033[96m  Thanks for learning with mlmath! Goodbye.\033[0m\n")
                break

            if choice == 'q' or choice == 'quit':
                print("\n\033[96m  Thanks for learning with mlmath! Goodbye.\033[0m\n")
                break
            elif choice == '1':
                blocks_browser()
            elif choice == '2':
                run_exercises_menu()
            elif choice == '3':
                search_topics()
            elif choice == '4':
                mark_complete_menu()
            elif choice == '5':
                reset_progress()
            else:
                pass  # invalid input — just redraw
    except KeyboardInterrupt:
        print("\n\n\033[96m  Interrupted. Goodbye!\033[0m\n")


if __name__ == "__main__":
    main()
