"""
Block 20: Model Evaluation Mathematics
Covers: Confusion matrix, ROC/AUC, PR curve, Calibration, AIC/BIC, CV, stat tests
"""
import numpy as np
import math
from collections import Counter


def run():
    topics = [
        ("Confusion Matrix & Metrics",      confusion_matrix_topic),
        ("ROC Curve and AUC",               roc_auc),
        ("Precision-Recall Curve",          precision_recall),
        ("Calibration & ECE",               calibration),
        ("AIC and BIC",                     aic_bic),
        ("Cross-Validation Analysis",       cross_validation),
        ("Statistical Tests for ML",        stat_tests),
        ("Multi-Model Radar Chart",         radar_chart),
    ]
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m  ║      BLOCK 20 — MODEL EVALUATION MATHEMATICS     ║\033[0m")
        print("\033[96m  ╚══════════════════════════════════════════════════╝\033[0m")
        print("\033[90mmmlmath > Block 20 > Model Evaluation\033[0m\n")
        for i, (name, _) in enumerate(topics, 1):
            print(f"  \033[93m{i:2d}.\033[0m {name}")
        print("\n  \033[90m[0] Back to main menu\033[0m")
        choice = input("\n\033[96mSelect topic: \033[0m").strip()
        if choice == "0":
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(topics):
                topics[idx][1]()
        except (ValueError, IndexError):
            print("\033[91mInvalid choice.\033[0m")


def confusion_matrix_topic():
    print("\n\033[95m━━━ Confusion Matrix & All Metrics ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
The confusion matrix is the foundation of all classification metrics.
It counts how predictions map to true labels:
  TP = predicted positive, actually positive
  FP = predicted positive, actually negative (Type I error)
  FN = predicted negative, actually positive (Type II error)
  TN = predicted negative, actually negative

Every other metric is a function of these four numbers.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Accuracy:     (TP+TN) / (TP+FP+FN+TN)
  Precision:    TP / (TP+FP)            ← of all predicted +, how many are +?
  Recall:       TP / (TP+FN)            ← of all actual +, how many found?
  Specificity:  TN / (TN+FP)
  F1:           2·P·R / (P+R)           ← harmonic mean of Precision & Recall
  Fβ:           (1+β²)·P·R / (β²P+R)   ← β>1 weights recall more
  MCC:          (TP·TN−FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  Cohen's κ:    (pₒ − pₑ) / (1 − pₑ)   [chance-corrected agreement]
  G-mean:       √(Recall · Specificity) ← good for imbalanced
""")

    # Example
    y_true = np.array([1,1,1,1,1,0,0,0,0,0,1,0,1,0,1])
    y_pred = np.array([1,1,0,1,0,0,0,1,0,0,1,0,0,1,1])

    TP = int(np.sum((y_pred==1)&(y_true==1)))
    FP = int(np.sum((y_pred==1)&(y_true==0)))
    FN = int(np.sum((y_pred==0)&(y_true==1)))
    TN = int(np.sum((y_pred==0)&(y_true==0)))

    acc = (TP+TN)/(TP+FP+FN+TN)
    prec = TP/(TP+FP) if (TP+FP)>0 else 0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0
    spec = TN/(TN+FP) if (TN+FP)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc  = (TP*TN - FP*FN)/denom if denom > 0 else 0

    print("\033[93mCONFUSION MATRIX\033[0m")
    print(f"""
          Predicted Pos  Predicted Neg
  Actual Pos   {TP:>4} (TP)      {FN:>4} (FN)
  Actual Neg   {FP:>4} (FP)      {TN:>4} (TN)
""")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  Specificity:{spec:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  MCC:        {mcc:.4f}")

    # ASCII bar chart of metrics
    print("\n\033[93mASCII Metric Bar Chart\033[0m")
    metrics = [("Accuracy",   acc), ("Precision", prec), ("Recall", rec),
               ("F1",         f1), ("MCC",       (mcc+1)/2)]  # normalize MCC to 0-1
    for name, val in metrics:
        bar = "█" * int(val * 30)
        print(f"  {name:<12} {bar:<30} {val:.3f}")

    # matplotlib heatmap
    try:
        import matplotlib.pyplot as plt2
        cm = np.array([[TP, FN], [FP, TN]])
        fig, ax = plt2.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred Pos','Pred Neg'])
        ax.set_yticklabels(['Actual Pos','Actual Neg'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                        fontsize=16, color='white' if cm[i,j] > cm.max()/2 else 'black')
        plt2.colorbar(im, ax=ax)
        ax.set_title('Confusion Matrix')
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for heatmap]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


def roc_auc():
    print("\n\033[95m━━━ ROC Curve and AUC ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
The ROC (Receiver Operating Characteristic) curve plots the trade-off
between True Positive Rate (TPR = Recall) and False Positive Rate (FPR)
as the classification threshold varies from 1 to 0.

AUC (Area Under Curve) is the probability that the model ranks a random
positive example higher than a random negative example.
AUC = 0.5 is random; AUC = 1.0 is perfect.

ROC is insensitive to class imbalance — Precision-Recall is better
for highly imbalanced datasets.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  TPR (Recall / Sensitivity) = TP / (TP + FN)
  FPR (1 - Specificity)      = FP / (FP + TN)

  AUC (trapezoidal rule):
    AUC ≈ Σᵢ (FPRᵢ₊₁ − FPRᵢ) · (TPRᵢ₊₁ + TPRᵢ) / 2

  Interpretation: P(f(x⁺) > f(x⁻)) = AUC
  where x⁺ is positive, x⁻ is negative example.
""")

    np.random.seed(7)
    n = 200
    y_true = np.random.binomial(1, 0.3, n)
    scores = y_true * np.random.beta(5, 2, n) + (1 - y_true) * np.random.beta(2, 5, n)

    thresholds = np.sort(scores)[::-1]
    tprs, fprs = [0.0], [0.0]
    P = y_true.sum(); N = n - P
    for t in thresholds:
        preds = (scores >= t).astype(int)
        TP = int(((preds==1)&(y_true==1)).sum())
        FP = int(((preds==1)&(y_true==0)).sum())
        FN = int(((preds==0)&(y_true==1)).sum())
        TN = int(((preds==0)&(y_true==0)).sum())
        tprs.append(TP/(TP+FN) if (TP+FN)>0 else 0)
        fprs.append(FP/(FP+TN) if (FP+TN)>0 else 0)
    tprs.append(1.0); fprs.append(1.0)
    auc = sum((fprs[i+1]-fprs[i])*(tprs[i+1]+tprs[i])/2 for i in range(len(fprs)-1))

    print(f"\n  Dataset: n={n}, P={P} positives, N={N} negatives")
    print(f"  AUC = {auc:.4f}")

    # ASCII ROC
    print("\n\033[93mASCII ROC Curve\033[0m")
    w, h = 50, 15
    grid = [[' '] * w for _ in range(h)]
    for i, (fpr, tpr) in enumerate(zip(fprs[::4], tprs[::4])):
        xi = int(fpr * (w-1))
        yi = h - 1 - int(tpr * (h-1))
        if 0 <= xi < w and 0 <= yi < h:
            grid[yi][xi] = '\033[96m*\033[0m'
    # diagonal
    for i in range(min(w, h)):
        xi = int(i / (min(w, h)-1) * (w-1))
        yi = h - 1 - int(i / (min(w, h)-1) * (h-1))
        if 0 <= xi < w and 0 <= yi < h and grid[yi][xi] == ' ':
            grid[yi][xi] = '\033[90m·\033[0m'
    print(f"  TPR\n  1.0 ┤")
    for row in grid:
        print("      │" + "".join(row))
    print(f"  0.0 └" + "─"*w + "> FPR")
    print(f"        0.0{'':>40}1.0")
    print(f"  AUC = {auc:.4f}  (random=0.5000)")

    # matplotlib
    try:
        import matplotlib.pyplot as plt2
        fig, ax = plt2.subplots(figsize=(6, 6))
        ax.plot(fprs, tprs, 'b-', lw=2, label=f'ROC (AUC={auc:.3f})')
        ax.plot([0,1],[0,1],'k--',label='Random (AUC=0.5)')
        ax.fill_between(fprs, tprs, alpha=0.1, color='blue')
        ax.set_xlabel('FPR (1-Specificity)'); ax.set_ylabel('TPR (Recall)')
        ax.set_title('ROC Curve'); ax.legend()
        ax.grid(True, alpha=0.3)
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for ROC plot]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • AUC = 1.0 perfect, 0.5 = random, <0.5 = worse than random
  • ROC insensitive to class imbalance: use PR curve instead
  • Optimal threshold: Youden's J = TPR − FPR (maximize)
  • Partial AUC: useful when only a specific FPR range matters
  • Multi-class ROC: one-vs-rest AUC for each class, then average
""")
    input("\033[90m[Enter to continue]\033[0m")


def precision_recall():
    print("\n\033[95m━━━ Precision-Recall Curve ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Precision-Recall curves are preferred over ROC when the dataset is
highly imbalanced (many more negatives than positives).

In imbalanced settings, a model can achieve high FPR without many
mistakes (since TN >> FP), making ROC look optimistic.
PR curves reveal the true performance on the minority class.

Average Precision (AP) = area under PR curve.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Precision = TP / (TP + FP)       ← quality of positive predictions
  Recall    = TP / (TP + FN)       ← coverage of actual positives

  PR curve: plot Precision vs Recall as threshold varies
  AP = Σᵢ (Rᵢ − Rᵢ₋₁) · Pᵢ         [area under PR curve]
  F1 = 2·P·R/(P+R)                  ← harmonic mean (maximized at threshold*)

  Imbalanced example:
    99% negative, 1% positive, predict ALL negative:
      ROC AUC ≈ 0.5 (bad)
      PR AUC  ≈ 0.01 (extremely bad — clearly seen!)
""")

    np.random.seed(3)
    n = 1000; pos_rate = 0.05  # 5% positive — imbalanced
    y_true = np.random.binomial(1, pos_rate, n)
    scores = y_true * np.random.beta(6, 2, n) + (1-y_true) * np.random.beta(2, 6, n)

    thresholds = np.sort(np.unique(scores))[::-1]
    precs, recs = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        TP = int(((preds==1)&(y_true==1)).sum())
        FP = int(((preds==1)&(y_true==0)).sum())
        FN = int(((preds==0)&(y_true==1)).sum())
        p = TP/(TP+FP) if (TP+FP)>0 else 1.0
        r = TP/(TP+FN) if (TP+FN)>0 else 0.0
        precs.append(p); recs.append(r)
    ap = sum((recs[i]-recs[i-1])*precs[i] for i in range(1, len(recs)) if recs[i]>recs[i-1])

    print(f"\n  n={n}, positives={y_true.sum()} ({100*y_true.mean():.1f}%)")
    print(f"  Average Precision (AP) = {ap:.4f}")

    # matplotlib
    try:
        import matplotlib.pyplot as plt2
        fig, axes = plt2.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(recs, precs, 'b-', lw=2)
        axes[0].axhline(y=y_true.mean(), color='r', linestyle='--', label='Random baseline')
        axes[0].fill_between(recs, precs, alpha=0.1, color='blue')
        axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
        axes[0].set_title(f'Precision-Recall Curve (AP={ap:.3f})')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        # PR vs ROC comparison
        axes[1].text(0.5, 0.7, f'Imbalanced dataset\n5% positive\n\nAP = {ap:.3f}',
                     ha='center', va='center', fontsize=14,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1].set_title('PR Curve preferred for imbalanced data')
        axes[1].axis('off')
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for PR plot]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


def calibration():
    print("\n\033[95m━━━ Calibration & ECE ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
A calibrated model has confidence scores that match empirical probabilities:
"When the model says 80% confident, it should be correct ~80% of the time."

Calibration is critical for:
• Medical diagnosis: 70% cancer risk means something specific
• Weather forecasting: "30% chance of rain" must be meaningful
• Risk management: financial models with probability outputs

Most modern deep learning models are OVERCONFIDENT (Guo et al. 2017).
Calibration methods: Temperature scaling, Platt scaling, isotonic regression.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Expected Calibration Error (ECE):
    ECE = Σₘ (|Bₘ|/n) · |acc(Bₘ) − conf(Bₘ)|

  where Bₘ = samples in bin m (e.g., 10 equal-width bins on [0,1])
  acc(Bₘ) = fraction correct in bin
  conf(Bₘ) = mean predicted confidence in bin

  Perfect calibration: reliability curve = diagonal y = x

  Temperature scaling:
    σ_T(z) = softmax(z / T)
    T > 1 → softer predictions (less confident, better calibrated)
    T < 1 → sharper predictions (more confident)
    Optimal T found by minimizing NLL on validation set.
""")

    np.random.seed(11)
    n = 1000
    # Simulate overconfident model
    true_probs = np.random.beta(2, 5, n)
    y_true = (np.random.rand(n) < true_probs).astype(int)
    pred_probs = np.clip(true_probs * 1.5 + 0.1, 0.01, 0.99)  # overconfident

    # Compute ECE
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (pred_probs >= bins[i]) & (pred_probs < bins[i+1])
        if mask.sum() > 0:
            bm_acc  = y_true[mask].mean()
            bm_conf = pred_probs[mask].mean()
            bm_size = mask.sum()
            ece += (bm_size / n) * abs(bm_acc - bm_conf)
            bin_data.append((bins[i], bins[i+1], bm_conf, bm_acc, bm_size))

    print(f"\n  Simulated overconfident model, n={n}")
    print(f"  ECE = {ece:.4f}  (0 = perfect, 0.1 = very poor)")
    print(f"\n  {'Bin':<14} {'Confidence':>12} {'Accuracy':>10} {'|diff|':>8} {'n':>6}")
    print("  " + "─" * 54)
    for lo, hi, conf, acc, sz in bin_data:
        diff = abs(conf - acc)
        bar = "█" if diff > 0.05 else "░"
        print(f"  [{lo:.1f},{hi:.1f})    {conf:>12.3f} {acc:>10.3f} {diff:>8.3f} {sz:>6} {bar}")

    # matplotlib reliability diagram
    try:
        import matplotlib.pyplot as plt2
        fig, axes = plt2.subplots(1, 2, figsize=(12, 5))
        confs = [b[2] for b in bin_data]
        accs  = [b[3] for b in bin_data]
        axes[0].plot([0,1],[0,1],'k--',label='Perfect calibration')
        axes[0].bar([b[0] for b in bin_data], accs,
                    width=0.1, alpha=0.7, align='edge', label='Actual accuracy', color='blue')
        axes[0].bar([b[0] for b in bin_data], confs,
                    width=0.1, alpha=0.4, align='edge', label='Confidence', color='red')
        axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Reliability Diagram (ECE={ece:.3f})')
        axes[0].legend()
        axes[1].hist(pred_probs[y_true==1], bins=20, alpha=0.6, color='blue', label='Positive')
        axes[1].hist(pred_probs[y_true==0], bins=20, alpha=0.6, color='red', label='Negative')
        axes[1].set_title('Predicted Probability Distribution')
        axes[1].set_xlabel('Predicted probability'); axes[1].legend()
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for calibration plot]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


def aic_bic():
    print("\n\033[95m━━━ AIC and BIC ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
are model selection criteria that balance goodness-of-fit against complexity.

Both penalize models with more parameters to prevent overfitting.
Lower AIC/BIC = better model.

AIC: derived from information theory (KL divergence to true distribution)
BIC: derived from Bayesian model comparison (Laplace approximation)
BIC penalizes model complexity more strongly for large datasets.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  AIC = −2·log L(θ̂) + 2k
  BIC = −2·log L(θ̂) + k·log n

  where: L(θ̂) = maximum likelihood, k = # parameters, n = # samples

  Derivation of AIC (Akaike):
    AIC ≈ −2 E[log P(x_new | θ̂)]  [expected log-likelihood on new data]
    KL(p_true || p_model) ≈ const − log L + k   [bias correction]

  ΔAIC = AICᵢ − AICₘᵢₙ
    ΔAIC < 2: substantial support for model i
    ΔAIC 4-7: considerably less support
    ΔAIC > 10: essentially no support

  BIC vs AIC:
    BIC is consistent: selects true model as n→∞
    AIC is efficient: minimizes prediction error (good for n < ∞)
""")

    np.random.seed(42)
    X = np.linspace(-3, 3, 80)
    y_true_func = 0.5 * X**2 + np.random.randn(80) * 0.5
    n = len(X)

    print("\n\033[93mMODEL COMPARISON — Polynomial Degree Selection\033[0m")
    print(f"  Data: y = 0.5x² + noise, n={n}")
    print(f"\n  {'Degree':<8} {'k':>4} {'Log L':>10} {'AIC':>10} {'BIC':>10}")
    print("  " + "─" * 46)
    for deg in range(1, 8):
        k = deg + 1  # degree + intercept
        coeffs = np.polyfit(X, y_true_func, deg)
        y_pred = np.polyval(coeffs, X)
        residuals = y_true_func - y_pred
        sigma2 = np.var(residuals)
        log_l = -n/2 * math.log(2*math.pi*sigma2) - 1/(2*sigma2) * np.sum(residuals**2)
        aic_val = -2 * log_l + 2 * k
        bic_val = -2 * log_l + k * math.log(n)
        marker = " ← BIC optimal" if deg == 2 else (" ← overfitting" if deg >= 6 else "")
        print(f"  {deg:<8} {k:>4} {log_l:>10.2f} {aic_val:>10.2f} {bic_val:>10.2f}{marker}")

    # plotext
    try:
        import plotext as plt
        plt.clf()
        degs = list(range(1, 8))
        aics, bics = [], []
        for deg in degs:
            k = deg + 1
            coeffs = np.polyfit(X, y_true_func, deg)
            y_pred = np.polyval(coeffs, X)
            residuals = y_true_func - y_pred
            sigma2 = np.var(residuals)
            log_l = -n/2*math.log(2*math.pi*sigma2) - 1/(2*sigma2)*np.sum(residuals**2)
            aics.append(-2*log_l + 2*k)
            bics.append(-2*log_l + k*math.log(n))
        plt.plot(degs, aics, label="AIC")
        plt.plot(degs, bics, label="BIC")
        plt.title("AIC and BIC vs Polynomial Degree")
        plt.xlabel("Degree"); plt.ylabel("Criterion")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for AIC/BIC plot]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • AIC minimization = choosing model closest to truth in KL sense
  • BIC implements Occam's Razor more aggressively than AIC
  • Neither handles regularized/Bayesian models well (k ill-defined)
  • MDL (Minimum Description Length) is the information-theoretic view
  • For neural nets: use validation loss / cross-validation instead
""")
    input("\033[90m[Enter to continue]\033[0m")


def cross_validation():
    print("\n\033[95m━━━ Cross-Validation Analysis ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Cross-validation (CV) estimates the true generalization error of a model
by partitioning data into training and validation subsets multiple times.

k-fold CV: split data into k equal folds, train on k-1, test on remaining,
repeat k times. Use average score as generalization estimate.

LOOCV (Leave-One-Out): special case k=n. Unbiased but high variance.
Stratified k-fold: maintains class proportions in each fold.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  k-fold CV error:   CV_k = (1/k) Σᵢ Err(f^{-i}, Dᵢ)
  where f^{-i} = model trained without fold i, Dᵢ = fold i

  Bias-Variance of CV:
    As k increases: bias ↓ (training closer to full dataset)
    As k increases: variance ↑ (smaller test folds, more correlated)
    k=5 or k=10 are typically optimal

  Variance formula:
    Var(CV_k) = Var(εᵢ)/k + Cov(εᵢ,εⱼ)·(k-1)/k
    ≈ σ²/k + ρσ²(k-1)/k  [ρ = correlation between fold errors]

  Nested CV: outer loop for model selection, inner loop for HPO
    Avoids data leakage from hyperparameter tuning
""")

    np.random.seed(42)
    n, d = 100, 5
    X = np.random.randn(n, d)
    w_true = np.array([1.5, -1.0, 0.5, 0.0, 0.0])
    y = X @ w_true + 0.5 * np.random.randn(n)

    k = 5
    fold_size = n // k
    cv_errors = []
    for fold in range(k):
        val_idx = list(range(fold * fold_size, (fold+1) * fold_size))
        train_idx = [i for i in range(n) if i not in val_idx]
        Xtr = X[train_idx]; ytr = y[train_idx]
        Xval = X[val_idx];   yval = y[val_idx]
        # OLS
        w = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        y_pred = Xval @ w
        mse = np.mean((yval - y_pred) ** 2)
        cv_errors.append(mse)

    print(f"\n  Linear Regression, n={n}, d={d}, {k}-fold CV")
    print(f"\n  {'Fold':<6} {'MSE':>8}  {'Bar'}")
    print("  " + "─" * 30)
    for i, e in enumerate(cv_errors):
        bar = "█" * int(e * 10)
        print(f"  {i+1:<6} {e:>8.4f}  {bar}")
    print("  " + "─" * 30)
    print(f"  CV MSE: {np.mean(cv_errors):.4f} ± {np.std(cv_errors):.4f}")

    # plotext
    try:
        import plotext as plt
        plt.clf()
        plt.bar(list(range(1, k+1)), cv_errors)
        plt.title(f"{k}-Fold CV Errors")
        plt.xlabel("Fold"); plt.ylabel("MSE")
        plt.show()
    except ImportError:
        print("  \033[90m[Install plotext for CV plot]\033[0m")

    input("\033[90m[Enter to continue]\033[0m")


def stat_tests():
    print("\n\033[95m━━━ Statistical Tests for ML ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
When comparing two ML models, we need to know if observed performance
differences are statistically significant or just random variation.

McNemar's test: proper test for comparing two classifiers on same test set.
Wilcoxon signed-rank: non-parametric test for two models across multiple datasets.
t-test: assumes normality; often used in practice despite violation.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  McNemar's Test (compare two classifiers A and B):
  Contingency table on n test samples:
         B correct  B wrong
  A correct  Nc,c       Nc,w
  A wrong    Nw,c       Nw,w

  χ² = (|Nc,w − Nw,c| − 1)² / (Nc,w + Nw,c)
  p-value from χ²(df=1) distribution
  Null: both classifiers have same error rate

  Permutation test:
  Shuffle class labels many times, compute test statistic.
  p = (# permutations with stat ≥ observed) / total permutations

  Bonferroni correction (multiple comparisons):
  α_adjusted = α / m  where m = number of tests
  Controls family-wise error rate (FWER)

  FDR (Benjamini-Hochberg):
  Sort p-values; reject all pᵢ ≤ (i/m)·α
  Controls false discovery rate (softer than Bonferroni)
""")

    np.random.seed(0)
    n_test = 200
    # Model A: 85% accuracy, Model B: 82% accuracy
    y_true = np.random.binomial(1, 0.5, n_test)
    correct_a = np.random.binomial(1, 0.85, n_test)
    correct_b = np.random.binomial(1, 0.82, n_test)
    Ncc = np.sum(correct_a & correct_b)
    Ncw = np.sum(correct_a & ~correct_b)
    Nwc = np.sum(~correct_a & correct_b)
    Nww = np.sum(~correct_a & ~correct_b)

    chi2 = (abs(Ncw - Nwc) - 1) ** 2 / (Ncw + Nwc + 1e-10)

    print(f"\n  Test set size: {n_test}")
    print(f"  Model A accuracy: {correct_a.mean():.4f}")
    print(f"  Model B accuracy: {correct_b.mean():.4f}")
    print(f"\n  McNemar contingency table:")
    print(f"               B correct  B wrong")
    print(f"  A correct       {Ncc:>4}      {Ncw:>4}")
    print(f"  A wrong         {Nwc:>4}      {Nww:>4}")
    print(f"\n  χ² = {chi2:.4f}")
    print(f"  Critical value at α=0.05: 3.841")
    print(f"  Significant: {chi2 > 3.841}")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • McNemar is specifically designed for paired classifier comparison
  • Never use 5×2 CV without proper statistical correction
  • p-values: report exact values, not just "significant/not significant"
  • Multiple testing: comparing 20 models at α=0.05 expects 1 false positive
  • Effect size (Cohen's d) matters as much as statistical significance
""")
    input("\033[90m[Enter to continue]\033[0m")


def radar_chart():
    print("\n\033[95m━━━ Multi-Model Radar/Spider Chart ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
A radar chart (spider chart) visualizes multiple metrics simultaneously
for multiple models. It's useful for multi-objective model comparison
where no single metric tells the whole story.

Different stakeholders may weight metrics differently:
• Medical: prioritize Recall (sensitivity)
• Fraud detection: prioritize Precision
• Recommendation: balance Precision, Recall, Coverage
""")

    models = {
        "Model A (Random Forest)": {"Accuracy": 0.87, "Precision": 0.85, "Recall": 0.82,
                                     "F1": 0.83, "AUC": 0.91, "Speed": 0.60},
        "Model B (Logistic Reg)":  {"Accuracy": 0.82, "Precision": 0.81, "Recall": 0.80,
                                     "F1": 0.80, "AUC": 0.88, "Speed": 0.95},
        "Model C (Neural Net)":    {"Accuracy": 0.91, "Precision": 0.90, "Recall": 0.89,
                                     "F1": 0.89, "AUC": 0.96, "Speed": 0.40},
    }
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Speed"]

    print("\n\033[93mMETRIC COMPARISON TABLE\033[0m")
    print("  " + "─" * 80)
    print(f"  {'Metric':<12} " + " ".join(f"{m[:16]:<18}" for m in models))
    print("  " + "─" * 80)
    for met in metrics:
        vals = [models[m][met] for m in models]
        bars = []
        for v in vals:
            bar = "█" * int(v * 12)
            bars.append(f"{bar:<12} {v:.2f}")
        print(f"  {met:<12} " + "  ".join(bars))
    print("  " + "─" * 80)

    # matplotlib radar chart
    try:
        import matplotlib.pyplot as plt2
        import matplotlib.patches as mpatches
        fig, ax = plt2.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        n_met = len(metrics)
        angles = [i * 2 * math.pi / n_met for i in range(n_met)]
        angles += angles[:1]
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        colors = ['blue', 'red', 'green']
        for (name, met_dict), color in zip(models.items(), colors):
            values = [met_dict[m] for m in metrics] + [met_dict[metrics[0]]]
            ax.plot(angles, values, 'o-', color=color, linewidth=2, label=name)
            ax.fill(angles, values, color=color, alpha=0.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title("Multi-Model Comparison Radar Chart", size=14, pad=20)
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib for radar chart]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • No single model wins on all metrics — always involves tradeoffs
  • Pareto frontier: set of models not dominated on all metrics
  • Stakeholder alignment: define metric weights BEFORE model selection
  • Time/memory costs are real metrics — "Speed" and complexity matter
  • Report confidence intervals, not point estimates, for honest comparison
""")
    input("\033[90m[Enter to continue]\033[0m")
