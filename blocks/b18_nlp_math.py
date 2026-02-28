"""
Block 18: NLP Mathematics
Covers: N-grams, TF-IDF, Word2Vec, GloVe, BPE, Attention, BERT, perplexity, metrics
"""
import numpy as np
import math
from collections import Counter, defaultdict


def run():
    topics = [
        ("N-gram Language Models",          ngram_lm),
        ("TF-IDF",                          tfidf),
        ("Word2Vec Skip-Gram",              word2vec),
        ("GloVe Embeddings",               glove),
        ("Byte-Pair Encoding (BPE)",        bpe),
        ("Attention in NLP",               attention_nlp),
        ("BERT-style Pretraining",          bert_pretraining),
        ("Language Model Perplexity",       perplexity),
        ("BLEU, ROUGE, BERTScore",          eval_metrics),
        ("Tokenization Strategies",         tokenization),
    ]
    while True:
        print("\n\033[96m╔══════════════════════════════════════════════════╗\033[0m")
        print("\033[96m║         BLOCK 18 — NLP MATHEMATICS               ║\033[0m")
        print("\033[96m╚══════════════════════════════════════════════════╝\033[0m")
        print("\033[90mmmlmath > Block 18 > NLP Math\033[0m\n")
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


def ngram_lm():
    print("\n\033[95m━━━ N-gram Language Models ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
N-gram models estimate the probability of the next word given the
previous (n-1) words. They are simple, interpretable baselines for
language modeling before neural LMs.

Unigram: P(wᵢ) = count(wᵢ)/N
Bigram:  P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁,wᵢ)/count(wᵢ₋₁)
Trigram: P(wᵢ|wᵢ₋₂,wᵢ₋₁) = count(wᵢ₋₂,wᵢ₋₁,wᵢ)/count(wᵢ₋₂,wᵢ₋₁)

Problems: sparse data (most n-grams never seen), exponential parameter
count, no generalization across similar words.
Smoothing: Add-k (Laplace), Kneser-Ney (state-of-the-art backoff).
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Sentence probability:   P(w₁...wₙ) = Π P(wᵢ | wᵢ₋ₙ₊₁...wᵢ₋₁)
  Log-probability:        log P = Σ log P(wᵢ | context)
  Perplexity:             PPL = exp(-1/N Σ log P(wᵢ|context))
  Add-1 smoothing:        P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁,wᵢ)+1)/(count(wᵢ₋₁)+V)
""")

    corpus = "the cat sat on the mat the cat ate the rat".split()
    bigrams = [(corpus[i], corpus[i+1]) for i in range(len(corpus)-1)]
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(corpus)
    V = len(set(corpus))

    print("\033[93mNUMERICAL\033[0m")
    print(f"  Corpus: {' '.join(corpus)}")
    print(f"  Vocab size V = {V}")
    print(f"\n  Top bigrams: {bigram_counts.most_common(5)}")
    test = [("the", "cat"), ("cat", "sat"), ("the", "mat")]
    print(f"\n  {'Bigram':<20} {'MLE P':>8} {'Add-1 P':>10}")
    print("  " + "─" * 42)
    for w1, w2 in test:
        mle = bigram_counts[(w1, w2)] / unigram_counts[w1]
        add1 = (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)
        print(f"  P({w2}|{w1}){'':>8} {mle:>8.4f} {add1:>10.4f}")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • N-grams can't handle "new" never-seen n-grams (data sparsity)
  • Kneser-Ney smoothing is best classical baseline for n-gram LMs
  • Neural LMs (LSTM, Transformer) generalize across words via embeddings
  • N-gram order: bigram≈OK, trigram≈better, 4+gram rarely helps
  • Still used: speech recognition decode, spell checking, domain adaptation
""")
    input("\033[90m[Enter to continue]\033[0m")


def tfidf():
    print("\n\033[95m━━━ TF-IDF ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
TF-IDF (Term Frequency - Inverse Document Frequency) is a weighting
scheme that measures how important a word is to a document in a corpus.

TF rewards words that appear often in a document.
IDF penalizes words that appear in many documents (common words).
Together, TF-IDF balances local relevance with global rarity.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  TF(t,d)  = count(t in d) / |d|
  IDF(t)   = log(N / (1 + df(t)))     [df = num docs containing t]
  TF-IDF(t,d) = TF(t,d) × IDF(t)

  Variants:
    Sublinear TF:   1 + log(count(t,d))   [dampens high frequencies]
    Smooth IDF:     log((1+N)/(1+df)) + 1
    BM25 (best):    TF·(k₁+1)/(TF+k₁(1-b+b·|d|/avgdl)) · IDF
""")

    docs = [
        "the cat sat on the mat",
        "the cat ate the rat",
        "the dog sat on the mat",
        "the dog chased the cat",
    ]
    vocab = set()
    for d in docs:
        vocab.update(d.split())
    vocab = sorted(vocab)
    N = len(docs)

    def compute_tfidf(docs, vocab):
        tf_idf = []
        tf_matrix = []
        idf = {}
        for t in vocab:
            df = sum(1 for d in docs if t in d.split())
            idf[t] = math.log(N / (1 + df))
        for d in docs:
            words = d.split()
            tf = {t: words.count(t)/len(words) for t in vocab}
            tf_matrix.append(tf)
            tf_idf.append({t: tf[t]*idf[t] for t in vocab})
        return tf_idf, idf

    tf_idf, idf = compute_tfidf(docs, vocab)
    print(f"\n  Corpus: {len(docs)} documents")
    print(f"\n  IDF values:")
    for t, v in sorted(idf.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * max(1, int(v * 10))
        print(f"    {t:<10} {v:+.3f} {bar}")

    print(f"\n  TF-IDF for doc 0: '{docs[0]}'")
    for t, v in sorted(tf_idf[0].items(), key=lambda x: -x[1])[:5]:
        if v > 0:
            print(f"    {t:<10} {v:.4f}")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • TF-IDF is surprisingly strong baseline for search & retrieval
  • BM25 generalizes TF-IDF with document-length normalization
  • Dense embeddings (BERT, ada) generally outperform TF-IDF
  • TF-IDF → sparse vector; embeddings → dense vector (dim ~768)
  • Cosine similarity: cos(u,v)=u·v/(||u||·||v||) measures doc similarity
""")
    input("\033[90m[Enter to continue]\033[0m")


def word2vec():
    print("\n\033[95m━━━ Word2Vec Skip-Gram ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Word2Vec (Mikolov 2013) learns dense word embeddings by training a
shallow neural network to predict context words (skip-gram) or predict
a word from context (CBOW).

Key insight: words with similar contexts get similar embeddings.
Result: king − man + woman ≈ queen (famous analogy).

Negative sampling: instead of full softmax over vocabulary (expensive),
sample k "negative" (noise) words and use binary classification.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Skip-gram objective (maximize):
    J = Σ_{(c,o)∈D} log P(o | c)

  Softmax (expensive):
    P(o|c) = exp(vₒᵀuᶜ) / Σ_{w∈V} exp(vwᵀuᶜ)

  Negative sampling objective:
    J_NEG = log σ(vₒᵀuᶜ) + Σₖ E[log σ(−vₙᵀuᶜ)]
    nₙ ~ P_noise(w) ∝ count(w)^(3/4)   [unigram^0.75]

  Analogy: king − man + woman ≈ queen
    find w* = argmax cos(vw, vking − vman + vwoman)
""")

    print("\033[93mTRAINING STEP WALKTHROUGH\033[0m")
    print("""
  Corpus: "the quick brown fox jumps"
  Window=2, center word = "fox", context = {quick, brown, jumps, over}

  [1] Center word "fox" → lookup embedding uᶜ ∈ ℝ^d
  [2] For each context word oᵢ → lookup vₒ ∈ ℝ^d
  [3] Compute score: s = vₒᵀuᶜ
  [4] Sigmoid: σ(s) → should be close to 1 for real pairs
  [5] Sample k negatives, sigmoid should be close to 0
  [6] Update uᶜ ← uᶜ + η∇J,  vₒ ← vₒ + η∇J
""")

    # matplotlib: simple embedding visualization
    try:
        import matplotlib.pyplot as plt2
        from sklearn.decomposition import PCA
        np.random.seed(42)
        words = ["king", "queen", "man", "woman", "dog", "cat", "car", "truck"]
        embeddings = np.random.randn(len(words), 50)
        embeddings[0] += [5, 3]; embeddings[1] += [5, 3.5]  # king,queen
        embeddings[2] += [3, -2]; embeddings[3] += [3, -1.5]
        embeddings[4] += [-3, 2]; embeddings[5] += [-3, 2.5]
        embeddings[6] += [-3, -3]; embeddings[7] += [-2.5, -3]
        pca = PCA(n_components=2)
        pts = pca.fit_transform(embeddings)
        fig, ax = plt2.subplots(figsize=(8, 6))
        ax.scatter(pts[:, 0], pts[:, 1], c='steelblue', s=100)
        for i, w in enumerate(words):
            ax.annotate(w, pts[i], textcoords="offset points", xytext=(5, 5))
        ax.set_title("Word Embeddings (PCA reduced to 2D)")
        ax.grid(True, alpha=0.3)
        plt2.tight_layout(); plt2.show()
    except ImportError:
        print("  \033[90m[Install matplotlib + sklearn for embedding plot]\033[0m")

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • Embedding dimension d = 100-300 is typical for Word2Vec
  • Negative sampling k=5-20 works well; more is better but slower
  • Subword models (FastText) handle morphology: "running" = "run"+"ing"
  • Contextual embeddings (BERT) give different vector per context
  • Word2Vec captures linear semantic relationships geometrically
""")
    input("\033[90m[Enter to continue]\033[0m")


def glove():
    print("\n\033[95m━━━ GloVe Embeddings ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
GloVe (Global Vectors, Pennington 2014) learns embeddings by
factorizing the global word co-occurrence matrix.

Key insight: log(P(ice|solid)/P(steam|solid)) encodes "solid is more
like ice than steam" — ratios of co-occurrence probabilities carry
meaning.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Co-occurrence matrix Xᵢⱼ = # times word j in context of word i

  GloVe objective:
    J = Σᵢⱼ f(Xᵢⱼ) · (wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²

  Weighting function (dampens frequent pairs):
    f(x) = (x/x_max)^α  if x < x_max,  else 1     [α = 0.75]

  vs Word2Vec:
    GloVe: factorize full matrix, efficient, good for rare words
    Word2Vec: local window, faster training, similar quality
""")
    np.random.seed(3)
    corpus = ["the cat sat on the mat",
              "the cat ate the rat",
              "dogs and cats are animals"]
    vocab = sorted(set(" ".join(corpus).split()))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    X = np.zeros((V, V))
    window = 2
    for sent in corpus:
        words = sent.split()
        for i, w in enumerate(words):
            for j in range(max(0, i-window), min(len(words), i+window+1)):
                if i != j:
                    X[w2i[w], w2i[words[j]]] += 1

    print(f"\n  Vocab: {vocab}")
    print(f"  Co-occurrence matrix X (shape {X.shape}):")
    print("  " + "  ".join(f"{v[:5]:>5}" for v in vocab[:8]))
    for i in range(min(8, V)):
        row = "  ".join(f"{int(X[i, j]):>5}" for j in range(min(8, V)))
        print(f"  {row}  ← {vocab[i]}")
    input("\033[90m[Enter to continue]\033[0m")


def bpe():
    print("\n\033[95m━━━ Byte-Pair Encoding (BPE) ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
BPE is a subword tokenization algorithm. It starts with character-level
vocabulary and repeatedly merges the most frequent adjacent pair.

This creates a vocabulary that:
• Handles OOV (out-of-vocabulary) words as subword pieces
• Balances vocabulary size vs sequence length
• Used in GPT-2, GPT-3, RoBERTa, CLIP, etc.
""")
    print("\033[93mALGORITHM STEP-BY-STEP\033[0m")
    print("""
  Input corpus: "low low low lowest newest newest"
  Initial vocab: {l, o, w, e, s, t, n, _}   [_ = word boundary]

  [1] Count all adjacent character pairs in corpus
  [2] Merge most frequent pair → add merged token to vocab
  [3] Re-tokenize corpus with new vocabulary
  [4] Repeat until vocab size reached
""")

    def bpe_demo():
        corpus = {"l o w </w>": 3, "l o w e s t </w>": 2, "n e w e s t </w>": 6,
                  "w i d e r </w>": 3, "n e w </w>": 2}
        print(f"  {'Iteration':<12} {'Merged pair':<20} {'New token'}")
        print("  " + "─" * 50)
        for step in range(6):
            pairs = defaultdict(int)
            for word, freq in corpus.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            print(f"  {step+1:<12} {str(best):<20} {''.join(best)}")
            new_corpus = {}
            bigram = " ".join(best)
            replacement = "".join(best)
            for word, freq in corpus.items():
                new_word = word.replace(bigram, replacement)
                new_corpus[new_word] = freq
            corpus = new_corpus
        print(f"\n  Final tokenization examples:")
        for word, freq in list(corpus.items())[:4]:
            print(f"    {word} (freq={freq})")

    bpe_demo()

    print("\n\033[93mKEY INSIGHTS\033[0m")
    print("""  • BPE vocab size: 32k-100k is typical (GPT-2 uses 50,257)
  • WordPiece (BERT): similar but uses likelihood instead of frequency
  • SentencePiece: language-independent, works on raw text
  • Tokenization affects model: "tokenizer is part of the model"
  • Averaging 4 chars/token in English → 1 token ≈ ¾ word
""")
    input("\033[90m[Enter to continue]\033[0m")


def attention_nlp():
    print("\n\033[95m━━━ Attention in NLP ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Attention (Bahdanau 2015) was introduced for seq2seq models to let the
decoder "look back" at encoder hidden states with learned weights.

Self-attention: every token attends to every other token in same sequence
Cross-attention: decoder queries encoder's key-value pairs
Masked self-attention: can only attend to past tokens (autoregressive)
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Bahdanau attention score:    eᵢⱼ = vᵀ tanh(W·sᵢ₋₁ + U·hⱼ)
  Attention weights:           αᵢⱼ = softmax(eᵢⱼ)
  Context vector:              cᵢ  = Σⱼ αᵢⱼ hⱼ

  Scaled dot-product (Transformer):
  Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
""")

    print("\033[93mASCII: Attention alignment matrix\033[0m")
    src = ["I", "ate", "the", "apple"]
    tgt = ["Je", "ai", "mangé", "la", "pomme"]
    # Simulated attention weights
    A = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.8,  0.05, 0.05],
        [0.05,0.1,  0.7,  0.15],
        [0.05,0.05, 0.15, 0.16],
        [0.05,0.1,  0.15, 0.7],
    ])
    print("  " + "    ".join(f"{w:>6}" for w in src))
    for i, row in enumerate(A):
        bar = "".join("█" if v > 0.4 else ("▓" if v > 0.2 else ("░" if v > 0.05 else " ")) for v in row)
        print(f"  {tgt[i]:<8} {bar}   {np.round(row, 2)}")

    input("\033[90m[Enter to continue]\033[0m")


def bert_pretraining():
    print("\n\033[95m━━━ BERT-style Pretraining ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
BERT (Bidirectional Encoder Representations from Transformers, Devlin 2019)
pretrained a Transformer encoder on two tasks using unlabeled text:

1. Masked Language Modeling (MLM): predict randomly masked tokens
2. Next Sentence Prediction (NSP): predict if sentence B follows A

Key insight: bidirectionality — attention over ALL directions simultaneously.
GPT uses causal/left-to-right attention; BERT uses full bidirectional.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  MLM objective (for each masked position i):
    L_MLM = −Σ_{i∈masked} log P(wᵢ | context)
    P(wᵢ | context) = softmax(H_i · E^T)ᵢ   [H_i = hidden state, E = embedding]

  Pre-train masking strategy:
    80% → [MASK] token
    10% → random word
    10% → unchanged word
  (to prevent "never seen non-[MASK] tokens" at fine-tune time)

  NSP:
    Input: [CLS] Sentence A [SEP] Sentence B [SEP]
    Label: IsNext (50%) or NotNext (50%)
    L_NSP = −Σ log P(IsNext | [CLS] embedding)
""")
    print("\n  Fine-tuning: add task-specific head on top of [CLS] token,")
    print("  train on small labeled dataset (thousands of examples suffice).")
    print("\n  BERT-base: 12 layers, 12 heads, d=768, 110M params")
    print("  BERT-large: 24 layers, 16 heads, d=1024, 340M params")
    input("\033[90m[Enter to continue]\033[0m")


def perplexity():
    print("\n\033[95m━━━ Language Model Perplexity ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Perplexity measures how well a language model predicts a test corpus.
Lower is better: a perplexity of 50 means the model is "as confused
as if choosing uniformly among 50 words at each step."

Perplexity is a log-scale measure of cross-entropy, making it the
standard intrinsic metric for LMs alongside bits-per-character.
""")
    print("\033[93mFORMULAS\033[0m")
    print("""
  Log-likelihood:   L = (1/N) Σᵢ log P(wᵢ | w₁...wᵢ₋₁)
  Perplexity:       PPL = exp(−L) = 2^(−(1/N)Σ log₂P(wᵢ|context))
  Bits-per-char:    BPC = −(1/N) Σ log₂ P(cᵢ | c₁...cᵢ₋₁)

  Example: if model assigns P(next word) = 0.01 for all N tokens:
    L = log(0.01) = −4.605
    PPL = exp(4.605) = 100

  Typical values (Penn Treebank):
    3-gram LM:  ~110 PPL
    LSTM:       ~60-80 PPL
    GPT-2:      ~35 PPL
    GPT-3:      ~20 PPL
""")

    probs = [0.05, 0.1, 0.02, 0.3, 0.15, 0.08, 0.2, 0.01, 0.05, 0.04]
    log_ll = np.mean([math.log(p) for p in probs])
    ppl = math.exp(-log_ll)
    print(f"\n  Example: 10 token probabilities: {probs}")
    print(f"  Avg log P = {log_ll:.4f}")
    print(f"  Perplexity = exp({-log_ll:.4f}) = {ppl:.2f}")

    print("\n\033[93mCODE SNIPPET — PERPLEXITY FOR A TORCH LM\033[0m")
    print("""\033[94m
import torch, math
import torch.nn as nn

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['labels'].to(device)
            logits    = model(input_ids)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss   += loss.item()
            total_tokens += shift_labels.numel()
    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)
\033[0m""")

    input("\033[90m[Enter to continue]\033[0m")


def eval_metrics():
    print("\n\033[95m━━━ BLEU, ROUGE, BERTScore ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Automatic evaluation metrics for NLP generation tasks.

BLEU (Papineni 2002): n-gram precision with brevity penalty.
Used for machine translation, suffers from exact-match bias.

ROUGE (Lin 2004): recall-oriented, used for summarization.
ROUGE-N: n-gram recall; ROUGE-L: longest common subsequence.

BERTScore (Zhang 2020): uses BERT embeddings for semantic similarity.
Better captures meaning than surface-form matching.
""")

    print("\033[93mFORMULAS\033[0m")
    print("""
  BLEU-N precision:
    pₙ = (Σ clipped count of n-grams in candidate) / (Σ n-gram count in candidate)

  Brevity Penalty:
    BP = 1 if c > r, else exp(1 - r/c)   [r=ref length, c=cand length]

  BLEU = BP · exp(Σₙ wₙ log pₙ)    [w_n = 1/N for uniform weights]

  ROUGE-N recall:
    ROUGE-N = (# n-gram matches) / (# n-grams in reference)

  ROUGE-L (F1):
    P = LCS/|candidate|,  R = LCS/|reference|
    F1 = 2PR/(P+R)

  BERTScore F1:
    P = (1/|ĉ|) Σ max_{r∈R} cosine(eĉ, eᵣ)
    R = (1/|R|) Σ max_{c∈ĉ} cosine(eᵣ, ec)
""")

    ref  = "the quick brown fox jumps over the lazy dog".split()
    cand = "the fast brown fox leaps over the lazy dog".split()

    def lcs_len(a, b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    lcs = lcs_len(ref, cand)
    rouge_p = lcs / len(cand)
    rouge_r = lcs / len(ref)
    rouge_l = 2*rouge_p*rouge_r/(rouge_p+rouge_r) if (rouge_p+rouge_r) > 0 else 0

    common_unigrams = len(set(ref) & set(cand))
    bleu1_approx = common_unigrams / len(cand)

    print(f"\n  Reference: {' '.join(ref)}")
    print(f"  Candidate: {' '.join(cand)}")
    print(f"  LCS length = {lcs}")
    print(f"  ROUGE-L = {rouge_l:.4f}")
    print(f"  Approx BLEU-1 = {bleu1_approx:.4f}")

    print("\n\033[93mCODE SNIPPET — BLEU / ROUGE / F1 (LIBRARIES)\033[0m")
    print("""\033[94m
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, classification_report

references = [["the cat sat on the mat".split()]]
hypotheses = ["the cat is on the mat".split()]
smoothie   = SmoothingFunction().method1
bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
scores = scorer.score("The cat sat on the mat", "A cat was sitting on the mat")

y_true = [1,0,1,1,0,1,0,0,1,0]
y_pred = [1,0,1,0,0,1,1,0,1,0]
print(classification_report(y_true, y_pred))
\033[0m""")

    input("\033[90m[Enter to continue]\033[0m")


def tokenization():
    print("\n\033[95m━━━ Tokenization Strategies ━━━\033[0m")
    print("""
\033[1mTHEORY\033[0m
Tokenization converts raw text to integer IDs the model processes.
The choice of tokenizer is literally part of the model — changing it
breaks compatibility.
""")
    print("\033[93mCOMPARISON TABLE\033[0m")
    rows = [
        ("Method",       "Unit",        "OOV?", "Vocab",  "Used in"),
        ("Word-level",   "word",        "Yes",  "20-200k","Word2Vec, older"),
        ("Char-level",   "character",   "No",   "100-300","Some LMs, CharRNN"),
        ("BPE",          "subword",     "No",   "32-100k","GPT-2/3, RoBERTa"),
        ("WordPiece",    "subword",     "No",   "30k",    "BERT, DistilBERT"),
        ("SentencePiece","subword/char","No",   "32-250k","T5, LLaMA, PaLM"),
        ("Unigram LM",   "subword",     "No",   "32-100k","XLNet, Albert"),
    ]
    print("  " + "─" * 72)
    for row in rows:
        print(f"  {row[0]:<15} {row[1]:<14} {row[2]:<8} {row[3]:<10} {row[4]}")
    print("  " + "─" * 72)

    text = "tokenization is fascinating!"
    print(f"\n  Example: '{text}'")
    print(f"  Word:       {text.split()}")
    print(f"  Char:       {list(text)}")
    print(f"  BPE-style:  ['token', 'ization', 'is', 'fasci', 'nating', '!']  (approx)")

    print("\n\033[93mBPE FROM SCRATCH (ALGORITHM SKETCH)\033[0m")
    print("""\033[94m
from collections import Counter, defaultdict
import re

def train_bpe(text, num_merges=50):
    # 1) build initial character vocab with </w> end-of-word token
    words = text.lower().split()
    vocab = Counter(" ".join(list(w) + ['</w>']) for w in words)

    def get_pairs(vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        bigram = re.escape(" ".join(best))
        pattern = re.compile(rf"(?<!\\S){bigram}(?!\\S)")
        vocab = Counter({pattern.sub("".join(best), w): f for w, f in vocab.items()})
    return vocab
\033[0m""")

    print("\n\033[93mTOKENISERS IN MODERN LLMs\033[0m")
    print("""
  • OpenAI GPT-4: tiktoken / BPE with ≈100k subword vocab.
  • LLaMA / T5:   SentencePiece (BPE or unigram) over raw text.
  • Trade-off: larger vocab → shorter sequences but bigger embedding table.
""")

    input("\033[90m[Enter to continue]\033[0m")
