"""Train three classifiers on CLINC150 and write results/metrics.json.

python train.py            # all three
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from intent_router import data
from intent_router.routing import pick_threshold, report
from intent_router.scratch_nn import ScratchNN

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def in_scope(texts, y, oos):
    keep = y != oos
    return [t for t, k in zip(texts, keep) if k], y[keep]


def score(name, proba_fn, train_seconds, val, test, oos):
    (vt, vy), (tt, ty) = val, test
    vp = proba_fn(vt)
    start = time.perf_counter()
    tp = proba_fn(tt)
    ms = (time.perf_counter() - start) / len(tt) * 1000
    threshold = pick_threshold(vp, vy, oos)
    scope = ty != oos
    accuracy = float((tp[scope].argmax(axis=1) == ty[scope]).mean())
    result = {
        "model": name,
        "in_scope_accuracy": accuracy,
        "train_seconds": round(train_seconds, 1),
        "ms_per_message": round(ms, 3),
        **asdict(report(tp, ty, oos, threshold)),
    }
    print(json.dumps(result, indent=2))
    return result


def align(proba, classes, n_labels):
    out = np.zeros((proba.shape[0], n_labels))
    out[:, classes] = proba
    return out


def main():
    train, val, test, names, oos = data.load()
    tx, ty = in_scope(*train, oos)
    n = len(names)
    results = []

    # 1. TF-IDF and logistic regression: the baseline any model has to beat.
    start = time.perf_counter()
    tfidf = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2),
    )
    lr = make_pipeline(tfidf, LogisticRegression(C=30, max_iter=2000))
    lr.fit(tx, ty)
    results.append(score(
        "TF-IDF + logistic regression",
        lambda t: align(lr.predict_proba(t), lr.classes_, n),
        time.perf_counter() - start, val, test, oos,
    ))

    # 2. The hand-written network, on compressed TF-IDF features.
    start = time.perf_counter()
    words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    svd = TruncatedSVD(400, random_state=0)
    Xs = svd.fit_transform(words.fit_transform(tx))
    label_map = np.unique(ty)
    y_dense = np.searchsorted(label_map, ty)
    nn = ScratchNN(hidden=512, epochs=40).fit(Xs, y_dense)
    results.append(score(
        "Hand-written neural network (NumPy)",
        lambda t: align(nn.predict_proba(svd.transform(words.transform(t))), label_map, n),
        time.perf_counter() - start, val, test, oos,
    ))
    (RESULTS / "scratch_nn_loss.json").write_text(json.dumps([round(x, 4) for x in nn.loss_curve]))

    # 3. Sentence embeddings and logistic regression.
    from sentence_transformers import SentenceTransformer
    start = time.perf_counter()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    Xe = encoder.encode(tx, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    emb_lr = LogisticRegression(C=30, max_iter=2000).fit(Xe, ty)
    results.append(score(
        "MiniLM embeddings + logistic regression",
        lambda t: align(emb_lr.predict_proba(encoder.encode(t, batch_size=128, normalize_embeddings=True, show_progress_bar=False)), emb_lr.classes_, n),
        time.perf_counter() - start, val, test, oos,
    ))

    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
