"""Export a small TF-IDF model that runs in a web page, and measure the pruned model honestly.

The browser demo uses word features only and keeps the strongest weights per intent,
so it is smaller and a little less accurate than the models in train.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from intent_router import data
from intent_router.routing import pick_threshold, report

KEEP_PER_CLASS = 140
OUT = Path("results/browser_model.json")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def main():
    train, val, test, names, oos = data.load()
    keep = train[1] != oos
    tx = [t for t, k in zip(train[0], keep) if k]
    ty = train[1][keep]

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(tx)
    clf = LogisticRegression(C=30, max_iter=2000).fit(X, ty)

    coef = clf.coef_.copy()
    for row in coef:
        cutoff = np.partition(np.abs(row), -KEEP_PER_CLASS)[-KEEP_PER_CLASS]
        row[np.abs(row) < cutoff] = 0

    # The page only ships the kept features, so normalise over those alone here as well.
    used = np.flatnonzero(np.abs(coef).sum(axis=0))
    terms = vec.get_feature_names_out()[used]
    idf = vec.idf_[used]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, vocabulary={t: i for i, t in enumerate(terms)})
    vec.idf_ = idf
    coef = coef[:, used]

    def proba(texts):
        z = np.asarray(vec.transform(texts) @ coef.T) + clf.intercept_
        out = np.zeros((len(texts), len(names)))
        out[:, clf.classes_] = softmax(z)
        return out

    threshold = pick_threshold(proba(val[0]), val[1], oos)
    tp = proba(test[0])
    scope = test[1] != oos
    metrics = {
        "model": f"Browser model (word TF-IDF, {KEEP_PER_CLASS} weights per intent)",
        "in_scope_accuracy": float((tp[scope].argmax(axis=1) == test[1][scope]).mean()),
        **asdict(report(tp, test[1], oos, threshold)),
    }
    print(json.dumps(metrics, indent=2))

    weights = [[[int(i), round(float(row[i]), 2)] for i in np.flatnonzero(row)] for row in coef]
    model = {
        "threshold": round(threshold, 3),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "labels": [names[int(c)] for c in clf.classes_],
        "vocab": {t: i for i, t in enumerate(terms)},
        "idf": [round(float(x), 3) for x in idf],
        "intercept": [round(float(x), 2) for x in clf.intercept_],
        "weights": weights,
    }
    OUT.write_text(json.dumps(model, separators=(",", ":")))
    print(OUT, f"{OUT.stat().st_size / 1024:.0f} KB", len(terms), "features")

    # A parity fixture so the JavaScript version can be tested against this one.
    samples = test[0][:400]
    fixture = [{"text": t, "label": model["labels"][int(np.argmax(p[clf.classes_]))], "conf": round(float(p.max()), 4)}
               for t, p in zip(samples, proba(samples))]
    Path("results/browser_parity.json").write_text(json.dumps(fixture))


if __name__ == "__main__":
    main()
