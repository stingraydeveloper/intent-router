"""A one-hidden-layer softmax classifier with backpropagation written by hand.

No framework: forward pass, gradients and Adam are all below, which makes it easy
to read what a framework would otherwise do for you.
"""
from __future__ import annotations

import numpy as np


class ScratchNN:
    def __init__(self, hidden: int = 256, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 64, weight_decay: float = 1e-4, seed: int = 0):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.batch_size, self.weight_decay = batch_size, weight_decay
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchNN":
        n, d = X.shape
        k = int(y.max()) + 1
        self.params = {
            "W1": self.rng.normal(0, 1 / np.sqrt(d), (d, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": self.rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, k)),
            "b2": np.zeros(k),
        }
        m = {name: np.zeros_like(p) for name, p in self.params.items()}
        v = {name: np.zeros_like(p) for name, p in self.params.items()}
        self.loss_curve: list[float] = []
        step = 0
        for _ in range(self.epochs):
            order = self.rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                loss, grads = self._loss_and_grads(X[idx], y[idx])
                total += loss * len(idx)
                step += 1
                for name, grad in grads.items():
                    m[name] = 0.9 * m[name] + 0.1 * grad
                    v[name] = 0.999 * v[name] + 0.001 * grad**2
                    mhat = m[name] / (1 - 0.9**step)
                    vhat = v[name] / (1 - 0.999**step)
                    self.params[name] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)
            self.loss_curve.append(total / n)
        return self

    def _loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        p = self.params
        h = np.maximum(0, X @ p["W1"] + p["b1"])
        logits = h @ p["W2"] + p["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        n = len(y)
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()

        err = probs.copy()
        err[np.arange(n), y] -= 1
        d_h = err @ p["W2"].T * (h > 0)
        return loss, {
            "W2": h.T @ err / n + self.weight_decay * p["W2"],
            "b2": err.mean(axis=0),
            "W1": X.T @ d_h / n + self.weight_decay * p["W1"],
            "b1": d_h.mean(axis=0),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        logits = np.maximum(0, X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)
