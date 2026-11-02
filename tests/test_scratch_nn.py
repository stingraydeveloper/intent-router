import numpy as np

from intent_router.routing import pick_threshold, report
from intent_router.scratch_nn import ScratchNN


def test_learns_a_problem_a_linear_model_cannot():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (600, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    model = ScratchNN(hidden=32, lr=0.01, epochs=120, batch_size=32, weight_decay=0).fit(X, y)
    assert (model.predict_proba(X).argmax(axis=1) == y).mean() > 0.95
    assert model.loss_curve[-1] < model.loss_curve[0]


def test_gradients_match_numerical_estimate():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(8, 3)), rng.integers(0, 3, 8)
    model = ScratchNN(hidden=4, epochs=1, weight_decay=0).fit(X, y)
    _, grads = model._loss_and_grads(X, y)
    eps = 1e-5
    for name in ("W1", "W2"):
        param = model.params[name]
        original = param[0, 0]
        param[0, 0] = original + eps
        up, _ = model._loss_and_grads(X, y)
        param[0, 0] = original - eps
        down, _ = model._loss_and_grads(X, y)
        param[0, 0] = original
        assert abs((up - down) / (2 * eps) - grads[name][0, 0]) < 1e-4


def test_low_confidence_goes_to_a_person():
    probs = np.array([[0.9, 0.1, 0], [0.4, 0.35, 0.25], [0.2, 0.3, 0.5]])
    y = np.array([0, 2, 2])
    r = report(probs, y, oos_label=2, threshold=0.8)
    assert r.coverage == 1.0
    assert r.routed_precision == 1.0
    assert r.out_of_scope_caught == 1.0
    assert r.wrong_auto_rate == 0.0


def test_threshold_meets_target_on_the_data_it_saw():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 500)
    probs = rng.dirichlet(np.ones(5), 500)
    probs[np.arange(500), y] += 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    t = pick_threshold(probs, y, oos_label=4, target_precision=0.9)
    assert report(probs, y, 4, t).routed_precision >= 0.9
