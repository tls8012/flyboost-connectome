import numpy as np
import torch
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flyboost import FlyBoostClassifier, FlyBoostRegressor


def _write_toy_graph(path):
    # Small directed graph in the same [post, pre] format as MaleCNS.
    n = 12
    pre = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,0,2,4,6,8,10])
    post = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,0,6,7,8,9,10,11])
    w = torch.tensor([
        0.5,0.4,0.3,0.4,0.5,0.3,0.4,0.5,0.3,0.4,0.5,0.3,
        -0.2,0.2,-0.2,0.2,-0.2,0.2,
    ], dtype=torch.float32)
    torch.save({
        "num_nodes": n,
        "edge_index": torch.stack([post, pre]),
        "base_weight": w,
        "sensory_idx": torch.tensor([0,1,2,3,4,5]),
        "readout_idx": torch.tensor([8,9,10,11]),
    }, path)


def test_classifier_modes_and_sklearn_plumbing(tmp_path):
    graph = tmp_path / "toy.pt"
    _write_toy_graph(graph)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    for mode in ["real", "shuffled", "frozen"]:
        clf = FlyBoostClassifier(
            graph_path=str(graph),
            n_flies=2,
            fly_steps=2,
            epochs_per_fly=1,
            batch_size=8,
            fly_lr=1e-2,
            fly_mode=mode,
            random_state=1,
            verbose=0,
        )
        clone(clf)
        clf.fit(X, y, eval_set=[(X, y)])
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        assert pred.shape == (len(X),)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
        assert len(clf.evals_result()["validation_0"]["logloss"]) == 2


def test_pipeline(tmp_path):
    graph = tmp_path / "toy.pt"
    _write_toy_graph(graph)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    pipe = make_pipeline(
        StandardScaler(),
        FlyBoostClassifier(
            graph_path=str(graph),
            n_flies=1,
            fly_steps=1,
            epochs_per_fly=1,
            batch_size=8,
            fly_lr=1e-2,
            verbose=0,
        ),
    )
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


def test_regressor(tmp_path):
    graph = tmp_path / "toy.pt"
    _write_toy_graph(graph)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=32)).astype(np.float32)
    reg = FlyBoostRegressor(
        graph_path=str(graph),
        n_flies=2,
        fly_steps=2,
        epochs_per_fly=1,
        batch_size=8,
        fly_lr=1e-2,
        fly_mode="real",
        verbose=0,
    )
    clone(reg)
    reg.fit(X, y, eval_set=[(X, y)])
    pred = reg.predict(X)
    assert pred.shape == (len(X),)
    assert len(reg.evals_result()["validation_0"]["rmse"]) == 2
