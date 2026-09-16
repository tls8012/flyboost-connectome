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


def _clf(graph, **kwargs):
    return FlyBoostClassifier(
        graph_path=str(graph),
        auto_download=False,
        n_flies=2,
        fly_steps=2,
        epochs_per_fly=1,
        batch_size=8,
        fly_lr=1e-2,
        random_state=1,
        verbose=0,
        **kwargs,
    )


def test_binary_classifier_modes_and_sklearn_plumbing(tmp_path):
    graph = tmp_path / "toy.pt"
    _write_toy_graph(graph)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    for mode in ["real", "shuffled", "frozen"]:
        clf = _clf(graph, fly_mode=mode)
        clone(clf)
        clf.fit(X, y, eval_set=[(X, y)])
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        assert pred.shape == (len(X),)
        assert proba.shape == (len(X), 2)
        assert clf.decision_function(X).shape == (len(X),)
        assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
        assert len(clf.evals_result()["validation_0"]["logloss"]) == 2


def test_multiclass_classifier_all_modes(tmp_path):
    graph = tmp_path / "toy.pt"
    _write_toy_graph(graph)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(48, 5)).astype(np.float32)
    score = np.column_stack([
        1.2 * X[:, 0] - 0.3 * X[:, 1],
        -0.8 * X[:, 0] + 1.1 * X[:, 2],
        0.4 * X[:, 1] - 0.9 * X[:, 2] + X[:, 3],
    ])
    y = np.array(["ant", "bee", "fly"])[score.argmax(axis=1)]

    for mode in ["real", "shuffled", "frozen"]:
        clf = _clf(graph, fly_mode=mode)
        clone(clf)
        clf.fit(X, y, eval_set=[(X, y)])
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        decision = clf.decision_function(X)
        assert clf.n_classes_ == 3
        assert pred.shape == (len(X),)
        assert proba.shape == (len(X), 3)
        assert decision.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
        assert set(np.unique(pred)).issubset(set(clf.classes_))
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
            auto_download=False,
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
        auto_download=False,
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


def test_auto_download_to_requested_path(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    target = tmp_path / "data" / "malecns_nomotor.pt"
    _write_toy_graph(source)

    class _Response:
        def __init__(self, path):
            self.f = open(path, "rb")
        def read(self, n=-1):
            return self.f.read(n)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.f.close()

    import flyboost.estimators as estimators
    monkeypatch.setattr(
        estimators.urllib.request,
        "urlopen",
        lambda url: _Response(source),
    )

    rng = np.random.default_rng(8)
    X = rng.normal(size=(16, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = FlyBoostClassifier(
        graph_path=str(target),
        graph_url="https://example.invalid/malecns_nomotor.pt",
        auto_download=True,
        n_flies=1,
        fly_steps=1,
        epochs_per_fly=1,
        batch_size=8,
        verbose=0,
    )
    clf.fit(X, y)
    assert target.exists()
