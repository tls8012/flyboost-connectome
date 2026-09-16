# FlyBoost

A small scikit-learn-compatible wrapper around the FlyBoost prototype from the
research notebook. The connectome tensor file is **not** bundled; point
`graph_path` at your existing `malecns_nomotor.pt`.

download here: https://github.com/tls8012/flyboost-connectome/releases/download/v0.1.0/malecns_nomotor.pt
wget https://github.com/tls8012/flyboost-connectome/releases/download/v0.1.0/malecns_nomotor.pt

## Install

From the project directory:

```bash
pip install .
```

Editable development install:

```bash
pip install -e .
```

You can also install the source archive directly:

```bash
pip install flyboost-connectome-0.1.0.tar.gz
```

## Binary classification

```python
from flyboost import FlyBoostClassifier

model = FlyBoostClassifier(
    graph_path="data/malecns_nomotor.pt",
    n_flies=8,
    epochs_per_fly=8,
    fly_steps=8,
    batch_size=32,
    fly_lr=2e-3,
    fly_mode="real",      # "real" | "shuffled" | "frozen"
    random_state=42,
    verbose=1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
)

pred = model.predict(X_val)
proba = model.predict_proba(X_val)
print(model.score(X_val, y_val))
print(model.evals_result())
```

Switching controls requires only one parameter:

```python
real = FlyBoostClassifier(fly_mode="real", ...)
shuffled = FlyBoostClassifier(fly_mode="shuffled", ...)
frozen = FlyBoostClassifier(fly_mode="frozen", ...)
```

All boosted shuffled flies share one shuffled topology for the same graph and
`shuffle_seed`; their sensory mappings still differ by stage seed.

## Regression

```python
from flyboost import FlyBoostRegressor

reg = FlyBoostRegressor(
    graph_path="data/malecns_nomotor.pt",
    n_flies=8,
    epochs_per_fly=8,
    fly_mode="real",
    verbose=1,
)
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
yhat = reg.predict(X_val)
print(reg.score(X_val, y_val))  # sklearn R^2
```

## sklearn ecosystem

Because the estimators inherit from `BaseEstimator` and the appropriate mixin,
normal sklearn parameter plumbing works:

```python
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

base = FlyBoostClassifier(graph_path="data/malecns_nomotor.pt")
clone(base)

pipe = make_pipeline(StandardScaler(), base)

search = GridSearchCV(
    base,
    {
        "fly_mode": ["real", "shuffled", "frozen"],
        "n_flies": [1, 8, 64],
        "epochs_per_fly": [1, 8, 64],
    },
    cv=3,
)
```

For the compute-matched control experiments, set only:

```python
FlyBoostClassifier(n_flies=1, epochs_per_fly=64, ...)
FlyBoostClassifier(n_flies=8, epochs_per_fly=8, ...)
FlyBoostClassifier(n_flies=64, epochs_per_fly=1, ...)
```

## Verbosity

- `verbose=0`: no training prints
- `verbose=1`: stage metrics
- `verbose>=2`: per-fly epoch metrics plus control-construction diagnostics

## Current scope

- classifier: binary targets
- regressor: single-output targets
- `eval_set=[(X_val, y_val), ...]` supported
- `evals_result()` stores stage-wise metrics
- graph file stays external to keep the package lightweight
