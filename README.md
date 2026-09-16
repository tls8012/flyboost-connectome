# FlyBoost

FlyBoost is a scikit-learn-compatible classifier/regressor built on a fixed
**Drosophila Male CNS connectome** implemented in PyTorch.

It can be used with ordinary sklearn-style code while switching between the
biological connectome and control models with one parameter.

## Install

From PyPI once the release is published:

```bash
pip install flyboost-connectome
```

For local development:

```bash
git clone https://github.com/tls8012/flyboost-connectome.git
cd flyboost-connectome
pip install -e .
```

## Connectome download

The ~122 MB connectome tensor is intentionally **not bundled in the wheel**.

By default, FlyBoost looks for:

```text
data/malecns_nomotor.pt
```

If that file does not exist, the first `fit()` automatically downloads the
latest GitHub Release asset named `malecns_nomotor.pt` and saves it there:

```text
https://github.com/tls8012/flyboost-connectome/releases/latest/download/malecns_nomotor.pt
```

So the minimal usage is simply:

```python
from flyboost import FlyBoostClassifier

model = FlyBoostClassifier()
model.fit(X_train, y_train)
```

To use a manually downloaded graph instead:

```python
model = FlyBoostClassifier(
    graph_path="/path/to/malecns_nomotor.pt",
    auto_download=False,
)
```

`graph_url=` can also be overridden if the asset is mirrored elsewhere.

## Classification

`FlyBoostClassifier` supports both **binary and multiclass** targets, including
non-numeric sklearn labels such as strings.

```python
from flyboost import FlyBoostClassifier

model = FlyBoostClassifier(
    n_flies=8,
    epochs_per_fly=8,
    fly_steps=8,
    batch_size=32,
    fly_lr=2e-3,
    learning_rate=0.2,
    fly_mode="real",      # "real" | "shuffled" | "frozen"
    random_state=42,
    device="cpu",        # or "cuda" / "auto"
    verbose=1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
)

pred = model.predict(X_val)
proba = model.predict_proba(X_val)
score = model.score(X_val, y_val)

print(model.classes_)
print(model.evals_result())
```

For multiclass classification, `predict_proba(X)` returns shape
`(n_samples, n_classes)` and each row sums to 1.

```python
# y may contain, for example:
# ["cat", "dog", "fly", "cat", ...]
model.fit(X_train, y_train)

print(model.classes_)
print(model.predict_proba(X_test).shape)
```

Binary classification keeps the original scalar-logit FlyBoost path. For 3+
classes, each fly gets an `n_classes`-dimensional readout and is trained with a
diagonal Newton approximation to multinomial log-loss.

## Control modes

The same estimator can switch between three fly types:

```python
real = FlyBoostClassifier(fly_mode="real")
shuffled = FlyBoostClassifier(fly_mode="shuffled")
frozen = FlyBoostClassifier(fly_mode="frozen")
```

### `real`

Uses the actual Male CNS topology with trainable pre-neuron, post-neuron and
sensory gains plus a trainable readout.

### `shuffled`

Preserves:

- the same neurons
- the same sensory/readout populations
- the same number of edge entries
- exact source out-degree counts
- exact destination in-degree counts
- source-associated signed weights
- the same trainable gain mechanism

but destroys the actual biological pre -> post pairing.

All boosted shuffled flies share one shuffled topology for the same graph and
`shuffle_seed`; their sensory mappings still differ by stage seed.

### `frozen`

Keeps the real topology and fixed internal gains. Only the final readout is
trainable.

## Regression

```python
from flyboost import FlyBoostRegressor

reg = FlyBoostRegressor(
    n_flies=8,
    epochs_per_fly=8,
    fly_mode="real",
    verbose=1,
)

reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
yhat = reg.predict(X_val)
print(reg.score(X_val, y_val))  # sklearn R^2
```

The current regressor supports single-output regression.

## sklearn compatibility

FlyBoost estimators support the normal sklearn estimator interface:

- `fit`
- `predict`
- `predict_proba` for classifiers
- `decision_function` for classifiers
- `score`
- `get_params` / `set_params`
- `clone`
- `Pipeline`
- `GridSearchCV` / `RandomizedSearchCV`
- `eval_set`
- `evals_result()`

Example:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    FlyBoostClassifier(verbose=0),
    {
        "fly_mode": ["real", "shuffled", "frozen"],
        "n_flies": [1, 8],
        "epochs_per_fly": [1, 8],
    },
    cv=3,
)
search.fit(X_train, y_train)
```

## Compute-matched fly/stage controls

The fly count and training epochs are independent parameters, so controls such
as these require no code changes:

```python
FlyBoostClassifier(n_flies=1, epochs_per_fly=64)
FlyBoostClassifier(n_flies=8, epochs_per_fly=8)
FlyBoostClassifier(n_flies=64, epochs_per_fly=1)
```

## Verbosity

- `verbose=0`: no training output
- `verbose=1`: download/status and stage metrics
- `verbose>=2`: per-fly epoch metrics plus control-construction diagnostics

## Development

Run tests from the repository root:

```bash
pip install -e ".[dev]"
pytest -q
```

Build wheel + source distribution:

```bash
rm -rf build dist src/*.egg-info
python -m build
```

The package is intentionally lightweight; the connectome graph remains a
separate GitHub Release asset.
