from __future__ import annotations

import copy
import math
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from .fly import FLY_MODES
from .graph import FlyGraph, load_graph_cached


DEFAULT_GRAPH_PATH = "data/malecns_nomotor.pt"
DEFAULT_GRAPH_URL = (
    "https://github.com/tls8012/flyboost-connectome/"
    "releases/latest/download/malecns_nomotor.pt"
)


@dataclass
class BoostStage:
    state_dict: dict[str, torch.Tensor]


def _as_float_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


class _FlyBoostBase(BaseEstimator):
    """Shared sklearn-facing plumbing for FlyBoost estimators."""

    def __init__(
        self,
        graph_path: str = DEFAULT_GRAPH_PATH,
        graph_url: str = DEFAULT_GRAPH_URL,
        auto_download: bool = True,
        n_flies: int = 8,
        learning_rate: float = 0.2,
        fly_steps: int = 8,
        epochs_per_fly: int = 8,
        batch_size: int = 32,
        fly_lr: float = 2e-3,
        fly_gain_lr: float = 0.0,
        fly_mode: str = "real",
        leak: float = 0.5,
        max_log_gain: float = math.log(2.0),
        shuffle_seed: int = 20260915,
        device: str = "cpu",
        random_state: int | None = 42,
        verbose: int = 0,
    ):
        self.graph_path = graph_path
        self.graph_url = graph_url
        self.auto_download = auto_download
        self.n_flies = n_flies
        self.learning_rate = learning_rate
        self.fly_steps = fly_steps
        self.epochs_per_fly = epochs_per_fly
        self.batch_size = batch_size
        self.fly_lr = fly_lr
        self.fly_gain_lr = fly_gain_lr
        self.fly_mode = fly_mode
        self.leak = leak
        self.max_log_gain = max_log_gain
        self.shuffle_seed = shuffle_seed
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _validate_hyperparameters(self) -> None:
        if self.fly_mode not in FLY_MODES:
            raise ValueError(
                f"fly_mode must be one of {sorted(FLY_MODES)}, got {self.fly_mode!r}"
            )
        if int(self.n_flies) < 1:
            raise ValueError("n_flies must be >= 1")
        if int(self.epochs_per_fly) < 1:
            raise ValueError("epochs_per_fly must be >= 1")
        if int(self.fly_steps) < 1:
            raise ValueError("fly_steps must be >= 1")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be > 0")
        if float(self.fly_lr) <= 0:
            raise ValueError("fly_lr must be > 0")
        if float(self.fly_gain_lr) < 0:
            raise ValueError("fly_gain_lr must be >= 0")
        if not (0 < float(self.leak) <= 1):
            raise ValueError("leak must be in (0, 1]")
        if self.auto_download and not str(self.graph_url):
            raise ValueError("graph_url must be non-empty when auto_download=True")

    @property
    def _seed(self) -> int:
        return 0 if self.random_state is None else int(self.random_state)

    def _ensure_graph_file(self) -> Path:
        path = Path(self.graph_path).expanduser().resolve()
        if path.exists():
            return path

        if not self.auto_download:
            raise FileNotFoundError(
                f"FlyBoost graph not found: {path}. "
                "Pass graph_path=... or set auto_download=True."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        if int(self.verbose) >= 1:
            print(f"[FlyBoost] downloading connectome -> {path}")

        try:
            with urllib.request.urlopen(str(self.graph_url)) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            os.replace(tmp, path)
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                "Could not download the FlyBoost connectome from "
                f"{self.graph_url!r}. Download malecns_nomotor.pt manually "
                f"and place it at {path}, or pass graph_path=..."
            ) from exc

        if int(self.verbose) >= 1:
            print(f"[FlyBoost] connectome ready: {path}")
        return path

    def _graph(self) -> FlyGraph:
        return load_graph_cached(str(self._ensure_graph_file()))

    def _new_fly(self, graph: FlyGraph, stage: int):
        fly_cls = FLY_MODES[self.fly_mode]
        kwargs = dict(
            graph=graph,
            n_features=self.n_features_in_,
            steps=self.fly_steps,
            leak=self.leak,
            max_log_gain=self.max_log_gain,
            seed=self._seed + int(stage),
            device=self.device_,
            verbose=self.verbose,
            output_dim=int(getattr(self, "output_dim_", 1)),
        )
        if self.fly_mode == "shuffled":
            kwargs["shuffle_seed"] = self.shuffle_seed

        init_seed = self._seed + 20000 + int(stage)
        cuda_devices = []
        if str(self.device_).startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(self.device_)
            cuda_devices = [
                dev.index if dev.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(init_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(init_seed)
            return fly_cls(**kwargs)

    def _make_optimizer(self, fly) -> torch.optim.Optimizer:
        gain_lr = float(self.fly_lr if self.fly_gain_lr == 0 else self.fly_gain_lr)
        gain_params = [
            p for p in (fly.pre_theta, fly.post_theta, fly.sensor_theta)
            if p.requires_grad
        ]
        readout_params = [p for p in fly.readout.parameters() if p.requires_grad]

        groups = []
        if gain_params:
            groups.append({"params": gain_params, "lr": gain_lr})
        if readout_params:
            groups.append({"params": readout_params, "lr": float(self.fly_lr)})
        return torch.optim.Adam(groups)

    def _batched_predict(self, fly, x: torch.Tensor) -> torch.Tensor:
        out = []
        with torch.no_grad():
            for start in range(0, len(x), int(self.batch_size)):
                xb = x[start : start + int(self.batch_size)].to(self.device_)
                out.append(fly(xb).cpu())
        if out:
            return torch.cat(out)
        if int(getattr(self, "output_dim_", 1)) == 1:
            return torch.empty(0, dtype=torch.float32)
        return torch.empty((0, int(self.output_dim_)), dtype=torch.float32)

    def _fit_fly_to_target(
        self,
        fly,
        x: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        stage: int,
    ) -> None:
        opt = self._make_optimizer(fly)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._seed + 10000 + int(stage))

        for epoch in range(int(self.epochs_per_fly)):
            perm = torch.randperm(len(x), generator=gen)
            weighted_loss_sum = 0.0
            seen_weight = 0.0

            for start in range(0, len(x), int(self.batch_size)):
                idx = perm[start : start + int(self.batch_size)]
                xb = x[idx].to(self.device_)
                tb = target[idx].to(self.device_)
                wb = weight[idx].to(self.device_)

                opt.zero_grad(set_to_none=True)
                pred = fly(xb)
                denom = wb.sum().clamp_min(1e-8)
                loss = (wb * (pred - tb).square()).sum() / denom
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in fly.parameters() if p.requires_grad], 5.0
                )
                opt.step()

                batch_weight = float(wb.sum().detach().cpu())
                weighted_loss_sum += float(loss.detach().cpu()) * batch_weight
                seen_weight += batch_weight

            if int(self.verbose) >= 2:
                mean_loss = weighted_loss_sum / max(seen_weight, 1e-12)
                print(
                    f"  fly {stage + 1}/{self.n_flies} "
                    f"epoch {epoch + 1}/{self.epochs_per_fly} "
                    f"weighted_mse={mean_loss:.5f}"
                )

    @staticmethod
    def _freeze_and_save(fly) -> BoostStage:
        fly.eval()
        for p in fly.parameters():
            p.requires_grad_(False)
        state = {k: v.detach().cpu().clone() for k, v in fly.state_dict().items()}
        return BoostStage(state)

    def _initial_raw(self, n_samples: int) -> torch.Tensor:
        base = torch.as_tensor(self.base_score_, dtype=torch.float32)
        if base.ndim == 0:
            return torch.full((n_samples,), float(base), dtype=torch.float32)
        return base.unsqueeze(0).expand(n_samples, -1).clone()

    def _predict_raw_tensor(self, X) -> torch.Tensor:
        check_is_fitted(self, attributes=["stages_", "n_features_in_"])
        Xv = validate_data(self, X, reset=False, dtype=np.float32, ensure_2d=True)
        x = _as_float_tensor(Xv)
        graph = self._graph()
        raw = self._initial_raw(len(x))
        for stage, saved in enumerate(self.stages_):
            fly = self._new_fly(graph, stage)
            fly.load_state_dict(saved.state_dict, strict=True)
            fly.eval()
            raw += float(self.learning_rate) * self._batched_predict(fly, x)
            del fly
        return raw

    def evals_result(self) -> dict:
        check_is_fitted(self, attributes=["evals_result_"])
        return copy.deepcopy(self.evals_result_)


class FlyBoostClassifier(ClassifierMixin, _FlyBoostBase):
    """Binary or multiclass classifier using stage-wise Fly weak learners."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.multi_class = True
            tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X, y, eval_set=None):
        self._validate_hyperparameters()
        Xv, yv = validate_data(
            self, X, y, reset=True, dtype=np.float32, ensure_2d=True
        )
        check_classification_targets(yv)
        y_type = type_of_target(yv, input_name="y", raise_unknown=True)
        if y_type not in {"binary", "multiclass"}:
            raise ValueError(
                "FlyBoostClassifier supports binary and multiclass targets; "
                f"got target type {y_type}."
            )

        self.classes_, y_idx_np = np.unique(yv, return_inverse=True)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("FlyBoostClassifier requires at least two classes; got 1 class")

        self.output_dim_ = 1 if self.n_classes_ == 2 else self.n_classes_
        self.device_ = _resolve_device(self.device)
        x = _as_float_tensor(Xv)
        graph = self._graph()
        self.stages_ = []
        self.evals_result_ = {"train": {"logloss": [], "accuracy": []}}

        eval_data = self._prepare_classifier_eval_set(eval_set)
        for i in range(len(eval_data)):
            self.evals_result_[f"validation_{i}"] = {
                "logloss": [],
                "accuracy": [],
            }

        if self.n_classes_ == 2:
            self._fit_binary(x, y_idx_np.astype(np.float32), graph, eval_data)
        else:
            self._fit_multiclass(x, y_idx_np.astype(np.int64), graph, eval_data)
        return self

    def _fit_binary(self, x, y_idx_np, graph, eval_data):
        y_t = torch.from_numpy(y_idx_np)
        sw = torch.ones(len(x), dtype=torch.float32)

        weighted_pos = (sw * y_t).sum() / sw.sum().clamp_min(1e-8)
        weighted_pos = weighted_pos.clamp(1e-4, 1 - 1e-4)
        self.base_score_ = float(torch.log(weighted_pos / (1 - weighted_pos)))
        ensemble = self._initial_raw(len(x))
        eval_ensembles = [self._initial_raw(len(ex)) for ex, _ in eval_data]

        for stage in range(int(self.n_flies)):
            p = torch.sigmoid(ensemble)
            g = p - y_t
            h = (p * (1 - p)).clamp_min(1e-3)
            z = (-g / h).clamp(-8.0, 8.0)
            fit_weight = h * sw

            fly = self._new_fly(graph, stage)
            self._fit_fly_to_target(fly, x, z, fit_weight, stage)
            ensemble += float(self.learning_rate) * self._batched_predict(fly, x)

            train_logloss = float(
                F.binary_cross_entropy_with_logits(
                    ensemble, y_t, weight=sw, reduction="sum"
                )
                / sw.sum().clamp_min(1e-8)
            )
            train_acc = float(((ensemble > 0).float() == y_t).float().mean())
            self.evals_result_["train"]["logloss"].append(train_logloss)
            self.evals_result_["train"]["accuracy"].append(train_acc)
            msg = (
                f"[stage {stage + 1}] train_logloss={train_logloss:.5f} "
                f"train_acc={train_acc:.3f}"
            )

            for i, (ex, ey) in enumerate(eval_data):
                eval_ensembles[i] += float(self.learning_rate) * self._batched_predict(
                    fly, ex
                )
                logits = eval_ensembles[i]
                val_logloss = float(F.binary_cross_entropy_with_logits(logits, ey.float()))
                val_acc = float(((logits > 0).long() == ey).float().mean())
                key = f"validation_{i}"
                self.evals_result_[key]["logloss"].append(val_logloss)
                self.evals_result_[key]["accuracy"].append(val_acc)
                msg += (
                    f" {key}_logloss={val_logloss:.5f} "
                    f"{key}_acc={val_acc:.3f}"
                )

            if int(self.verbose) >= 1:
                print(msg)
            self.stages_.append(self._freeze_and_save(fly))
            del fly

    def _fit_multiclass(self, x, y_idx_np, graph, eval_data):
        y_t = torch.from_numpy(y_idx_np)
        onehot = F.one_hot(y_t, num_classes=self.n_classes_).float()

        counts = torch.bincount(y_t, minlength=self.n_classes_).float()
        priors = (counts / counts.sum()).clamp_min(1e-6)
        base = torch.log(priors)
        base -= base.mean()
        self.base_score_ = base.numpy()
        ensemble = self._initial_raw(len(x))
        eval_ensembles = [self._initial_raw(len(ex)) for ex, _ in eval_data]

        for stage in range(int(self.n_flies)):
            p = torch.softmax(ensemble, dim=1)
            g = p - onehot
            h = (p * (1.0 - p)).clamp_min(1e-3)
            z = (-g / h).clamp(-8.0, 8.0)

            fly = self._new_fly(graph, stage)
            self._fit_fly_to_target(fly, x, z, h, stage)
            ensemble += float(self.learning_rate) * self._batched_predict(fly, x)

            train_logloss = float(F.cross_entropy(ensemble, y_t))
            train_acc = float((ensemble.argmax(dim=1) == y_t).float().mean())
            self.evals_result_["train"]["logloss"].append(train_logloss)
            self.evals_result_["train"]["accuracy"].append(train_acc)
            msg = (
                f"[stage {stage + 1}] train_logloss={train_logloss:.5f} "
                f"train_acc={train_acc:.3f}"
            )

            for i, (ex, ey) in enumerate(eval_data):
                eval_ensembles[i] += float(self.learning_rate) * self._batched_predict(
                    fly, ex
                )
                logits = eval_ensembles[i]
                val_logloss = float(F.cross_entropy(logits, ey))
                val_acc = float((logits.argmax(dim=1) == ey).float().mean())
                key = f"validation_{i}"
                self.evals_result_[key]["logloss"].append(val_logloss)
                self.evals_result_[key]["accuracy"].append(val_acc)
                msg += (
                    f" {key}_logloss={val_logloss:.5f} "
                    f"{key}_acc={val_acc:.3f}"
                )

            if int(self.verbose) >= 1:
                print(msg)
            self.stages_.append(self._freeze_and_save(fly))
            del fly

    def _prepare_classifier_eval_set(self, eval_set):
        if eval_set is None:
            return []
        out = []
        for X_eval, y_eval in eval_set:
            Xe = validate_data(
                self, X_eval, reset=False, dtype=np.float32, ensure_2d=True
            )
            ye = np.asarray(y_eval)
            unknown = np.setdiff1d(np.unique(ye), self.classes_)
            if len(unknown):
                raise ValueError(
                    f"eval_set contains labels not seen in fit(): {unknown.tolist()}"
                )
            if len(Xe) != len(ye):
                raise ValueError("X_eval and y_eval have inconsistent lengths")
            indices = np.searchsorted(self.classes_, ye).astype(np.int64)
            out.append((_as_float_tensor(Xe), torch.from_numpy(indices)))
        return out

    def decision_function(self, X) -> np.ndarray:
        return self._predict_raw_tensor(X).numpy()

    def predict_proba(self, X) -> np.ndarray:
        logits = self._predict_raw_tensor(X)
        if self.n_classes_ == 2:
            p1 = torch.sigmoid(logits).numpy()
            return np.column_stack([1.0 - p1, p1])
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        logits = self._predict_raw_tensor(X)
        if self.n_classes_ == 2:
            idx = (logits.numpy() > 0).astype(np.int64)
        else:
            idx = logits.argmax(dim=1).numpy()
        return self.classes_[idx]


class FlyBoostRegressor(RegressorMixin, _FlyBoostBase):
    """Single-output squared-error regressor using Fly connectome weak learners."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        return tags

    def fit(self, X, y, eval_set=None):
        self._validate_hyperparameters()
        Xv, yv = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=np.float32,
            ensure_2d=True,
            y_numeric=True,
        )
        y_arr = np.asarray(yv, dtype=np.float32)
        if y_arr.ndim != 1:
            raise ValueError("FlyBoostRegressor supports single-output regression only")

        self.device_ = _resolve_device(self.device)
        x = _as_float_tensor(Xv)
        y_t = torch.from_numpy(y_arr)
        sw = torch.ones(len(x), dtype=torch.float32)

        graph = self._graph()
        self.stages_ = []
        self.evals_result_ = {"train": {"rmse": [], "mae": []}}

        self.base_score_ = float((sw * y_t).sum() / sw.sum().clamp_min(1e-8))
        ensemble = torch.full_like(y_t, self.base_score_)

        eval_data = self._prepare_regression_eval_set(eval_set)
        eval_ensembles = [
            torch.full((len(ex),), self.base_score_, dtype=torch.float32)
            for ex, _ in eval_data
        ]
        for i in range(len(eval_data)):
            self.evals_result_[f"validation_{i}"] = {"rmse": [], "mae": []}

        for stage in range(int(self.n_flies)):
            residual = y_t - ensemble
            fly = self._new_fly(graph, stage)
            self._fit_fly_to_target(fly, x, residual, sw, stage)

            ensemble += float(self.learning_rate) * self._batched_predict(fly, x)
            err = ensemble - y_t
            train_rmse = float(torch.sqrt((sw * err.square()).sum() / sw.sum()))
            train_mae = float((sw * err.abs()).sum() / sw.sum())
            self.evals_result_["train"]["rmse"].append(train_rmse)
            self.evals_result_["train"]["mae"].append(train_mae)

            msg = (
                f"[stage {stage + 1}] train_rmse={train_rmse:.5f} "
                f"train_mae={train_mae:.5f}"
            )

            for i, (ex, ey) in enumerate(eval_data):
                eval_ensembles[i] += float(self.learning_rate) * self._batched_predict(
                    fly, ex
                )
                err_eval = eval_ensembles[i] - ey
                rmse = float(torch.sqrt(torch.mean(err_eval.square())))
                mae = float(torch.mean(err_eval.abs()))
                key = f"validation_{i}"
                self.evals_result_[key]["rmse"].append(rmse)
                self.evals_result_[key]["mae"].append(mae)
                msg += f" {key}_rmse={rmse:.5f} {key}_mae={mae:.5f}"

            if int(self.verbose) >= 1:
                print(msg)

            self.stages_.append(self._freeze_and_save(fly))
            del fly

        return self

    def _prepare_regression_eval_set(self, eval_set):
        if eval_set is None:
            return []
        out = []
        for X_eval, y_eval in eval_set:
            Xe = validate_data(
                self, X_eval, reset=False, dtype=np.float32, ensure_2d=True
            )
            ye = np.asarray(y_eval, dtype=np.float32)
            if ye.ndim != 1:
                raise ValueError("Each regression y_eval must be 1-dimensional")
            if len(Xe) != len(ye):
                raise ValueError("X_eval and y_eval have inconsistent lengths")
            out.append((_as_float_tensor(Xe), torch.from_numpy(ye)))
        return out

    def predict(self, X) -> np.ndarray:
        return self._predict_raw_tensor(X).numpy()
