from __future__ import annotations

import math

import torch
from torch import nn

from .graph import FlyGraph


class TrainableFly(nn.Module):
    def __init__(
        self,
        graph: FlyGraph,
        n_features: int,
        steps: int = 4,
        leak: float = 0.5,
        max_log_gain: float = math.log(2.0),
        seed: int = 0,
        device: str = "cpu",
        verbose: int = 0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.n = graph.num_nodes
        self.n_features = int(n_features)
        self.steps = int(steps)
        self.leak = float(leak)
        self.max_log_gain = float(max_log_gain)
        self.device_name = device
        self.verbose = int(verbose)
        self.output_dim = int(output_dim)
        if self.output_dim < 1:
            raise ValueError("output_dim must be >= 1")

        W = torch.sparse_coo_tensor(
            graph.edge_index.to(device),
            graph.base_weight.to(device),
            size=(self.n, self.n),
            dtype=torch.float32,
            device=device,
        ).coalesce()
        W.requires_grad_(False)

        # Large fixed graph tensors are deliberately excluded from state_dict.
        self.register_buffer("W", W, persistent=False)
        self.register_buffer("sensory_idx", graph.sensory_idx.to(device), persistent=False)
        self.register_buffer("readout_idx", graph.readout_idx.to(device), persistent=False)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        perm = graph.sensory_idx[
            torch.randperm(len(graph.sensory_idx), generator=gen)
        ]
        feat = torch.arange(len(perm), dtype=torch.long) % self.n_features
        inv = torch.full((graph.num_nodes,), -1, dtype=torch.long)
        inv[perm] = feat
        sensory_feature = inv[graph.sensory_idx]
        self.register_buffer(
            "sensory_feature", sensory_feature.to(device), persistent=False
        )

        self.pre_theta = nn.Parameter(torch.zeros(self.n, device=device))
        self.post_theta = nn.Parameter(torch.zeros(self.n, device=device))
        self.sensor_theta = nn.Parameter(
            torch.zeros(len(graph.sensory_idx), device=device)
        )
        self.readout = nn.Linear(
            len(graph.readout_idx), self.output_dim, bias=True, device=device
        )

    def _gain(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.max_log_gain * torch.tanh(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.pre_theta.device, dtype=torch.float32)
        b = x.shape[0]
        h = torch.zeros((b, self.n), dtype=torch.float32, device=x.device)

        drive = torch.zeros_like(h)
        drive[:, self.sensory_idx] = (
            x[:, self.sensory_feature] * self._gain(self.sensor_theta)
        )

        pre_gain = self._gain(self.pre_theta)
        post_gain = self._gain(self.post_theta)

        for _ in range(self.steps):
            h_pre = h * pre_gain
            recurrent = torch.sparse.mm(self.W, h_pre.T).T
            recurrent = recurrent * post_gain
            proposal = torch.tanh(recurrent + drive)
            h = (1.0 - self.leak) * h + self.leak * proposal

        out = self.readout(h[:, self.readout_idx])
        return out.squeeze(-1) if self.output_dim == 1 else out


class ShuffledTrainableFly(TrainableFly):
    """
    Null-model fly.

    Preserves the neuron populations, edge count, exact source out-degrees,
    exact destination in-degrees, and source-associated signed weights while
    destroying the biological pre -> post pairing.
    """

    _cached_graph: FlyGraph | None = None
    _cached_source_id: int | None = None
    _cached_shuffle_seed: int | None = None

    def __init__(
        self,
        graph: FlyGraph,
        n_features: int,
        steps: int = 4,
        leak: float = 0.5,
        max_log_gain: float = math.log(2.0),
        seed: int = 0,
        device: str = "cpu",
        verbose: int = 0,
        shuffle_seed: int = 20260915,
        output_dim: int = 1,
    ):
        cls = self.__class__
        if (
            cls._cached_graph is None
            or cls._cached_source_id != id(graph)
            or cls._cached_shuffle_seed != int(shuffle_seed)
        ):
            if verbose >= 2:
                print("[ShuffledTrainableFly] building shuffled connectome...")

            edge_index = graph.edge_index.cpu()
            base_weight = graph.base_weight.cpu()

            # prepare_malecns.py convention:
            # edge_index[0] = post, edge_index[1] = pre
            post = edge_index[0]
            pre = edge_index[1]
            E = post.numel()

            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(shuffle_seed))
            perm = torch.randperm(E, generator=gen)
            shuffled_post = post[perm]
            shuffled_weight = base_weight.clone()

            incoming_abs = torch.bincount(
                shuffled_post,
                weights=shuffled_weight.abs(),
                minlength=graph.num_nodes,
            )
            shuffled_weight = (
                shuffled_weight
                / incoming_abs[shuffled_post].clamp_min(1e-12)
            )

            shuffled_edge_index = torch.stack([shuffled_post, pre], dim=0)
            cls._cached_graph = FlyGraph(
                num_nodes=graph.num_nodes,
                edge_index=shuffled_edge_index,
                base_weight=shuffled_weight,
                sensory_idx=graph.sensory_idx,
                readout_idx=graph.readout_idx,
            )
            cls._cached_source_id = id(graph)
            cls._cached_shuffle_seed = int(shuffle_seed)

            if verbose >= 2:
                print(
                    f"[ShuffledTrainableFly] done: "
                    f"{graph.num_nodes:,} neurons, {E:,} shuffled edges"
                )

        super().__init__(
            graph=cls._cached_graph,
            n_features=n_features,
            steps=steps,
            leak=leak,
            max_log_gain=max_log_gain,
            seed=seed,
            device=device,
            verbose=verbose,
            output_dim=output_dim,
        )


class FrozenTrainableFly(TrainableFly):
    """Real connectome with all neuron/sensory gains frozen; readout only trains."""

    def __init__(
        self,
        graph: FlyGraph,
        n_features: int,
        steps: int = 4,
        leak: float = 0.5,
        max_log_gain: float = math.log(2.0),
        seed: int = 0,
        device: str = "cpu",
        verbose: int = 0,
        output_dim: int = 1,
    ):
        super().__init__(
            graph=graph,
            n_features=n_features,
            steps=steps,
            leak=leak,
            max_log_gain=max_log_gain,
            seed=seed,
            device=device,
            verbose=verbose,
            output_dim=output_dim,
        )

        self.pre_theta.requires_grad_(False)
        self.post_theta.requires_grad_(False)
        self.sensor_theta.requires_grad_(False)

        if verbose >= 2:
            n_trainable = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(
                f"[FrozenTrainableFly] trainable params = "
                f"{n_trainable:,} (readout only)"
            )


FLY_MODES = {
    "real": TrainableFly,
    "shuffled": ShuffledTrainableFly,
    "frozen": FrozenTrainableFly,
}
