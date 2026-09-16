from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


@dataclass
class FlyGraph:
    num_nodes: int
    edge_index: torch.Tensor
    base_weight: torch.Tensor
    sensory_idx: torch.Tensor
    readout_idx: torch.Tensor

    @classmethod
    def load(cls, path: str | Path) -> "FlyGraph":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            int(d["num_nodes"]),
            d["edge_index"].long(),
            d["base_weight"].float(),
            d["sensory_idx"].long(),
            d["readout_idx"].long(),
        )


@lru_cache(maxsize=4)
def load_graph_cached(path: str) -> FlyGraph:
    """Load a graph once per absolute path inside the current Python process."""
    resolved = str(Path(path).expanduser().resolve())
    return FlyGraph.load(resolved)
