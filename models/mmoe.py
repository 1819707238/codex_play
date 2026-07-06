"""PyTorch implementation of a Multi-gate Mixture-of-Experts model.

MMOE is commonly used in recommendation systems for multi-task learning, for
example jointly optimizing CTR, CVR, watch time, or retention objectives. Each
task owns an independent gate over shared experts, so tasks can reuse common
representations while keeping task-specific expert mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MMOEConfig:
    """Configuration for the MMOE model."""

    input_dim: int
    expert_hidden_units: Sequence[int]
    tower_hidden_units: Sequence[int]
    task_names: Sequence[str]
    num_experts: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not self.task_names:
            raise ValueError("task_names must not be empty")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must be unique")
        if not self.expert_hidden_units:
            raise ValueError("expert_hidden_units must not be empty")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def build_mlp(
    input_dim: int,
    hidden_units: Iterable[int],
    dropout: float,
    output_dim: int | None = None,
) -> nn.Sequential:
    """Build a feed-forward network with ReLU activations."""

    layers: List[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_units:
        if hidden_dim <= 0:
            raise ValueError("hidden units must be positive")
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    if output_dim is not None:
        layers.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*layers)


class Expert(nn.Module):
    """Single expert network shared by all tasks."""

    def __init__(self, input_dim: int, hidden_units: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.network = build_mlp(input_dim, hidden_units, dropout)
        self.output_dim = hidden_units[-1]

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class TaskGate(nn.Module):
    """Task-specific softmax gate over the shared experts."""

    def __init__(self, input_dim: int, num_experts: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, num_experts)

    def forward(self, features: Tensor) -> Tensor:
        return torch.softmax(self.projection(features), dim=-1)


class MMOEModel(nn.Module):
    """Multi-gate Mixture-of-Experts model for recommendation tasks."""

    def __init__(self, config: MMOEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.experts = nn.ModuleList(
            [
                Expert(config.input_dim, config.expert_hidden_units, config.dropout)
                for _ in range(config.num_experts)
            ]
        )
        expert_output_dim = config.expert_hidden_units[-1]

        self.gates = nn.ModuleDict(
            {
                task_name: TaskGate(config.input_dim, config.num_experts)
                for task_name in config.task_names
            }
        )
        self.towers = nn.ModuleDict(
            {
                task_name: build_mlp(
                    expert_output_dim,
                    config.tower_hidden_units,
                    config.dropout,
                    output_dim=1,
                )
                for task_name in config.task_names
            }
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """Return logits for each task.

        Args:
            features: Dense feature tensor shaped ``[batch_size, input_dim]``.

        Returns:
            Mapping from task name to a 1-D logit tensor shaped ``[batch_size]``.
        """

        if features.ndim != 2 or features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"features must have shape [batch_size, {self.config.input_dim}]"
            )

        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts],
            dim=1,
        )
        logits: Dict[str, Tensor] = {}

        for task_name, gate in self.gates.items():
            gate_weights = gate(features).unsqueeze(-1)
            task_representation = torch.sum(expert_outputs * gate_weights, dim=1)
            logits[task_name] = self.towers[task_name](task_representation).squeeze(-1)

        return logits

    @torch.no_grad()
    def predict_proba(self, features: Tensor) -> Dict[str, Tensor]:
        """Return sigmoid probabilities for binary recommendation tasks."""

        return {
            task_name: torch.sigmoid(logit)
            for task_name, logit in self.forward(features).items()
        }


def build_default_mmoe(input_dim: int) -> MMOEModel:
    """Create a practical default MMOE for CTR/CVR style ranking tasks."""

    return MMOEModel(
        MMOEConfig(
            input_dim=input_dim,
            expert_hidden_units=(128, 64),
            tower_hidden_units=(32,),
            task_names=("ctr", "cvr", "watch_time"),
            num_experts=4,
            dropout=0.1,
        )
    )


def _smoke_test() -> None:
    model = build_default_mmoe(input_dim=16)
    batch = torch.randn(8, 16)
    outputs = model(batch)

    assert set(outputs) == {"ctr", "cvr", "watch_time"}
    assert all(value.shape == (8,) for value in outputs.values())


if __name__ == "__main__":
    _smoke_test()
