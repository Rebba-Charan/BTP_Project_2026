"""Small LoRA adapters for selected ViFiCLIP visual projection layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return result + update.to(dtype=result.dtype) * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        update = self.lora_b @ self.lora_a
        return self.base.weight + update.to(dtype=self.base.weight.dtype) * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Freeze the model and attach adapters to visual attention/MLP projections."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    wrapped = 0
    for name, module in list(model.image_encoder.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(("attn.out_proj", "mlp.c_fc", "mlp.c_proj")):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.image_encoder.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        wrapped += 1
    if wrapped == 0:
        raise RuntimeError("No visual layers matched the LoRA target modules")
    return wrapped


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in result.unexpected_keys if "lora_" in key]
    missing = [key for key in result.missing_keys if "lora_" in key]
    if unexpected or missing:
        raise ValueError(
            f"LoRA adapter mismatch: missing={missing}, unexpected={unexpected}"
        )
