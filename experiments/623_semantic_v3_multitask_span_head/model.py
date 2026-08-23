from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from multitask_head import SpanConceptHead
from torch import nn


class Qwen35VerdictSpanModel(nn.Module):
    """Verdict-primary wrapper that preserves the parent's digit-logit classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        concept_count: int,
        token_zero: int,
        token_one: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.auxiliary_head = SpanConceptHead(hidden_size, concept_count)
        self.token_zero = int(token_zero)
        self.token_one = int(token_one)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        text_token_mask: torch.Tensor,
        **model_inputs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **model_inputs,
        )
        hidden = outputs.hidden_states[-1]
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
        last = torch.where(attention_mask.bool(), positions, -1).max(dim=1).values
        row = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        last_logits = outputs.logits[row, last]
        verdict_logits = last_logits[:, self.token_one] - last_logits[:, self.token_zero]
        auxiliary = self.auxiliary_head(hidden, attention_mask, text_token_mask)
        return {"verdict_logits": verdict_logits, **auxiliary}


class TinyBackbone(nn.Module):
    """CPU-only contract double used by preflight; never a training substitute."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))
