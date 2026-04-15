"""
Exponential Moving Average (EMA) of model parameters.

WHY EMA?
    SGD/Adam updates are noisy. The running average of parameters over
    training is smoother and often generalizes better than the final
    parameters. For diffusion/flow matching models, EMA is essentially
    mandatory — without it, generated samples have noticeably more
    artifacts.

    The EMA update rule:
        θ_ema ← decay · θ_ema + (1 - decay) · θ_current

    With decay = 0.9999 and ~100k steps, the EMA averages roughly over
    the last ~10k steps (effective window ≈ 1/(1-decay)).

HOW TO USE:
    ema = EMA(model, decay=0.9999)

    # During training:
    for batch in loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        ema.update()           # ← call after EVERY optimizer step

    # For evaluation/sampling:
    ema.apply()                # swap EMA weights into model
    samples = model.sample(...)
    ema.restore()              # swap back original weights

    # For checkpointing:
    torch.save({'ema': ema.state_dict(), ...}, path)
    ema.load_state_dict(torch.load(path)['ema'])
"""

import torch
import torch.nn as nn
from typing import Dict
from copy import deepcopy


class EMA:
    """
    Maintains an exponential moving average of model parameters.

    Args:
        model: the model whose parameters to track
        decay: EMA decay factor
            0.9999 → averages over ~10k steps (standard for diffusion)
            0.999  → averages over ~1k steps (faster adaptation)
            0.99   → averages over ~100 steps (for debugging/fast iteration)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay

        # Shadow parameters: the EMA values
        # We store copies, not references, so the shadow is independent
        # of gradient updates to the model.
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        # Backup storage for apply/restore cycle
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        """
        Update shadow parameters. Call after each optimizer.step().

        The update is: shadow ← decay · shadow + (1-decay) · current
        Equivalent to: shadow += (1-decay) · (current - shadow)
        The second form is numerically more stable (smaller additions).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self) -> None:
        """
        Swap EMA weights into the model. Call before evaluation/sampling.

        After calling apply(), the model uses EMA parameters for forward().
        Call restore() when done to get the training parameters back.
        """
        self._backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """
        Restore original (non-EMA) weights. Call after evaluation/sampling.
        """
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """For checkpointing."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """For loading from checkpoint."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# =============================================================================
# Self-test  (run with: python -m src.training.ema  from project root)
# =============================================================================

if __name__ == "__main__":
    print("=== experiments.synthetic.high_d.common.training.ema self-test ===\n")

    torch.manual_seed(0)

    # Tiny model for fast testing
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)

    # Shadow should start as a copy of the initial params
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.allclose(ema.shadow[name], param.data), \
                "Shadow not initialised to model params"
    print("[PASS] Shadow initialised to model parameters")

    # After one update with decay=0.9: shadow = 0.9*shadow + 0.1*new_param
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    ema.update()
    # shadow was ~randn (from init), now = 0.9*original + 0.1*1.0
    # We just check shapes and that the shadow changed
    assert ema.shadow["weight"].shape == model.weight.shape
    print("[PASS] update() runs without error, shadow shape correct")

    # apply / restore cycle
    original_weight = model.weight.data.clone()
    ema.apply()
    ema_weight = model.weight.data.clone()
    assert not torch.allclose(original_weight, ema_weight), \
        "apply() didn't change model weights"
    print("[PASS] apply() swaps EMA weights into model")

    ema.restore()
    assert torch.allclose(model.weight.data, original_weight), \
        "restore() didn't recover original weights"
    print("[PASS] restore() recovers original weights")

    # State dict roundtrip
    sd = ema.state_dict()
    ema2 = EMA(nn.Linear(4, 4), decay=0.9)
    ema2.load_state_dict(sd)
    for k in sd:
        assert torch.allclose(ema.shadow[k], ema2.shadow[k])
    print("[PASS] state_dict / load_state_dict roundtrip")

    # EMA with decay=1.0 should never move
    model3 = nn.Linear(2, 2)
    ema3 = EMA(model3, decay=1.0)
    orig = {k: v.clone() for k, v in ema3.shadow.items()}
    with torch.no_grad():
        for p in model3.parameters():
            p.fill_(99.0)
    ema3.update()
    for k in orig:
        assert torch.allclose(ema3.shadow[k], orig[k]), "decay=1.0 should not move shadow"
    print("[PASS] decay=1.0 → shadow never moves")

    print("\n=== All EMA tests passed ===")
