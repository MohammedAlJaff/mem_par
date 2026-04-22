"""
Loss probes for checkpoint evaluation.

These helpers evaluate model loss on the saved held-out/reference tensor at the
same checkpoint steps used for memorization metrics. They report two kinds of
loss:

1. random-time loss, which follows the model's usual training-time sampling;
2. fixed-time losses, which hold `t` fixed so checkpoints can be compared at
   the same noise levels.
"""

import torch

from mp_runtime import autocast_context


DEFAULT_LOSS_EVAL_TIMES = (0.5, 0.75)


def time_key(value):
    """
    Convert a scalar time into a compact metric suffix, e.g. 0.5 -> 0_5.
    """
    return str(float(value)).replace(".", "_")


def loss_eval_times(config, fixed_times=None):
    """
    Resolve fixed-t loss probes from an explicit override or config defaults.
    """
    if fixed_times is not None:
        values = fixed_times
    else:
        values = getattr(
            getattr(config, "evaluation", None),
            "loss_eval_times",
            DEFAULT_LOSS_EVAL_TIMES,
        )
    return tuple(float(value) for value in values)


def _loss_eval_batch_size(config, eval_tensor):
    """Resolve the batch size used for loss evaluation."""
    evaluation = getattr(config, "evaluation", None)
    requested = int(getattr(evaluation, "loss_eval_batch_size", 0) or 0)
    if requested <= 0:
        requested = int(getattr(evaluation, "batch_size", eval_tensor.shape[0]))
    return max(1, min(requested, int(eval_tensor.shape[0])))


def _fork_rng_devices_for(device):
    """Return CUDA devices that need RNG isolation for loss evaluation."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return []
    if torch_device.index is None:
        return [torch.cuda.current_device()]
    return [int(torch_device.index)]


def collect_loss_eval_record(
    step,
    model,
    eval_tensor,
    config,
    device,
    *,
    fixed_times=None,
):
    """
    Evaluate random-t and fixed-t losses on a held-out reference tensor.

    The RNG is forked and seeded from run seed plus checkpoint step. That keeps
    loss probes reproducible without changing the caller's surrounding random
    state.
    """
    step = int(step)
    batch_size = _loss_eval_batch_size(config, eval_tensor)
    batch = eval_tensor[:batch_size].to(device)
    probes = loss_eval_times(config, fixed_times=fixed_times)

    seed = int(getattr(config.training, "seed", 0))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            with torch.random.fork_rng(
                devices=_fork_rng_devices_for(batch.device),
                enabled=True,
            ):
                torch.manual_seed(seed + step + 1009)
                with autocast_context(config, batch.device):
                    random_t_loss = float(
                        model.compute_loss(batch).detach().cpu().item()
                    )

                fixed = {}
                for value in probes:
                    t_value = float(value)
                    torch.manual_seed(seed + step + int(round(t_value * 10000)) + 2027)
                    eps = torch.randn_like(batch)
                    t = torch.full((batch.shape[0],), t_value, device=batch.device)
                    with autocast_context(config, batch.device):
                        loss = model.compute_loss(batch, eps=eps, t=t)
                    fixed[time_key(t_value)] = float(loss.detach().cpu().item())
    finally:
        model.train(was_training)

    fixed_values = list(fixed.values())
    record = {
        "step": step,
        "batch_size": int(batch_size),
        "random_t_loss": random_t_loss,
        "fixed_t_loss_mean": float(sum(fixed_values) / max(1, len(fixed_values))),
        "fixed_t": fixed,
    }
    for key, value in fixed.items():
        record[f"loss_t_{key}"] = value
    return record


__all__ = [
    "DEFAULT_LOSS_EVAL_TIMES",
    "collect_loss_eval_record",
    "loss_eval_times",
    "time_key",
]


if __name__ == "__main__":
    print("mp_metrics.loss_eval example")
    print("default fixed times:", DEFAULT_LOSS_EVAL_TIMES)
    print("time key for 0.5:", time_key(0.5))
