"""
Checkpoint schedules for training runs.

The training loop needs a concrete set of gradient steps where checkpoints are
saved. Most runs use a simple interval through `ckpt_every`. Long memorization
runs can instead use the Bonnaire-style log-spaced schedule: dense early,
coarser later, and capped at one million optimizer steps.
"""

import math


CHECKPOINT_SCHEDULE_CHOICES = ("interval", "bonnaire-1m")
BONNAIRE_1M_MAX_STEPS = 1_000_000


def parse_checkpoint_steps(value):
    """
    Parse a user-provided checkpoint-step list.

    The CLI form is a comma-separated string such as `"0,250,500"`. Programmatic
    callers may pass a list or tuple of integers. An empty value means no
    explicit list was requested.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        parts = [part.strip() for part in text.split(",")]
        if any(part == "" for part in parts):
            raise ValueError("--ckpt-steps must be a comma-separated list of integers.")
        return tuple(_parse_single_step(part) for part in parts)
    return tuple(_parse_single_step(item) for item in value)


def bonnaire_1m_checkpoint_steps(n_steps=BONNAIRE_1M_MAX_STEPS):
    """
    Return the Bonnaire-style checkpoint schedule clipped to `n_steps`.

    Phase 1 covers 251 to 10,000 with 10 log-spaced points snapped down to a
    multiple of 250. Phase 2 covers 10,000 to 1,000,000 with 90 log-spaced
    points snapped down to a multiple of 5,000.
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError(f"training.n_training_steps must be >= 1, got {n_steps}")
    if n_steps > BONNAIRE_1M_MAX_STEPS:
        raise ValueError(
            "bonnaire-1m checkpoint schedule requires training.n_training_steps <= 1000000."
        )

    phase1 = _snapped_logspace(math.log10(251), 4.0, 10, 250)
    phase2 = _snapped_logspace(4.0, 6.0, 90, 5000)
    steps = {0, n_steps}
    for step in phase1 + phase2:
        if 0 <= step <= n_steps:
            steps.add(step)
    return tuple(sorted(steps))


def resolve_checkpoint_steps(config):
    """
    Resolve the concrete checkpoint steps for a training config.

    Priority is explicit `training.ckpt_steps`, then named schedule, then the
    interval schedule controlled by `training.ckpt_every`. Step 0 and final
    step are always included.
    """
    training = config.training
    n_steps = int(training.n_training_steps)
    if n_steps < 1:
        raise ValueError(f"training.n_training_steps must be >= 1, got {n_steps}")

    explicit_steps = parse_checkpoint_steps(getattr(training, "ckpt_steps", ()))
    if explicit_steps:
        return _finalize_steps(explicit_steps, n_steps)

    schedule = str(getattr(training, "ckpt_schedule", "interval")).strip()
    if schedule == "bonnaire-1m":
        return bonnaire_1m_checkpoint_steps(n_steps)
    if schedule != "interval":
        raise ValueError(
            f"Unknown checkpoint schedule {schedule!r}. "
            f"Expected one of {CHECKPOINT_SCHEDULE_CHOICES}."
        )

    ckpt_every = int(training.ckpt_every)
    if ckpt_every < 1:
        raise ValueError(f"training.ckpt_every must be >= 1, got {ckpt_every}")
    return _finalize_steps(range(0, n_steps + 1, ckpt_every), n_steps)


def checkpoint_schedule_metadata(config, checkpoint_steps=None):
    """Return the schedule fields saved in status and run summary."""
    if checkpoint_steps is None:
        checkpoint_steps = resolve_checkpoint_steps(config)
    return {
        "ckpt_schedule": str(getattr(config.training, "ckpt_schedule", "interval")),
        "ckpt_every": int(getattr(config.training, "ckpt_every", 0)),
        "ckpt_steps": list(parse_checkpoint_steps(getattr(config.training, "ckpt_steps", ()))),
        "resolved_ckpt_steps": [int(step) for step in checkpoint_steps],
        "num_checkpoints": int(len(checkpoint_steps)),
    }


def _parse_single_step(value):
    """Parse one checkpoint step and reject non-integer values."""
    if isinstance(value, bool):
        raise ValueError("checkpoint steps must be integers, not booleans.")
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid checkpoint step {value!r}; expected an integer.") from exc


def _finalize_steps(steps, n_steps):
    """Validate, sort, deduplicate, and add required endpoints."""
    resolved = {0, int(n_steps)}
    for step in steps:
        step = int(step)
        if step < 0 or step > int(n_steps):
            raise ValueError(
                f"checkpoint step {step} is outside valid range [0, {int(n_steps)}]."
            )
        resolved.add(step)
    return tuple(sorted(resolved))


def _snapped_logspace(start_power, end_power, num, modulus):
    """Return log-spaced values snapped down to multiples of `modulus`."""
    if num < 1:
        return []
    if num == 1:
        values = [10.0**start_power]
    else:
        values = [
            10.0 ** (start_power + index * (end_power - start_power) / (num - 1))
            for index in range(num)
        ]
    snapped = []
    for value in values:
        snapped.append(int(value - (value % int(modulus))))
    return sorted(set(snapped))


__all__ = [
    "BONNAIRE_1M_MAX_STEPS",
    "CHECKPOINT_SCHEDULE_CHOICES",
    "bonnaire_1m_checkpoint_steps",
    "checkpoint_schedule_metadata",
    "parse_checkpoint_steps",
    "resolve_checkpoint_steps",
]


if __name__ == "__main__":
    steps = bonnaire_1m_checkpoint_steps(1_000_000)
    print("mp_training.checkpoint_schedule example")
    print("schedule:", "bonnaire-1m")
    print("num_checkpoints:", len(steps))
    print("first_steps:", list(steps[:10]))
    print("last_steps:", list(steps[-5:]))
