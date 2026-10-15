"""Dependency-light learning-rate schedule formulas."""

import math


def warmup_cosine_factor(step, total_steps, warmup_steps, min_factor):
    """Return an LR multiplier that warms to 1 and decays to ``min_factor``."""
    if total_steps <= 0:
        raise ValueError(f'total_steps must be positive, got {total_steps}')
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            f'warmup_steps must satisfy 0 <= warmup_steps < total_steps, got '
            f'{warmup_steps} for total_steps={total_steps}')
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError(f'min_factor must be in [0, 1], got {min_factor}')

    step = min(max(step, 0), total_steps)
    if warmup_steps > 0 and step <= warmup_steps:
        progress = step / warmup_steps
        return min_factor + (1.0 - min_factor) * progress

    cosine = 0.5 * (
        1.0 + math.cos(
            (step - warmup_steps) / (total_steps - warmup_steps) * math.pi
        )
    )
    return min_factor + (1.0 - min_factor) * cosine
