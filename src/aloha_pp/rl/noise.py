"""Initial-noise sources for the three BC comparisons.

``bc_zero`` is the zero tensor, the point an untrained tanh actor maps its
mean to. ``bc_seeded`` draws ``N(0, I)`` from a generator built from the
evaluation seed, one chunk at a time, so a rerun of that seed repeats the
sequence. Neither path changes ``DiffusionPolicy.sample(noise=None)``.
"""

from __future__ import annotations

import numpy as np


# Mixed into SeedSequence so an eval seed is not the same stream as anything
# else that seeds numpy with that integer.
SEEDED_STREAM = 0xB7C0


def zero_noise(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def seeded_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), SEEDED_STREAM]))


def clip_standard_normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    """``N(0, I)`` clipped to ``[-scale, scale]``, the DSRL warmup action.

    The official collector clips the standard normal to ``action_magnitude``
    before storing it. Values outside the tanh range cannot be actions.
    """
    noise = rng.standard_normal(shape).astype(np.float32)
    np.clip(noise, -float(scale), float(scale), out=noise)
    return noise
