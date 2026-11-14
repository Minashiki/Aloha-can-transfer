"""Actor / critic fields and the three seed ranges.

The actor observation is the frozen image encoding plus proprioception.
The critic adds the privileged state. Those two sets are fixed here so a
test can check the names and the widths without constructing a network.
"""

from __future__ import annotations

from aloha_pp.robot import ARM_JOINTS

# Read by ``DiffusionPolicy.encode_actor_obs``. Anything else in the dict is ignored.
ACTOR_OBS_KEYS: tuple[str, ...] = (
    "agentview",
    "wrist_cam_left",
    "tcp_pose",
    "gripper_opening",
)

# Object pose and joint state are privileged. TCP pose and gripper opening are
# also in the actor feature (as normalized proprio); the critic gets the raw
# values as well, in this order.
PRIVILEGED_FIELDS: tuple[tuple[str, int], ...] = (
    ("can_pose", 7),
    ("can_linvel", 3),
    ("tcp_pose", 7),
    ("gripper_opening", 1),
    ("q", len(ARM_JOINTS)),
    ("qvel", len(ARM_JOINTS)),
)

# Names the actor must not see. TCP and gripper are proprioception, not object truth.
ACTOR_FORBIDDEN_FIELDS: tuple[str, ...] = ("can_pose", "can_linvel", "q", "qvel")

STAGE_NAMES: tuple[str, ...] = (
    "both_contact",
    "lifted",
    "above_cell",
    "released_in_cell",
)


def privileged_dim() -> int:
    return sum(width for _name, width in PRIVILEGED_FIELDS)


def pack_privileged(obs: dict, privileged: dict) -> "np.ndarray":
    """Float32 vector in ``PRIVILEGED_FIELDS`` order.

    ``obs`` is one ``get_obs`` dict (joint state and TCP). ``privileged`` is
    one ``get_privileged`` dict (can pose and linear velocity).
    """
    import numpy as np

    source = {}
    source.update(obs)
    source.update(privileged)
    parts = []
    for name, width in PRIVILEGED_FIELDS:
        if name not in source:
            raise KeyError(f"privileged state is missing {name}")
        value = np.asarray(source[name], dtype=np.float32).reshape(-1)
        if value.shape != (width,):
            raise ValueError(f"{name} has shape {value.shape}, expected ({width},)")
        parts.append(value)
    return np.concatenate(parts, axis=0)


class SeedStream:
    """Training episode seeds. Skips the validation and evaluation sets."""

    def __init__(self, start: int, banned: set[int]) -> None:
        self.next = int(start)
        self.banned = set(int(s) for s in banned)

    def take(self, n: int = 1) -> list[int]:
        if n < 1:
            raise ValueError("n must be positive")
        out: list[int] = []
        while len(out) < n:
            seed = self.next
            self.next += 1
            if seed not in self.banned:
                out.append(seed)
        return out


def eval_seed_set(start: int = 0, count: int = 50) -> set[int]:
    return set(range(int(start), int(start) + int(count)))


def val_seed_set(start: int, n_episodes: int) -> set[int]:
    return set(range(int(start), int(start) + int(n_episodes)))


def assert_seed_ranges(train_start: int, val_start: int, n_val: int, eval_start: int = 0, eval_count: int = 50) -> None:
    """The configured ranges are pairwise disjoint. The stream also skips collisions."""
    evaluation = eval_seed_set(eval_start, eval_count)
    validation = val_seed_set(val_start, n_val)
    if evaluation & validation:
        raise ValueError(f"eval and val seeds overlap: {sorted(evaluation & validation)[:8]}")
    if int(train_start) in evaluation or int(train_start) in validation:
        raise ValueError("train_seed_start lies in the val or eval set")
