"""Leak-probe holdout: deliberately truncated x where V_true should drop.

Plan rationale: hold out 10% of encounters, truncate x to ~30% of its
original length, and verify that any downstream estimator's V_true drops
on this subset. Failure to drop indicates the agent is not actually using
x and there is hidden leakage somewhere.
"""

from __future__ import annotations

import random
from copy import deepcopy

from ccema.data.schema import Encounter


def make_leak_probe_holdout(
    encounters: list[Encounter],
    holdout_frac: float = 0.10,
    truncate_to_frac: float = 0.30,
    seed: int = 42,
) -> tuple[list[Encounter], list[Encounter]]:
    """Split encounters into (main, leak_probe).

    leak_probe encounters have x truncated to `truncate_to_frac` of original
    character length; case_id gets a "_truncated" suffix.
    """
    rng = random.Random(seed)
    indices = list(range(len(encounters)))
    rng.shuffle(indices)

    n_holdout = max(1, int(round(holdout_frac * len(encounters))))
    holdout_idx = set(indices[:n_holdout])

    main: list[Encounter] = []
    leak_probe: list[Encounter] = []

    for i, enc in enumerate(encounters):
        if i in holdout_idx:
            truncated = deepcopy(enc)
            new_len = max(40, int(round(truncate_to_frac * len(enc.x))))
            truncated.x = enc.x[:new_len].rstrip()
            truncated.case_id = f"{enc.case_id}_truncated"
            truncated.metadata = {**enc.metadata, "leak_probe": True, "orig_len": len(enc.x)}
            leak_probe.append(truncated)
        else:
            main.append(enc)

    return main, leak_probe
