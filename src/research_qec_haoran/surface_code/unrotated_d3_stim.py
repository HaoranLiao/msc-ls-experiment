"""Stim-native helpers for the distance-3 unrotated surface code."""

from __future__ import annotations

import stim
from research_qec_haoran.decoding.memory_circuit_stim import (
    _MEMORY_TASK_BY_BASIS,
    normalized_basis,
)


def generate_d3_unrotated_memory_circuit(rounds: int = 1, basis: str = "Z") -> stim.Circuit:
    """Return Stim's native d=3 unrotated memory circuit (noiseless).

    ``basis='Z'`` prepares and stores a logical ``|0_L>`` memory experiment.
    ``basis='X'`` prepares and stores a logical ``|+_L>`` memory experiment.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    basis = normalized_basis(basis)
    return stim.Circuit.generated(
        _MEMORY_TASK_BY_BASIS[basis],
        distance=3,
        rounds=rounds,
    )
