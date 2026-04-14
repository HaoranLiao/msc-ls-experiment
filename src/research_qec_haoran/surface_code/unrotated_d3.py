"""Stim-native helpers for the distance-3 unrotated surface code."""

from __future__ import annotations

import stim


D3_UNROTATED_DISTANCE = 3
D3_UNROTATED_NUM_QUBITS = 25

_UNROTATED_MEMORY_TASK_BY_BASIS = {
    "Z": "surface_code:unrotated_memory_z",
    "X": "surface_code:unrotated_memory_x",
}


def generate_d3_unrotated_memory_circuit(rounds: int = 1, basis: str = "Z") -> stim.Circuit:
    """Return Stim's native d=3 unrotated memory circuit.

    ``basis='Z'`` prepares and stores a logical ``|0_L>`` memory experiment.
    ``basis='X'`` prepares and stores a logical ``|+_L>`` memory experiment.
    """

    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    basis = basis.upper()
    if basis not in _UNROTATED_MEMORY_TASK_BY_BASIS:
        raise ValueError("basis must be 'X' or 'Z'")

    return stim.Circuit.generated(
        _UNROTATED_MEMORY_TASK_BY_BASIS[basis],
        distance=D3_UNROTATED_DISTANCE,
        rounds=rounds,
    )
