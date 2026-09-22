"""Append full rotated-surface stabilizer rounds in Stim.

This module appends the repeated measurement schedule and returns the Z and X
measurement positions for every round. It does not add temporal detectors:
the correct reference for round 1 depends on what happened before these
rounds.

The d=5 lattice-surgery callers complete that boundary differently::

    append_post_transfer_stabilization(...)
    ├── append_rotated_full_stabilizer_rounds(...)  # this module
    └── compare round 1 with merge-round and m_ab results

    append_ideal_logical_verification(...)
    ├── append_rotated_full_stabilizer_rounds(...)  # this module
    └── compare the ideal round with the final noisy round
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import stim

CircuitAppender = Callable[..., None]


def append_rotated_full_stabilizer_rounds(
    circuit: stim.Circuit,
    *,
    surface_qubits: Sequence[int],
    z_ancillas: Sequence[int],
    x_ancillas: Sequence[int],
    z_stabilizer_names: Sequence[str],
    x_stabilizer_names: Sequence[str],
    stabilizer_rounds: int,
    append_scheduled_stabilizer_round: CircuitAppender,
    add_ticks: bool = True,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Append full rounds and return their Z/X measurement-position histories.

    This helper owns only the repeated rotated-patch schedule. It cannot append
    the first temporal detectors because their references depend on the
    preceding protocol: lattice surgery, code expansion, or another boundary.
    """
    if stabilizer_rounds < 0:
        raise ValueError("stabilizer_rounds must be nonnegative.")
    if stabilizer_rounds == 0:
        return (), ()

    first_round_measurement = circuit.num_measurements
    round_circuit = stim.Circuit()
    stabilizer_names = tuple(z_stabilizer_names) + tuple(x_stabilizer_names)
    ancillas = tuple(z_ancillas) + tuple(x_ancillas)
    append_scheduled_stabilizer_round(
        round_circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=stabilizer_names,
        ancillas=ancillas,
        round_label="post-transfer round",
        add_ticks=add_ticks,
    )
    if stabilizer_rounds == 1:
        circuit += round_circuit
    else:
        circuit.append(stim.CircuitRepeatBlock(stabilizer_rounds, round_circuit))

    z_measurement_rounds: list[tuple[int, ...]] = []
    x_measurement_rounds: list[tuple[int, ...]] = []
    z_round_width = len(z_stabilizer_names)
    round_width = z_round_width + len(x_stabilizer_names)
    for round_index in range(stabilizer_rounds):
        round_start = first_round_measurement + round_index * round_width
        z_measurement_rounds.append(tuple(range(round_start, round_start + z_round_width)))
        x_measurement_rounds.append(
            tuple(range(round_start + z_round_width, round_start + round_width))
        )
    return tuple(z_measurement_rounds), tuple(x_measurement_rounds)
