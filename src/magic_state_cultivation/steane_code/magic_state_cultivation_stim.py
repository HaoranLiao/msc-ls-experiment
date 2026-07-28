"""Stim port of Steane fold-measure-unfold magic-state cultivation.

This module parallels :mod:`magic_state_cultivation_qiskit`. The fold schedule,
patch sites, measurement count, and optional-CX filtering are shared with that
module. Qiskit's transversal ``T``/``T_DAG`` rotations are replaced by
``S``/``S_DAG``, so Stim projects onto the logical Pauli-Y axis of the
Clifford ``S``-state proxy instead of the non-Clifford ``H_XY`` axis.

Each round appends five measurements to Stim's record in this order:

1. folded root at site 7;
2. helper sites 1, 2, 6, and 7.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import stim

from magic_state_cultivation.steane_code.magic_state_cultivation_qiskit import (
    DATA_SITES,
    FOLD_LAYERS,
    FOLDED_MEASUREMENT_SITE,
    HELPER_SITES,
    filter_fold_layer,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import NUM_PATCH_QUBITS
from magic_state_cultivation.steane_code.state_injection_stim import (
    DEFAULT_STIM_INJECTED_STATE,
    DEFAULT_STIM_PATCH_QUBITS,
    StimInjectedStateName,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from magic_state_cultivation.utils.normalize import resolve_patch_qubits


def measure_in_x_basis(circuit: stim.Circuit, qubit: int) -> None:
    """Measure ``qubit`` in the X basis."""
    circuit.append("MX", [qubit])


def reset_to_plus(circuit: stim.Circuit, qubit: int) -> None:
    """Reset a qubit and prepare ``|+>``."""
    circuit.append("RX", [qubit])


def append_cnot_layer_from_site_labels(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    cnot_layer: Sequence[tuple[int, int]],
) -> None:
    """Append one CNOT layer using 11-site patch labels."""
    targets: list[int] = []
    for control_site, target_site in cnot_layer:
        targets.extend(
            (
                patch_qubits[control_site],
                patch_qubits[target_site],
            )
        )
    if targets:
        circuit.append("CX", targets)


def append_fold_network(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    *,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append the forward 4-3-2-1 fold network."""
    for layer in FOLD_LAYERS:
        append_cnot_layer_from_site_labels(
            circuit,
            patch_qubits,
            filter_fold_layer(
                layer,
                skip_cx_1_4=skip_cx_1_4,
                skip_cx=skip_cx,
            ),
        )
        if add_ticks:
            circuit.append("TICK", [])


def append_unfold_network(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    *,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append the inverse fold network in fully reversed order."""
    for layer in reversed(FOLD_LAYERS):
        filtered_layer = filter_fold_layer(
            layer,
            skip_cx_1_4=skip_cx_1_4,
            skip_cx=skip_cx,
        )
        append_cnot_layer_from_site_labels(
            circuit,
            patch_qubits,
            list(reversed(filtered_layer)),
        )
        if add_ticks:
            circuit.append("TICK", [])


def append_h_xy_projection_round(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_PATCH_QUBITS,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append one Clifford-proxy ``H_XY`` cultivation round.

    The stage order and fold network parallel Qiskit's
    :func:`append_h_xy_projection_round`. Stim replaces transversal ``T`` and
    ``T_DAG`` with ``S`` and ``S_DAG``, thereby projecting onto logical Y.

    Parameters
    ----------
    circuit
        Stim circuit being modified.
    patch_qubits
        Exactly eleven qubit indices ordered by the patch site labels.
    add_ticks
        Whether to insert ``TICK`` separators between conceptual stages.
    skip_cx_1_4
        Whether to omit ``CX 1 -> 4`` from both fold and unfold.
    skip_cx
        Optional ``(control_site, target_site)`` gate to omit. This takes
        precedence over ``skip_cx_1_4``.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_PATCH_QUBITS,
    )

    for helper_site in HELPER_SITES:
        reset_to_plus(circuit, patch_qubits[helper_site])
    if add_ticks:
        circuit.append("TICK", [])

    circuit.append("S", [patch_qubits[site] for site in DATA_SITES])
    if add_ticks:
        circuit.append("TICK", [])

    append_fold_network(
        circuit,
        patch_qubits,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    measure_in_x_basis(circuit, patch_qubits[FOLDED_MEASUREMENT_SITE])
    if add_ticks:
        circuit.append("TICK", [])

    reset_to_plus(circuit, patch_qubits[FOLDED_MEASUREMENT_SITE])
    if add_ticks:
        circuit.append("TICK", [])

    append_unfold_network(
        circuit,
        patch_qubits,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    for helper_site in HELPER_SITES:
        measure_in_x_basis(circuit, patch_qubits[helper_site])
    if add_ticks:
        circuit.append("TICK", [])

    circuit.append(
        "S_DAG",
        [patch_qubits[site] for site in DATA_SITES],
    )
    if add_ticks:
        circuit.append("TICK", [])


def append_steane_injection_and_cultivation(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    cultivation_rounds: int = 1,
    measure_encoder_helpers: bool = False,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append patch-Crumble injection and one or more cultivation rounds.

    Each cultivation round contributes
    ``CULTIVATION_MEASUREMENTS_PER_ROUND`` entries. When
    ``measure_encoder_helpers`` is true, four encoder-helper measurements
    precede them.
    """
    if cultivation_rounds < 1:
        raise ValueError("cultivation_rounds must be at least 1.")

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        measure_helpers=measure_encoder_helpers,
        add_ticks=False,
    )
    if add_ticks:
        circuit.append("TICK", [])

    for _ in range(cultivation_rounds):
        append_h_xy_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            add_ticks=add_ticks,
            skip_cx_1_4=skip_cx_1_4,
            skip_cx=skip_cx,
        )


def postselect_success_shots(
    shots: np.ndarray,
    cult_bit_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Return sampled shots whose selected postselection bits are all zero.

    If ``cult_bit_positions`` is omitted, every measurement column is included
    in the all-zero condition. Otherwise, only the selected zero-based Stim
    measurement-record columns are checked.
    """
    if shots.ndim != 2:
        raise ValueError("shots must be a two-dimensional array.")

    positions = (
        tuple(range(shots.shape[1])) if cult_bit_positions is None else tuple(cult_bit_positions)
    )
    if any(position < 0 or position >= shots.shape[1] for position in positions):
        raise ValueError("cult_bit_positions contains an out-of-range position.")
    if not positions:
        return shots.copy()

    return shots[~shots[:, positions].any(axis=1)]
