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
    append_patch_stabilizer_measurement,
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
    add_qed_round: bool = True,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append injection, optional QED, and zero or more cultivation rounds.

    By default, all three X and all three Z Steane stabilizers are measured
    after injection and before cultivation. The papers call this a QEC cycle,
    but this circuit uses its six outcomes only for error detection and
    postselection. When ``measure_encoder_helpers`` is true, the four encoder
    helper outcomes come first; the six QED outcomes come next; each
    cultivation round then contributes five outcomes. Set
    ``cultivation_rounds=0`` to stop after injection and optional QED.

    Parameters
    ----------
    circuit
        Caller-owned Stim circuit being modified.
    patch_qubits
        Eleven Stim qubit indices ordered by the patch site labels.
    injected_state
        Clifford state injected on the center data site before encoding.
    cultivation_rounds
        Number of Clifford-proxy cultivation rounds. Must be non-negative.
    measure_encoder_helpers
        Whether to record the four encoder-helper outcomes.
    add_qed_round
        Whether to append the six-outcome post-injection Steane QED round.
    add_ticks
        Whether to preserve the protocol's logical time layers with ``TICK``.
    skip_cx_1_4
        Whether to omit ``CX 1 -> 4`` from every fold/unfold network.
    skip_cx
        Optional ``(control_site, target_site)`` CX gate to omit.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        measure_helpers=measure_encoder_helpers,
        add_ticks=add_ticks,
    )
    if add_ticks:
        circuit.append("TICK", [])

    if add_qed_round:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            basis="both",
            add_ticks=add_ticks,
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


def build_steane_injection_and_cultivation_circuit(
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    *,
    cultivation_rounds: int = 1,
    add_qed_round: bool = True,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> stim.Circuit:
    """This is the authoritative Stim qubit-layout and appending example.

    This is the circuit-owning counterpart to
    :func:`append_steane_injection_and_cultivation`. Stim has no Qiskit-style
    quantum or classical registers: the builder uses qubit indices ``0..10``
    for the 11-site patch and appends measurements directly to Stim's record.

    Measurement-record layout
    -------------------------
    The first four entries are the state-injection encoder-helper
    measurements. When ``add_qed_round`` is true, the next six entries are the
    three X and three Z Steane stabilizer outcomes. Each cultivation round then
    appends five entries: the folded root measurement followed by the four
    cultivation-helper measurements.

    Parameters
    ----------
    injected_state
        Clifford state injected on the center data site before encoding.
    cultivation_rounds
        Number of Clifford-proxy cultivation rounds. Zero appends injection and
        optional QED without cultivation.
    add_qed_round
        Whether to append the six-outcome post-injection Steane QED round.
        Enabled by default.
    add_ticks
        Whether to insert ``TICK`` separators between conceptual stages.
    skip_cx_1_4
        Whether to omit ``CX 1 -> 4`` from every fold/unfold network.
    skip_cx
        Optional ``(control_site, target_site)`` CX gate to omit.

    Returns
    -------
    stim.Circuit
        New circuit containing injection and all requested cultivation rounds.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    # Quantum layout: Stim qubit indices 0..10 represent patch sites 0..10.
    circuit = stim.Circuit()

    # Measurement-record layout:
    # - 4 encoder-helper entries;
    # - 6 post-injection Steane QED entries, when enabled;
    # - 5 entries for each cultivation round.
    append_steane_injection_and_cultivation(
        circuit,
        patch_qubits=DEFAULT_STIM_PATCH_QUBITS,
        injected_state=injected_state,
        cultivation_rounds=cultivation_rounds,
        measure_encoder_helpers=True,
        add_qed_round=add_qed_round,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )
    return circuit


def postselect_success_shots(
    shots: np.ndarray,
    postselection_bit_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Return sampled shots whose selected error-detection bits are all zero.

    If ``postselection_bit_positions`` is omitted, every measurement column is
    included in the all-zero condition. Otherwise, only the selected zero-based
    Stim measurement-record columns are checked.
    """
    if shots.ndim != 2:
        raise ValueError("shots must be a two-dimensional array.")

    positions = (
        tuple(range(shots.shape[1]))
        if postselection_bit_positions is None
        else tuple(postselection_bit_positions)
    )
    if any(position < 0 or position >= shots.shape[1] for position in positions):
        raise ValueError("postselection_bit_positions contains an out-of-range position.")
    if not positions:
        return shots.copy()

    return shots[~shots[:, positions].any(axis=1)]
