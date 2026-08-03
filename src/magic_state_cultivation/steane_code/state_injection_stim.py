"""Stim port of the Steane state-injection patch-Crumble encoder.

This is the Clifford-only Stim model of the 11-site patch state injection. The
non-Clifford ``T`` injection used on hardware is replaced by its Clifford proxy
``S``: the encoder prepares ``S|+> = |i>`` (a Pauli-``Y`` eigenstate) as the
stand-in for ``T|+> = |T>``. Stim analysis reports the native ``S``-state
logical error rate without converting it to a ``T``-state estimate.

The public helpers intentionally have the same names as their Qiskit
counterparts. The module name identifies the framework at import time. Shared
encoder schedules, patch labels, logical supports, and validation helpers are
imported from :mod:`state_injection_qiskit` so the two implementations have one
source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import stim

from magic_state_cultivation.steane_code.state_injection_qiskit import (
    NUM_PATCH_QUBITS,
    PATCH_CRUMBLE_ENCODER_LAYERS,
    PATCH_CRUMBLE_INJECTION_SITE,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    PATCH_HELPER_SITES,
    PATCH_STABILIZER_ANCILLA_SITES,
    PATCH_X_STABILIZERS,
    PATCH_Z_STABILIZERS,
    PatchStabilizerBasis,
    resolve_patch_stabilizer_indices,
)
from magic_state_cultivation.utils.circuit_stim import (
    append_logical_measurement_on_indexed_qubits,
    append_parity_measurement,
)
from magic_state_cultivation.utils.normalize import resolve_patch_qubits

# Clifford single-qubit states injectable in Stim.  This is the Clifford subset
# of the Qiskit ``InjectedStateName``; the non-Clifford ``"T"``/``"Tdg"``/``"Ty"``
# magic states cannot be simulated by Stim.  ``"S"`` is the Clifford proxy used
# in place of ``"T"``.
StimInjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "Sdg"]

DEFAULT_STIM_INJECTED_STATE: StimInjectedStateName = "S"
DEFAULT_STIM_PATCH_QUBITS: tuple[int, ...] = tuple(range(NUM_PATCH_QUBITS))


def prepare_physical_state(
    circuit: stim.Circuit,
    qubit: int,
    state: StimInjectedStateName,
) -> None:
    """Prepare a Clifford single-qubit ``state`` on ``qubit`` (assumed ``|0>``).

    Mirrors ``prepare_physical_state`` restricted to Clifford states. ``"S"``
    (equivalently ``"i"``/``"Y+"``) prepares ``S|+> = |i>``, the Clifford proxy
    for the ``T`` magic state.
    """
    if state == "0":
        return
    if state == "1":
        circuit.append("X", [qubit])
        return
    if state == "+":
        circuit.append("H", [qubit])
        return
    if state == "-":
        circuit.append("X", [qubit])
        circuit.append("H", [qubit])
        return
    if state in ("i", "Y+", "S"):
        # |i> = S|+> = +1 eigenstate of Pauli Y (Clifford proxy for |T>).
        circuit.append("H", [qubit])
        circuit.append("S", [qubit])
        return
    if state in ("-i", "Y-", "Sdg"):
        # |-i> = S_dag|+> = -1 eigenstate of Pauli Y.
        circuit.append("H", [qubit])
        circuit.append("S_DAG", [qubit])
        return
    raise ValueError(f"Unknown Clifford state name: {state!r}")


def append_patch_steane_state_injection_encoder_crumble_sequences(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    measure_flags: bool = False,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the Clifford (Stim) patch-Crumble Steane injection encoder.

    Mirrors ``append_patch_steane_state_injection_encoder_crumble_sequences``.
    The injection site is prepared in ``injected_state`` and the Rosenfeld
    Crumble ``H``/``CZ`` layers (imported from the Qiskit module) then entangle
    the 11-site patch into the encoded state.

    Parameters
    ----------
    circuit
        Stim circuit being appended to.
    patch_qubits
        Eleven Stim qubit indices ordered by the 11-site patch labels. Defaults
        to ``range(11)``.
    injected_state
        Clifford state to inject on patch site 4 (the center data site).
        Defaults to ``"S"``, the proxy for the ``T`` magic state.
    measure_flags
        Whether to measure helper sites 1, 2, 6, and 7 into the Stim
        measurement record after encoding.
    add_ticks
        Whether to insert ``TICK`` layer separators (mirrors the Qiskit
        ``add_barriers`` argument).

    Returns
    -------
    stim.Circuit
        The same circuit object, after appending the encoder.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_PATCH_QUBITS,
    )

    prepare_physical_state(
        circuit,
        patch_qubits[PATCH_CRUMBLE_INJECTION_SITE],
        injected_state,
    )
    if add_ticks:
        circuit.append("TICK", [])

    for gate_name, operands in PATCH_CRUMBLE_ENCODER_LAYERS:
        if gate_name == "h":
            circuit.append("H", [patch_qubits[site] for site in operands])
        elif gate_name == "cz":
            targets: list[int] = []
            for site_a, site_b in operands:
                targets.extend((patch_qubits[site_a], patch_qubits[site_b]))
            circuit.append("CZ", targets)
        else:  # pragma: no cover - defensive if the shared schedule is edited.
            raise ValueError(f"Unsupported Crumble injection gate: {gate_name!r}")

        if add_ticks:
            circuit.append("TICK", [])

    if measure_flags:
        circuit.append(
            "M",
            [patch_qubits[site] for site in PATCH_HELPER_SITES],
        )

    return circuit


def append_patch_stabilizer_measurement(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_PATCH_QUBITS,
    basis: PatchStabilizerBasis = "both",
    stabilizer_indices: Sequence[int] | None = None,
    add_ticks: bool = True,
) -> None:
    """Append non-destructive Stim stabilizer measurements on the 11-site patch.

    Mirrors ``append_patch_stabilizer_measurement``: the helper sites (2, 6, 7)
    act as ancillas measuring the X- and/or Z-type stabilizers. Measurements are
    appended to the record in order X0, X1, X2 then Z0, Z1, Z2 (three per basis).

    Parameters
    ----------
    circuit
        Stim circuit being appended to.
    patch_qubits
        Eleven Stim qubit indices ordered by the 11-site patch labels. Defaults
        to ``range(11)``.
    basis
        Stabilizer basis to measure (``"x"``, ``"z"``, or ``"both"``).
    stabilizer_indices
        Ordered subset of the three canonical stabilizers to measure. ``None``
        measures all three.
    add_ticks
        Whether to insert a ``TICK`` between X and Z rounds in ``"both"``
        mode.
    """
    if basis not in ("x", "z", "both"):
        raise ValueError("basis must be 'x', 'z', or 'both'.")

    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_PATCH_QUBITS,
    )
    selected_indices = resolve_patch_stabilizer_indices(stabilizer_indices)
    anc = PATCH_STABILIZER_ANCILLA_SITES
    selected_ancilla_sites = tuple(anc[index] for index in selected_indices)

    if basis in ("x", "both"):
        circuit.append(
            "R",
            [patch_qubits[site] for site in selected_ancilla_sites],
        )
        for stabilizer_index in selected_indices:
            ancilla_site = anc[stabilizer_index]
            support = PATCH_X_STABILIZERS[stabilizer_index]
            append_parity_measurement(
                circuit,
                "X",
                data_qubits=[patch_qubits[site] for site in support],
                ancilla_qubit=patch_qubits[ancilla_site],
            )
        if basis == "both" and add_ticks:
            circuit.append("TICK", [])

    if basis in ("z", "both"):
        circuit.append(
            "R",
            [patch_qubits[site] for site in selected_ancilla_sites],
        )
        for stabilizer_index in selected_indices:
            ancilla_site = anc[stabilizer_index]
            support = PATCH_Z_STABILIZERS[stabilizer_index]
            append_parity_measurement(
                circuit,
                "Z",
                data_qubits=[patch_qubits[site] for site in support],
                ancilla_qubit=patch_qubits[ancilla_site],
            )


def append_patch_logical_measurement(
    circuit: stim.Circuit,
    basis: str = "Z",
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_PATCH_QUBITS,
    support: tuple[int, ...] | None = None,
    ancilla_qubit: int | None = None,
    destructive: bool = True,
) -> None:
    """Append a logical Pauli measurement on the 11-site patch.

    Measurement results are appended to Stim's measurement record. A
    destructive measurement appends one result per support qubit; a
    non-destructive measurement appends one parity result from ``ancilla_qubit``.
    As in the Qiskit implementation, a corrected weight-three logical-Y result
    uses ``0`` for the logical ``+1`` eigenvalue.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_PATCH_QUBITS,
    )
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=patch_qubits,
        default_logical_x=PATCH_DEFAULT_LOGICAL_X,
        default_logical_z=PATCH_DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        destructive=destructive,
    )
