"""Circuit-level utility helpers for Stim circuits."""

from __future__ import annotations

from collections.abc import Sequence

import stim

from magic_state_cultivation.utils.normalize import normalize_pauli_char


def logical_y_parity_is_inverted(support: tuple[int, ...]) -> bool:
    """Return whether measuring the physical Y string reverses logical Y."""
    if len(support) % 2 == 0:
        raise ValueError("Y logical support must have odd weight.")
    return len(support) % 4 == 3


def append_parity_measurement(
    circuit: stim.Circuit,
    basis: str,
    *,
    data_qubits: Sequence[int],
    ancilla_qubit: int,
    invert_result: bool = False,
) -> None:
    """Append a non-destructive Pauli-string parity measurement.

    The ancilla is assumed to start in ``|0>``. The measurement result is
    appended to Stim's measurement record. Set ``invert_result`` when the
    physical Pauli-string parity has the opposite sign from the represented
    logical observable.
    """
    normalized_basis = normalize_pauli_char(basis)

    if normalized_basis == "Z":
        for data_qubit in data_qubits:
            circuit.append("CX", [data_qubit, ancilla_qubit])
    elif normalized_basis in {"X", "Y"}:
        circuit.append("H", [ancilla_qubit])
        entangling_gate = "CX" if normalized_basis == "X" else "CY"
        for data_qubit in data_qubits:
            circuit.append(entangling_gate, [ancilla_qubit, data_qubit])
        circuit.append("H", [ancilla_qubit])

    measurement_target: int | stim.GateTarget = ancilla_qubit
    if invert_result:
        measurement_target = stim.target_inv(measurement_target)
    circuit.append("M", [measurement_target])


def append_logical_measurement_on_indexed_qubits(
    circuit: stim.Circuit,
    basis: str,
    *,
    indexed_qubits: Sequence[int],
    default_logical_x: tuple[int, ...],
    default_logical_z: tuple[int, ...],
    support: tuple[int, ...] | None,
    ancilla_qubit: int | None,
    destructive: bool,
) -> None:
    """Append a logical Pauli measurement to an existing Stim circuit.

    ``support`` contains indices into ``indexed_qubits``. A destructive
    measurement appends one result per support qubit; a non-destructive
    measurement appends one parity result from ``ancilla_qubit``. Logical-Y
    outcomes are sign-corrected so parity zero denotes the logical ``+1``
    eigenvalue.
    """
    normalized_basis = normalize_pauli_char(basis)
    if normalized_basis == "Y" and set(default_logical_x) != set(default_logical_z):
        raise ValueError("logical Y requires common logical X and Z support.")

    qubits = list(indexed_qubits)
    selected_support = support
    if selected_support is None:
        selected_support = default_logical_z if normalized_basis == "Z" else default_logical_x

    physical_support = [qubits[index] for index in selected_support]
    invert_y_parity = normalized_basis == "Y" and logical_y_parity_is_inverted(selected_support)

    if destructive:
        targets: list[int | stim.GateTarget] = list(physical_support)
        if invert_y_parity:
            targets[0] = stim.target_inv(targets[0])
        circuit.append(
            {"X": "MX", "Y": "MY", "Z": "M"}[normalized_basis],
            targets,
        )
        return

    if ancilla_qubit is None:
        raise ValueError("ancilla_qubit is required for non-destructive measurement.")

    append_parity_measurement(
        circuit,
        normalized_basis,
        data_qubits=physical_support,
        ancilla_qubit=ancilla_qubit,
        invert_result=invert_y_parity,
    )
