"""Distance-3 unrotated surface-code circuit building blocks.

This module exposes reusable pieces for three general tasks:

- setting up the fixed data/ancilla layout
- appending one or more ancilla-based stabilizer-measurement rounds
- appending a final logical ``X_L`` or ``Z_L`` readout

These helpers are meant to be composed by notebooks, examples, or later
protocol code instead of hiding the full workflow inside one monolithic
builder.
"""

from __future__ import annotations

import stim

from research_qec_haoran.surface_code.unrotated_d3_layout import (
    D3_UNROTATED_DATA_COORDS,
    D3_UNROTATED_LOGICAL_X_COORDS,
    D3_UNROTATED_LOGICAL_Z_COORDS,
    D3_UNROTATED_X_STABILIZERS,
    D3_UNROTATED_Z_STABILIZERS,
)


D3_UNROTATED_NUM_DATA_QUBITS = len(D3_UNROTATED_DATA_COORDS)
D3_UNROTATED_NUM_X_ANCILLAS = len(D3_UNROTATED_X_STABILIZERS)
D3_UNROTATED_NUM_Z_ANCILLAS = len(D3_UNROTATED_Z_STABILIZERS)
D3_UNROTATED_NUM_QUBITS = (
    D3_UNROTATED_NUM_DATA_QUBITS + D3_UNROTATED_NUM_X_ANCILLAS + D3_UNROTATED_NUM_Z_ANCILLAS
)

D3_UNROTATED_DATA_QUBITS: tuple[int, ...] = tuple(range(D3_UNROTATED_NUM_DATA_QUBITS))
D3_UNROTATED_X_ANCILLAS: tuple[int, ...] = tuple(
    range(
        D3_UNROTATED_NUM_DATA_QUBITS,
        D3_UNROTATED_NUM_DATA_QUBITS + D3_UNROTATED_NUM_X_ANCILLAS,
    )
)
D3_UNROTATED_Z_ANCILLAS: tuple[int, ...] = tuple(
    range(
        D3_UNROTATED_NUM_DATA_QUBITS + D3_UNROTATED_NUM_X_ANCILLAS,
        D3_UNROTATED_NUM_QUBITS,
    )
)

_DATA_INDEX_BY_COORD = {coord: index for index, coord in enumerate(D3_UNROTATED_DATA_COORDS)}
_X_ANCILLA_BY_COORD = {
    stabilizer.ancilla_coord: ancilla
    for stabilizer, ancilla in zip(D3_UNROTATED_X_STABILIZERS, D3_UNROTATED_X_ANCILLAS)
}
_Z_ANCILLA_BY_COORD = {
    stabilizer.ancilla_coord: ancilla
    for stabilizer, ancilla in zip(D3_UNROTATED_Z_STABILIZERS, D3_UNROTATED_Z_ANCILLAS)
}

D3_UNROTATED_LOGICAL_Z_QUBITS: tuple[int, ...] = tuple(
    _DATA_INDEX_BY_COORD[coord] for coord in D3_UNROTATED_LOGICAL_Z_COORDS
)
D3_UNROTATED_LOGICAL_X_QUBITS: tuple[int, ...] = tuple(
    _DATA_INDEX_BY_COORD[coord] for coord in D3_UNROTATED_LOGICAL_X_COORDS
)

D3_UNROTATED_STABILIZER_MEASUREMENT_LABELS: tuple[str, ...] = tuple(
    stabilizer.name for stabilizer in D3_UNROTATED_Z_STABILIZERS
) + tuple(stabilizer.name for stabilizer in D3_UNROTATED_X_STABILIZERS)

# Backward-compatible alias for the stabilizer-only measurement order.
D3_UNROTATED_MEASUREMENT_LABELS = D3_UNROTATED_STABILIZER_MEASUREMENT_LABELS

D3_UNROTATED_LOGICAL_X_LABEL = "LX"
D3_UNROTATED_LOGICAL_Z_LABEL = "LZ"


def get_d3_unrotated_measurement_labels(logical_readout_basis: str | None = None) -> tuple[str, ...]:
    """Return measurement labels for the chosen circuit configuration."""

    if logical_readout_basis is None:
        return D3_UNROTATED_STABILIZER_MEASUREMENT_LABELS

    basis = logical_readout_basis.upper()
    if basis == "Z":
        return D3_UNROTATED_MEASUREMENT_LABELS + (D3_UNROTATED_LOGICAL_Z_LABEL,)
    if basis == "X":
        return D3_UNROTATED_MEASUREMENT_LABELS + (D3_UNROTATED_LOGICAL_X_LABEL,)
    raise ValueError("logical_readout_basis must be None, 'Z', or 'X'")


def _mpp_targets(pauli: str, qubits: tuple[int, ...]) -> list[stim.GateTarget]:
    """Return combined MPP targets for a multi-qubit Pauli product."""

    pauli = pauli.upper()
    if pauli == "Z":
        target_factory = stim.target_z
    elif pauli == "X":
        target_factory = stim.target_x
    else:
        raise ValueError("pauli must be 'X' or 'Z'")

    targets: list[stim.GateTarget] = []
    for index, qubit in enumerate(qubits):
        if index > 0:
            targets.append(stim.target_combiner())
        targets.append(target_factory(qubit))
    return targets


def append_d3_unrotated_qubit_coords(circuit: stim.Circuit) -> None:
    """Append the fixed d=3 unrotated data/ancilla layout to a circuit."""

    for coord, qubit in zip(D3_UNROTATED_DATA_COORDS, D3_UNROTATED_DATA_QUBITS):
        circuit.append("QUBIT_COORDS", [qubit], [float(coord[1]), float(coord[0])])

    for coord, qubit in _X_ANCILLA_BY_COORD.items():
        circuit.append("QUBIT_COORDS", [qubit], [float(coord[1]), float(coord[0])])

    for coord, qubit in _Z_ANCILLA_BY_COORD.items():
        circuit.append("QUBIT_COORDS", [qubit], [float(coord[1]), float(coord[0])])


def build_d3_unrotated_base_circuit() -> stim.Circuit:
    """Return a fresh circuit containing only the d=3 unrotated qubit layout."""

    circuit = stim.Circuit()
    append_d3_unrotated_qubit_coords(circuit)
    return circuit


def append_d3_unrotated_stabilizer_round(circuit: stim.Circuit, *, reset_ancillas: bool) -> None:
    """Append one ancilla-based stabilizer-measurement round to a circuit."""

    if reset_ancillas:
        circuit.append("R", list(D3_UNROTATED_Z_ANCILLAS))
    circuit.append("TICK")
    for stabilizer, ancilla in zip(D3_UNROTATED_Z_STABILIZERS, D3_UNROTATED_Z_ANCILLAS):
        for data_coord in stabilizer.data_coords:
            circuit.append("CX", [_DATA_INDEX_BY_COORD[data_coord], ancilla])
    circuit.append("TICK")
    circuit.append("M", list(D3_UNROTATED_Z_ANCILLAS))

    circuit.append("TICK")

    if reset_ancillas:
        circuit.append("R", list(D3_UNROTATED_X_ANCILLAS))
    circuit.append("H", list(D3_UNROTATED_X_ANCILLAS))
    circuit.append("TICK")
    for stabilizer, ancilla in zip(D3_UNROTATED_X_STABILIZERS, D3_UNROTATED_X_ANCILLAS):
        for data_coord in stabilizer.data_coords:
            circuit.append("CX", [ancilla, _DATA_INDEX_BY_COORD[data_coord]])
    circuit.append("TICK")
    circuit.append("H", list(D3_UNROTATED_X_ANCILLAS))
    circuit.append("M", list(D3_UNROTATED_X_ANCILLAS))


def append_d3_unrotated_stabilizer_rounds(
    circuit: stim.Circuit,
    rounds: int = 1,
    *,
    reset_before_first_round: bool = False,
) -> None:
    """Append one or more standard ancilla-based stabilizer rounds.

    This helper is reusable outside initialization. For example, if the same
    ancillas were already used earlier in a protocol, set
    ``reset_before_first_round=True`` before stabilizing again.
    """

    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    for round_index in range(rounds):
        if round_index > 0:
            circuit.append("TICK")
        append_d3_unrotated_stabilizer_round(
            circuit,
            reset_ancillas=reset_before_first_round or round_index > 0,
        )


def append_d3_unrotated_logical_readout(circuit: stim.Circuit, basis: str) -> None:
    """Append a final logical ``X_L`` or ``Z_L`` readout to a circuit."""

    basis = basis.upper()
    if basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'")

    circuit.append("TICK")
    if basis == "Z":
        circuit.append("MPP", _mpp_targets("Z", D3_UNROTATED_LOGICAL_Z_QUBITS))
    else:
        circuit.append("MPP", _mpp_targets("X", D3_UNROTATED_LOGICAL_X_QUBITS))
