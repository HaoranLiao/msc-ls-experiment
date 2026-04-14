"""Distance-3 unrotated surface-code circuit building blocks.

This module follows Stim's unrotated planar-code conventions for:

- qubit indexing on the full 5x5 grid
- logical-state preparation in the X or Z basis
- the four-substep stabilizer-measurement schedule

The helpers stay modular so notebooks and examples can compose:

- qubit layout
- initialization
- one or more stabilizer rounds
- either full data readout or logical-product readout
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


D3_UNROTATED_GRID_WIDTH = 5
D3_UNROTATED_NUM_DATA_QUBITS = len(D3_UNROTATED_DATA_COORDS)
D3_UNROTATED_NUM_X_ANCILLAS = len(D3_UNROTATED_X_STABILIZERS)
D3_UNROTATED_NUM_Z_ANCILLAS = len(D3_UNROTATED_Z_STABILIZERS)
D3_UNROTATED_NUM_QUBITS = D3_UNROTATED_GRID_WIDTH**2


def _grid_qubit(coord: tuple[int, int]) -> int:
    """Return Stim's row-major qubit index for a grid coordinate."""

    row, col = coord
    return col + D3_UNROTATED_GRID_WIDTH * row


D3_UNROTATED_DATA_QUBITS: tuple[int, ...] = tuple(_grid_qubit(coord) for coord in D3_UNROTATED_DATA_COORDS)
D3_UNROTATED_X_ANCILLAS: tuple[int, ...] = tuple(
    _grid_qubit(stabilizer.ancilla_coord) for stabilizer in D3_UNROTATED_X_STABILIZERS
)
D3_UNROTATED_Z_ANCILLAS: tuple[int, ...] = tuple(
    _grid_qubit(stabilizer.ancilla_coord) for stabilizer in D3_UNROTATED_Z_STABILIZERS
)
D3_UNROTATED_ANCILLA_QUBITS: tuple[int, ...] = tuple(
    sorted(D3_UNROTATED_X_ANCILLAS + D3_UNROTATED_Z_ANCILLAS)
)

_DATA_INDEX_BY_COORD = {coord: _grid_qubit(coord) for coord in D3_UNROTATED_DATA_COORDS}
_X_ANCILLA_BY_COORD = {
    stabilizer.ancilla_coord: _grid_qubit(stabilizer.ancilla_coord)
    for stabilizer in D3_UNROTATED_X_STABILIZERS
}
_Z_ANCILLA_BY_COORD = {
    stabilizer.ancilla_coord: _grid_qubit(stabilizer.ancilla_coord)
    for stabilizer in D3_UNROTATED_Z_STABILIZERS
}

D3_UNROTATED_LOGICAL_Z_QUBITS: tuple[int, ...] = tuple(
    _DATA_INDEX_BY_COORD[coord] for coord in D3_UNROTATED_LOGICAL_Z_COORDS
)
D3_UNROTATED_LOGICAL_X_QUBITS: tuple[int, ...] = tuple(
    _DATA_INDEX_BY_COORD[coord] for coord in D3_UNROTATED_LOGICAL_X_COORDS
)

_ANCILLA_LABEL_BY_QUBIT = {
    **{
        _grid_qubit(stabilizer.ancilla_coord): stabilizer.name
        for stabilizer in D3_UNROTATED_X_STABILIZERS
    },
    **{
        _grid_qubit(stabilizer.ancilla_coord): stabilizer.name
        for stabilizer in D3_UNROTATED_Z_STABILIZERS
    },
}

D3_UNROTATED_STABILIZER_MEASUREMENT_LABELS: tuple[str, ...] = tuple(
    _ANCILLA_LABEL_BY_QUBIT[qubit] for qubit in D3_UNROTATED_ANCILLA_QUBITS
)

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
    """Append the full 5x5 Stim qubit layout for the d=3 unrotated code."""

    for row in range(D3_UNROTATED_GRID_WIDTH):
        for col in range(D3_UNROTATED_GRID_WIDTH):
            circuit.append("QUBIT_COORDS", [_grid_qubit((row, col))], [float(col), float(row)])


def build_d3_unrotated_base_circuit() -> stim.Circuit:
    """Return a fresh circuit containing only the d=3 unrotated qubit layout."""

    circuit = stim.Circuit()
    append_d3_unrotated_qubit_coords(circuit)
    return circuit


def append_d3_unrotated_initial_reset(circuit: stim.Circuit, basis: str = "Z") -> None:
    """Append Stim-style initialization for a chosen data basis.

    ``basis='Z'`` matches ``surface_code:unrotated_memory_z``.
    ``basis='X'`` matches ``surface_code:unrotated_memory_x``.
    """

    basis = basis.upper()
    if basis == "Z":
        circuit.append("R", list(D3_UNROTATED_DATA_QUBITS) + list(D3_UNROTATED_ANCILLA_QUBITS))
        return
    if basis == "X":
        circuit.append("RX", list(D3_UNROTATED_DATA_QUBITS))
        circuit.append("R", list(D3_UNROTATED_ANCILLA_QUBITS))
        return
    raise ValueError("basis must be 'X' or 'Z'")


_SCHEDULE_DIRECTION_ORDER: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1),
)


def _schedule_key(stabilizer) -> tuple[int, int]:
    row, col = stabilizer.ancilla_coord
    return (col, row)


def _cx_targets_for_direction(
    stabilizers,
    ancilla_by_coord: dict[tuple[int, int], int],
    *,
    delta: tuple[int, int],
    ancilla_as_control: bool,
) -> list[int]:
    """Return the Stim CX targets for one directional substep."""

    dr, dc = delta
    targets: list[int] = []
    for stabilizer in sorted(stabilizers, key=_schedule_key):
        row, col = stabilizer.ancilla_coord
        neighbor = (row + dr, col + dc)
        if neighbor not in stabilizer.data_coords:
            continue
        ancilla = ancilla_by_coord[stabilizer.ancilla_coord]
        data = _DATA_INDEX_BY_COORD[neighbor]
        if ancilla_as_control:
            targets.extend((ancilla, data))
        else:
            targets.extend((data, ancilla))
    return targets


def append_d3_unrotated_stabilizer_round(circuit: stim.Circuit, *, reset_ancillas: bool) -> None:
    """Append one Stim-style stabilizer-measurement round to a circuit."""

    if reset_ancillas:
        circuit.append("R", list(D3_UNROTATED_ANCILLA_QUBITS))
    circuit.append("TICK")
    circuit.append("H", list(D3_UNROTATED_X_ANCILLAS))
    circuit.append("TICK")
    for delta in _SCHEDULE_DIRECTION_ORDER:
        targets = _cx_targets_for_direction(
            D3_UNROTATED_X_STABILIZERS,
            _X_ANCILLA_BY_COORD,
            delta=delta,
            ancilla_as_control=True,
        )
        targets.extend(
            _cx_targets_for_direction(
                D3_UNROTATED_Z_STABILIZERS,
                _Z_ANCILLA_BY_COORD,
                delta=delta,
                ancilla_as_control=False,
            )
        )
        circuit.append("CX", targets)
        circuit.append("TICK")
    circuit.append("H", list(D3_UNROTATED_X_ANCILLAS))
    circuit.append("TICK")
    circuit.append("MR", list(D3_UNROTATED_ANCILLA_QUBITS))


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
            reset_ancillas=reset_before_first_round and round_index == 0,
        )


def append_d3_unrotated_data_readout(circuit: stim.Circuit, basis: str) -> None:
    """Append Stim-style final data readout in the chosen basis."""

    basis = basis.upper()
    if basis == "Z":
        circuit.append("M", list(D3_UNROTATED_DATA_QUBITS))
        return
    if basis == "X":
        circuit.append("MX", list(D3_UNROTATED_DATA_QUBITS))
        return
    raise ValueError("basis must be 'X' or 'Z'")


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
