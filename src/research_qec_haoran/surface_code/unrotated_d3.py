"""Distance-3 unrotated surface-code initialization circuits.

This module builds a simple circuit that prepares the distance-3 unrotated
surface code in the logical ``|0_L>`` sector by:

- leaving data qubits in their default ``|0>`` state
- measuring all ``Z`` stabilizers
- measuring all ``X`` stabilizers using ancillas

Important subtlety:

- The initial all-zero product state is already in the ``+1`` eigenspace of
  all ``Z`` stabilizers and of the logical ``Z_L`` operator.
- The ``X`` stabilizer outcomes are *not* deterministic on that product state.
  Measuring them projects into the code space but returns random signs.
- Therefore this circuit prepares logical ``|0_L>`` up to a known Pauli frame
  or a later corrective ``Z`` chain derived from the measured ``X`` syndrome.

This is the standard first-step initialization picture often used in surface
code discussions.
"""

from __future__ import annotations

import stim

from research_qec_haoran.surface_code.unrotated_d3_layout import (
    D3_UNROTATED_DATA_COORDS,
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

D3_UNROTATED_MEASUREMENT_LABELS: tuple[str, ...] = tuple(
    stabilizer.name for stabilizer in D3_UNROTATED_Z_STABILIZERS
) + tuple(stabilizer.name for stabilizer in D3_UNROTATED_X_STABILIZERS)


def build_d3_unrotated_logical_zero_init_circuit(rounds: int = 1) -> stim.Circuit:
    """Build a circuit for distance-3 unrotated surface-code ``|0_L>`` init.

    Args:
        rounds: Number of stabilizer-measurement rounds to perform.

    Returns:
        A ``stim.Circuit`` using:

        - 13 data qubits
        - 6 ``X``-stabilizer ancillas
        - 6 ``Z``-stabilizer ancillas

    Measurement order within each round:

    1. all ``Z``-stabilizer ancilla measurements, in the order given by
       ``D3_UNROTATED_Z_STABILIZERS``
    2. all ``X``-stabilizer ancilla measurements, in the order given by
       ``D3_UNROTATED_X_STABILIZERS``

    Notes:
        Data qubits are left in Stim's default ``|0>`` initialization. The
        first round therefore does not need explicit ancilla resets. Ancillas
        are only reset before later rounds, after they have already been used
        and measured once.
    """

    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    circuit = stim.Circuit()

    for coord, qubit in zip(D3_UNROTATED_DATA_COORDS, D3_UNROTATED_DATA_QUBITS):
        circuit.append("QUBIT_COORDS", [qubit], [float(coord[1]), float(coord[0])])

    for coord, qubit in _X_ANCILLA_BY_COORD.items():
        circuit.append("QUBIT_COORDS", [qubit], [float(coord[1]), float(coord[0])])

    for coord, qubit in _Z_ANCILLA_BY_COORD.items():
        circuit.append("QUBIT_COORDS", [qubit], [float(coord[1]), float(coord[0])])

    for round_index in range(rounds):
        if round_index > 0:
            circuit.append("TICK")

        if round_index > 0:
            circuit.append("R", list(D3_UNROTATED_Z_ANCILLAS))
        circuit.append("TICK")
        for stabilizer, ancilla in zip(D3_UNROTATED_Z_STABILIZERS, D3_UNROTATED_Z_ANCILLAS):
            for data_coord in stabilizer.data_coords:
                circuit.append("CX", [_DATA_INDEX_BY_COORD[data_coord], ancilla])
        circuit.append("TICK")
        circuit.append("M", list(D3_UNROTATED_Z_ANCILLAS))

        circuit.append("TICK")

        if round_index > 0:
            circuit.append("R", list(D3_UNROTATED_X_ANCILLAS))
        circuit.append("H", list(D3_UNROTATED_X_ANCILLAS))
        circuit.append("TICK")
        for stabilizer, ancilla in zip(D3_UNROTATED_X_STABILIZERS, D3_UNROTATED_X_ANCILLAS):
            for data_coord in stabilizer.data_coords:
                circuit.append("CX", [ancilla, _DATA_INDEX_BY_COORD[data_coord]])
        circuit.append("TICK")
        circuit.append("H", list(D3_UNROTATED_X_ANCILLAS))
        circuit.append("M", list(D3_UNROTATED_X_ANCILLAS))

    return circuit
