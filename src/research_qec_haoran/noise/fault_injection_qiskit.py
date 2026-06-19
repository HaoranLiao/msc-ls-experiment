"""Deterministic Pauli fault-injection helpers for Qiskit circuits.

Provides a single general-purpose function that appends physical Pauli gates
to named qubits on any code patch.  Works for rotated surface codes (any
distance) and the Steane 11-site patch alike — the caller supplies the
qubit-index mapping.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from qiskit import QuantumCircuit


def append_pauli_faults(
    circuit: QuantumCircuit,
    faults: Sequence[tuple[str, Hashable]],
    *,
    qubit_map: Mapping[Hashable, object] | Sequence[object] | None = None,
) -> None:
    """Append deterministic Pauli faults to qubits on any code patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    faults
        Sequence of ``(pauli, qubit_key)`` pairs where *pauli* is
        ``"X"``, ``"Y"``, or ``"Z"`` and *qubit_key* identifies the
        target qubit via *qubit_map*.  Multiple faults are applied in
        order, supporting arbitrary-weight error injection.
    qubit_map
        Mapping (dict or sequence) from qubit key to Qiskit qubit object.
        If a dict, *qubit_key* is used as the dict key.
        If a sequence (list of qubits), *qubit_key* is used as an index.
        If ``None``, *qubit_key* is passed directly as the qubit argument
        (useful when keys are already raw qubit indices).

    Raises
    ------
    ValueError
        If a Pauli string is unrecognized.
    KeyError / IndexError
        If a qubit key is not found in *qubit_map*.
    """
    for pauli, key in faults:
        if qubit_map is None:
            qubit = key
        elif isinstance(qubit_map, Mapping):
            qubit = qubit_map[key]
        else:
            qubit = qubit_map[key]

        p = pauli.upper()
        if p == "X":
            circuit.x(qubit)
        elif p == "Y":
            circuit.y(qubit)
        elif p == "Z":
            circuit.z(qubit)
        else:
            raise ValueError(f"unsupported Pauli {pauli!r}; expected 'X', 'Y', or 'Z'")

