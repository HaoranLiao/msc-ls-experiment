"""Common gate decompositions for Qiskit circuits."""
from __future__ import annotations

from qiskit import QuantumCircuit


def apply_ty(circuit: QuantumCircuit, qubit) -> None:
    """Apply ``T_y = RY(pi/4) = exp(-i*pi*Y/8)``.

    This is the rotation satisfying ``H = T_y^dagger X T_y``, used in
    magic-state cultivation to map the Hadamard-axis observable onto the X basis.

    The decomposition uses only named Clifford+T gates (no parameterized
    ``rz`` or ``ry``), making it compatible with Aer's ``extended_stabilizer``
    simulation method.
    """
    # RY(pi/4) = Sdg · H · T · H · S  (up to global phase)
    circuit.sdg(qubit)
    circuit.h(qubit)
    circuit.t(qubit)
    circuit.h(qubit)
    circuit.s(qubit)


def apply_ty_dagger(circuit: QuantumCircuit, qubit) -> None:
    """Apply ``T_y^dagger = RY(-pi/4)``."""
    # RY(-pi/4) = Sdg · H · Tdg · H · S  (up to global phase)
    circuit.sdg(qubit)
    circuit.h(qubit)
    circuit.tdg(qubit)
    circuit.h(qubit)
    circuit.s(qubit)