from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class CircuitBuildResult:
    """A circuit together with the semantic labels of its measurement bits."""

    circuit: QuantumCircuit
    measurement_labels: tuple[str, ...]


def measure_z_parity(
    circuit: QuantumCircuit,
    *,
    ancilla,
    data_qubits,
    classical_bit,
) -> None:
    """Measure one ad hoc multi-qubit Z parity into ``classical_bit``.

    Performs reset → CX fan-in → measure on a single ancilla qubit.
    Used by the lattice-surgery transfer for shared-boundary checks.
    """
    circuit.reset(ancilla)
    for qubit in data_qubits:
        circuit.cx(qubit, ancilla)
    circuit.measure(ancilla, classical_bit)


def measure_x_parity(
    circuit: QuantumCircuit,
    *,
    ancilla,
    data_qubits,
    classical_bit,
) -> None:
    """Measure one ad hoc multi-qubit X parity into ``classical_bit``.

    Performs reset → H → CX fan-out → H → measure on a single ancilla qubit.
    Used by the lattice-surgery transfer for shared-boundary checks.
    """
    circuit.reset(ancilla)
    circuit.h(ancilla)
    for qubit in data_qubits:
        circuit.cx(ancilla, qubit)
    circuit.h(ancilla)
    circuit.measure(ancilla, classical_bit)

