from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector

from research_qec_haoran.msc_by_lattice_surgery.common_qiskit import CircuitBuildResult
from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    ROTATED_D3_LOGICAL_X,
    ROTATED_D3_LOGICAL_Z,
    ROTATED_D3_SURFACE_ORDER,
    ROTATED_D3_X_STABILIZERS,
    ROTATED_D3_Z_STABILIZERS,
    rotated_d3_surface_label_map,
)

ROTATED_D3_INIT_Z_LABELS: tuple[str, ...] = tuple(
    f"init_{name.lower()}" for name in ROTATED_D3_Z_STABILIZERS
)


def build_rotated_d3_pauli_operator(
    pauli_char: str,
    support: tuple[str, ...] | list[str],
) -> SparsePauliOp:
    """Build a 9-qubit Pauli on the rotated-d3 data qubits."""
    label = ["I"] * len(ROTATED_D3_SURFACE_ORDER)
    for qubit_label in support:
        label[ROTATED_D3_SURFACE_ORDER.index(qubit_label)] = pauli_char
    return SparsePauliOp("".join(reversed(label)))


def rotated_d3_z_syndrome_bits_from_measurements(
    measurements: dict[str, int | bool],
) -> dict[str, int]:
    """Convert labeled initialization measurements into Z-stabilizer bits.

    The returned dict is keyed by the canonical stabilizer names from
    ``ROTATED_D3_Z_STABILIZERS`` such as ``"Z_abde"``.
    """
    syndrome: dict[str, int] = {}
    for name in ROTATED_D3_Z_STABILIZERS:
        init_label = f"init_{name.lower()}"
        if init_label in measurements:
            bit = measurements[init_label]
        elif name in measurements:
            bit = measurements[name]
        else:
            raise KeyError(f"missing measurement for {name!r} / {init_label!r}")
        if bit not in (0, 1, False, True):
            raise ValueError(f"stabilizer bit must be 0 or 1, got {bit!r}")
        syndrome[name] = int(bit)
    return syndrome


def build_rotated_d3_logical_plus_statevector(
    z_syndrome_bits: dict[str, int | bool] | None = None,
) -> Statevector:
    """Return the exact logical-``|+_L>`` branch for a chosen Z-syndrome.

    Parameters
    ----------
    z_syndrome_bits
        Optional mapping from the canonical Z-stabilizer names
        ``"Z_abde"``, ``"Z_cf"``, ``"Z_dg"``, ``"Z_efhi"`` to bits.
        Bit ``0`` means the ``+1`` eigenvalue branch and bit ``1`` means the
        ``-1`` eigenvalue branch.
    """
    state = Statevector.from_label("+" * len(ROTATED_D3_SURFACE_ORDER))
    identity = np.eye(2 ** len(ROTATED_D3_SURFACE_ORDER), dtype=complex)
    syndrome = {name: 0 for name in ROTATED_D3_Z_STABILIZERS}
    if z_syndrome_bits is not None:
        syndrome.update(rotated_d3_z_syndrome_bits_from_measurements(z_syndrome_bits))

    for name, support in ROTATED_D3_Z_STABILIZERS.items():
        op = build_rotated_d3_pauli_operator("Z", support).to_matrix()
        sign = -1 if syndrome[name] else 1
        state = Statevector(((identity + sign * op) @ state.data) / 2)
        norm = np.linalg.norm(state.data)
        state = Statevector(state.data / norm)

    return state


def measure_z_parity(
    circuit: QuantumCircuit,
    *,
    ancilla,
    data_qubits,
    classical_bit,
) -> None:
    """Measure a multi-qubit Z parity into ``classical_bit``."""
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
    """Measure a multi-qubit X parity into ``classical_bit``."""
    circuit.reset(ancilla)
    circuit.h(ancilla)
    for qubit in data_qubits:
        circuit.cx(ancilla, qubit)
    circuit.h(ancilla)
    circuit.measure(ancilla, classical_bit)


def append_initialize_rotated_d3_logical_plus(
    circuit: QuantumCircuit,
    *,
    surface_qubits,
    ancilla,
    classical_bits,
    add_barriers: bool = True,
) -> None:
    """Initialize logical ``|+_L>`` and record the first Z-stabilizer round.

    This does *not* require postselection on all-zero Z stabilizers. The first
    measured Z-stabilizer bits define the reference syndrome for the branch.
    Later Z-stabilizer measurements should be compared relative to that record.
    """
    if len(classical_bits) != len(ROTATED_D3_Z_STABILIZERS):
        raise ValueError("classical_bits must contain 4 bits for Z-stabilizer projection.")

    label_to_qubit = rotated_d3_surface_label_map(surface_qubits)

    for qubit in surface_qubits:
        circuit.h(qubit)

    if add_barriers:
        circuit.barrier(label="surface d3 |+> product state")

    for classical_bit, (name, support) in zip(
        classical_bits,
        ROTATED_D3_Z_STABILIZERS.items(),
        strict=False,
    ):
        measure_z_parity(
            circuit,
            ancilla=ancilla,
            data_qubits=[label_to_qubit[label] for label in support],
            classical_bit=classical_bit,
        )
        if add_barriers:
            circuit.barrier(label=f"init {name}")


def append_prepare_rotated_d3_logical_plus(
    circuit: QuantumCircuit,
    *,
    surface_qubits,
    ancilla,
    classical_bits,
    add_barriers: bool = True,
) -> None:
    """Backward-compatible alias for :func:`append_initialize_rotated_d3_logical_plus`."""
    append_initialize_rotated_d3_logical_plus(
        circuit,
        surface_qubits=surface_qubits,
        ancilla=ancilla,
        classical_bits=classical_bits,
        add_barriers=add_barriers,
    )


def append_surface_d3_stabilizer_round(
    circuit: QuantumCircuit,
    *,
    surface_qubits,
    ancilla,
    z_bits,
    x_bits,
    round_label: str,
    add_barriers: bool = True,
) -> None:
    """Append one full stabilizer round for the rotated d=3 surface code."""
    if len(z_bits) != len(ROTATED_D3_Z_STABILIZERS):
        raise ValueError("z_bits must contain 4 bits.")
    if len(x_bits) != len(ROTATED_D3_X_STABILIZERS):
        raise ValueError("x_bits must contain 4 bits.")

    label_to_qubit = rotated_d3_surface_label_map(surface_qubits)

    for classical_bit, (name, support) in zip(
        z_bits,
        ROTATED_D3_Z_STABILIZERS.items(),
        strict=False,
    ):
        measure_z_parity(
            circuit,
            ancilla=ancilla,
            data_qubits=[label_to_qubit[label] for label in support],
            classical_bit=classical_bit,
        )
        if add_barriers:
            circuit.barrier(label=f"{round_label} {name}")

    for classical_bit, (name, support) in zip(
        x_bits,
        ROTATED_D3_X_STABILIZERS.items(),
        strict=False,
    ):
        measure_x_parity(
            circuit,
            ancilla=ancilla,
            data_qubits=[label_to_qubit[label] for label in support],
            classical_bit=classical_bit,
        )
        if add_barriers:
            circuit.barrier(label=f"{round_label} {name}")


def build_rotated_d3_logical_plus_initialization_circuit(
    *,
    add_barriers: bool = True,
) -> CircuitBuildResult:
    """Return a rotated-d3 logical-``|+_L>`` initialization circuit.

    The returned classical bits are the initial Z-stabilizer reference record:

    ``init_z_abde``, ``init_z_cf``, ``init_z_dg``, ``init_z_efhi``.
    """
    surface = QuantumRegister(9, "surface")
    ancilla = QuantumRegister(1, "anc")
    labels = ROTATED_D3_INIT_Z_LABELS
    classical = ClassicalRegister(len(labels), "m")
    circuit = QuantumCircuit(surface, ancilla, classical)

    append_initialize_rotated_d3_logical_plus(
        circuit,
        surface_qubits=list(surface),
        ancilla=ancilla[0],
        classical_bits=list(classical),
        add_barriers=add_barriers,
    )

    return CircuitBuildResult(circuit=circuit, measurement_labels=labels)


def build_rotated_d3_logical_plus_preparation_circuit(
    *,
    add_barriers: bool = True,
) -> CircuitBuildResult:
    """Backward-compatible alias for the initialization circuit builder."""
    return build_rotated_d3_logical_plus_initialization_circuit(add_barriers=add_barriers)


def rotated_d3_logical_plus_expectations(
    z_syndrome_bits: dict[str, int | bool] | None = None,
) -> dict[str, float]:
    """Return stabilizer and logical expectations for a chosen syndrome branch."""
    syndrome = {name: 0 for name in ROTATED_D3_Z_STABILIZERS}
    if z_syndrome_bits is not None:
        syndrome.update(rotated_d3_z_syndrome_bits_from_measurements(z_syndrome_bits))

    state = build_rotated_d3_logical_plus_statevector(syndrome)
    expectations: dict[str, float] = {}

    for name, support in ROTATED_D3_Z_STABILIZERS.items():
        expectations[name] = state.expectation_value(
            build_rotated_d3_pauli_operator("Z", support)
        ).real

    for name, support in ROTATED_D3_X_STABILIZERS.items():
        expectations[name] = state.expectation_value(
            build_rotated_d3_pauli_operator("X", support)
        ).real

    expectations["logical_X"] = state.expectation_value(
        build_rotated_d3_pauli_operator("X", ROTATED_D3_LOGICAL_X)
    ).real
    expectations["logical_Z"] = state.expectation_value(
        build_rotated_d3_pauli_operator("Z", ROTATED_D3_LOGICAL_Z)
    ).real
    return expectations
