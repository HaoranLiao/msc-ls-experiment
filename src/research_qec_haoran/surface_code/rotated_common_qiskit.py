"""Shared Qiskit circuit primitives for rotated surface-code patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qiskit import QuantumCircuit


def label2qubit(
    surface_qubits: Sequence[object],
    *,
    surface_index: Mapping[str, int],
) -> dict[str, object]:
    """Map surface labels to supplied Qiskit qubits."""
    if len(surface_qubits) != len(surface_index):
        raise ValueError(
            f"surface_qubits must contain exactly {len(surface_index)} qubits."
        )
    return {
        label: surface_qubits[index]
        for label, index in surface_index.items()
    }


def reset_to_zero(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Reset a collection of qubits to ``|0>``."""
    for qubit in qubits:
        circuit.reset(qubit)
    if add_barriers:
        circuit.barrier(label=barrier_label or "prepare |0>")


def prepare_plus(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Prepare a collection of qubits in ``|+>``."""
    for qubit in qubits:
        circuit.h(qubit)
    if add_barriers:
        circuit.barrier(label=barrier_label or "prepare |+>")


def append_parallel_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    label_to_qubit: Mapping[str, object],
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    schedules: Mapping[str, tuple[str, ...]],
    basis: str,
    prepare_ancillas: bool,
    add_barriers: bool,
    barrier_label: str | None,
) -> None:
    """Append one padded parallel stabilizer-measurement round."""
    ancilla_list = list(ancillas)
    classical_list = list(classical_bits)
    stabilizer_names = tuple(schedules)
    n = len(stabilizer_names)
    if len(ancilla_list) < n:
        raise ValueError(
            f"{basis}-check requires at least {n} ancilla qubits, got {len(ancilla_list)}."
        )
    if len(classical_list) < n:
        raise ValueError(
            f"{basis}-check requires at least {n} classical bits, got {len(classical_list)}."
        )

    if prepare_ancillas:
        reset_to_zero(circuit, qubits=ancilla_list, add_barriers=False)

    if basis == "X":
        for ancilla in ancilla_list:
            circuit.h(ancilla)

    num_steps = len(next(iter(schedules.values())))
    for step in range(num_steps):
        for ancilla, stabilizer_name in zip(ancilla_list, stabilizer_names, strict=False):
            data_label = schedules[stabilizer_name][step]
            if not data_label:
                continue
            if basis == "Z":
                circuit.cx(label_to_qubit[data_label], ancilla)
            elif basis == "X":
                circuit.cx(ancilla, label_to_qubit[data_label])
            else:
                raise ValueError(f"unsupported stabilizer basis {basis!r}")

    if basis == "X":
        for ancilla in ancilla_list:
            circuit.h(ancilla)

    for ancilla, classical_bit in zip(ancilla_list, classical_list, strict=False):
        circuit.measure(ancilla, classical_bit)

    if add_barriers:
        circuit.barrier(label=barrier_label or f"parallel {basis} stabilizer round")


def append_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Z",
    *,
    surface_qubits: Sequence[object],
    surface_index: Mapping[str, int],
    default_logical_x: Sequence[str],
    default_logical_z: Sequence[str],
    support: tuple[str, ...] | None = None,
    ancilla_qubit=None,
    classical_bits: Sequence[object],
    destructive: bool = True,
) -> None:
    """Append a logical measurement on a rotated surface-code patch."""
    if support is None:
        support = tuple(default_logical_z if basis == "Z" else default_logical_x)

    label_to_qubit = label2qubit(surface_qubits, surface_index=surface_index)
    cb = list(classical_bits)
    if destructive:
        if len(cb) < len(support):
            raise ValueError(
                "classical_bits must have at least "
                f"{len(support)} entries for destructive measurement."
            )
        if basis == "X":
            for label in support:
                circuit.h(label_to_qubit[label])
        for index, label in enumerate(support):
            circuit.measure(label_to_qubit[label], cb[index])
        return

    if ancilla_qubit is None:
        raise ValueError("ancilla_qubit is required for non-destructive measurement.")
    if len(cb) < 1:
        raise ValueError(
            "classical_bits must have at least 1 entry for non-destructive measurement."
        )
    if basis == "Z":
        for label in support:
            circuit.cx(label_to_qubit[label], ancilla_qubit)
        circuit.measure(ancilla_qubit, cb[0])
    elif basis == "X":
        circuit.h(ancilla_qubit)
        for label in support:
            circuit.cx(ancilla_qubit, label_to_qubit[label])
        circuit.h(ancilla_qubit)
        circuit.measure(ancilla_qubit, cb[0])
    else:
        raise ValueError("basis must be 'X' or 'Z'.")
