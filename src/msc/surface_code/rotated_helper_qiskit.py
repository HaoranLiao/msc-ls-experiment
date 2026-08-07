"""Qiskit circuit primitives for rotated surface-code patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qiskit import QuantumCircuit

from msc.surface_code import rotated_common
from msc.utils.circuit_qiskit import (
    append_logical_measurement_on_indexed_qubits,
    append_scheduled_parity_measurements,
)


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


def prepare_product_plus(
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


def append_scheduled_stabilizer_round(
    circuit: QuantumCircuit,
    *,
    label_to_qubit: Mapping[str, object],
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    selected_stabilizer_schedules: Mapping[str, tuple[str, ...]],
    reset_ancillas_initially: bool,
    add_barriers: bool,
    barrier_label: str | None,
) -> None:
    """Append any selected surface stabilizers in four scheduled steps.

    ``selected_stabilizer_schedules`` may contain all X checks, all Z checks,
    every X and Z check, or any same-basis or mixed-basis subset selected from
    the full canonical schedules. Mapping order defines the corresponding
    order of ``ancillas`` and ``classical_bits``; every selected stabilizer
    requires a distinct ancilla. Each stabilizer name begins with its physical
    basis, ``"X"`` or ``"Z"``. If ``reset_ancillas_initially`` is true, every
    used ancilla is reset to ``|0>`` immediately before the scheduled
    interactions; otherwise, the caller must supply ancillas already in
    ``|0>``.

    Call structure::

        full X_STABILIZER_SCHEDULE and Z_STABILIZER_SCHEDULE
        └── select any same-basis or mixed-basis stabilizers
            └── selected_stabilizer_schedules
                └── append_scheduled_stabilizer_round
                    └── rotated_common.resolve_selected_stabilizer_schedules
                        └── append_scheduled_parity_measurements

    """
    data_qubit_schedules, used_ancillas = rotated_common.resolve_selected_stabilizer_schedules(
        label_to_qubit=label_to_qubit,
        selected_stabilizer_schedules=selected_stabilizer_schedules,
        ancillas=ancillas,
    )
    num_checks = len(selected_stabilizer_schedules)
    if len(classical_bits) < num_checks:
        raise ValueError(
            f"stabilizer round requires at least {num_checks} classical bits, "
            f"got {len(classical_bits)}."
        )
    used_bits = tuple(classical_bits[:num_checks])
    if reset_ancillas_initially:
        reset_to_zero(circuit, qubits=used_ancillas, add_barriers=False)

    append_scheduled_parity_measurements(
        circuit,
        bases=tuple(stabilizer_name[0] for stabilizer_name in selected_stabilizer_schedules),
        data_qubit_schedules=data_qubit_schedules,
        ancilla_qubits=used_ancillas,
        classical_bits=used_bits,
    )
    if add_barriers:
        circuit.barrier(label=barrier_label or "scheduled stabilizer round")


# ---------------------------------------------------------------------------
# Logical measurement
# ---------------------------------------------------------------------------


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
    """Append a logical Pauli measurement on a rotated surface-code patch.

    X and Z use their corresponding logical representative. Y measures
    ``i X_L Z_L``: physical Y on the intersection of the logical-X and
    logical-Z supports, X on the X-only sites, and Z on the Z-only sites.
    Consequently, a destructive Y measurement requires one classical bit per
    qubit in the union of the two supports.

    ``support`` overrides the representative for X or Z. For Y, omit it so
    the distinct surface-code X and Z representatives are both used.
    """
    indexed_support = None if support is None else tuple(surface_index[label] for label in support)
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=surface_qubits,
        default_logical_x=tuple(surface_index[label] for label in default_logical_x),
        default_logical_z=tuple(surface_index[label] for label in default_logical_z),
        support=indexed_support,
        ancilla_qubit=ancilla_qubit,
        classical_bits=classical_bits,
        destructive=destructive,
    )
