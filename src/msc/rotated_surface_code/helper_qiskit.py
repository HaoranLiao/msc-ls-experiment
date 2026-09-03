"""Qiskit circuit primitives for rotated surface-code patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qiskit import QuantumCircuit

from msc.rotated_surface_code import common
from msc.utils.circuit_qiskit import (
    append_logical_measurement_on_indexed_qubits,
    append_scheduled_parity_measurements,
)
from msc.utils.normalize import normalize_pauli_char


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
    bits: Sequence[object],
    selected_stabilizer_schedules: Mapping[str, tuple[str, ...]],
    reset_ancillas_initially: bool,
    add_barriers: bool,
    barrier_label: str | None,
) -> None:
    """Append any selected surface stabilizers in four scheduled steps.

    ``selected_stabilizer_schedules`` may contain all X checks, all Z checks,
    every X and Z check, or any same-basis or mixed-basis subset selected from
    the full canonical schedules. Mapping order defines the corresponding
    order of ``ancillas`` and ``bits``; every selected stabilizer requires a
    distinct ancilla. Each stabilizer name begins with its physical basis,
    ``"X"`` or ``"Z"``. If ``reset_ancillas_initially`` is true, every used
    ancilla is reset to ``|0>`` immediately before the scheduled interactions;
    otherwise, the caller must supply ancillas already in ``|0>``.

    Call structure::

        full X_STABILIZER_SCHEDULE and Z_STABILIZER_SCHEDULE
        └── select any same-basis or mixed-basis stabilizers
            └── selected_stabilizer_schedules
                └── append_scheduled_stabilizer_round
                    └── common.resolve_selected_stabilizer_schedules
                        └── append_scheduled_parity_measurements

    """
    data_qubit_schedules, used_ancillas = common.resolve_selected_stabilizer_schedules(
        label_to_qubit=label_to_qubit,
        selected_stabilizer_schedules=selected_stabilizer_schedules,
        ancillas=ancillas,
    )
    num_checks = len(selected_stabilizer_schedules)
    if len(bits) < num_checks:
        raise ValueError(
            f"stabilizer round requires at least {num_checks} classical bits, " f"got {len(bits)}."
        )
    used_bits = tuple(bits[:num_checks])
    if reset_ancillas_initially:
        for ancilla in used_ancillas:
            circuit.reset(ancilla)

    append_scheduled_parity_measurements(
        circuit,
        bases=tuple(stabilizer_name[0] for stabilizer_name in selected_stabilizer_schedules),
        data_qubit_schedules=data_qubit_schedules,
        ancilla_qubits=used_ancillas,
        bits=used_bits,
    )
    if add_barriers:
        circuit.barrier(label=barrier_label or "scheduled stabilizer round")


def append_full_data_readout(
    circuit: QuantumCircuit,
    basis: str,
    *,
    surface_qubits: Sequence[object],
    surface_order: Sequence[str],
    bits: Sequence[object],
) -> None:
    """Destructively measure every data qubit in canonical surface order.

    Results for ``surface_order[i]`` are stored in ``bits[i]``.
    Qiskit has no detector annotations; callers reconstruct compatible final
    stabilizer parities from these data bits during analysis.
    """
    basis = normalize_pauli_char(basis)
    if basis not in {"X", "Z"}:
        raise ValueError("full-data readout basis must be X or Z.")
    num_data_qubits = len(surface_order)
    if len(surface_qubits) < num_data_qubits:
        raise ValueError("surface_qubits does not cover surface_order.")
    if len(bits) < num_data_qubits:
        raise ValueError("bits does not cover surface_order.")
    selected_surface_qubits = tuple(surface_qubits[:num_data_qubits])
    selected_bits = tuple(bits[:num_data_qubits])
    if basis == "X":
        circuit.h(selected_surface_qubits)
    circuit.measure(selected_surface_qubits, selected_bits)


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
    bits: Sequence[object],
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
        bits=bits,
        destructive=destructive,
    )
