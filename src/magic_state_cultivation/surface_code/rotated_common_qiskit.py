"""Shared Qiskit circuit primitives for rotated surface-code patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qiskit import QuantumCircuit

from magic_state_cultivation.utils.circuit_qiskit import (
    append_logical_measurement_on_indexed_qubits,
    append_scheduled_parity_measurements,
)

# ---------------------------------------------------------------------------
# Patch-wide preparation
# ---------------------------------------------------------------------------


def label2qubit(
    surface_qubits: Sequence[object],
    *,
    surface_index: Mapping[str, int],
) -> dict[str, object]:
    """Map surface labels to supplied Qiskit qubits."""
    if len(surface_qubits) < len(surface_index):
        raise ValueError(f"surface_qubits must contain at least {len(surface_index)} qubits.")
    return {label: surface_qubits[index] for label, index in surface_index.items()}


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


# ---------------------------------------------------------------------------
# Canonical stabilizer schedules
# ---------------------------------------------------------------------------


def build_full_canonical_stabilizer_schedule(
    *,
    stabilizers: Mapping[str, tuple[str, ...]],
    surface_coords: Mapping[str, tuple[int, int]],
    basis: str,
) -> dict[str, tuple[str, ...]]:
    """Build the full canonical four-step schedule for one stabilizer basis.

    X checks use ``NW, NE, SW, SE`` in the bulk, with top-boundary checks
    assigned to steps 3-4 and bottom-boundary checks to steps 1-2. Z checks use
    ``NW, SW, NE, SE`` in the bulk, with right-boundary checks assigned to
    steps 1-2 and left-boundary checks to steps 3-4.

    Data-flow structure::

        build_full_canonical_stabilizer_schedule
        └── full X_STABILIZER_SCHEDULE or Z_STABILIZER_SCHEDULE
            └── select any same-basis or mixed-basis subset
                └── selected_stabilizer_schedules
                    └── append_scheduled_stabilizer_round

    Reference CanonicalMeasurementScheduleSurfaceCode.png
    """
    if basis not in {"X", "Z"}:
        raise ValueError(f"unsupported stabilizer basis {basis!r}")
    if not surface_coords:
        raise ValueError("surface_coords must not be empty.")

    min_x = min(x for x, _ in surface_coords.values())
    max_x = max(x for x, _ in surface_coords.values())
    min_y = min(y for _, y in surface_coords.values())
    max_y = max(y for _, y in surface_coords.values())
    full_stabilizer_schedule: dict[str, tuple[str, ...]] = {}

    for stabilizer_name, support in stabilizers.items():
        try:
            label_by_coord = {surface_coords[label]: label for label in support}
        except KeyError as error:
            raise ValueError(
                f"{stabilizer_name} contains unknown data-qubit label {error.args[0]!r}."
            ) from error

        if len(support) == 4:
            support_xs = {x for x, _ in label_by_coord}
            support_ys = {y for _, y in label_by_coord}
            if len(support_xs) != 2 or len(support_ys) != 2:
                raise ValueError(f"{stabilizer_name} is not a four-corner bulk check.")
            west, east = min(support_xs), max(support_xs)
            north, south = min(support_ys), max(support_ys)
            try:
                northwest = label_by_coord[(west, north)]
                northeast = label_by_coord[(east, north)]
                southwest = label_by_coord[(west, south)]
                southeast = label_by_coord[(east, south)]
            except KeyError as error:
                raise ValueError(f"{stabilizer_name} is not a four-corner bulk check.") from error
            full_stabilizer_schedule[stabilizer_name] = (
                (northwest, northeast, southwest, southeast)
                if basis == "X"
                else (northwest, southwest, northeast, southeast)
            )
            continue

        if len(support) != 2:
            raise ValueError(
                f"{stabilizer_name} must be a weight-2 boundary or weight-4 bulk check."
            )

        first, second = sorted(
            support,
            key=lambda label: (
                surface_coords[label][0] if basis == "X" else surface_coords[label][1]
            ),
        )
        first_x, first_y = surface_coords[first]
        second_x, second_y = surface_coords[second]
        if basis == "X":
            if first_y != second_y:
                raise ValueError(f"{stabilizer_name} is not a horizontal X-boundary check.")
            if first_y == min_y:
                full_stabilizer_schedule[stabilizer_name] = ("", "", first, second)
            elif first_y == max_y:
                full_stabilizer_schedule[stabilizer_name] = (first, second, "", "")
            else:
                raise ValueError(f"{stabilizer_name} is not on an X boundary.")
        else:
            if first_x != second_x:
                raise ValueError(f"{stabilizer_name} is not a vertical Z-boundary check.")
            if first_x == max_x:
                full_stabilizer_schedule[stabilizer_name] = (first, second, "", "")
            elif first_x == min_x:
                full_stabilizer_schedule[stabilizer_name] = ("", "", first, second)
            else:
                raise ValueError(f"{stabilizer_name} is not on a Z boundary.")

    for step in range(4):
        active_labels = [
            schedule[step] for schedule in full_stabilizer_schedule.values() if schedule[step]
        ]
        if len(active_labels) != len(set(active_labels)):
            raise ValueError(f"canonical schedule has a data-qubit collision at step {step + 1}.")

    return full_stabilizer_schedule


# ---------------------------------------------------------------------------
# Scheduled stabilizer measurement
# ---------------------------------------------------------------------------


def _resolve_selected_stabilizer_schedules(
    *,
    label_to_qubit: Mapping[str, object],
    selected_stabilizer_schedules: Mapping[str, tuple[str, ...]],
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
) -> tuple[
    tuple[tuple[object | None, ...], ...],
    tuple[object, ...],
    tuple[object, ...],
]:
    """Resolve selected label schedules and align their ancillas and result bits.

    Data-flow structure::

        selected label schedules + label_to_qubit
        └── Qiskit data-qubit schedules + aligned ancillas + result bits
            └── append_scheduled_parity_measurements
    """
    num_checks = len(selected_stabilizer_schedules)
    if len(ancillas) < num_checks:
        raise ValueError(
            f"stabilizer round requires at least {num_checks} ancillas, " f"got {len(ancillas)}."
        )
    if len(classical_bits) < num_checks:
        raise ValueError(
            f"stabilizer round requires at least {num_checks} classical bits, "
            f"got {len(classical_bits)}."
        )

    data_qubit_schedules = tuple(
        tuple(label_to_qubit[label] if label else None for label in schedule)
        for schedule in selected_stabilizer_schedules.values()
    )
    return (
        data_qubit_schedules,
        tuple(ancillas[:num_checks]),
        tuple(classical_bits[:num_checks]),
    )


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
                    └── _resolve_selected_stabilizer_schedules
                        └── append_scheduled_parity_measurements

    """
    data_qubit_schedules, used_ancillas, used_bits = _resolve_selected_stabilizer_schedules(
        label_to_qubit=label_to_qubit,
        selected_stabilizer_schedules=selected_stabilizer_schedules,
        ancillas=ancillas,
        classical_bits=classical_bits,
    )
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
    """Append a logical measurement on a rotated surface-code patch."""
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
