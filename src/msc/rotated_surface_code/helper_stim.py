"""Stim circuit primitives for rotated surface-code patches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import stim

from msc.rotated_surface_code import common
from msc.utils import circuit_stim
from msc.utils.circuit_stim import (
    append_logical_measurement_on_indexed_qubits,
    append_scheduled_parity_measurements,
)
from msc.utils.normalize import normalize_pauli_char


def reset_to_zero(
    circuit: stim.Circuit,
    *,
    qubits: Sequence[int],
    add_ticks: bool = False,
    tick_label: str | None = None,
) -> None:
    """Reset a collection of qubits to ``|0>``."""
    del tick_label
    selected_qubits = tuple(qubits)
    if selected_qubits:
        circuit.append("R", selected_qubits)
    if add_ticks:
        circuit.append("TICK", [])


def prepare_product_plus(
    circuit: stim.Circuit,
    *,
    qubits: Sequence[int],
    add_ticks: bool = False,
    tick_label: str | None = None,
) -> None:
    """Prepare a collection of qubits in ``|+>`` from caller-supplied ``|0>`` states."""
    del tick_label
    selected_qubits = tuple(qubits)
    if selected_qubits:
        circuit.append("H", selected_qubits)
    if add_ticks:
        circuit.append("TICK", [])


def append_scheduled_stabilizer_round(
    circuit: stim.Circuit,
    *,
    label_to_qubit: Mapping[str, int],
    ancillas: Sequence[int],
    selected_stabilizer_schedules: Mapping[str, tuple[str, ...]],
    reset_ancillas_initially: bool,
    add_ticks: bool,
    tick_label: str | None,
) -> tuple[int, ...]:
    """Measure any selected surface stabilizers in four scheduled steps.

    ``selected_stabilizer_schedules`` may contain all X checks, all Z checks,
    every X and Z check, or any same-basis or mixed-basis subset. Mapping order
    defines both ancilla alignment and Stim measurement-record order; every
    selected stabilizer requires a distinct ancilla and retains its canonical
    schedule row. Each stabilizer name begins with its physical basis,
    ``"X"`` or ``"Z"``. If ``reset_ancillas_initially`` is true, every used
    ancilla is reset immediately before the interactions; otherwise the caller
    must supply ancillas already in ``|0>``.

    This measures only. Pass the returned record positions to
    :func:`append_stabilizer_detectors` to annotate known-``+1`` or change
    detectors.

    Call structure::

        full X and Z stabilizer schedules
        └── selected_stabilizer_schedules
            └── common.resolve_selected_stabilizer_schedules
                └── append_scheduled_parity_measurements
    """
    del tick_label
    first_measurement_position = circuit.num_measurements
    data_qubit_schedules, used_ancillas = common.resolve_selected_stabilizer_schedules(
        label_to_qubit=label_to_qubit,
        selected_stabilizer_schedules=selected_stabilizer_schedules,
        ancillas=ancillas,
    )
    if reset_ancillas_initially and used_ancillas:
        circuit.append("R", used_ancillas)
        if add_ticks:
            circuit.append("TICK", [])

    append_scheduled_parity_measurements(
        circuit,
        bases=tuple(name[0] for name in selected_stabilizer_schedules),
        data_qubit_schedules=data_qubit_schedules,
        ancilla_qubits=used_ancillas,
        add_ticks=add_ticks,
    )
    if add_ticks:
        circuit.append("TICK", [])
    return tuple(range(first_measurement_position, circuit.num_measurements))


def append_stabilizer_detectors(
    circuit: stim.Circuit,
    *,
    rounds: Sequence[Sequence[int]],
    first_round_reference: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Append change detectors across one or more equal-width stabilizer rounds.

    Each round is compared elementwise with the round before it. The first
    round is compared with ``first_round_reference`` when supplied, or with the
    known +1 eigenvalue otherwise. Passing a single round is the common special
    case. Returns the detector indices added by this call.
    """
    first_detector = circuit.num_detectors
    previous_positions = None if first_round_reference is None else tuple(first_round_reference)
    for current_round in rounds:
        current_positions = tuple(current_round)
        if previous_positions is not None and len(previous_positions) != len(current_positions):
            raise ValueError("previous and current stabilizer records must have equal length.")
        for index, current_position in enumerate(current_positions):
            measurement_positions = (
                (current_position,)
                if previous_positions is None
                else (previous_positions[index], current_position)
            )
            circuit_stim.append_detector(
                circuit,
                measurement_positions=measurement_positions,
            )
        previous_positions = current_positions
    return tuple(range(first_detector, circuit.num_detectors))


def append_logical_measurement(
    circuit: stim.Circuit,
    basis: str = "Z",
    *,
    surface_qubits: Sequence[int],
    surface_index: Mapping[str, int],
    default_logical_x: Sequence[str],
    default_logical_z: Sequence[str],
    support: tuple[str, ...] | None = None,
    ancilla_qubit: int | None = None,
    destructive: bool = True,
    observable_index: int | None = None,
) -> None:
    """Append a logical Pauli measurement on a rotated surface-code patch.

    X and Z use their corresponding logical representative. Y measures
    ``i X_L Z_L``: physical Y on the intersection of the logical-X and
    logical-Z supports, X on X-only sites, and Z on Z-only sites. A destructive
    measurement appends one Stim result per physical term; a non-destructive
    measurement appends one ancilla-parity result.

    ``support`` overrides the representative for X or Z. For Y, omit it so the
    distinct surface-code X and Z representatives are both used.
    Set ``observable_index`` to include the resulting parity as a Stim logical
    observable.
    """
    indexed_support = None if support is None else tuple(surface_index[label] for label in support)
    first_measurement_position = circuit.num_measurements
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=surface_qubits,
        default_logical_x=tuple(surface_index[label] for label in default_logical_x),
        default_logical_z=tuple(surface_index[label] for label in default_logical_z),
        support=indexed_support,
        ancilla_qubit=ancilla_qubit,
        destructive=destructive,
    )
    if observable_index is not None:
        circuit_stim.append_observable_include(
            circuit,
            measurement_positions=tuple(
                range(first_measurement_position, circuit.num_measurements)
            ),
            observable_index=observable_index,
        )


def append_all_data_readout(
    circuit: stim.Circuit,
    *,
    data_qubits: Sequence[int],
    basis: str,
    add_ticks: bool = True,
) -> None:
    """Destructively measure every surface data qubit in a single Pauli basis.

    Unlike a logical-support readout, this records one result per data qubit in
    ``data_qubits`` order, so the full record both supplies a logical
    representative and closes matching-basis stabilizer detectors. Pair an
    ``X``- or ``Z``-basis readout with
    :func:`append_all_data_readout_detectors_and_observable` to annotate those
    detectors and the logical observable. When ``add_ticks`` is true a trailing
    ``TICK`` separates this readout from any following layer.
    """
    basis = normalize_pauli_char(basis)
    selected_qubits = tuple(data_qubits)
    if selected_qubits:
        circuit.append({"X": "MX", "Y": "MY", "Z": "M"}[basis], selected_qubits)
    if add_ticks:
        circuit.append("TICK", [])


def append_all_data_readout_detectors_and_observable(
    circuit: stim.Circuit,
    basis: str,
    *,
    x_stabilizers: Mapping[str, Sequence[str]],
    z_stabilizers: Mapping[str, Sequence[str]],
    logical_x: Sequence[str],
    logical_z: Sequence[str],
    final_stabilizer_measurement_positions: Mapping[str, int],
    data_measurement_positions: Mapping[str, int],
    logical_frame_measurement_positions: Sequence[int] = (),
    observable_index: int = 0,
) -> tuple[int, ...]:
    """Annotate a completed destructive all-data readout with detectors and an observable.

    Appends no measurements. It assumes every surface data qubit was just
    measured destructively in the ``X`` or ``Z`` basis, with the final
    stabilizer ancillas measured in the preceding round.

    Each matching-basis stabilizer gets one detector comparing its ancilla
    record with the parity of its measured data support, so a fault-free
    stabilizer stays consistent across the readout. The logical observable is
    the parity of the basis representative's data qubits plus any
    caller-supplied Pauli-frame correction records
    (``logical_frame_measurement_positions``).

    ``basis`` must be ``X`` or ``Z``; ``Y`` is rejected because an all-data
    readout is single-basis. Returns the detector indices added by this call.
    """
    basis = normalize_pauli_char(basis)
    if basis == "Y":
        raise ValueError("all-data surface detectors require an X or Z basis.")
    stabilizers = x_stabilizers if basis == "X" else z_stabilizers
    logical_support = logical_x if basis == "X" else logical_z

    if set(final_stabilizer_measurement_positions) != set(stabilizers):
        raise ValueError("final stabilizer positions do not match the stabilizer labels.")
    required_data_labels = set(logical_support)
    for support in stabilizers.values():
        required_data_labels.update(support)
    if not required_data_labels <= set(data_measurement_positions):
        raise ValueError("data measurement positions do not cover every required label.")

    first_detector = circuit.num_detectors
    for label, support in stabilizers.items():
        circuit_stim.append_detector(
            circuit,
            measurement_positions=(final_stabilizer_measurement_positions[label],)
            + tuple(data_measurement_positions[data_label] for data_label in support),
        )
    circuit_stim.append_observable_include(
        circuit,
        measurement_positions=tuple(data_measurement_positions[label] for label in logical_support)
        + tuple(logical_frame_measurement_positions),
        observable_index=observable_index,
    )
    return tuple(range(first_detector, circuit.num_detectors))
