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


def prepare_product_plus(
    circuit: stim.Circuit,
    *,
    qubits: Sequence[int],
    add_ticks: bool = False,
    tick_label: str | None = None,
) -> None:
    """Reset a collection of qubits directly into ``|+>``."""
    del tick_label
    selected_qubits = tuple(qubits)
    if selected_qubits:
        circuit.append("RX", selected_qubits)
    if add_ticks:
        circuit.append("TICK", [])


def append_scheduled_stabilizer_round(
    circuit: stim.Circuit,
    *,
    label_to_qubit: Mapping[str, int],
    ancillas: Sequence[int],
    selected_stabilizer_schedules: Mapping[str, tuple[str, ...]],
    add_ticks: bool,
    tick_label: str | None,
) -> tuple[int, ...]:
    """Measure any selected surface stabilizers in four scheduled steps.

    ``selected_stabilizer_schedules`` may contain all X checks, all Z checks,
    every X and Z check, or any same-basis or mixed-basis subset. Mapping order
    defines both ancilla alignment and Stim measurement-record order; every
    selected stabilizer requires a distinct ancilla and retains its canonical
    schedule row. Each stabilizer name begins with its physical basis,
    ``"X"`` or ``"Z"``. Every ancilla is reset and measured directly in that
    basis with ``RX``/``RZ`` and ``MX``/``MZ``; no Hadamard basis changes are
    inserted.

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


def append_full_data_readout(
    circuit: stim.Circuit,
    basis: str,
    *,
    surface_qubits: Sequence[int],
    surface_order: Sequence[str],
    surface_index: Mapping[str, int],
    stabilizers: Mapping[str, Sequence[str]],
    previous_stabilizer_measurement_positions: Sequence[int],
) -> tuple[int, ...]:
    """Destructively read every data qubit and close compatible checks.

    X- or Z-basis data measurements reconstruct the stabilizers of the same
    basis. Each reconstructed parity is compared with the corresponding final
    syndrome-round result, producing the readout-boundary detectors used by
    Hirano's non-full-postselection experiment. The returned record positions
    follow ``surface_order``.
    """
    basis = normalize_pauli_char(basis)
    if basis not in {"X", "Z"}:
        raise ValueError("full-data readout basis must be X or Z.")
    selected_surface_order = tuple(surface_order)
    selected_surface_qubits = tuple(surface_qubits[: len(selected_surface_order)])
    if len(selected_surface_qubits) != len(selected_surface_order):
        raise ValueError("surface_qubits does not cover surface_order.")
    previous_positions = tuple(previous_stabilizer_measurement_positions)
    if len(previous_positions) != len(stabilizers):
        raise ValueError("previous stabilizer positions do not match the readout basis.")

    first_measurement_position = circuit.num_measurements
    circuit.append("MX" if basis == "X" else "M", selected_surface_qubits)
    readout_positions = tuple(
        range(first_measurement_position, first_measurement_position + len(selected_surface_order))
    )
    readout_position_by_label = {
        label: readout_positions[surface_index[label]] for label in selected_surface_order
    }
    for previous_position, support in zip(
        previous_positions,
        stabilizers.values(),
        strict=True,
    ):
        circuit_stim.append_detector(
            circuit,
            measurement_positions=(
                previous_position,
                *(readout_position_by_label[label] for label in support),
            ),
        )
    return readout_positions


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
