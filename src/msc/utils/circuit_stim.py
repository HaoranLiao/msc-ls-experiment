"""Circuit-level utility helpers for Stim circuits."""

from __future__ import annotations

from collections.abc import Sequence

import stim

from msc.utils.circuit_common import (
    resolve_logical_measurement_terms,
)
from msc.utils.normalize import normalize_pauli_char

# ---------------------------------------------------------------------------
# Detector and observable annotations
# ---------------------------------------------------------------------------


def _measurement_record_targets(
    circuit: stim.Circuit,
    measurement_positions: Sequence[int],
) -> tuple[stim.GateTarget, ...]:
    """Resolve absolute measurement positions into Stim record targets."""
    positions = tuple(measurement_positions)
    if not positions:
        raise ValueError("at least one measurement position is required.")
    if any(not 0 <= position < circuit.num_measurements for position in positions):
        raise ValueError("a measurement position is outside the circuit record.")
    return tuple(stim.target_rec(position - circuit.num_measurements) for position in positions)


def append_detector(
    circuit: stim.Circuit,
    *,
    measurement_positions: Sequence[int],
) -> None:
    """Append one detector over absolute measurement-record positions."""
    circuit.append(
        "DETECTOR",
        _measurement_record_targets(circuit, measurement_positions),
    )


def append_observable_include(
    circuit: stim.Circuit,
    *,
    measurement_positions: Sequence[int],
    observable_index: int = 0,
) -> None:
    """Append a logical observable over absolute measurement positions."""
    if observable_index < 0:
        raise ValueError("observable_index must be nonnegative.")
    circuit.append(
        "OBSERVABLE_INCLUDE",
        _measurement_record_targets(circuit, measurement_positions),
        observable_index,
    )


# ---------------------------------------------------------------------------
# Parity measurements
# ---------------------------------------------------------------------------


def append_scheduled_parity_measurements(
    circuit: stim.Circuit,
    *,
    bases: Sequence[str],
    data_qubit_schedules: Sequence[Sequence[int | None]],
    ancilla_qubits: Sequence[int],
    invert_results: Sequence[bool] | None = None,
    add_ticks: bool = False,
) -> None:
    """Append parallel Pauli-parity measurements from a schedule matrix.

    Parameters
    ----------
    circuit
        Stim circuit to modify.
    bases
        Case-insensitive Pauli basis for each row. Z uses data-controlled CX,
        X uses ancilla-controlled CX, and Y uses ancilla-controlled CY.
    data_qubit_schedules
        Matrix with one parity-measurement row and one interaction-time-step
        column. Use ``None`` for idle positions. Rows must have equal length;
        each column must be collision-free.
    ancilla_qubits
        The first row-count indices are used, one distinct ancilla per row.
        Every used ancilla is reset in its measurement basis with ``RX``,
        ``RY``, or ``RZ`` and read out natively with ``MX``, ``MY``, or ``MZ``.
    invert_results
        Optional result-sign inversion per row. Omitted entries default to
        false only when the entire argument is omitted; a supplied sequence
        must contain one value per row.
    add_ticks
        Whether to separate the reset, interaction, and measurement layers with
        ticks. Consecutive interaction columns are always tick-separated
        regardless, so their shared-ancilla CNOTs are not fused.

    Notes
    -----
    Row order defines Stim measurement-record order. Additional basis,
    ancilla, and inversion entries are untouched.
    """
    schedules = tuple(tuple(schedule) for schedule in data_qubit_schedules)
    num_measurements = len(schedules)
    if len(bases) < num_measurements:
        raise ValueError(f"bases must have at least {num_measurements} entries, got {len(bases)}.")
    if len(ancilla_qubits) < num_measurements:
        raise ValueError(
            "ancilla_qubits must have at least "
            f"{num_measurements} entries, got {len(ancilla_qubits)}."
        )
    selected_inversions = (
        (False,) * num_measurements
        if invert_results is None
        else tuple(invert_results[:num_measurements])
    )
    if len(selected_inversions) < num_measurements:
        raise ValueError(
            "invert_results must have at least "
            f"{num_measurements} entries, got {len(selected_inversions)}."
        )

    selected_bases = tuple(normalize_pauli_char(basis) for basis in bases[:num_measurements])
    used_ancillas = tuple(ancilla_qubits[:num_measurements])
    if len(used_ancillas) != len(set(used_ancillas)):
        raise ValueError("each parity-measurement schedule row requires a unique ancilla.")

    schedule_widths = {len(schedule) for schedule in schedules}
    if len(schedule_widths) > 1:
        raise ValueError("all data-qubit schedules must have the same number of steps.")

    schedule_layers = tuple(zip(*schedules, strict=True))
    for data_qubits_at_step in schedule_layers:
        active_data_qubits = tuple(
            data_qubit for data_qubit in data_qubits_at_step if data_qubit is not None
        )
        if len(active_data_qubits) != len(set(active_data_qubits)):
            raise ValueError("a data qubit cannot participate twice in one parallel step.")
        if set(active_data_qubits) & set(used_ancillas):
            raise ValueError("a measurement ancilla cannot also be data in one parallel step.")

    for basis, ancilla in zip(selected_bases, used_ancillas, strict=True):
        circuit.append({"X": "RX", "Y": "RY", "Z": "RZ"}[basis], [ancilla])
    if used_ancillas and add_ticks:
        circuit.append("TICK", [])

    for layer_index, data_qubits_at_step in enumerate(schedule_layers):
        if layer_index > 0 and not add_ticks:
            # Separate sequential shared-ancilla columns so Stim does not fuse
            # them into one multi-pair gate (which would hide hook faults).
            circuit.append("TICK", [])
        cx_targets: list[int] = []
        cy_targets: list[int] = []
        for data_qubit, basis, ancilla in zip(
            data_qubits_at_step,
            selected_bases,
            used_ancillas,
            strict=True,
        ):
            if data_qubit is None:
                continue
            if basis == "Z":
                cx_targets.extend((data_qubit, ancilla))
            elif basis == "X":
                cx_targets.extend((ancilla, data_qubit))
            elif basis == "Y":
                cy_targets.extend((ancilla, data_qubit))
        if cx_targets:
            circuit.append("CX", cx_targets)
        if cy_targets:
            circuit.append("CY", cy_targets)
        if add_ticks:
            circuit.append("TICK", [])
    for basis, ancilla, invert_result in zip(
        selected_bases,
        used_ancillas,
        selected_inversions,
        strict=True,
    ):
        target = stim.target_inv(ancilla) if invert_result else ancilla
        circuit.append({"X": "MX", "Y": "MY", "Z": "MZ"}[basis], [target])


def append_parity_measurement(
    circuit: stim.Circuit,
    basis: str,
    *,
    data_qubits: Sequence[int],
    ancilla_qubit: int,
    invert_result: bool = False,
) -> None:
    """Append a non-destructive Pauli-string parity measurement.

    The ancilla is reset and measured directly in ``basis`` without Hadamard
    basis changes. The result is appended to Stim's measurement record. Set
    ``invert_result`` when the physical Pauli-string parity has the opposite
    sign from the represented logical observable.
    """
    basis = normalize_pauli_char(basis)
    append_scheduled_parity_measurements(
        circuit,
        bases=(basis,),
        data_qubit_schedules=(tuple(data_qubits),),
        ancilla_qubits=(ancilla_qubit,),
        invert_results=(invert_result,),
        # Tick between the shared-ancilla CNOTs so noise sees each separately.
        add_ticks=True,
    )


# ---------------------------------------------------------------------------
# Logical measurements
# ---------------------------------------------------------------------------


def append_pauli_string_measurement(
    circuit: stim.Circuit,
    *,
    paulis: Sequence[str],
    data_qubits: Sequence[int],
    ancilla_qubit: int,
    invert_result: bool = False,
) -> None:
    """Append a non-destructive measurement of a mixed physical Pauli string.

    ``paulis`` and ``data_qubits`` give the Pauli acting on each physical
    qubit. Homogeneous strings reuse :func:`append_parity_measurement`. Mixed
    strings reset the ancilla with ``RX``, apply the corresponding controlled
    Pauli from the ancilla to each data qubit, and measure it with ``MX``.
    ``invert_result`` reverses the recorded sign.
    """
    selected_paulis = tuple(normalize_pauli_char(pauli) for pauli in paulis)
    selected_data_qubits = tuple(data_qubits)
    if len(selected_paulis) != len(selected_data_qubits):
        raise ValueError("paulis and data_qubits must have the same length.")
    if len(selected_data_qubits) != len(set(selected_data_qubits)):
        raise ValueError("a Pauli string can act on each data qubit at most once.")

    if selected_paulis and len(set(selected_paulis)) == 1:
        append_parity_measurement(
            circuit,
            selected_paulis[0],
            data_qubits=selected_data_qubits,
            ancilla_qubit=ancilla_qubit,
            invert_result=invert_result,
        )
        return

    circuit.append("RX", [ancilla_qubit])
    for index, (pauli, data_qubit) in enumerate(
        zip(selected_paulis, selected_data_qubits, strict=True)
    ):
        if index > 0:
            # Tick between shared-ancilla gates so same-Pauli runs are not fused.
            circuit.append("TICK", [])
        circuit.append({"X": "CX", "Y": "CY", "Z": "CZ"}[pauli], [ancilla_qubit, data_qubit])
    measurement_target: int | stim.GateTarget = ancilla_qubit
    if invert_result:
        measurement_target = stim.target_inv(measurement_target)
    circuit.append("MX", [measurement_target])


def append_logical_measurement_on_indexed_qubits(
    circuit: stim.Circuit,
    basis: str,
    *,
    indexed_qubits: Sequence[int],
    default_logical_x: tuple[int, ...],
    default_logical_z: tuple[int, ...],
    support: tuple[int, ...] | None,
    ancilla_qubit: int | None,
    destructive: bool,
) -> None:
    """Append a logical Pauli measurement to an existing Stim circuit.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Logical measurement basis ``"X"``, ``"Y"``, or ``"Z"``.
    indexed_qubits
        Physical qubits addressed by integer support indices.
    default_logical_x, default_logical_z
        Default representatives used when ``support`` is omitted. Logical Y
        measures ``i X_L Z_L`` using both.
    support
        Optional index support overriding X or Z. For Y it denotes a common
        X/Z support, as used by the self-dual Steane code.
    ancilla_qubit
        Ancilla required for non-destructive measurement.
    destructive
        Directly measure physical terms when true; otherwise append one
        ancilla-parity result.

    Logical Y uses Y on the X/Z intersection, X on X-only sites, and Z on
    Z-only sites. Outcomes are sign-corrected so parity zero denotes the
    logical ``+1`` eigenvalue.
    """
    basis = normalize_pauli_char(basis)
    qubits = tuple(indexed_qubits)
    if support is not None:
        logical_x_support = support if basis in {"X", "Y"} else default_logical_x
        logical_z_support = support if basis in {"Z", "Y"} else default_logical_z
    else:
        logical_x_support = default_logical_x
        logical_z_support = default_logical_z

    measurement_terms, invert_parity = resolve_logical_measurement_terms(
        basis,
        logical_x_support=logical_x_support,
        logical_z_support=logical_z_support,
    )

    if destructive:
        for output_index, (index, physical_basis) in enumerate(measurement_terms):
            target: int | stim.GateTarget = qubits[index]
            if invert_parity and output_index == 0:
                target = stim.target_inv(target)
            circuit.append({"X": "MX", "Y": "MY", "Z": "M"}[physical_basis], [target])
        return

    if ancilla_qubit is None:
        raise ValueError("ancilla_qubit is required for non-destructive measurement.")

    append_pauli_string_measurement(
        circuit,
        paulis=tuple(physical_basis for _, physical_basis in measurement_terms),
        data_qubits=tuple(qubits[index] for index, _ in measurement_terms),
        ancilla_qubit=ancilla_qubit,
        invert_result=invert_parity,
    )


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------


def two_qubit_depth(circuit: stim.Circuit) -> int:
    """Return the two-qubit gate depth of a circuit.

    Each batched two-qubit instruction is one parallel layer; noise channels are
    excluded so applying a noise model does not change the reported depth.
    """
    return sum(
        stim.gate_data(instruction.name).is_two_qubit_gate
        and not stim.gate_data(instruction.name).is_noisy_gate
        for instruction in circuit.flattened()
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def show_timeline(
    circuit: stim.Circuit,
    *,
    zoom: float = 1.0,
    max_height: int = 600,
    annotation_font_size: int = 18,
):
    """Return a notebook-renderable Stim timeline diagram of ``circuit``.

    Prefer this over ``print(circuit)`` when inspecting a Stim circuit in a
    notebook: gates, measurement records, detectors, and observables are shown
    in execution order on a white background that stays legible under dark
    editor themes.

    Parameters
    ----------
    zoom
        Horizontal scale factor; the surrounding box scrolls and is resizable.
    max_height
        Maximum rendered height in pixels before vertical scrolling.
    annotation_font_size
        Stim draws record, detector, and coordinate annotations at font size 8;
        they are rescaled to this value. Large values can overlap neighbouring
        columns, so raise ``zoom`` together with the font size.
    """
    from IPython.display import HTML

    svg = str(circuit.diagram("timeline-svg"))
    svg = svg.replace('font-size="8"', f'font-size="{annotation_font_size}"')
    return HTML(
        f'<div style="overflow:auto; max-height:{max_height}px; resize:both;'
        f' background:#ffffff; border:1px solid #cccccc; padding:8px">'
        f'<div style="width:{zoom * 100:.0f}%">{svg}</div></div>'
    )
