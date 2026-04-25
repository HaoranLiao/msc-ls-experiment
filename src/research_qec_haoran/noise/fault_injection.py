"""Deterministic fault-injection helpers for Stim circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import stim

Coord = tuple[int, int]


def inject_pauli_errors_after_instruction_occurrence(
    circuit: stim.Circuit,
    *,
    pauli: str,
    coords: Iterable[Coord],
    instruction_name: str,
    occurrence: int,
) -> stim.Circuit:
    """Insert deterministic Pauli errors after the ``occurrence``-th matching instruction."""
    if occurrence < 1:
        raise ValueError("occurrence must be at least 1")

    flat = circuit.flattened()
    instructions = list(flat)

    seen = 0
    insert_idx: int | None = None
    for i, inst in enumerate(instructions):
        if isinstance(inst, stim.CircuitRepeatBlock):
            continue
        if inst.name == instruction_name:
            seen += 1
            if seen == occurrence:
                insert_idx = i + 1
                break
    if insert_idx is None:
        raise ValueError(f"circuit only has {seen} occurrences of {instruction_name!r}")

    return _build_circuit_with_error(instructions, flat, pauli, coords, insert_idx)


def inject_pauli_errors_after_measurement_round(
    circuit: stim.Circuit,
    *,
    pauli: str,
    coords: Iterable[Coord],
    measurement_round: int,
) -> stim.Circuit:
    """Insert deterministic Pauli errors right after the chosen ancilla-measurement round."""
    return inject_pauli_errors_after_instruction_occurrence(
        circuit,
        pauli=pauli,
        coords=coords,
        instruction_name="MR",
        occurrence=measurement_round,
    )


def inject_pauli_errors_before_final_measurement(
    circuit: stim.Circuit,
    *,
    pauli: str,
    coords: Iterable[Coord],
    measurement_names: Sequence[str] = ("M", "MX", "MY"),
) -> stim.Circuit:
    """Insert deterministic Pauli errors before the final measurement block."""
    name_set = set(measurement_names)
    if not name_set:
        raise ValueError("measurement_names must not be empty")

    flat = circuit.flattened()
    instructions = list(flat)

    # Find the start of the final contiguous measurement block by scanning backward.
    last_meas_idx: int | None = None
    for i in range(len(instructions) - 1, -1, -1):
        inst = instructions[i]
        if isinstance(inst, stim.CircuitRepeatBlock):
            continue
        if inst.name in name_set:
            last_meas_idx = i
            break
    if last_meas_idx is None:
        raise ValueError(f"circuit has no terminal measurement in {{{', '.join(sorted(name_set))}}}")

    # Walk backward from the last measurement to find the block start.
    block_start = last_meas_idx
    while block_start > 0:
        prev = instructions[block_start - 1]
        if isinstance(prev, stim.CircuitRepeatBlock) or prev.name not in name_set:
            break
        block_start -= 1

    return _build_circuit_with_error(instructions, flat, pauli, coords, block_start)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_circuit_with_error(
    instructions: list[stim.CircuitInstruction | stim.CircuitRepeatBlock],
    circuit: stim.Circuit,
    pauli: str,
    coords: Iterable[Coord],
    insert_before: int,
) -> stim.Circuit:
    """Build a new circuit with a Pauli error instruction inserted at ``insert_before``."""
    error_gate = _pauli_error_gate(pauli)
    qubit_indices = _resolve_qubit_indices(circuit, coords)

    out = stim.Circuit()
    for i, inst in enumerate(instructions):
        if i == insert_before:
            out.append(error_gate, qubit_indices, 1.0)
        out.append(inst)
    # Handle insertion at the very end.
    if insert_before == len(instructions):
        out.append(error_gate, qubit_indices, 1.0)
    return out


def _resolve_qubit_indices(circuit: stim.Circuit, coords: Iterable[Coord]) -> list[int]:
    """Map ``(x, y)`` coordinates to Stim qubit indices."""
    coord_to_qubit = {
        (int(xy[0]), int(xy[1])): q
        for q, xy in circuit.get_final_qubit_coordinates().items()
    }
    resolved = list(coords)
    if not resolved:
        raise ValueError("coords must not be empty")
    return [coord_to_qubit[c] for c in resolved]


def _pauli_error_gate(pauli: str) -> str:
    """Return the Stim error gate name for a single-qubit Pauli."""
    p = pauli.upper()
    if p not in ("X", "Y", "Z"):
        raise ValueError("pauli must be 'X', 'Y', or 'Z'")
    return f"{p}_ERROR"
