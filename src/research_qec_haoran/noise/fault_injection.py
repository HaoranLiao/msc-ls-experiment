"""Manual fault-injection helpers for Stim circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import stim

Coord = tuple[int, int]
Instruction = stim.CircuitInstruction | stim.CircuitRepeatBlock


def _qubit_index_by_coord(circuit: stim.Circuit) -> dict[Coord, int]:
    """Map each qubit coordinate in ``circuit`` to its Stim qubit index."""

    return {
        tuple(int(value) for value in coords[:2]): qubit
        for qubit, coords in circuit.get_final_qubit_coordinates().items()
    }


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

    flattened = circuit.flattened()
    instructions = list(flattened)
    insert_before_index = _insertion_index_after_instruction_occurrence(
        instructions=instructions,
        instruction_name=instruction_name,
        occurrence=occurrence,
    )
    return _circuit_with_inserted_error_instruction(
        instructions=instructions,
        pauli=pauli,
        coords=coords,
        circuit=flattened,
        insert_before_index=insert_before_index,
    )


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
    """Insert deterministic Pauli errors before the final matching measurement block."""

    measurement_name_set = set(measurement_names)
    if not measurement_name_set:
        raise ValueError("measurement_names must not be empty")

    flattened = circuit.flattened()
    instructions = list(flattened)
    insert_before_index = _first_instruction_index_of_final_measurement_block(
        instructions=instructions,
        measurement_names=measurement_name_set,
    )
    return _circuit_with_inserted_error_instruction(
        instructions=instructions,
        pauli=pauli,
        coords=coords,
        circuit=flattened,
        insert_before_index=insert_before_index,
    )


def _circuit_with_inserted_error_instruction(
    *,
    instructions: list[Instruction],
    pauli: str,
    coords: Iterable[Coord],
    circuit: stim.Circuit,
    insert_before_index: int,
) -> stim.Circuit:
    qubits = _qubits_for_coords(circuit, coords)
    error_instruction = _pauli_error_instruction(pauli)

    out = stim.Circuit()
    for index, instruction in enumerate(instructions):
        if index == insert_before_index:
            out.append(error_instruction, qubits, 1.0)
        out.append(instruction)

    if insert_before_index == len(instructions):
        out.append(error_instruction, qubits, 1.0)

    return out


def _insertion_index_after_instruction_occurrence(
    *,
    instructions: list[Instruction],
    instruction_name: str,
    occurrence: int,
) -> int:
    seen_matches = 0
    for index, instruction in enumerate(instructions):
        if isinstance(instruction, stim.CircuitRepeatBlock):
            continue
        if instruction.name == instruction_name:
            seen_matches += 1
            if seen_matches == occurrence:
                return index + 1

    raise ValueError(f"circuit only has {seen_matches} occurrences of {instruction_name!r}")


def _first_instruction_index_of_final_measurement_block(
    *,
    instructions: list[Instruction],
    measurement_names: set[str],
) -> int:
    final_measurement_index = None
    for index, instruction in enumerate(instructions):
        if isinstance(instruction, stim.CircuitRepeatBlock):
            continue
        if instruction.name in measurement_names:
            final_measurement_index = index

    if final_measurement_index is None:
        names = ", ".join(sorted(measurement_names))
        raise ValueError(
            f"circuit does not contain a terminal measurement with name in {{{names}}}"
        )

    block_start = final_measurement_index
    while block_start > 0:
        previous_instruction = instructions[block_start - 1]
        if isinstance(previous_instruction, stim.CircuitRepeatBlock):
            break
        if previous_instruction.name not in measurement_names:
            break
        block_start -= 1

    return block_start


def _qubits_for_coords(circuit: stim.Circuit, coords: Iterable[Coord]) -> list[int]:
    coord_to_qubit = _qubit_index_by_coord(circuit)
    resolved_coords = list(coords)
    if not resolved_coords:
        raise ValueError("coords must not be empty")
    return [coord_to_qubit[coord] for coord in resolved_coords]


def _pauli_error_instruction(pauli: str) -> str:
    normalized = pauli.upper()
    if normalized not in {"X", "Y", "Z"}:
        raise ValueError("pauli must be 'X', 'Y', or 'Z'")
    return f"{normalized}_ERROR"
