"""Circuit-level noise and deterministic Pauli faults for Stim."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import stim

from msc.utils.normalize import normalize_pauli_char

DEFAULT_PHYSICAL_ERROR_PROBABILITY = 0.001


# ---------------------------------------------------------------------------
# Stochastic circuit noise
# ---------------------------------------------------------------------------


def add_circuit_depolarizing_noise(
    circuit: stim.Circuit,
    *,
    p: float = DEFAULT_PHYSICAL_ERROR_PROBABILITY,
) -> stim.Circuit:
    """Return a uniformly noised circuit without an idle-error model.

    Every active location uses the same ``p``. One-qubit gates
    receive ``DEPOLARIZE1`` and two-qubit gates receive ``DEPOLARIZE2``.
    Z- and X-basis resets receive post-reset ``X_ERROR`` and ``Z_ERROR``,
    respectively. Measurements receive result flips; each MPP term receives an
    independent anticommuting Pauli fault. ``TICK`` instructions are preserved
    as schedule boundaries but do not produce noise. ``REPEAT`` bodies are
    transformed recursively without changing their structure.
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1].")

    def add_noise(selected_circuit: stim.Circuit) -> stim.Circuit:
        noisy_circuit = stim.Circuit()
        for instruction in selected_circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                noisy_circuit.append(
                    stim.CircuitRepeatBlock(
                        instruction.repeat_count,
                        add_noise(instruction.body_copy()),
                        tag=instruction.tag,
                    )
                )
                continue

            instruction_name = instruction.name
            gate = stim.gate_data(instruction_name)
            targets = instruction.targets_copy()
            qubits = [
                target.qubit_value
                for target in targets
                if (
                    target.is_qubit_target
                    or target.is_x_target
                    or target.is_y_target
                    or target.is_z_target
                )
            ]

            if instruction_name == "MPP" and p > 0:
                # Fault each Pauli term with an anticommuting error, matching
                # Hirano's product-measurement model.
                for target in targets:
                    if target.is_x_target or target.is_y_target:
                        noisy_circuit.append("Z_ERROR", [target.qubit_value], p)
                    elif target.is_z_target:
                        noisy_circuit.append("X_ERROR", [target.qubit_value], p)
                noisy_circuit.append(instruction)
            elif gate.produces_measurements and p > 0:
                # A Stim measurement argument flips the reported result.
                noisy_circuit.append(
                    instruction_name,
                    targets,
                    p,
                    tag=instruction.tag,
                )
            else:
                noisy_circuit.append(instruction)

            if not p:
                continue
            if gate.is_reset:
                # Apply reset noise afterward so it corrupts the prepared state.
                reset_error = {
                    "R": "X_ERROR",
                    "RZ": "X_ERROR",
                    "RX": "Z_ERROR",
                    "RY": "X_ERROR",
                }.get(instruction_name)
                if reset_error is None:
                    raise ValueError(f"unsupported reset gate for Hirano noise: {instruction_name}")
                noisy_circuit.append(reset_error, qubits, p)
            elif gate.is_unitary and gate.is_single_qubit_gate:
                noisy_circuit.append("DEPOLARIZE1", qubits, p)
            elif gate.is_unitary and gate.is_two_qubit_gate:
                if len(qubits) != len(set(qubits)):
                    # A multi-pair gate sharing a qubit runs sequentially, but one
                    # DEPOLARIZE2 after it cannot model faults between the gates.
                    raise ValueError(
                        f"two-qubit gate {instruction_name!r} reuses a qubit "
                        f"across its pairs ({qubits}); circuit-level noise cannot "
                        "place a fault location between the physically sequential "
                        "gates. Separate the pairs into distinct instructions."
                    )
                noisy_circuit.append("DEPOLARIZE2", qubits, p)

        return noisy_circuit

    return add_noise(circuit)


# ---------------------------------------------------------------------------
# Deterministic fault insertion
# ---------------------------------------------------------------------------


def append_pauli_faults(
    circuit: stim.Circuit,
    faults: Sequence[tuple[str, Hashable]],
    *,
    qubit_map: Mapping[Hashable, int] | Sequence[int] | None = None,
    as_error_channel: bool = False,
) -> None:
    """Append ordered deterministic Pauli faults in place.

    ``qubit_map`` resolves each ``(pauli, key)`` target. With
    ``as_error_channel=True``, append probability-1 error channels so detectors
    observe the faults; otherwise append Pauli gates.

    Call structure::

        circuit + [(pauli, qubit_key), ...]
        └── append_pauli_faults(...)
            └── mutated circuit
    """
    for pauli, qubit_key in faults:
        qubit = qubit_key if qubit_map is None else qubit_map[qubit_key]
        pauli = normalize_pauli_char(pauli)
        if as_error_channel:
            if pauli != "I":
                circuit.append(f"{pauli}_ERROR", [qubit], 1.0)
        else:
            circuit.append(pauli, [qubit])
