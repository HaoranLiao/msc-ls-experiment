"""Circuit-level noise and deterministic Pauli faults for Stim."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import stim

from msc.utils.normalize import normalize_pauli_char

NOISE_P_1Q = 0.0005
NOISE_P_2Q = 0.005
NOISE_MEASURE = 0.005
NOISE_RESET = 0.005


# ---------------------------------------------------------------------------
# Stochastic circuit noise
# ---------------------------------------------------------------------------


def add_circuit_depolarizing_noise(
    circuit: stim.Circuit,
    *,
    p_1q: float = NOISE_P_1Q,
    p_2q: float = NOISE_P_2Q,
    p_measure: float = NOISE_MEASURE,
    p_reset: float = NOISE_RESET,
) -> stim.Circuit:
    """Return a noisy copy of a circuit, including nested ``REPEAT`` bodies.

    Adds ``DEPOLARIZE1`` after one-qubit gates and resets,
    ``DEPOLARIZE2`` after two-qubit gates, and measurement-result flips.
    The default rates are ``NOISE_P_1Q``, ``NOISE_P_2Q``, ``NOISE_MEASURE``,
    and ``NOISE_RESET``.

    Call structure::

        noiseless circuit
        └── add_circuit_depolarizing_noise(...)
            ├── recursively add noise inside each REPEAT body
            └── noisy circuit with the same REPEAT structure
    """
    for probability_name, probability in (
        ("p_1q", p_1q),
        ("p_2q", p_2q),
        ("p_measure", p_measure),
        ("p_reset", p_reset),
    ):
        if not 0 <= probability <= 1:
            raise ValueError(f"{probability_name} must be in [0, 1].")

    noisy_circuit = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            noisy_circuit.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    add_circuit_depolarizing_noise(
                        instruction.body_copy(),
                        p_1q=p_1q,
                        p_2q=p_2q,
                        p_measure=p_measure,
                        p_reset=p_reset,
                    ),
                    tag=instruction.tag,
                )
            )
            continue

        instruction_name = instruction.name
        gate = stim.gate_data(instruction_name)
        targets = instruction.targets_copy()

        if gate.produces_measurements and p_measure > 0:
            noisy_circuit.append(instruction_name, targets, p_measure, tag=instruction.tag)
        else:
            noisy_circuit.append(
                instruction_name,
                targets,
                instruction.gate_args_copy(),
                tag=instruction.tag,
            )

        if gate.is_reset and p_reset > 0:
            qubits = [target.qubit_value for target in targets]
            noisy_circuit.append("DEPOLARIZE1", qubits, p_reset)
        elif gate.is_unitary and gate.is_single_qubit_gate and p_1q > 0:
            qubits = [target.qubit_value for target in targets]
            noisy_circuit.append("DEPOLARIZE1", qubits, p_1q)
        elif gate.is_unitary and gate.is_two_qubit_gate and p_2q > 0:
            qubits = [target.qubit_value for target in targets]
            noisy_circuit.append("DEPOLARIZE2", qubits, p_2q)

    return noisy_circuit


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
