"""Fault-injection and circuit-level noise helpers for Stim circuits.

The helpers append deterministic physical Pauli faults or return a noisy copy
of a noiseless ``stim.Circuit`` with standard circuit-level noise:

* ``DEPOLARIZE1`` after every single-qubit gate and after every reset,
* ``DEPOLARIZE2`` after every two-qubit gate,
* a measurement bit-flip probability on every measurement.

This is the noise source for native S-state cultivation logical-error-rate
(LER) estimates. No conversion to a T-state estimate is applied.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import stim

from magic_state_cultivation.utils.normalize import normalize_pauli_char


def add_circuit_depolarizing_noise(
    circuit: stim.Circuit,
    *,
    p_1q: float,
    p_2q: float,
    p_measure: float | None = None,
    p_reset: float | None = None,
) -> stim.Circuit:
    """Return a noisy copy of ``circuit`` with circuit-level depolarizing noise.

    Parameters
    ----------
    circuit
        Noiseless Stim circuit to add noise to.
    p_1q, p_2q
        Depolarizing probabilities after single- and two-qubit gates.
    p_measure
        Probability a measurement result is reported flipped. Defaults to
        ``p_2q``.
    p_reset
        Depolarizing probability after a reset. Defaults to ``p_2q``.
    """
    p_measure = p_2q if p_measure is None else p_measure
    p_reset = p_2q if p_reset is None else p_reset

    noisy = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise NotImplementedError(
                "add_circuit_depolarizing_noise expects a flat circuit (no REPEAT blocks)."
            )

        name = instruction.name
        gate = stim.gate_data(name)
        targets = instruction.targets_copy()
        qubits = [target.qubit_value for target in targets]

        if gate.produces_measurements:
            if p_measure > 0:
                noisy.append(name, targets, p_measure)
            else:
                noisy.append(name, targets, instruction.gate_args_copy())
            if gate.is_reset and p_reset > 0:
                noisy.append("DEPOLARIZE1", qubits, p_reset)
            continue

        noisy.append(name, targets, instruction.gate_args_copy())

        if not qubits:
            continue
        if gate.is_reset and p_reset > 0:
            noisy.append("DEPOLARIZE1", qubits, p_reset)
        elif gate.is_unitary and gate.is_single_qubit_gate and p_1q > 0:
            noisy.append("DEPOLARIZE1", qubits, p_1q)
        elif gate.is_unitary and gate.is_two_qubit_gate and p_2q > 0:
            noisy.append("DEPOLARIZE2", qubits, p_2q)

    return noisy


def append_pauli_faults(
    circuit: stim.Circuit,
    faults: Sequence[tuple[str, Hashable]],
    *,
    qubit_map: Mapping[Hashable, int] | Sequence[int] | None = None,
) -> None:
    """Append deterministic Pauli faults to qubits in a Stim circuit.

    Parameters
    ----------
    circuit
        Circuit being modified.
    faults
        Sequence of ``(pauli, qubit_key)`` pairs where *pauli* is ``"I"``,
        ``"X"``, ``"Y"``, or ``"Z"`` and *qubit_key* identifies the target
        through *qubit_map*. Multiple faults are applied in order.
    qubit_map
        Mapping from qubit keys to Stim qubit indices, or a sequence indexed
        by the qubit keys. If ``None``, each key is used directly as a Stim
        qubit index.

    Raises
    ------
    ValueError
        If a Pauli string is unrecognized.
    KeyError / IndexError
        If a qubit key is not found in *qubit_map*.
    """
    for pauli, key in faults:
        if qubit_map is None:
            qubit = key
        elif isinstance(qubit_map, Mapping):
            qubit = qubit_map[key]
        else:
            qubit = qubit_map[key]

        normalized_pauli = normalize_pauli_char(pauli)
        circuit.append(normalized_pauli, [qubit])
