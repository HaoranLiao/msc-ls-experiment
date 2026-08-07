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
from dataclasses import dataclass
from itertools import product

import stim

from msc.utils.normalize import normalize_pauli_char


@dataclass(frozen=True)
class CircuitSingleFault:
    """One elementary fault mechanism from a noisy Stim circuit.

    ``instruction_index`` identifies the stochastic instruction in the input
    circuit. ``target_group_index`` identifies one independently sampled
    qubit, qubit pair, or measurement result within that instruction.
    ``paulis`` is empty for a classical measurement-result flip.
    """

    instruction_index: int
    instruction_name: str
    target_group_index: int
    qubits: tuple[int, ...]
    paulis: tuple[str, ...]


_NONIDENTITY_SINGLE_QUBIT_PAULIS = ("X", "Y", "Z")
_NONIDENTITY_TWO_QUBIT_PAULIS = tuple(
    paulis for paulis in product(("I", "X", "Y", "Z"), repeat=2) if paulis != ("I", "I")
)


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

        if gate.produces_measurements and p_measure > 0:
            noisy.append(name, targets, p_measure)
        else:
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


def enumerate_circuit_single_faults(
    circuit: stim.Circuit,
) -> tuple[CircuitSingleFault, ...]:
    """Enumerate elementary faults represented by this circuit's noise channels.

    The supported channels are exactly those emitted by
    :func:`add_circuit_depolarizing_noise`: independent ``DEPOLARIZE1`` and
    ``DEPOLARIZE2`` locations plus independent reported-result flips on
    measurement targets. Zero-probability channels are ignored.
    """
    faults: list[CircuitSingleFault] = []
    for instruction_index, instruction in enumerate(circuit):
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError("single-fault enumeration requires a flat circuit.")
        name = instruction.name
        targets = instruction.targets_copy()
        probabilities = instruction.gate_args_copy()
        is_active_channel = bool(probabilities) and probabilities[0] > 0
        if name == "DEPOLARIZE1":
            if is_active_channel:
                for target_index, target in enumerate(targets):
                    for pauli in _NONIDENTITY_SINGLE_QUBIT_PAULIS:
                        faults.append(
                            CircuitSingleFault(
                                instruction_index=instruction_index,
                                instruction_name=name,
                                target_group_index=target_index,
                                qubits=(target.qubit_value,),
                                paulis=(pauli,),
                            )
                        )
            continue
        if name == "DEPOLARIZE2":
            if len(targets) % 2:
                raise ValueError("DEPOLARIZE2 targets must form qubit pairs.")
            if is_active_channel:
                for pair_index in range(len(targets) // 2):
                    pair = targets[2 * pair_index : 2 * pair_index + 2]
                    for paulis in _NONIDENTITY_TWO_QUBIT_PAULIS:
                        faults.append(
                            CircuitSingleFault(
                                instruction_index=instruction_index,
                                instruction_name=name,
                                target_group_index=pair_index,
                                qubits=tuple(target.qubit_value for target in pair),
                                paulis=paulis,
                            )
                        )
            continue
        gate = stim.gate_data(name)
        if gate.produces_measurements and is_active_channel:
            if not all(target.is_qubit_target for target in targets):
                raise NotImplementedError(
                    "single-fault enumeration does not support composite "
                    f"measurement targets for {name}."
                )
            for target_index, target in enumerate(targets):
                faults.append(
                    CircuitSingleFault(
                        instruction_index=instruction_index,
                        instruction_name=name,
                        target_group_index=target_index,
                        qubits=(target.qubit_value,),
                        paulis=(),
                    )
                )
    return tuple(faults)


def append_pauli_faults(
    circuit: stim.Circuit,
    faults: Sequence[tuple[str, Hashable]],
    *,
    qubit_map: Mapping[Hashable, int] | Sequence[int] | None = None,
    as_error_channel: bool = False,
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
    as_error_channel
        When ``False`` (default) each fault is a deterministic Pauli gate. When
        ``True`` each fault is a probability-1 ``X_ERROR``/``Y_ERROR``/
        ``Z_ERROR`` channel instead (identity faults are skipped). Stim folds a
        deterministic gate into the noiseless reference, so its change detectors
        stay silent; a probability-1 error channel registers as a detector
        deviation and therefore makes those detectors fire.

    Raises
    ------
    ValueError
        If a Pauli string is unrecognized.
    KeyError / IndexError
        If a qubit key is not found in *qubit_map*.
    """
    for pauli, key in faults:
        qubit = key if qubit_map is None else qubit_map[key]
        pauli = normalize_pauli_char(pauli)
        if as_error_channel:
            if pauli != "I":
                circuit.append(f"{pauli}_ERROR", [qubit], 1.0)
        else:
            circuit.append(pauli, [qubit])
