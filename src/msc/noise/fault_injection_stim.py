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


# Frame changes on IBM devices (virtual Z rotations): no moment of their own and no noise.
_VIRTUAL_GATES = frozenset({"Z", "S", "S_DAG"})
_MEASUREMENTS = frozenset({"M", "MX"})
_RESETS = frozenset({"R", "RX"})


def schedule_si1000_moments(circuit: stim.Circuit) -> tuple[stim.Circuit, list[int]]:
    """Lay a noiseless model without schedule into moments, as Google's circuits are laid out.

    ``add_si1000_noise`` needs moments, and a converted hardware model has none. Each moment
    holds one kind of operation, scheduled greedily in dependency order: single-qubit gates
    whenever any are ready (they unblock two-qubit gates), else two-qubit gates. Measurements
    wait until no gate is ready anywhere, so a round's ancillas are measured in one moment
    rather than whenever each finishes; resets of measured qubits follow in their own moment,
    as in Google's circuits. Virtual gates take no moment; each is emitted just before its
    qubit's next operation.

    Measurements are recorded in moment order, since a converted model's record order need not
    follow time. Annotations move to the end, re-targeted to their measurements, so detector and
    observable indices are unchanged. Also returns the input record of each output record.
    """
    # Per qubit, the indices of its operations in program order.
    operations: list[tuple[str, tuple[int, ...], list[str], int | None]] = []
    chains: dict[int, list[int]] = {}
    pending_virtual_gates: dict[int, list[str]] = {}
    coordinates: list[str] = []
    annotations: list[tuple[str, list[int], list[float]]] = []
    measurement_count = 0
    for instruction in circuit.flattened():
        name = instruction.name
        targets = [target.value for target in instruction.targets_copy()]
        if name == "QUBIT_COORDS":
            coordinates.append(str(instruction))
            continue
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            # Input records, re-targeted once every measurement is placed.
            annotations.append(
                (name, [measurement_count + target for target in targets], instruction.gate_args_copy())
            )
            continue
        if name == "TICK":
            continue
        if name in _VIRTUAL_GATES:
            for qubit in targets:
                pending_virtual_gates.setdefault(qubit, []).append(f"{name} {qubit}")
            continue
        gate = stim.gate_data(name)
        if gate.is_unitary and gate.is_two_qubit_gate:
            units = list(zip(targets[::2], targets[1::2], strict=True))
        elif (gate.is_unitary and gate.is_single_qubit_gate) or (
            name in _MEASUREMENTS | _RESETS and not instruction.gate_args_copy()
        ):
            units = [(qubit,) for qubit in targets]
        else:
            raise ValueError(f"schedule_si1000_moments expects a noiseless circuit; got {instruction}.")
        for qubits in units:
            for qubit in qubits:
                chains.setdefault(qubit, []).append(len(operations))
            operations.append(
                (
                    name,
                    qubits,
                    [line for qubit in qubits for line in pending_virtual_gates.pop(qubit, [])],
                    measurement_count if name in _MEASUREMENTS else None,
                )
            )
            measurement_count += name in _MEASUREMENTS

    def kind(name: str) -> str:
        if name in _MEASUREMENTS:
            return "measure"
        if name in _RESETS:
            return "reset"
        return "two_qubit" if stim.gate_data(name).is_two_qubit_gate else "single_qubit"

    heads = dict.fromkeys(chains, 0)
    lines = list(coordinates)
    measurement_order: list[int] = []
    while True:
        ready_by_kind: dict[str, list[int]] = {}
        for qubit, head in heads.items():
            if head < len(chains[qubit]):
                index = chains[qubit][head]
                name, qubits, _, _ = operations[index]
                if qubits[0] == qubit and all(chains[q][heads[q]] == index for q in qubits):
                    ready_by_kind.setdefault(kind(name), []).append(index)
        moment_kind = next(
            (k for k in ("single_qubit", "two_qubit", "measure", "reset") if k in ready_by_kind), None
        )
        if moment_kind is None:
            break
        for index in sorted(ready_by_kind[moment_kind]):
            name, qubits, virtual_gates, input_record = operations[index]
            lines += [*virtual_gates, f"{name} {' '.join(map(str, qubits))}"]
            if input_record is not None:
                measurement_order.append(input_record)
            for qubit in qubits:
                heads[qubit] += 1
        lines.append("TICK")
    lines.extend(line for virtual_gates in pending_virtual_gates.values() for line in virtual_gates)
    output_records = {record: index for index, record in enumerate(measurement_order)}
    for name, records, arguments in annotations:
        arguments_text = f"({', '.join(map(repr, arguments))})" if arguments else ""
        lines.append(
            f"{name}{arguments_text} "
            + " ".join(f"rec[{output_records[record] - measurement_count}]" for record in records)
        )
    return stim.Circuit("\n".join(lines)), measurement_order


def add_si1000_noise(circuit: stim.Circuit, *, p: float) -> stim.Circuit:
    """Add SI1000 noise moment by moment, in the convention of Google's released grafting DEMs.

    SI1000 (Gidney, Newman, and McEwen, Quantum 6, 813 (2022)) is superconducting-inspired
    circuit noise; this convention reproduces the detector error models Rosenfeld et al.
    released with their Fig. 4 Stim twins (built at ``p = 1e-3``) from the noiseless twins:
    ``DEPOLARIZE2(p)`` after two-qubit gates, ``DEPOLARIZE1(p/10)`` after single-qubit gates
    (Google's echo identities included), result flips of ``5p``, flips of ``2p`` after resets,
    and idle ``DEPOLARIZE1`` on every qubit of the circuit a moment leaves alone, from the first
    moment on: ``2p`` in reset moments and ``p/10`` in gate moments. In measurement moments,
    Google echoes exactly the qubits already in use (``p/10``) and leaves the qubits not yet used
    idle (``2p``), so idle qubits get ``p/10`` once in use, as if dynamical decoupling ran during
    readout, and ``2p`` before.

    ``TICK`` separates moments (``schedule_si1000_moments`` lays out a model without them).
    Virtual gates (``Z``, ``S``, ``S_DAG``) are frame changes: no noise, and their qubits still
    idle.
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1].")
    moments: list[list[stim.CircuitInstruction]] = [[]]
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            moments.append([])
        else:
            moments[-1].append(instruction)
    noisy_circuit = stim.Circuit()
    used_qubits: set[int] = set()
    for moment_index, moment in enumerate(moments):
        names = {instruction.name for instruction in moment}
        if names & _MEASUREMENTS and names & _RESETS:
            raise ValueError("a moment must not mix measurements and resets.")
        acted_qubits: set[int] = set()
        after_moment = stim.Circuit()
        for instruction in moment:
            name = instruction.name
            targets = [target.value for target in instruction.targets_copy()]
            if name in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS") or (
                name in _VIRTUAL_GATES
            ):
                noisy_circuit.append(instruction)
                continue
            gate = stim.gate_data(name)
            if instruction.gate_args_copy():
                raise ValueError(f"add_si1000_noise expects a noiseless circuit; got {instruction}.")
            acted_qubits.update(targets)
            if name in _MEASUREMENTS:
                noisy_circuit.append(name, targets, 5 * p)
            elif name in _RESETS:
                noisy_circuit.append(instruction)
                after_moment.append("X_ERROR" if name == "R" else "Z_ERROR", targets, 2 * p)
            elif gate.is_unitary and gate.is_two_qubit_gate:
                # Pairs apply in parallel, so each pair's fault follows the whole instruction.
                noisy_circuit.append(instruction)
                after_moment.append("DEPOLARIZE2", targets, p)
            elif gate.is_unitary and gate.is_single_qubit_gate:
                noisy_circuit.append(instruction)
                after_moment.append("DEPOLARIZE1", targets, p / 10)
            else:
                raise ValueError(f"unsupported operation {instruction}.")
        if acted_qubits:
            for qubit in range(circuit.num_qubits):
                if qubit in acted_qubits:
                    continue
                if names & _RESETS or (names & _MEASUREMENTS and qubit not in used_qubits):
                    after_moment.append("DEPOLARIZE1", [qubit], 2 * p)
                else:
                    after_moment.append("DEPOLARIZE1", [qubit], p / 10)
        used_qubits |= acted_qubits
        noisy_circuit += after_moment
        if moment_index < len(moments) - 1:
            noisy_circuit.append("TICK")
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
