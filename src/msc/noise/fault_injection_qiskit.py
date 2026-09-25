"""Fault-injection and circuit-level noise helpers for Qiskit circuits."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from msc.utils.normalize import normalize_pauli_char

NOISE_P_1Q = 0.0005
NOISE_P_2Q = 0.005
NOISE_MEASURE = 0.005
NOISE_RESET = 0.005


# ---------------------------------------------------------------------------
# Stochastic circuit noise
# ---------------------------------------------------------------------------


def build_depolarizing_noise_model(
    circuit: QuantumCircuit,
    *,
    p_1q: float = NOISE_P_1Q,
    p_2q: float = NOISE_P_2Q,
    p_measure: float = NOISE_MEASURE,
    p_reset: float = NOISE_RESET,
) -> NoiseModel:
    """Build a circuit-level noise model for gates, measurements, and resets.

    Gate names are discovered from ``circuit`` instead of maintained in a
    hard-coded list. One- and two-qubit gates receive depolarizing errors,
    measurements receive symmetric readout errors, and resets receive
    single-qubit depolarizing errors. The default rates are ``NOISE_P_1Q``,
    ``NOISE_P_2Q``, ``NOISE_MEASURE``, and ``NOISE_RESET``.

    Parameters
    ----------
    circuit
        Circuit whose one- and two-qubit gates should receive noise.
    p_1q, p_2q
        Depolarizing probabilities after single- and two-qubit gates.
    p_measure
        Probability that a measurement result is reported flipped.
    p_reset
        Depolarizing probability after a reset.
    """
    gate_names_by_size: dict[int, set[str]] = {1: set(), 2: set()}
    has_measurements = False
    has_resets = False
    for circuit_instruction in circuit.data:
        operation = circuit_instruction.operation
        if isinstance(operation, Gate) and operation.num_qubits in gate_names_by_size:
            gate_names_by_size[operation.num_qubits].add(operation.name)
        elif operation.name == "measure":
            has_measurements = True
        elif operation.name == "reset":
            has_resets = True

    noise_model = NoiseModel()
    for num_qubits, probability in ((1, p_1q), (2, p_2q)):
        gate_names = gate_names_by_size[num_qubits]
        if not gate_names or probability == 0:
            continue

        error = depolarizing_error(probability, num_qubits)
        for gate_name in sorted(gate_names):
            noise_model.add_all_qubit_quantum_error(error, gate_name)

    if has_resets and p_reset != 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(p_reset, 1),
            "reset",
        )

    if has_measurements and p_measure > 0:
        noise_model.add_all_qubit_readout_error(
            ReadoutError(
                [
                    [1 - p_measure, p_measure],
                    [p_measure, 1 - p_measure],
                ]
            )
        )

    return noise_model


def build_noisy_aer_simulator(
    circuit: QuantumCircuit,
    *,
    p_1q: float = NOISE_P_1Q,
    p_2q: float = NOISE_P_2Q,
    p_measure: float = NOISE_MEASURE,
    p_reset: float = NOISE_RESET,
    seed_simulator: int | None = None,
) -> AerSimulator:
    """Build an Aer simulator with circuit-aware depolarizing noise.

    The noise-channel arguments have the same meaning and defaults as in
    :func:`build_depolarizing_noise_model`.
    """
    noise_model = build_depolarizing_noise_model(
        circuit,
        p_1q=p_1q,
        p_2q=p_2q,
        p_measure=p_measure,
        p_reset=p_reset,
    )
    return AerSimulator(
        noise_model=noise_model,
        seed_simulator=seed_simulator,
    )


# ---------------------------------------------------------------------------
# Deterministic fault insertion
# ---------------------------------------------------------------------------


def append_pauli_faults(
    circuit: QuantumCircuit,
    faults: Sequence[tuple[str, Hashable]],
    *,
    qubit_map: Mapping[Hashable, object] | Sequence[object] | None = None,
) -> None:
    """Append deterministic Pauli faults to qubits on any code patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    faults
        Sequence of ``(pauli, qubit_key)`` pairs where *pauli* is
        ``"I"``, ``"X"``, ``"Y"``, or ``"Z"`` and *qubit_key* identifies the
        target qubit via *qubit_map*.  Multiple faults are applied in
        order, supporting arbitrary-weight error injection.
    qubit_map
        Mapping (dict or sequence) from qubit key to Qiskit qubit object.
        If a dict, *qubit_key* is used as the dict key.
        If a sequence (list of qubits), *qubit_key* is used as an index.
        If ``None``, *qubit_key* is passed directly as the qubit argument
        (useful when keys are already raw qubit indices).

    Raises
    ------
    ValueError
        If a Pauli string is unrecognized.
    KeyError / IndexError
        If a qubit key is not found in *qubit_map*.
    """
    for pauli, qubit_key in faults:
        qubit = qubit_key if qubit_map is None else qubit_map[qubit_key]
        pauli = normalize_pauli_char(pauli)
        if pauli == "I":
            circuit.id(qubit)
        elif pauli == "X":
            circuit.x(qubit)
        elif pauli == "Y":
            circuit.y(qubit)
        elif pauli == "Z":
            circuit.z(qubit)
