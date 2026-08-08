"""Tests for Stim circuit-level noise insertion."""

import random

import stim

from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise


def test_depolarizing_noise_covers_every_supported_operation() -> None:
    """Check every supported operation after randomizing order and targets."""
    rng = random.Random(104_729)
    gate_data = stim.gate_data()
    operation_names = sorted(
        {
            gate.name
            for gate in gate_data.values()
            if (
                gate.is_reset
                or (gate.is_unitary and gate.is_single_qubit_gate)
                or (gate.is_unitary and gate.is_two_qubit_gate)
                or (gate.produces_measurements and not gate.is_noisy_gate)
            )
        }
    )
    rng.shuffle(operation_names)

    circuit = stim.Circuit()
    expected_noisy_circuit = stim.Circuit()
    for operation_name in operation_names:
        gate = gate_data[operation_name]
        if operation_name == "MPP":
            qubit, other_qubit = rng.sample(range(20), 2)
            targets = [
                stim.target_x(qubit),
                stim.target_combiner(),
                stim.target_z(other_qubit),
            ]
            qubits: list[int] = []
        elif operation_name == "MPAD":
            targets = [rng.randrange(2)]
            qubits = []
        elif gate.is_two_qubit_gate or operation_name in {"MXX", "MYY", "MZZ"}:
            qubits = rng.sample(range(20), 2)
            targets = qubits
        else:
            qubits = [rng.randrange(20)]
            targets = qubits

        circuit.append(operation_name, targets)
        circuit.append("TICK")

        if gate.produces_measurements:
            expected_noisy_circuit.append(operation_name, targets, 0.03)
        else:
            expected_noisy_circuit.append(operation_name, targets)
        if gate.is_reset:
            expected_noisy_circuit.append("DEPOLARIZE1", qubits, 0.04)
        elif gate.is_unitary and gate.is_single_qubit_gate:
            expected_noisy_circuit.append("DEPOLARIZE1", qubits, 0.01)
        elif gate.is_unitary and gate.is_two_qubit_gate:
            expected_noisy_circuit.append("DEPOLARIZE2", qubits, 0.02)
        expected_noisy_circuit.append("TICK")

    noisy_circuit = add_circuit_depolarizing_noise(
        circuit,
        p_1q=0.01,
        p_2q=0.02,
        p_measure=0.03,
        p_reset=0.04,
    )

    assert noisy_circuit == expected_noisy_circuit
