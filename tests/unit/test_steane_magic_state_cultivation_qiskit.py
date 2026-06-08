"""Tests for the Steane-code cultivation helpers."""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    DATA_SITES,
    FOLDED_MEASUREMENT_SITE,
    HELPER_SITES,
    append_fold_network,
    append_steane_state_injection_encoder,
    append_unfold_network,
    apply_google_ty,
    apply_google_ty_dagger,
    build_steane_injection_with_h_cultivation,
    is_successful_cultivation_bitstring,
    postselect_success_counts,
)


NUM_PATCH_QUBITS = 11
_PROJECTOR_ZERO = np.array([[1.0, 0.0], [0.0, 0.0]])

# The H-axis magic state angle: cos(pi/8)|0> + sin(pi/8)|1>.
_H_STATE_ANGLE = math.pi / 4


def _build_h_state_on_patch() -> Statevector:
    """Build |H_L> embedded in the 11-site patch using the encoder."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    append_steane_state_injection_encoder(
        circuit,
        patch_qubits=list(circuit.qubits),
        injected_state="H",
    )
    return Statevector.from_instruction(circuit)


def _project_onto_x_plus(state: Statevector, qubit: int) -> tuple[Statevector, float]:
    """Project onto the ``+1`` eigenspace of ``X`` on one qubit."""
    basis_change = QuantumCircuit(NUM_PATCH_QUBITS)
    basis_change.h(qubit)
    rotated = state.evolve(basis_change)

    ops = [np.eye(2)] * NUM_PATCH_QUBITS
    ops[qubit] = _PROJECTOR_ZERO
    projector = reduce(np.kron, [ops[NUM_PATCH_QUBITS - 1 - i] for i in range(NUM_PATCH_QUBITS)])

    projected = projector @ rotated.data
    probability = float(np.linalg.norm(projected) ** 2)
    if probability == 0:
        raise ValueError("projection onto X=+1 had zero probability")

    return Statevector(projected / math.sqrt(probability)), probability


def _helper_x_probabilities_after_successful_root_projection() -> dict[int, float]:
    """Return the helper ``X=+1`` probabilities after a successful root check."""
    state = _build_h_state_on_patch()

    forward = QuantumCircuit(NUM_PATCH_QUBITS)
    for helper_site in HELPER_SITES:
        forward.h(helper_site)
    for data_site in DATA_SITES:
        apply_google_ty(forward, data_site)
    append_fold_network(forward, list(range(NUM_PATCH_QUBITS)), add_barriers=False)
    state = state.evolve(forward)

    state, _ = _project_onto_x_plus(state, FOLDED_MEASUREMENT_SITE)

    backward = QuantumCircuit(NUM_PATCH_QUBITS)
    backward.h(FOLDED_MEASUREMENT_SITE)
    append_unfold_network(backward, list(range(NUM_PATCH_QUBITS)), add_barriers=False)
    state = state.evolve(backward)

    helper_probabilities: dict[int, float] = {}
    for helper_site in HELPER_SITES:
        _, probability = _project_onto_x_plus(state, helper_site)
        helper_probabilities[helper_site] = probability
    return helper_probabilities


def test_build_steane_injection_with_h_cultivation_shape() -> None:
    circuit = build_steane_injection_with_h_cultivation(injected_state="H")

    assert circuit.num_qubits == 11
    assert circuit.num_clbits == 5
    assert circuit.count_ops()["measure"] == 5
    assert circuit.count_ops()["reset"] == 5


def test_encoder_prepares_noiseless_data_only_state() -> None:
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    append_steane_state_injection_encoder(
        circuit,
        patch_qubits=list(circuit.qubits),
        injected_state="H",
    )
    state = Statevector.from_instruction(circuit)

    helper_probabilities = state.probabilities(HELPER_SITES)
    assert helper_probabilities[0] == pytest.approx(1.0)
    assert helper_probabilities[1:].sum() == pytest.approx(0.0)


def test_folded_root_measurement_accepts_logical_h_with_probability_one() -> None:
    state = _build_h_state_on_patch()

    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    for helper_site in HELPER_SITES:
        circuit.h(helper_site)
    for data_site in DATA_SITES:
        apply_google_ty(circuit, data_site)
    append_fold_network(circuit, list(range(NUM_PATCH_QUBITS)), add_barriers=False)
    state = state.evolve(circuit)

    _, probability = _project_onto_x_plus(state, FOLDED_MEASUREMENT_SITE)
    assert probability == pytest.approx(1.0)


def test_helper_measurements_are_deterministic_after_successful_root_projection() -> None:
    probabilities = _helper_x_probabilities_after_successful_root_projection()

    assert probabilities == {
        1: pytest.approx(1.0),
        2: pytest.approx(1.0),
        6: pytest.approx(1.0),
        7: pytest.approx(1.0),
    }


def test_ty_and_inverse_cancel_on_data_after_successful_projection_round() -> None:
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    for data_site in DATA_SITES:
        apply_google_ty(circuit, data_site)
        apply_google_ty_dagger(circuit, data_site)

    state = Statevector.from_instruction(circuit)
    assert state == Statevector.from_label("0" * NUM_PATCH_QUBITS)


def test_full_circuit_postselects_noiseless_logical_h_with_probability_one() -> None:
    circuit = build_steane_injection_with_h_cultivation(
        injected_state="H",
        add_barriers=False,
    )

    simulator = AerSimulator()
    shots = 512
    counts = simulator.run(circuit, shots=shots).result().get_counts()

    assert counts == {"00000": shots}


def test_postselection_helpers_filter_all_zero_bitstrings() -> None:
    counts = {
        "00000": 8,
        "10000": 3,
        "00001": 2,
        "00 000": 5,
    }

    assert is_successful_cultivation_bitstring("00000")
    assert is_successful_cultivation_bitstring("00 000")
    assert not is_successful_cultivation_bitstring("10000")
    assert not is_successful_cultivation_bitstring("0000")
    assert postselect_success_counts(counts) == {"00000": 8, "00 000": 5}
