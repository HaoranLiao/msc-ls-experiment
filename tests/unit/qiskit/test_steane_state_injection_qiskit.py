"""Tests for the Steane-code state-injection encoder."""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from research_qec_haoran.steane_code.state_injection_qiskit import (
    LOGICAL_X_REPRESENTATIVES,
    LOGICAL_Z_REPRESENTATIVES,
    NUM_PATCH_QUBITS,
    PATCH_DATA_SITES,
    PATCH_LOGICAL_X_REPRESENTATIVES,
    PATCH_LOGICAL_Z_REPRESENTATIVES,
    PATCH_X_STABILIZERS,
    PATCH_Z_STABILIZERS,
    X_STABILIZERS,
    Z_STABILIZERS,
    InjectedStateName,
    append_logical_measurement,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
    append_stabilizer_measurement,
    append_patch_steane_state_injection_encoder,
    build_steane_state_injection_encoder,
)

NUM_DATA_QUBITS = 7
TOLERANCE = 1e-10


def _expected_logical_expectation_values(
    state: InjectedStateName,
) -> tuple[float, float]:
    """Return (⟨Z_L⟩, ⟨X_L⟩) for the given injected state."""
    inv_sqrt2 = 1.0 / math.sqrt(2)
    expectations: dict[str, tuple[float, float]] = {
        "0": (1.0, 0.0),
        "1": (-1.0, 0.0),
        "+": (0.0, 1.0),
        "-": (0.0, -1.0),
        "i": (0.0, 0.0),
        "-i": (0.0, 0.0),
        "Y+": (0.0, 0.0),
        "Y-": (0.0, 0.0),
        "S": (0.0, 0.0),
        "T": (0.0, inv_sqrt2),
        "Tdg": (0.0, inv_sqrt2),
        "H": (inv_sqrt2, inv_sqrt2),
    }
    return expectations[state]


ALL_STATES: list[InjectedStateName] = [
    "0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "T", "Tdg", "H",
]


# -----------------------------------------------
# Destructive logical expectation values
# -----------------------------------------------


def _logical_pauli_label(support: tuple[int, ...], pauli: str, n_qubits: int) -> str:
    """
    Build a Qiskit SparsePauliOp label for a multi-qubit Pauli.
    Qiskit labels are little-endian: label[0] acts on qubit 0 (rightmost in string).
    """
    chars = ["I"] * n_qubits
    for q in support:
        chars[q] = pauli
    # SparsePauliOp uses reversed ordering (qubit 0 is rightmost).
    return "".join(reversed(chars))


def _measure_logical_expectation_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
) -> float:
    """Encode |state_L⟩ and compute ⟨basis_L⟩ via statevector simulator."""
    encoder = build_steane_state_injection_encoder(injected_state=state, add_barriers=False)
    encoder.save_expectation_value(
        SparsePauliOp(_logical_pauli_label(support, basis, NUM_DATA_QUBITS)),
        list(range(NUM_DATA_QUBITS)),
        label="ev",
    )

    simulator = AerSimulator(method="statevector")
    result = simulator.run(encoder).result()
    return result.data()["ev"].real


@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_z_expectation(state: InjectedStateName) -> None:
    """Verify ⟨Z_L⟩ matches the expected value for all weight-3 representatives."""
    expected_z, _ = _expected_logical_expectation_values(state)
    for rep in LOGICAL_Z_REPRESENTATIVES:
        measured_z = _measure_logical_expectation_destructive(state, "Z", support=rep)
        assert measured_z == pytest.approx(expected_z, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_L⟩ expected {expected_z:.4f}, got {measured_z:.4f}"
        )


@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_x_expectation(state: InjectedStateName) -> None:
    """Verify ⟨X_L⟩ matches the expected value for all weight-3 representatives."""
    _, expected_x = _expected_logical_expectation_values(state)
    for rep in LOGICAL_X_REPRESENTATIVES:
        measured_x = _measure_logical_expectation_destructive(state, "X", support=rep)
        assert measured_x == pytest.approx(expected_x, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨X_L⟩ expected {expected_x:.4f}, got {measured_x:.4f}"
        )


# -----------------------------------------------
# Non-destructive logical measurement via append_logical_measurement
# -----------------------------------------------


def _measure_logical_expectation_non_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
) -> float:
    """
    Encode |state_L⟩, append non-destructive logical measurement via
    append_logical_measurement, and compute ⟨Z⟩ on the ancilla via
    save_expectation_value.
    """
    encoder = build_steane_state_injection_encoder(injected_state=state, add_barriers=False)

    # 7 data + 1 ancilla + 1 classical bit.
    circuit = QuantumCircuit(NUM_DATA_QUBITS + 1, 1)
    circuit.compose(encoder, qubits=list(range(NUM_DATA_QUBITS)), inplace=True)

    anc = NUM_DATA_QUBITS
    append_logical_measurement(
        circuit,
        basis,
        data_qubits=list(range(NUM_DATA_QUBITS)),
        support=support,
        ancilla_qubit=anc,
        classical_bits=[0],
        destructive=False,
    )

    # Remove measurements so we can use save_expectation_value.
    circuit.remove_final_measurements(inplace=True)

    circuit.save_expectation_value(SparsePauliOp("Z"), [anc], label="ev")

    simulator = AerSimulator(method="statevector")
    result = simulator.run(circuit).result()
    return result.data()["ev"].real


@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_z_via_measurement_circuit(state: InjectedStateName) -> None:
    """Verify ⟨Z_anc⟩ after non-destructive Z_L circuit equals ⟨Z_L⟩ for all representatives."""
    expected_z, _ = _expected_logical_expectation_values(state)
    for rep in LOGICAL_Z_REPRESENTATIVES:
        measured_z = _measure_logical_expectation_non_destructive(state, "Z", support=rep)
        assert measured_z == pytest.approx(expected_z, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_anc⟩ expected {expected_z:.4f}, got {measured_z:.4f}"
        )


@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_x_via_measurement_circuit(state: InjectedStateName) -> None:
    """Verify ⟨Z_anc⟩ after non-destructive X_L circuit equals ⟨X_L⟩ for all representatives."""
    _, expected_x = _expected_logical_expectation_values(state)
    for rep in LOGICAL_X_REPRESENTATIVES:
        measured_x = _measure_logical_expectation_non_destructive(state, "X", support=rep)
        assert measured_x == pytest.approx(expected_x, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_anc⟩ expected {expected_x:.4f}, got {measured_x:.4f}"
        )


# -----------------------------------------------
# Stabilizer syndrome via append_stabilizer_measurement
# -----------------------------------------------


@pytest.mark.parametrize("state", ALL_STATES)
def test_all_stabilizers_plus_one(state: InjectedStateName) -> None:
    """Verify all 6 syndrome ancillas give ⟨Z⟩ = +1 using append_stabilizer_measurement."""

    encoder = build_steane_state_injection_encoder(injected_state=state, add_barriers=False)

    n_ancilla = 6
    circuit = QuantumCircuit(NUM_DATA_QUBITS + n_ancilla, n_ancilla)
    circuit.compose(encoder, qubits=list(range(NUM_DATA_QUBITS)), inplace=True)

    append_stabilizer_measurement(
        circuit,
        data_qubits=list(range(NUM_DATA_QUBITS)),
        ancilla_qubits=list(range(NUM_DATA_QUBITS, NUM_DATA_QUBITS + n_ancilla)),
        classical_bits=list(range(n_ancilla)),
    )

    # Remove measurements so we can use save_expectation_value.
    circuit.remove_final_measurements(inplace=True)

    # Check ⟨Z⟩ = +1 on each ancilla individually.
    simulator = AerSimulator(method="statevector")
    for idx in range(n_ancilla):
        anc = NUM_DATA_QUBITS + idx
        circ_copy = circuit.copy()
        circ_copy.save_expectation_value(SparsePauliOp("Z"), [anc], label="ev")
        result = simulator.run(circ_copy).result()
        ev = result.data()["ev"].real

        stab_type = "X" if idx < 3 else "Z"
        stab_idx = idx if idx < 3 else idx - 3
        assert ev == pytest.approx(1.0, abs=TOLERANCE), (
            f"State {state!r}: {stab_type}-stabilizer {stab_idx} "
            f"⟨Z_anc⟩ expected 1.0, got {ev:.6f}"
        )


# -----------------------------------------------
# Logical representative consistency check
# -----------------------------------------------


def test_logical_representatives_are_x7_times_stabilizer() -> None:
    """Verify each X_L representative equals X^7 times a stabilizer group element."""
    from itertools import combinations

    stab_supports = [set(s) for s in X_STABILIZERS]

    # Generate all 8 elements of the stabilizer group (as sets of qubit indices).
    stab_group: list[set[int]] = [set()]
    for r in range(1, len(stab_supports) + 1):
        for combo in combinations(stab_supports, r):
            elem: set[int] = set()
            for s in combo:
                elem = elem.symmetric_difference(s)
            stab_group.append(elem)

    all_7 = set(range(7))

    for rep in LOGICAL_X_REPRESENTATIVES:
        rep_set = set(rep)
        found = any(all_7.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"X_L representative {rep} is not X^7 times any stabilizer group element"
        )


def test_logical_representatives_are_z7_times_stabilizer() -> None:
    """Verify each Z_L representative equals Z^7 times a stabilizer group element."""
    from itertools import combinations

    stab_supports = [set(s) for s in Z_STABILIZERS]

    # Generate all 8 elements of the stabilizer group (as sets of qubit indices).
    stab_group: list[set[int]] = [set()]
    for r in range(1, len(stab_supports) + 1):
        for combo in combinations(stab_supports, r):
            elem: set[int] = set()
            for s in combo:
                elem = elem.symmetric_difference(s)
            stab_group.append(elem)

    all_7 = set(range(7))

    for rep in LOGICAL_Z_REPRESENTATIVES:
        rep_set = set(rep)
        found = any(all_7.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"Z_L representative {rep} is not Z^7 times any stabilizer group element"
        )


def test_steane_code_is_css_self_dual() -> None:
    """Verify X and Z stabilizers have the same supports (CSS self-dual)."""
    assert X_STABILIZERS == Z_STABILIZERS
    assert LOGICAL_X_REPRESENTATIVES == LOGICAL_Z_REPRESENTATIVES


# =============================================================================
# 11-site patch versions of all tests above
# =============================================================================

N_PATCH_ANCILLA = 6


def _patch_measure_logical_expectation_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
) -> float:
    """Encode on 11-site patch and compute ⟨basis_L⟩ directly via statevector."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    append_patch_steane_state_injection_encoder(
        circuit, patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state, add_barriers=False,
    )

    circuit.save_expectation_value(
        SparsePauliOp(_logical_pauli_label(support, basis, NUM_PATCH_QUBITS)),
        list(range(NUM_PATCH_QUBITS)),
        label="ev",
    )

    simulator = AerSimulator(method="statevector")
    result = simulator.run(circuit).result()
    return result.data()["ev"].real


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_logical_z_expectation(state: InjectedStateName) -> None:
    """Verify ⟨Z_L⟩ on 11-site patch for all mapped representatives."""
    expected_z, _ = _expected_logical_expectation_values(state)
    for rep in PATCH_LOGICAL_Z_REPRESENTATIVES:
        print(rep)
        measured_z = _patch_measure_logical_expectation_destructive(state, "Z", support=rep)
        assert measured_z == pytest.approx(expected_z, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_L⟩ expected {expected_z:.4f}, got {measured_z:.4f}"
        )


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_logical_x_expectation(state: InjectedStateName) -> None:
    """Verify ⟨X_L⟩ on 11-site patch for all mapped representatives."""
    _, expected_x = _expected_logical_expectation_values(state)
    for rep in PATCH_LOGICAL_X_REPRESENTATIVES:
        measured_x = _patch_measure_logical_expectation_destructive(state, "X", support=rep)
        assert measured_x == pytest.approx(expected_x, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨X_L⟩ expected {expected_x:.4f}, got {measured_x:.4f}"
        )


# -----------------------------------------------
# Non-destructive logical measurement on 11-site patch
# -----------------------------------------------


def _patch_measure_logical_expectation_non_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
) -> float:
    """Encode on 11-site patch, append non-destructive logical measurement,
    and compute ⟨Z⟩ on the ancilla via Statevector."""
    anc_qubit = NUM_PATCH_QUBITS  # qubit index 11

    circuit = QuantumCircuit(NUM_PATCH_QUBITS + 1, 1)
    append_patch_steane_state_injection_encoder(
        circuit, patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state, add_barriers=False,
    )

    append_patch_logical_measurement(
        circuit,
        basis,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        support=support,
        ancilla_qubit=anc_qubit,
        classical_bits=[0],
        destructive=False,
    )

    # Remove measurements so Statevector can be computed.
    circuit.remove_final_measurements(inplace=True)

    sv = Statevector(circuit)
    return sv.expectation_value(SparsePauliOp("Z"), [anc_qubit]).real


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_logical_z_via_measurement_circuit(state: InjectedStateName) -> None:
    """Verify ⟨Z_anc⟩ after non-destructive Z_L on patch for all representatives."""
    expected_z, _ = _expected_logical_expectation_values(state)
    for rep in PATCH_LOGICAL_Z_REPRESENTATIVES:
        measured_z = _patch_measure_logical_expectation_non_destructive(state, "Z", support=rep)
        assert measured_z == pytest.approx(expected_z, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_anc⟩ expected {expected_z:.4f}, got {measured_z:.4f}"
        )


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_logical_x_via_measurement_circuit(state: InjectedStateName) -> None:
    """Verify ⟨Z_anc⟩ after non-destructive X_L on patch for all representatives."""
    _, expected_x = _expected_logical_expectation_values(state)
    for rep in PATCH_LOGICAL_X_REPRESENTATIVES:
        measured_x = _patch_measure_logical_expectation_non_destructive(state, "X", support=rep)
        assert measured_x == pytest.approx(expected_x, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_anc⟩ expected {expected_x:.4f}, got {measured_x:.4f}"
        )


# -----------------------------------------------
# Stabilizer syndrome on 11-site patch
# -----------------------------------------------


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_all_stabilizers_plus_one(state: InjectedStateName) -> None:
    """Verify all 6 syndrome ancillas give ⟨Z⟩ = +1 on the 11-site patch."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS + N_PATCH_ANCILLA, N_PATCH_ANCILLA)
    append_patch_steane_state_injection_encoder(
        circuit, patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state, add_barriers=False,
    )

    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        ancilla_qubits=list(range(NUM_PATCH_QUBITS, NUM_PATCH_QUBITS + N_PATCH_ANCILLA)),
        classical_bits=list(range(N_PATCH_ANCILLA)),
        add_barriers=False,
    )

    circuit.remove_final_measurements(inplace=True)

    sv = Statevector(circuit)
    for idx in range(N_PATCH_ANCILLA):
        anc = NUM_PATCH_QUBITS + idx
        ev = sv.expectation_value(SparsePauliOp("Z"), [anc]).real

        stab_type = "X" if idx < 3 else "Z"
        stab_idx = idx if idx < 3 else idx - 3
        assert ev == pytest.approx(1.0, abs=TOLERANCE), (
            f"State {state!r}: {stab_type}-stabilizer {stab_idx} "
            f"⟨Z_anc⟩ expected 1.0, got {ev:.6f}"
        )


# -----------------------------------------------
# Patch logical representative consistency check
# -----------------------------------------------


def test_patch_logical_representatives_are_x7_times_stabilizer() -> None:
    """Verify each patch X_L representative equals X^7 times a patch stabilizer group element."""
    from itertools import combinations

    stab_supports = [set(s) for s in PATCH_X_STABILIZERS]

    stab_group: list[set[int]] = [set()]
    for r in range(1, len(stab_supports) + 1):
        for combo in combinations(stab_supports, r):
            elem: set[int] = set()
            for s in combo:
                elem = elem.symmetric_difference(s)
            stab_group.append(elem)

    all_data = set(PATCH_DATA_SITES)

    for rep in PATCH_LOGICAL_X_REPRESENTATIVES:
        rep_set = set(rep)
        found = any(all_data.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"Patch X_L representative {rep} is not X^7 times any stabilizer group element"
        )


def test_patch_logical_representatives_are_z7_times_stabilizer() -> None:
    """Verify each patch Z_L representative equals Z^7 times a patch stabilizer group element."""
    from itertools import combinations

    stab_supports = [set(s) for s in PATCH_Z_STABILIZERS]

    stab_group: list[set[int]] = [set()]
    for r in range(1, len(stab_supports) + 1):
        for combo in combinations(stab_supports, r):
            elem: set[int] = set()
            for s in combo:
                elem = elem.symmetric_difference(s)
            stab_group.append(elem)

    all_data = set(PATCH_DATA_SITES)

    for rep in PATCH_LOGICAL_Z_REPRESENTATIVES:
        rep_set = set(rep)
        found = any(all_data.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"Patch Z_L representative {rep} is not Z^7 times any stabilizer group element"
        )


def test_patch_steane_code_is_css_self_dual() -> None:
    """Verify patch X and Z stabilizers have the same supports (CSS self-dual)."""
    assert PATCH_X_STABILIZERS == PATCH_Z_STABILIZERS
    assert PATCH_LOGICAL_X_REPRESENTATIVES == PATCH_LOGICAL_Z_REPRESENTATIVES




