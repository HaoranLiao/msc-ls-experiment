"""Tests for the Steane-code state-injection encoder."""

from __future__ import annotations

import math

import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from research_qec_haoran.steane_code.state_injection_qiskit import (
    LOGICAL_X_REPRESENTATIVES,
    LOGICAL_Z_REPRESENTATIVES,
    X_STABILIZERS,
    Z_STABILIZERS,
    InjectedStateName,
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


def _logical_pauli_label(support: tuple[int, ...], pauli: str) -> str:
    """Build a Qiskit SparsePauliOp label for a multi-qubit Pauli on 7 qubits.

    Qiskit labels are little-endian: label[0] acts on qubit 0 (rightmost in string).
    """
    chars = ["I"] * NUM_DATA_QUBITS
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
        SparsePauliOp(_logical_pauli_label(support, basis)),
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
# Non-destructive logical measurement via ancilla
# -----------------------------------------------


def _measure_logical_expectation_non_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
) -> float:
    """Encode |state_L⟩, apply non-destructive logical measurement gates, and
    compute ⟨Z⟩ on the ancilla (which equals ⟨basis_L⟩ on the data)."""
    from qiskit import QuantumCircuit


    encoder = build_steane_state_injection_encoder(injected_state=state, add_barriers=False)

    # Build 8-qubit circuit: 7 data + 1 ancilla (qubit 7).
    circuit = QuantumCircuit(NUM_DATA_QUBITS + 1)
    circuit.compose(encoder, qubits=list(range(NUM_DATA_QUBITS)), inplace=True)

    anc = NUM_DATA_QUBITS

    if basis == "Z":
        # Ancilla in |0⟩, CX from each support qubit to ancilla.
        for q in support:
            circuit.cx(q, anc)
    else:
        # Ancilla in |+⟩, CX from ancilla to each support qubit, then H.
        circuit.h(anc)
        for q in support:
            circuit.cx(anc, q)
        circuit.h(anc)

    # ⟨Z_ancilla⟩ should equal ⟨basis_L⟩ on the encoded state.
    circuit.save_expectation_value(
        SparsePauliOp("Z"),
        [anc],
        label="ev",
    )

    simulator = AerSimulator(method="statevector")
    result = simulator.run(circuit).result()
    return result.data()["ev"].real


@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_z_via_ancilla(state: InjectedStateName) -> None:
    """Verify ⟨Z_ancilla⟩ after non-destructive Z_L circuit equals ⟨Z_L⟩ for all representatives."""
    expected_z, _ = _expected_logical_expectation_values(state)
    for rep in LOGICAL_Z_REPRESENTATIVES:
        measured_z = _measure_logical_expectation_non_destructive(state, "Z", support=rep)
        assert measured_z == pytest.approx(expected_z, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_anc⟩ expected {expected_z:.4f}, got {measured_z:.4f}"
        )


@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_x_via_ancilla(state: InjectedStateName) -> None:
    """Verify ⟨Z_ancilla⟩ after non-destructive X_L circuit equals ⟨X_L⟩ for all representatives."""
    _, expected_x = _expected_logical_expectation_values(state)
    for rep in LOGICAL_X_REPRESENTATIVES:
        measured_x = _measure_logical_expectation_non_destructive(state, "X", support=rep)
        assert measured_x == pytest.approx(expected_x, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: ⟨Z_anc⟩ expected {expected_x:.4f}, got {measured_x:.4f}"
        )

# -----------------------------------------------
# Stabilizer syndrome expectation values
# -----------------------------------------------


def _measure_stabilizer_ancilla_expectations(
    state: InjectedStateName,
) -> list[float]:
    """Encode |state_L⟩, apply stabilizer measurement gates, and return
    ⟨Z⟩ for each of the 6 syndrome ancillas.

    Returns a list of 6 floats: [X_stab_0, X_stab_1, X_stab_2,
    Z_stab_0, Z_stab_1, Z_stab_2].
    """
    from qiskit import QuantumCircuit

    n_ancilla = 6
    encoder = build_steane_state_injection_encoder(injected_state=state, add_barriers=False)

    # 7 data + 6 ancillas (3 for X stabilizers, 3 for Z stabilizers).
    circuit = QuantumCircuit(NUM_DATA_QUBITS + n_ancilla)
    circuit.compose(encoder, qubits=list(range(NUM_DATA_QUBITS)), inplace=True)

    # X-stabilizer measurement: ancilla in |+⟩, CX(anc→data), H on ancilla.
    for i, support in enumerate(X_STABILIZERS):
        anc = NUM_DATA_QUBITS + i
        circuit.h(anc)
        for data_q in support:
            circuit.cx(anc, data_q)
        circuit.h(anc)

    # Z-stabilizer measurement: ancilla in |0⟩, CX(data→anc).
    for i, support in enumerate(Z_STABILIZERS):
        anc = NUM_DATA_QUBITS + 3 + i
        for data_q in support:
            circuit.cx(data_q, anc)

    # Save ⟨Z⟩ on each ancilla.
    ancilla_qubits = list(range(NUM_DATA_QUBITS, NUM_DATA_QUBITS + n_ancilla))
    label = "I" * (n_ancilla - 1)
    for idx, anc in enumerate(ancilla_qubits):
        circuit.save_expectation_value(
            SparsePauliOp("Z"),
            [anc],
            label=f"stab_{idx}",
        )

    simulator = AerSimulator(method="statevector")
    result = simulator.run(circuit).result()
    return [result.data()[f"stab_{idx}"].real for idx in range(n_ancilla)]


@pytest.mark.parametrize("state", ALL_STATES)
def test_all_stabilizers_plus_one(state: InjectedStateName) -> None:
    """Verify all 6 syndrome ancillas give ⟨Z⟩ = +1 for a valid codeword."""
    expectations = _measure_stabilizer_ancilla_expectations(state)
    for idx, ev in enumerate(expectations):
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
    """Verify each logical representative equals X^7 (or Z^7) times a stabilizer group element."""
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

    # X_L representatives should be X^7 times a stabilizer element.
    for rep in LOGICAL_X_REPRESENTATIVES:
        rep_set = set(rep)
        # X^7 * stabilizer_element = symmetric_difference(all_7, stab_elem)
        found = any(all_7.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"X_L representative {rep} is not X^7 times any stabilizer group element"
        )

    # Z_L representatives should be Z^7 times a stabilizer element.
    for rep in LOGICAL_Z_REPRESENTATIVES:
        rep_set = set(rep)
        found = any(all_7.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"Z_L representative {rep} is not Z^7 times any stabilizer group element"
        )

