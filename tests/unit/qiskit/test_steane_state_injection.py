"""Tests for the Steane-code state-injection encoder."""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from msc.noise import append_pauli_faults
from msc.steane_code.state_injection_qiskit import (
    LOGICAL_X_REPRESENTATIVES,
    LOGICAL_Z_REPRESENTATIVES,
    NUM_PATCH_QUBITS,
    PATCH_CRUMBLE_ENCODER_LAYERS,
    PATCH_DATA_SITES,
    PATCH_HELPER_SITES,
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
    append_patch_steane_state_injection_encoder,
    append_patch_steane_state_injection_encoder_crumble_sequences,
    append_stabilizer_measurement,
    append_steane_state_injection_encoder,
)
from msc.utils.circuit_qiskit import (
    ibm_backend,
    ibm_backend_available,
    two_qubit_depth,
)
from msc.utils.normalize import normalize_pauli_char, parse_qiskit_bitstring

NUM_DATA_QUBITS = 7
TOLERANCE = 1e-10


def _expected_logical_expectation_values(
    state: InjectedStateName,
) -> tuple[float, float, float]:
    """Return (⟨Z_L⟩, ⟨X_L⟩, ⟨Y_L⟩) for the given injected state."""
    inv_sqrt2 = 1.0 / math.sqrt(2)
    expectations: dict[str, tuple[float, float, float]] = {
        "0": (1.0, 0.0, 0.0),
        "1": (-1.0, 0.0, 0.0),
        "+": (0.0, 1.0, 0.0),
        "-": (0.0, -1.0, 0.0),
        "i": (0.0, 0.0, 1.0),
        "-i": (0.0, 0.0, -1.0),
        "Y+": (0.0, 0.0, 1.0),
        "Y-": (0.0, 0.0, -1.0),
        "S": (0.0, 0.0, 1.0),
        "T": (0.0, inv_sqrt2, inv_sqrt2),
        "Tdg": (0.0, inv_sqrt2, -inv_sqrt2),
        "Ty": (inv_sqrt2, inv_sqrt2, 0.0),
    }
    return expectations[state]


ALL_STATES: list[InjectedStateName] = [
    "0",
    "1",
    "+",
    "-",
    "i",
    "-i",
    "Y+",
    "Y-",
    "S",
    "T",
    "Tdg",
    "Ty",
]

LOGICAL_EXPECTATION_INDEX = {"Z": 0, "X": 1, "Y": 2}

LOGICAL_BASIS_CASES = [
    pytest.param("Z", LOGICAL_Z_REPRESENTATIVES, id="Z"),
    pytest.param("X", LOGICAL_X_REPRESENTATIVES, id="X"),
    # Steane self-duality gives Y_L the same common X_L/Z_L supports.
    pytest.param("Y", LOGICAL_X_REPRESENTATIVES, id="Y"),
]

PATCH_LOGICAL_BASIS_CASES = [
    pytest.param("Z", PATCH_LOGICAL_Z_REPRESENTATIVES, id="Z"),
    pytest.param("X", PATCH_LOGICAL_X_REPRESENTATIVES, id="X"),
    pytest.param("Y", PATCH_LOGICAL_X_REPRESENTATIVES, id="Y"),
]

# --------------------------------------
# Destructive logical expectation values
# --------------------------------------


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


def _logical_pauli_observable(
    support: tuple[int, ...],
    pauli: str,
    n_qubits: int,
) -> SparsePauliOp:
    """Build the signed logical Pauli observable on a Steane support."""
    observable = SparsePauliOp(_logical_pauli_label(support, pauli, n_qubits))
    if pauli == "Y" and len(support) % 4 == 3:
        # For Y_L = i X_L Z_L on a weight-3 common X/Z support,
        # Y_L = -prod(Y_j).
        observable = -observable
    return observable


def _measure_logical_expectation_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
) -> float:
    """Encode |state_L⟩ and compute ⟨basis_L⟩ via statevector simulator."""
    encoder = QuantumCircuit(NUM_DATA_QUBITS)
    append_steane_state_injection_encoder(
        encoder,
        data_qubits=list(range(NUM_DATA_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )
    encoder.save_expectation_value(
        _logical_pauli_observable(support, basis, NUM_DATA_QUBITS),
        list(range(NUM_DATA_QUBITS)),
        label="ev",
    )

    simulator = AerSimulator(method="statevector")
    result = simulator.run(encoder).result()
    return result.data()["ev"].real


@pytest.mark.parametrize(("basis", "representatives"), LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_expectation(
    state: InjectedStateName,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify ⟨P_L⟩ for every state, Pauli basis, and weight-3 representative."""
    expected = _expected_logical_expectation_values(state)[LOGICAL_EXPECTATION_INDEX[basis]]
    for rep in representatives:
        measured = _measure_logical_expectation_destructive(state, basis, support=rep)
        assert measured == pytest.approx(expected, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: "
            f"⟨{basis}_L⟩ expected {expected:.4f}, got {measured:.4f}"
        )


# ------------------------------------------------------------------
# Non-destructive logical measurement via append_logical_measurement
# ------------------------------------------------------------------


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
    # 7 data + 1 ancilla + 1 classical bit.
    circuit = QuantumCircuit(NUM_DATA_QUBITS + 1, 1)
    append_steane_state_injection_encoder(
        circuit,
        data_qubits=list(range(NUM_DATA_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )

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

    if basis == "Y":
        # Statevector cannot apply Store, and remove_final_measurements leaves
        # it behind. Remove it here and apply its sign below.
        assert circuit.data[-1].operation.name == "store"
        circuit.data.pop()

    # Remove measurements so we can use save_expectation_value.
    circuit.remove_final_measurements(inplace=True)

    circuit.save_expectation_value(SparsePauliOp("Z"), [anc], label="ev")

    simulator = AerSimulator(method="statevector")
    result = simulator.run(circuit).result()
    expectation = result.data()["ev"].real
    return -expectation if basis == "Y" else expectation


@pytest.mark.parametrize(("basis", "representatives"), LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_via_measurement_circuit(
    state: InjectedStateName,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify ancilla readout equals ⟨P_L⟩ for every logical Pauli representative."""
    expected = _expected_logical_expectation_values(state)[LOGICAL_EXPECTATION_INDEX[basis]]
    for rep in representatives:
        measured = _measure_logical_expectation_non_destructive(state, basis, support=rep)
        assert measured == pytest.approx(expected, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: "
            f"⟨Z_anc⟩ expected {expected:.4f}, got {measured:.4f}"
        )


# -----------------------------------------------------
# Stabilizer syndrome via append_stabilizer_measurement
# -----------------------------------------------------


@pytest.mark.parametrize("state", ALL_STATES)
def test_all_stabilizers_plus_one(state: InjectedStateName) -> None:
    """Verify all 6 syndrome ancillas give ⟨Z⟩ = +1 using append_stabilizer_measurement."""

    n_ancilla = 6
    circuit = QuantumCircuit(NUM_DATA_QUBITS + n_ancilla, n_ancilla)
    append_steane_state_injection_encoder(
        circuit,
        data_qubits=list(range(NUM_DATA_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )

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
        assert ev == pytest.approx(
            1.0, abs=TOLERANCE
        ), f"State {state!r}: {stab_type}-stabilizer {stab_idx} ⟨Z_anc⟩ expected 1.0, got {ev:.6f}"


# ----------------------------------------
# Logical representative consistency check
# ----------------------------------------


@pytest.mark.parametrize(
    ("basis", "stabilizers", "representatives"),
    [
        pytest.param("X", X_STABILIZERS, LOGICAL_X_REPRESENTATIVES, id="X"),
        pytest.param("Z", Z_STABILIZERS, LOGICAL_Z_REPRESENTATIVES, id="Z"),
    ],
)
def test_logical_representatives_are_weight_seven_times_stabilizer(
    basis: str,
    stabilizers: tuple[tuple[int, ...], ...],
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify each logical representative equals P^7 times a stabilizer."""
    from itertools import combinations

    stab_supports = [set(s) for s in stabilizers]

    # Generate all 8 elements of the stabilizer group (as sets of qubit indices).
    stab_group: list[set[int]] = [set()]
    for r in range(1, len(stab_supports) + 1):
        for combo in combinations(stab_supports, r):
            elem: set[int] = set()
            for s in combo:
                elem = elem.symmetric_difference(s)
            stab_group.append(elem)

    all_7 = set(range(7))

    for rep in representatives:
        rep_set = set(rep)
        found = any(all_7.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"{basis}_L representative {rep} is not "
            f"{basis}^7 times any stabilizer group element"
        )


def test_steane_code_is_css_self_dual() -> None:
    """Verify X and Z stabilizers have the same supports (CSS self-dual)."""
    assert X_STABILIZERS == Z_STABILIZERS
    assert LOGICAL_X_REPRESENTATIVES == LOGICAL_Z_REPRESENTATIVES


# =============================================================================
# 11-site patch versions of all tests above
# =============================================================================

N_PATCH_ANCILLA = 6

PATCH_ENCODERS = ["abstract", "crumble"]


def _apply_patch_encoder(
    circuit: QuantumCircuit,
    encoder: str,
    *,
    patch_qubits: list[int],
    injected_state: InjectedStateName,
    add_barriers: bool = False,
) -> None:
    """Dispatch to the abstract or Crumble encoder on the 11-site patch."""
    if encoder == "abstract":
        append_patch_steane_state_injection_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=add_barriers,
        )
    elif encoder == "crumble":
        append_patch_steane_state_injection_encoder_crumble_sequences(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=add_barriers,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder!r}")


def _patch_measure_logical_expectation_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
    encoder: str = "abstract",
) -> float:
    """Encode on 11-site patch and compute ⟨basis_L⟩ directly via statevector."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    _apply_patch_encoder(
        circuit,
        encoder,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )

    circuit.save_expectation_value(
        _logical_pauli_observable(support, basis, NUM_PATCH_QUBITS),
        list(range(NUM_PATCH_QUBITS)),
        label="ev",
    )

    simulator = AerSimulator(method="statevector")
    result = simulator.run(circuit).result()
    return result.data()["ev"].real


@pytest.mark.parametrize(("basis", "representatives"), PATCH_LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("encoder", PATCH_ENCODERS)
@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_logical_expectation(
    state: InjectedStateName,
    encoder: str,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify ⟨P_L⟩ on both 11-site patch encoders for every representative."""
    expected = _expected_logical_expectation_values(state)[LOGICAL_EXPECTATION_INDEX[basis]]
    for rep in representatives:
        measured = _patch_measure_logical_expectation_destructive(
            state,
            basis,
            support=rep,
            encoder=encoder,
        )
        assert measured == pytest.approx(expected, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: "
            f"⟨{basis}_L⟩ expected {expected:.4f}, got {measured:.4f}"
        )


# ----------------------------------------------------
# Non-destructive logical measurement on 11-site patch
# ----------------------------------------------------


def _patch_measure_logical_expectation_non_destructive(
    state: InjectedStateName,
    basis: str,
    support: tuple[int, ...],
    encoder: str = "abstract",
) -> float:
    """Encode on 11-site patch, append non-destructive logical measurement,
    and compute ⟨Z⟩ on the ancilla via Statevector."""
    anc_qubit = NUM_PATCH_QUBITS  # qubit index 11

    circuit = QuantumCircuit(NUM_PATCH_QUBITS + 1, 1)
    _apply_patch_encoder(
        circuit,
        encoder,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state,
        add_barriers=False,
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

    if basis == "Y":
        # Statevector cannot apply Store, and remove_final_measurements leaves
        # it behind. Remove it here and apply its sign below.
        assert circuit.data[-1].operation.name == "store"
        circuit.data.pop()

    # Remove measurements so Statevector can be computed.
    circuit.remove_final_measurements(inplace=True)

    sv = Statevector(circuit)
    expectation = sv.expectation_value(SparsePauliOp("Z"), [anc_qubit]).real
    return -expectation if basis == "Y" else expectation


@pytest.mark.parametrize(("basis", "representatives"), PATCH_LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("encoder", PATCH_ENCODERS)
@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_logical_via_measurement_circuit(
    state: InjectedStateName,
    encoder: str,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify patch ancilla readout equals ⟨P_L⟩ for every representative."""
    expected = _expected_logical_expectation_values(state)[LOGICAL_EXPECTATION_INDEX[basis]]
    for rep in representatives:
        measured = _patch_measure_logical_expectation_non_destructive(
            state,
            basis,
            support=rep,
            encoder=encoder,
        )
        assert measured == pytest.approx(expected, abs=TOLERANCE), (
            f"State {state!r}, support {rep}: "
            f"⟨Z_anc⟩ expected {expected:.4f}, got {measured:.4f}"
        )


# ----------------------------------------------
# Patch logical representative consistency check
# ----------------------------------------------


@pytest.mark.parametrize(
    ("basis", "stabilizers", "representatives"),
    [
        pytest.param("X", PATCH_X_STABILIZERS, PATCH_LOGICAL_X_REPRESENTATIVES, id="X"),
        pytest.param("Z", PATCH_Z_STABILIZERS, PATCH_LOGICAL_Z_REPRESENTATIVES, id="Z"),
    ],
)
def test_patch_logical_representatives_are_weight_seven_times_stabilizer(
    basis: str,
    stabilizers: tuple[tuple[int, ...], ...],
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify each patch logical representative equals P^7 times a stabilizer."""
    from itertools import combinations

    stab_supports = [set(s) for s in stabilizers]

    stab_group: list[set[int]] = [set()]
    for r in range(1, len(stab_supports) + 1):
        for combo in combinations(stab_supports, r):
            elem: set[int] = set()
            for s in combo:
                elem = elem.symmetric_difference(s)
            stab_group.append(elem)

    all_data = set(PATCH_DATA_SITES)

    for rep in representatives:
        rep_set = set(rep)
        found = any(all_data.symmetric_difference(g) == rep_set for g in stab_group)
        assert found, (
            f"Patch {basis}_L representative {rep} is not "
            f"{basis}^7 times any stabilizer group element"
        )


def test_patch_steane_code_is_css_self_dual() -> None:
    """Verify patch X and Z stabilizers have the same supports (CSS self-dual)."""
    assert PATCH_X_STABILIZERS == PATCH_Z_STABILIZERS
    assert PATCH_LOGICAL_X_REPRESENTATIVES == PATCH_LOGICAL_Z_REPRESENTATIVES


# ------------------------------------
# Stabilizer syndrome on 11-site patch
# ------------------------------------


@pytest.mark.parametrize("encoder", PATCH_ENCODERS)
@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_all_stabilizers_via_ancilla_measurement(
    state: InjectedStateName,
    encoder: str,
) -> None:
    """Verify all 6 syndrome bits are 0 (eigenvalue +1) via measurement circuit."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS, N_PATCH_ANCILLA)
    _apply_patch_encoder(
        circuit,
        encoder,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )

    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        classical_bits=list(range(N_PATCH_ANCILLA)),
        add_barriers=False,
    )

    sim = AerSimulator()
    counts = sim.run(circuit, shots=1).result().get_counts()
    bitstring = next(iter(counts))
    for idx in range(N_PATCH_ANCILLA):
        bit_val = int(bitstring[-(idx + 1)])
        stab_type = "X" if idx < 3 else "Z"
        stab_idx = idx if idx < 3 else idx - 3
        assert (
            bit_val == 0
        ), f"State {state!r}: {stab_type}-stabilizer {stab_idx} expected 0, got {bit_val}"


@pytest.mark.parametrize("basis", ["x", "z"])
def test_patch_single_basis_stabilizer_measurement(basis: str) -> None:
    """The shared patch helper supports the final one-basis KT check."""

    circuit = QuantumCircuit(NUM_PATCH_QUBITS, 3)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state="Ty",
        add_barriers=False,
    )
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        classical_bits=list(range(3)),
        basis=basis,
        add_barriers=False,
    )

    counts = AerSimulator().run(circuit, shots=1).result().get_counts()

    assert counts == {"000": 1}


@pytest.mark.parametrize("basis", ["x", "z"])
def test_patch_selected_stabilizer_measurement(basis: str) -> None:
    """The patch helper measures only the requested ordered stabilizer subset."""

    circuit = QuantumCircuit(NUM_PATCH_QUBITS, 2)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state="Ty",
        add_barriers=False,
    )
    measurement_start = len(circuit.data)
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        classical_bits=list(range(2)),
        basis=basis,
        stabilizer_indices=(0, 2),
        add_barriers=False,
    )

    appended_instructions = circuit.data[measurement_start:]
    assert all(circuit.qubits[6] not in instruction.qubits for instruction in appended_instructions)
    assert (
        sum(instruction.operation.name == "measure" for instruction in appended_instructions) == 2
    )
    assert AerSimulator().run(circuit, shots=1).result().get_counts() == {"00": 1}


@pytest.mark.parametrize("encoder", PATCH_ENCODERS)
@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_stabilizer_expectation_values(state: InjectedStateName, encoder: str) -> None:
    """Verify ⟨S⟩ = +1 for all 6 stabilizer operators directly via Statevector."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    _apply_patch_encoder(
        circuit,
        encoder,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )

    sv = Statevector(circuit)

    for i, support in enumerate(PATCH_X_STABILIZERS):
        label = _logical_pauli_label(support, "X", NUM_PATCH_QUBITS)
        ev = sv.expectation_value(SparsePauliOp(label)).real
        assert ev == pytest.approx(1.0, abs=TOLERANCE), (
            f"State {state!r}: X-stabilizer {i} (support {support}) "
            f"⟨X_S⟩ expected 1.0, got {ev:.6f}"
        )

    for i, support in enumerate(PATCH_Z_STABILIZERS):
        label = _logical_pauli_label(support, "Z", NUM_PATCH_QUBITS)
        ev = sv.expectation_value(SparsePauliOp(label)).real
        assert ev == pytest.approx(1.0, abs=TOLERANCE), (
            f"State {state!r}: Z-stabilizer {i} (support {support}) "
            f"⟨Z_S⟩ expected 1.0, got {ev:.6f}"
        )


# ---------------------------------------------------------
# Stabilizer syndrome on 11-site patch with injected faults
# ---------------------------------------------------------


def _expected_patch_syndrome(
    faults: list[tuple[str, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute expected (x_syndrome, z_syndrome) for a list of (pauli, site) faults.

    x_syndrome: 3 bits from X-stabilizer measurements (flag Z-type errors).
    z_syndrome: 3 bits from Z-stabilizer measurements (flag X-type errors).
    """
    x_syn = [0, 0, 0]
    z_syn = [0, 0, 0]
    for pauli, site in faults:
        pauli = normalize_pauli_char(pauli)
        for i, support in enumerate(PATCH_X_STABILIZERS):
            if site in support and pauli in ("Z", "Y"):
                x_syn[i] ^= 1
        for i, support in enumerate(PATCH_Z_STABILIZERS):
            if site in support and pauli in ("X", "Y"):
                z_syn[i] ^= 1
    return tuple(x_syn), tuple(z_syn)


def _build_and_run_patch_syndrome(
    faults: list[tuple[str, int]],
    injected_state: InjectedStateName = "0",
    encoder: str = "abstract",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Inject faults after state injection, measure stabilizers, return (x_syn, z_syn)."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS, N_PATCH_ANCILLA)

    _apply_patch_encoder(
        circuit,
        encoder,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=injected_state,
        add_barriers=False,
    )

    # Inject faults.
    append_pauli_faults(circuit, faults)

    # Measure all 6 stabilizers.
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        classical_bits=list(range(N_PATCH_ANCILLA)),
        add_barriers=False,
    )

    # Run 1 shot (noiseless, deterministic).
    sim = AerSimulator()
    bitstring = next(iter(sim.run(circuit, shots=1).result().get_counts()))
    bit_labels = tuple(f"b{i}" for i in range(N_PATCH_ANCILLA))
    bits = parse_qiskit_bitstring(bitstring, bit_labels)
    # bits[0..2] = X-stabilizer outcomes, bits[3..5] = Z-stabilizer outcomes.
    x_syn = tuple(bits[f"b{i}"] for i in range(3))
    z_syn = tuple(bits[f"b{i}"] for i in range(3, 6))
    return x_syn, z_syn


# -- Fault parametrizations --

_SINGLE_FAULTS = [[(p, s)] for p in ("X", "Y", "Z") for s in PATCH_DATA_SITES]
_SINGLE_IDS = [f"{p}_{s}" for p in ("X", "Y", "Z") for s in PATCH_DATA_SITES]

_TWO_SAME_FAULTS: list[list[tuple[str, int]]] = []
_TWO_SAME_IDS: list[str] = []
for _p in ("X", "Y", "Z"):
    for _i, _s1 in enumerate(PATCH_DATA_SITES):
        for _s2 in PATCH_DATA_SITES[_i + 1 :]:
            _TWO_SAME_FAULTS.append([(_p, _s1), (_p, _s2)])
            _TWO_SAME_IDS.append(f"{_p}{_p}_{_s1}_{_s2}")

_TWO_MIXED_FAULTS: list[list[tuple[str, int]]] = []
_TWO_MIXED_IDS: list[str] = []
for _p1, _p2 in [("X", "Z"), ("X", "Y"), ("Z", "Y")]:
    for _i, _s1 in enumerate(PATCH_DATA_SITES):
        for _s2 in PATCH_DATA_SITES[_i + 1 :]:
            _TWO_MIXED_FAULTS.append([(_p1, _s1), (_p2, _s2)])
            _TWO_MIXED_IDS.append(f"{_p1}{_p2}_{_s1}_{_s2}")


class TestPatchSyndromeExtraction:
    """Verify stabilizer measurement outputs match expected syndromes for injected faults."""

    @pytest.mark.parametrize("encoder", PATCH_ENCODERS)
    @pytest.mark.parametrize("state", ["0", "T", "Ty"])
    def test_no_fault_trivial(self, state: InjectedStateName, encoder: str) -> None:
        """No faults → all syndrome bits zero for any encoded state."""
        x_syn, z_syn = _build_and_run_patch_syndrome([], injected_state=state, encoder=encoder)
        assert x_syn == (0, 0, 0) and z_syn == (
            0,
            0,
            0,
        ), f"State {state!r}: expected trivial syndrome, got X={x_syn}, Z={z_syn}"

    @pytest.mark.parametrize("encoder", PATCH_ENCODERS)
    @pytest.mark.parametrize("state", ["0", "T", "Ty"])
    @pytest.mark.parametrize("faults", _SINGLE_FAULTS, ids=_SINGLE_IDS)
    def test_single_fault(
        self,
        state: InjectedStateName,
        faults: list[tuple[str, int]],
        encoder: str,
    ) -> None:
        """Single-qubit fault produces the correct unique syndrome."""
        expected_x, expected_z = _expected_patch_syndrome(faults)
        measured_x, measured_z = _build_and_run_patch_syndrome(
            faults,
            injected_state=state,
            encoder=encoder,
        )
        pauli, site = faults[0]
        assert (measured_x, measured_z) == (expected_x, expected_z), (
            f"State {state!r}, {pauli}@{site}: "
            f"got X={measured_x} Z={measured_z}, expected X={expected_x} Z={expected_z}"
        )

    @pytest.mark.parametrize("encoder", PATCH_ENCODERS)
    @pytest.mark.parametrize("state", ["0", "T", "Ty"])
    @pytest.mark.parametrize("faults", _TWO_SAME_FAULTS, ids=_TWO_SAME_IDS)
    def test_two_qubit_same_pauli(
        self,
        state: InjectedStateName,
        faults: list[tuple[str, int]],
        encoder: str,
    ) -> None:
        """Two same-Pauli faults: syndrome is XOR of individual syndromes."""
        expected_x, expected_z = _expected_patch_syndrome(faults)
        measured_x, measured_z = _build_and_run_patch_syndrome(
            faults,
            injected_state=state,
            encoder=encoder,
        )
        p, s1, s2 = faults[0][0], faults[0][1], faults[1][1]
        assert (measured_x, measured_z) == (expected_x, expected_z), (
            f"State {state!r}, {p}{p}@{s1},{s2}: "
            f"got X={measured_x} Z={measured_z}, expected X={expected_x} Z={expected_z}"
        )

    @pytest.mark.parametrize("encoder", PATCH_ENCODERS)
    @pytest.mark.parametrize("state", ["0", "T", "Ty"])
    @pytest.mark.parametrize("faults", _TWO_MIXED_FAULTS, ids=_TWO_MIXED_IDS)
    def test_two_qubit_mixed_pauli(
        self,
        state: InjectedStateName,
        faults: list[tuple[str, int]],
        encoder: str,
    ) -> None:
        """Two different-Pauli faults: X and Z sectors detect independently."""
        expected_x, expected_z = _expected_patch_syndrome(faults)
        measured_x, measured_z = _build_and_run_patch_syndrome(
            faults,
            injected_state=state,
            encoder=encoder,
        )
        p1, s1 = faults[0]
        p2, s2 = faults[1]
        assert (measured_x, measured_z) == (expected_x, expected_z), (
            f"State {state!r}, {p1}@{s1}+{p2}@{s2}: "
            f"got X={measured_x} Z={measured_z}, expected X={expected_x} Z={expected_z}"
        )


# ============================================================================
# Specific to Crumble state-injection schedule on the 11-site patch
# ============================================================================

# Site layout (x right, y down):
#
#            0
#          /   \
#        1       2
#      /   \   /   \
#    3       4       5
#      \   /   \   /
#        6       7
#      /   \   /   \
#    8       9      10
#
_PATCH_CRUMBLE_SITE_COORDINATES: dict[int, tuple[float, float]] = {
    0: (1.0, 0.0),
    1: (0.5, 0.5),
    2: (1.5, 0.5),
    3: (0.0, 1.0),
    4: (1.0, 1.0),
    5: (2.0, 1.0),
    6: (0.5, 1.5),
    7: (1.5, 1.5),
    8: (0.0, 2.0),
    9: (1.0, 2.0),
    10: (2.0, 2.0),
}


@pytest.mark.parametrize("layer", PATCH_CRUMBLE_ENCODER_LAYERS)
def test_patch_crumble_cz_layers_are_nearest_neighbor_data_helper_edges(layer) -> None:
    """Every Crumble CZ layer is a nearest-neighbor data-helper interaction."""
    gate_name, operands = layer
    if gate_name != "cz":
        return

    data_sites = set(PATCH_DATA_SITES)
    helper_sites = set(PATCH_HELPER_SITES)
    for site_a, site_b in operands:
        ax, ay = _PATCH_CRUMBLE_SITE_COORDINATES[site_a]
        bx, by = _PATCH_CRUMBLE_SITE_COORDINATES[site_b]
        assert abs(ax - bx) + abs(ay - by) == 1
        assert {site_a in data_sites, site_b in data_sites} == {True, False}
        assert {site_a in helper_sites, site_b in helper_sites} == {True, False}


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_crumble_helper_sites_expectation_value(state: InjectedStateName) -> None:
    """After Crumble encoding, all helper sites are in |0⟩ (⟨Z⟩ = +1)."""
    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )
    sv = Statevector(circuit)
    for site in PATCH_HELPER_SITES:
        ev = sv.expectation_value(SparsePauliOp("Z"), [site]).real
        assert ev == pytest.approx(
            1.0, abs=TOLERANCE
        ), f"State {state!r}: helper site {site} ⟨Z⟩ expected 1.0, got {ev:.6f}"


@pytest.mark.parametrize("state", ALL_STATES)
def test_patch_crumble_helper_sites_bit_value(state: InjectedStateName) -> None:
    """Measured Crumble helper sites always return the all-zero outcome."""
    n_helpers = len(PATCH_HELPER_SITES)
    circuit = QuantumCircuit(NUM_PATCH_QUBITS, n_helpers)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state=state,
        flag_classical_bits=list(range(n_helpers)),
        add_barriers=False,
    )
    shots = 1024
    counts = AerSimulator().run(circuit, shots=shots).result().get_counts()
    assert counts == {"0000": shots}


@pytest.mark.skipif(
    not ibm_backend_available(),
    reason="QiskitRuntimeService account 'QR' or backend 'ibm_miami' unavailable",
)
def test_patch_crumble_encoder_two_qubit_depth_preserved_on_ibm_miami() -> None:
    """Transpiling the Crumble encoder to ibm_miami must not increase two-qubit depth.

    The Crumble schedule is already nearest-neighbor on a square lattice, so a
    square-lattice device like ibm_miami should be able to map it without
    inserting any additional SWAP gates.
    """
    from qiskit import transpile
    from qiskit.transpiler.passes import RemoveBarriers

    circuit = QuantumCircuit(NUM_PATCH_QUBITS)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(range(NUM_PATCH_QUBITS)),
        injected_state="0",
        add_barriers=True,
    )

    original_depth = two_qubit_depth(RemoveBarriers()(circuit))

    transpiled = transpile(
        RemoveBarriers()(circuit),
        backend=ibm_backend(),
        optimization_level=3,
    )

    transpiled_depth = two_qubit_depth(transpiled)
    assert transpiled_depth <= original_depth, (
        f"Two-qubit depth increased from {original_depth} to {transpiled_depth} "
        f"after transpilation to ibm_miami — expected no increase for a "
        f"nearest-neighbor schedule on a square-lattice device"
    )


# ============================================================================
# Other tests
# ============================================================================


@pytest.mark.parametrize(
    ("angle", "equivalent_state"),
    [
        (0.0, "+"),
        (math.pi / 4, "T"),
        (math.pi / 2, "Y+"),
        (-math.pi / 4, "Tdg"),
        (-math.pi / 2, "Y-"),
        (math.pi, "-"),
    ],
)
def test_patch_injection_azimuth_angle_matches_named_state(
    angle: float, equivalent_state: InjectedStateName
) -> None:
    """injection_azimuth_angle=θ injects Rz(θ)|+⟩, matching the equivalent named state.

    Rz(θ)|+⟩ = (|0⟩ + e^{iθ}|1⟩)/√2 sweeps the equator (XY plane): Rz(0)|+⟩=|+⟩,
    Rz(π/4)|+⟩=|T⟩, Rz(π/2)|+⟩=|Y+⟩, Rz(-π/4)|+⟩=|Tdg⟩, Rz(-π/2)|+⟩=|Y-⟩, and
    Rz(π)|+⟩=|-⟩, so the encoded 11-site state must be identical (up to global
    phase) to the corresponding named injection.

    The Crumble and abstract encoders both prepare the unique Steane codeword for
    the injected state, so all four encoded statevectors (angle/named × patch_crumble/
    patch_abstract) must agree up to global phase.
    """

    def _encode(encoder, *, injected_state=None, injection_azimuth_angle=None) -> Statevector:
        circuit = QuantumCircuit(NUM_PATCH_QUBITS)
        encoder(
            circuit,
            patch_qubits=list(range(NUM_PATCH_QUBITS)),
            injected_state=injected_state,
            injection_azimuth_angle=injection_azimuth_angle,
            add_barriers=False,
        )
        return Statevector(circuit)

    statevectors = {
        "crumble_angle": _encode(
            append_patch_steane_state_injection_encoder_crumble_sequences,
            injection_azimuth_angle=angle,
        ),
        "crumble_named": _encode(
            append_patch_steane_state_injection_encoder_crumble_sequences,
            injected_state=equivalent_state,
        ),
        "abstract_angle": _encode(
            append_patch_steane_state_injection_encoder,
            injection_azimuth_angle=angle,
        ),
        "abstract_named": _encode(
            append_patch_steane_state_injection_encoder,
            injected_state=equivalent_state,
        ),
    }

    reference_name, reference_sv = next(iter(statevectors.items()))
    for name, sv in statevectors.items():
        assert reference_sv.equiv(sv), (
            f"{name} statevector disagrees with {reference_name}: "
            f"injection_azimuth_angle={angle} should encode the same state as "
            f"injected_state={equivalent_state!r} for both encoders"
        )
