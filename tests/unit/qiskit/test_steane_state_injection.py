"""Tests for the Steane-code state-injection encoder."""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import NamedTuple

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator

from msc.noise.fault_injection_qiskit import append_pauli_faults
from msc.steane_code.state_injection_common import (
    LOGICAL_X_REPRESENTATIVES,
    LOGICAL_Z_REPRESENTATIVES,
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_11_SITE_COORDS,
    PATCH_11_SITE_DATA_SITES,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES,
    PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES,
    PATCH_11_SITE_X_STABILIZERS,
    PATCH_11_SITE_Z_STABILIZERS,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_COORDS,
    PATCH_14_SITE_DATA_SITES,
    PATCH_14_SITE_HELPER_SITES,
    PATCH_14_SITE_HIRANO_ENCODER_COORDS,
    PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES,
    PATCH_14_SITE_LOGICAL_Z_REPRESENTATIVES,
    PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS,
    PATCH_14_SITE_X_STABILIZERS,
    PATCH_14_SITE_Z_STABILIZERS,
    X_STABILIZERS,
    Z_STABILIZERS,
)
from msc.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_11_site_patch_abstract_encoder,
    append_11_site_patch_logical_measurement,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_abstract_encoder,
    append_14_site_patch_abstract_routed_encoder,
    append_14_site_patch_hirano_encoder,
    append_14_site_patch_logical_measurement,
    append_14_site_patch_superdense_stabilizer_measurement,
    append_abstract_encoder,
)
from msc.utils.circuit_qiskit import two_qubit_depth
from msc.utils.normalize import normalize_pauli_char, parse_qiskit_bitstring

NUM_DATA_QUBITS = 7
TOLERANCE = 1e-10


ALL_STATES: tuple[InjectedStateName, ...] = (
    "0",
    "1",
    "+",
    "-",
    "i",
    "-i",
    "Y+",
    "Y-",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "Ty",
)
HIRANO_STATES: tuple[InjectedStateName, ...] = (
    "S",
    "i",
    "Y+",
    "Sdg",
    "-i",
    "Y-",
)

_INV_SQRT_2 = 1.0 / math.sqrt(2)
LOGICAL_EXPECTATION_VALUES: dict[InjectedStateName, tuple[float, float, float]] = {
    "0": (1.0, 0.0, 0.0),
    "1": (-1.0, 0.0, 0.0),
    "+": (0.0, 1.0, 0.0),
    "-": (0.0, -1.0, 0.0),
    "i": (0.0, 0.0, 1.0),
    "-i": (0.0, 0.0, -1.0),
    "Y+": (0.0, 0.0, 1.0),
    "Y-": (0.0, 0.0, -1.0),
    "S": (0.0, 0.0, 1.0),
    "Sdg": (0.0, 0.0, -1.0),
    "T": (0.0, _INV_SQRT_2, _INV_SQRT_2),
    "Tdg": (0.0, _INV_SQRT_2, -_INV_SQRT_2),
    "Ty": (_INV_SQRT_2, _INV_SQRT_2, 0.0),
}
LOGICAL_EXPECTATION_INDEX = {"Z": 0, "X": 1, "Y": 2}


# =============================================================================
# Abstract
# =============================================================================


# --------------------------
# Logical expectation values
# --------------------------


def _logical_pauli_label(support: tuple[int, ...], pauli: str, n_qubits: int) -> str:
    """
    Build a Qiskit SparsePauliOp label for a multi-qubit Pauli.
    Qiskit labels are little-endian: label[0] acts on qubit 0 (rightmost in string).
    """
    characters = ["I"] * n_qubits
    for site in support:
        characters[site] = pauli
    # SparsePauliOp uses reversed ordering (qubit 0 is rightmost).
    return "".join(reversed(characters))


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


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize("state", ALL_STATES)
def test_logical_expectation(
    state: InjectedStateName,
    basis: str,
) -> None:
    """Verify ⟨P_L⟩ for every state, Pauli basis, and weight-3 representative."""
    circuit = QuantumCircuit(NUM_DATA_QUBITS)
    append_abstract_encoder(
        circuit,
        data_qubits=list(range(NUM_DATA_QUBITS)),
        injected_state=state,
        add_barriers=False,
    )
    statevector = Statevector(circuit)
    representatives = LOGICAL_Z_REPRESENTATIVES if basis == "Z" else LOGICAL_X_REPRESENTATIVES
    expected = LOGICAL_EXPECTATION_VALUES[state][LOGICAL_EXPECTATION_INDEX[basis]]
    for support in representatives:
        measured = statevector.expectation_value(
            _logical_pauli_observable(support, basis, NUM_DATA_QUBITS)
        ).real
        assert measured == pytest.approx(expected, abs=TOLERANCE)


# =============================================================================
# 11-site and 14-site patches
# =============================================================================


class PatchTestSpec(NamedTuple):
    """Geometry and operator supports shared by the parameterized patch tests."""

    name: str
    # The full patch width, including data and any encoder helper sites.
    num_qubits: int
    data_sites: tuple[int, ...]
    x_stabilizers: tuple[tuple[int, ...], ...]
    z_stabilizers: tuple[tuple[int, ...], ...]
    logical_x_representatives: tuple[tuple[int, ...], ...]
    logical_z_representatives: tuple[tuple[int, ...], ...]


PATCH_11_SITE_TEST_SPEC = PatchTestSpec(
    "11-site",
    NUM_QUBITS_PATCH_11_SITES,
    PATCH_11_SITE_DATA_SITES,
    PATCH_11_SITE_X_STABILIZERS,
    PATCH_11_SITE_Z_STABILIZERS,
    PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES,
    PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES,
)
PATCH_14_SITE_TEST_SPEC = PatchTestSpec(
    "14-site",
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_14_SITE_DATA_SITES,
    PATCH_14_SITE_X_STABILIZERS,
    PATCH_14_SITE_Z_STABILIZERS,
    PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES,
    PATCH_14_SITE_LOGICAL_Z_REPRESENTATIVES,
)

PATCH_ENCODERS = [
    pytest.param(PATCH_11_SITE_TEST_SPEC, "abstract", id="11-site-abstract"),
    pytest.param(PATCH_11_SITE_TEST_SPEC, "rosenfeld", id="11-site-rosenfeld"),
    pytest.param(PATCH_14_SITE_TEST_SPEC, "abstract", id="14-site-abstract"),
    pytest.param(
        PATCH_14_SITE_TEST_SPEC,
        "abstract-routed",
        id="14-site-abstract-routed",
    ),
    pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", id="14-site-hirano"),
]
ARBITRARY_STATE_PATCH_ENCODERS = (
    (PATCH_11_SITE_TEST_SPEC, "abstract"),
    (PATCH_11_SITE_TEST_SPEC, "rosenfeld"),
    (PATCH_14_SITE_TEST_SPEC, "abstract"),
    (PATCH_14_SITE_TEST_SPEC, "abstract-routed"),
)
PATCH_ENCODER_STATE_CASES = [
    pytest.param(spec, encoder, state, id=f"{spec.name}-{encoder}-{state}")
    for spec, encoder in ARBITRARY_STATE_PATCH_ENCODERS
    for state in ALL_STATES
] + [
    pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", state, id=f"14-site-hirano-{state}")
    for state in HIRANO_STATES
]
PATCH_FAULT_ENCODER_STATE_CASES = [
    pytest.param(spec, encoder, state, id=f"{spec.name}-{encoder}-{state}")
    for spec, encoder in ARBITRARY_STATE_PATCH_ENCODERS
    for state in ("0", "T", "Ty")
] + [
    pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", state, id=f"14-site-hirano-{state}")
    for state in ("S", "Sdg")
]
PATCH_ROUTED_ENCODERS = [
    pytest.param(PATCH_11_SITE_TEST_SPEC, "rosenfeld", id="11-site-rosenfeld"),
    pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", id="14-site-hirano"),
    pytest.param(PATCH_14_SITE_TEST_SPEC, "abstract-routed", id="14-site-abstract-routed"),
]
PATCH_ROUTED_ENCODER_STATE_CASES = (
    [
        pytest.param(
            PATCH_11_SITE_TEST_SPEC,
            "rosenfeld",
            state,
            id=f"11-site-rosenfeld-{state}",
        )
        for state in ALL_STATES
    ]
    + [
        pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", state, id=f"14-site-hirano-{state}")
        for state in HIRANO_STATES
    ]
    + [
        pytest.param(
            PATCH_14_SITE_TEST_SPEC,
            "abstract-routed",
            state,
            id=f"14-site-abstract-routed-{state}",
        )
        for state in ALL_STATES
    ]
)


def _apply_patch_encoder(
    circuit: QuantumCircuit,
    patch_spec: PatchTestSpec,
    encoder: str,
    injected_state: InjectedStateName,
) -> None:
    """Append one encoder covered by the shared patch tests."""
    patch_qubits = list(range(patch_spec.num_qubits))
    if patch_spec.name == "11-site" and encoder == "abstract":
        append_11_site_patch_abstract_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=False,
        )
    elif patch_spec.name == "11-site" and encoder == "rosenfeld":
        append_11_site_patch_rosenfeld_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=False,
        )
    elif patch_spec.name == "14-site" and encoder == "abstract":
        append_14_site_patch_abstract_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=False,
        )
    elif patch_spec.name == "14-site" and encoder == "abstract-routed":
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=False,
        )
    elif patch_spec.name == "14-site" and encoder == "hirano":
        append_14_site_patch_hirano_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=False,
        )
    else:
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}")


# --------------------------
# Logical expectation values
# --------------------------


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ENCODER_STATE_CASES)
def test_patch_logical_expectation(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: InjectedStateName,
    basis: str,
) -> None:
    """Verify every encoded patch state against every logical representative."""
    circuit = QuantumCircuit(patch_spec.num_qubits)
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    statevector = Statevector(circuit)
    representatives = (
        patch_spec.logical_z_representatives
        if basis == "Z"
        else patch_spec.logical_x_representatives
    )
    expected = LOGICAL_EXPECTATION_VALUES[state][LOGICAL_EXPECTATION_INDEX[basis]]
    for support in representatives:
        measured = statevector.expectation_value(
            _logical_pauli_observable(support, basis, patch_spec.num_qubits)
        ).real
        assert measured == pytest.approx(expected, abs=TOLERANCE)


# -----------------------------------
# Non-destructive logical measurement
# -----------------------------------


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ENCODER_STATE_CASES)
def test_patch_logical_via_measurement_circuit(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: InjectedStateName,
    basis: str,
) -> None:
    """Verify ancilla readout against every patch logical representative."""
    ancilla_qubit = patch_spec.num_qubits
    representatives = (
        patch_spec.logical_z_representatives
        if basis == "Z"
        else patch_spec.logical_x_representatives
    )
    expected = LOGICAL_EXPECTATION_VALUES[state][LOGICAL_EXPECTATION_INDEX[basis]]
    for support in representatives:
        circuit = QuantumCircuit(patch_spec.num_qubits + 1, 1)
        _apply_patch_encoder(circuit, patch_spec, encoder, state)
        measurement = (
            append_11_site_patch_logical_measurement
            if patch_spec.name == "11-site"
            else append_14_site_patch_logical_measurement
        )
        measurement(
            circuit,
            basis,
            patch_qubits=list(range(patch_spec.num_qubits)),
            support=support,
            ancilla_qubit=ancilla_qubit,
            bits=[0],
            destructive=False,
        )
        if basis == "Y":
            assert circuit.data[-1].operation.name == "store"
            circuit.data.pop()
        circuit.remove_final_measurements(inplace=True)
        measured = Statevector(circuit).expectation_value(SparsePauliOp("Z"), [ancilla_qubit]).real
        if basis == "Y":
            measured = -measured
        assert measured == pytest.approx(expected, abs=TOLERANCE)


# ----------------------------------
# Logical representative consistency
# ----------------------------------


@pytest.mark.parametrize(
    (
        "data_sites",
        "x_stabilizers",
        "z_stabilizers",
        "x_representatives",
        "z_representatives",
    ),
    [
        pytest.param(
            tuple(range(NUM_DATA_QUBITS)),
            X_STABILIZERS,
            Z_STABILIZERS,
            LOGICAL_X_REPRESENTATIVES,
            LOGICAL_Z_REPRESENTATIVES,
            id="abstract",
        ),
        pytest.param(
            PATCH_11_SITE_DATA_SITES,
            PATCH_11_SITE_X_STABILIZERS,
            PATCH_11_SITE_Z_STABILIZERS,
            PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES,
            PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES,
            id="11-site",
        ),
        pytest.param(
            PATCH_14_SITE_DATA_SITES,
            PATCH_14_SITE_X_STABILIZERS,
            PATCH_14_SITE_Z_STABILIZERS,
            PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES,
            PATCH_14_SITE_LOGICAL_Z_REPRESENTATIVES,
            id="14-site",
        ),
    ],
)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_logical_representatives_are_weight_seven_times_stabilizer(
    data_sites: tuple[int, ...],
    x_stabilizers: tuple[tuple[int, ...], ...],
    z_stabilizers: tuple[tuple[int, ...], ...],
    x_representatives: tuple[tuple[int, ...], ...],
    z_representatives: tuple[tuple[int, ...], ...],
    basis: str,
) -> None:
    """Verify every logical representative equals P^7 times a stabilizer."""
    stabilizers = x_stabilizers if basis == "X" else z_stabilizers
    representatives = x_representatives if basis == "X" else z_representatives
    stabilizer_group: list[set[int]] = [set()]
    for count in range(1, len(stabilizers) + 1):
        for combination in combinations((set(support) for support in stabilizers), count):
            element: set[int] = set()
            for support in combination:
                element = element.symmetric_difference(support)
            stabilizer_group.append(element)

    all_data = set(data_sites)
    for representative in representatives:
        assert any(
            all_data.symmetric_difference(element) == set(representative)
            for element in stabilizer_group
        )


@pytest.mark.parametrize(
    ("x_stabilizers", "z_stabilizers", "x_representatives", "z_representatives"),
    [
        pytest.param(
            X_STABILIZERS,
            Z_STABILIZERS,
            LOGICAL_X_REPRESENTATIVES,
            LOGICAL_Z_REPRESENTATIVES,
            id="abstract",
        ),
        pytest.param(
            PATCH_11_SITE_X_STABILIZERS,
            PATCH_11_SITE_Z_STABILIZERS,
            PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES,
            PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES,
            id="11-site",
        ),
        pytest.param(
            PATCH_14_SITE_X_STABILIZERS,
            PATCH_14_SITE_Z_STABILIZERS,
            PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES,
            PATCH_14_SITE_LOGICAL_Z_REPRESENTATIVES,
            id="14-site",
        ),
    ],
)
def test_steane_code_is_css_self_dual(
    x_stabilizers: tuple[tuple[int, ...], ...],
    z_stabilizers: tuple[tuple[int, ...], ...],
    x_representatives: tuple[tuple[int, ...], ...],
    z_representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify X and Z operators use the same supports."""
    assert x_stabilizers == z_stabilizers
    assert x_representatives == z_representatives


# -------------------
# Stabilizer syndrome
# -------------------


def _build_and_run_patch_syndrome(
    patch_spec: PatchTestSpec,
    encoder: str,
    injected_state: InjectedStateName,
    faults: list[tuple[str, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode, inject faults, and return the measured X/Z syndromes."""
    bit_count = 6 if patch_spec.name == "11-site" else 7
    circuit = QuantumCircuit(patch_spec.num_qubits, bit_count)
    _apply_patch_encoder(circuit, patch_spec, encoder, injected_state)
    append_pauli_faults(circuit, faults)
    if patch_spec.name == "11-site":
        append_11_site_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            bits=list(range(6)),
            add_barriers=False,
        )
    elif patch_spec.name == "14-site":
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            stabilizer_bits=list(range(6)),
            d2_copy_flag_bit=6,
            add_barriers=False,
        )
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")

    bitstring = next(iter(AerSimulator().run(circuit, shots=1).result().get_counts()))
    bits = parse_qiskit_bitstring(
        bitstring,
        tuple(f"b{i}" for i in range(bit_count)),
    )
    if patch_spec.name == "11-site":
        return (
            tuple(bits[f"b{i}"] for i in range(3)),
            tuple(bits[f"b{i}"] for i in range(3, 6)),
        )
    return (
        tuple(bits[f"b{i}"] for i in (0, 2, 4)),
        tuple(bits[f"b{i}"] for i in (1, 3, 5)),
    )


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ENCODER_STATE_CASES)
def test_patch_all_stabilizers_via_ancilla_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: InjectedStateName,
) -> None:
    """Verify all six measured stabilizer outcomes are trivial."""
    assert _build_and_run_patch_syndrome(patch_spec, encoder, state, []) == (
        (0, 0, 0),
        (0, 0, 0),
    )


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ENCODER_STATE_CASES)
def test_patch_stabilizer_expectation_values(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: InjectedStateName,
) -> None:
    """Verify all six stabilizer operators have expectation +1."""
    circuit = QuantumCircuit(patch_spec.num_qubits)
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    statevector = Statevector(circuit)
    for basis, stabilizers in (
        ("X", patch_spec.x_stabilizers),
        ("Z", patch_spec.z_stabilizers),
    ):
        for support in stabilizers:
            label = _logical_pauli_label(support, basis, patch_spec.num_qubits)
            assert statevector.expectation_value(SparsePauliOp(label)).real == pytest.approx(
                1.0, abs=TOLERANCE
            )


@pytest.mark.parametrize(("patch_spec", "encoder"), PATCH_ENCODERS)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_patch_single_basis_stabilizer_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    basis: str,
) -> None:
    """Verify a complete one-basis stabilizer round on either patch."""
    bit_count = 3 if patch_spec.name == "11-site" else 7
    circuit = QuantumCircuit(patch_spec.num_qubits, bit_count)
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    if patch_spec.name == "11-site":
        append_11_site_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            bits=list(range(3)),
            basis=basis,
            add_barriers=False,
        )
    elif patch_spec.name == "14-site":
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            stabilizer_bits=list(range(3)),
            bell_partner_flag_bits=list(range(3, 6)),
            d2_copy_flag_bit=6,
            basis=basis,
            add_barriers=False,
        )
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")
    assert AerSimulator().run(circuit, shots=1).result().get_counts() == {"0" * bit_count: 1}


@pytest.mark.parametrize(("patch_spec", "encoder"), PATCH_ENCODERS)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_patch_selected_stabilizer_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    basis: str,
) -> None:
    """Verify either patch measures only the requested stabilizer subset."""
    bit_count = 5 if patch_spec.name == "14-site" else 2
    circuit = QuantumCircuit(patch_spec.num_qubits, bit_count)
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    measurement_start = len(circuit.data)
    if patch_spec.name == "11-site":
        append_11_site_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            bits=list(range(2)),
            basis=basis,
            stabilizer_indices=(0, 2),
            add_barriers=False,
        )
        unselected_ancilla_sites = {6}
    elif patch_spec.name == "14-site":
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            stabilizer_bits=list(range(2)),
            bell_partner_flag_bits=list(range(2, 4)),
            d2_copy_flag_bit=4,
            basis=basis,
            stabilizer_indices=(0, 2),
            add_barriers=False,
        )
        unselected_ancilla_sites = {11, 12}
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")

    appended_instructions = circuit.data[measurement_start:]
    assert all(
        not unselected_ancilla_sites.intersection(
            circuit.find_bit(qubit).index for qubit in instruction.qubits
        )
        for instruction in appended_instructions
    )
    assert (
        sum(instruction.operation.name == "measure" for instruction in appended_instructions)
        == bit_count
    )
    assert AerSimulator().run(circuit, shots=1).result().get_counts() == {"0" * bit_count: 1}


# ---------------------------------------
# Stabilizer syndrome with injected faults
# ---------------------------------------


def _expected_patch_syndrome(
    patch_spec: PatchTestSpec,
    faults: list[tuple[str, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the expected X/Z stabilizer outcomes for physical faults."""
    x_syndrome = [0, 0, 0]
    z_syndrome = [0, 0, 0]
    for pauli, site in faults:
        pauli = normalize_pauli_char(pauli)
        for index, support in enumerate(patch_spec.x_stabilizers):
            if site in support and pauli in ("Z", "Y"):
                x_syndrome[index] ^= 1
        for index, support in enumerate(patch_spec.z_stabilizers):
            if site in support and pauli in ("X", "Y"):
                z_syndrome[index] ^= 1
    return tuple(x_syndrome), tuple(z_syndrome)


_DATA_SITE_PAIRS = [(first, second) for first in range(7) for second in range(first + 1, 7)]


class TestPatchSyndromeExtraction:
    """Verify both patch measurements against expected fault syndromes."""

    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_no_fault_trivial(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: InjectedStateName,
    ) -> None:
        """No faults produce a trivial syndrome."""
        assert _build_and_run_patch_syndrome(patch_spec, encoder, state, []) == (
            (0, 0, 0),
            (0, 0, 0),
        )

    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    @pytest.mark.parametrize("site_index", range(7))
    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_single_fault(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: InjectedStateName,
        site_index: int,
        pauli: str,
    ) -> None:
        """Single-qubit faults produce the expected syndrome."""
        faults = [(pauli, patch_spec.data_sites[site_index])]
        assert _build_and_run_patch_syndrome(
            patch_spec, encoder, state, faults
        ) == _expected_patch_syndrome(patch_spec, faults)

    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    @pytest.mark.parametrize(("first", "second"), _DATA_SITE_PAIRS)
    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_two_qubit_same_pauli(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: InjectedStateName,
        first: int,
        second: int,
        pauli: str,
    ) -> None:
        """Two same-Pauli faults produce the XOR of their syndromes."""
        faults = [
            (pauli, patch_spec.data_sites[first]),
            (pauli, patch_spec.data_sites[second]),
        ]
        assert _build_and_run_patch_syndrome(
            patch_spec, encoder, state, faults
        ) == _expected_patch_syndrome(patch_spec, faults)

    @pytest.mark.parametrize("paulis", [("X", "Z"), ("X", "Y"), ("Z", "Y")])
    @pytest.mark.parametrize(("first", "second"), _DATA_SITE_PAIRS)
    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_two_qubit_mixed_pauli(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: InjectedStateName,
        first: int,
        second: int,
        paulis: tuple[str, str],
    ) -> None:
        """Two mixed-Pauli faults are detected independently by both sectors."""
        faults = [
            (paulis[0], patch_spec.data_sites[first]),
            (paulis[1], patch_spec.data_sites[second]),
        ]
        assert _build_and_run_patch_syndrome(
            patch_spec, encoder, state, faults
        ) == _expected_patch_syndrome(patch_spec, faults)


# --------------------------------
# Injection azimuth / polar angles
# --------------------------------


@pytest.mark.parametrize(
    ("patch_spec", "encoder"),
    [
        pytest.param(spec, encoder, id=f"{spec.name}-{encoder}")
        for spec, encoder in ARBITRARY_STATE_PATCH_ENCODERS
    ],
)
@pytest.mark.parametrize(
    ("angle_argument", "angle", "equivalent_state"),
    [
        # Azimuth sweeps Rz(angle)|+> around the XY plane containing |T>.
        ("injection_azimuth_angle", 0.0, "+"),
        ("injection_azimuth_angle", math.pi / 4, "T"),
        ("injection_azimuth_angle", math.pi / 2, "Y+"),
        ("injection_azimuth_angle", -math.pi / 4, "Tdg"),
        ("injection_azimuth_angle", -math.pi / 2, "Y-"),
        ("injection_azimuth_angle", math.pi, "-"),
        # Polar sweeps Ry(angle)|0> around the XZ plane containing |Ty>.
        ("injection_polar_angle", 0.0, "0"),
        ("injection_polar_angle", math.pi / 4, "Ty"),
        ("injection_polar_angle", math.pi / 2, "+"),
        ("injection_polar_angle", math.pi, "1"),
        ("injection_polar_angle", -math.pi / 2, "-"),
    ],
)
def test_patch_injection_angle_matches_named_state(
    patch_spec: PatchTestSpec,
    encoder: str,
    angle_argument: str,
    angle: float,
    equivalent_state: InjectedStateName,
) -> None:
    """Verify each swept-angle input encodes the same patch state as its named input."""
    angle_circuit = QuantumCircuit(patch_spec.num_qubits)
    named_circuit = QuantumCircuit(patch_spec.num_qubits)
    if patch_spec.name == "11-site" and encoder == "abstract":
        encoder_function = append_11_site_patch_abstract_encoder
    elif patch_spec.name == "11-site" and encoder == "rosenfeld":
        encoder_function = append_11_site_patch_rosenfeld_encoder
    elif patch_spec.name == "14-site" and encoder == "abstract":
        encoder_function = append_14_site_patch_abstract_encoder
    elif patch_spec.name == "14-site" and encoder == "abstract-routed":
        encoder_function = append_14_site_patch_abstract_routed_encoder
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    encoder_arguments = {
        "patch_qubits": list(range(patch_spec.num_qubits)),
        "add_barriers": False,
    }
    encoder_function(angle_circuit, **encoder_arguments, **{angle_argument: angle})
    encoder_function(named_circuit, **encoder_arguments, injected_state=equivalent_state)
    assert Statevector(angle_circuit).equiv(Statevector(named_circuit))


@pytest.mark.parametrize(
    "injection_arguments",
    [
        {"injected_state": "T", "injection_azimuth_angle": math.pi / 4},
        {"injected_state": "Ty", "injection_polar_angle": math.pi / 4},
        {"injection_azimuth_angle": math.pi / 4, "injection_polar_angle": math.pi / 4},
    ],
    ids=["state-azimuth", "state-polar", "azimuth-polar"],
)
def test_patch_injection_inputs_are_mutually_exclusive(
    injection_arguments: dict[str, object],
) -> None:
    """Reject any two injection inputs, even when they name the same state.

    Every arbitrary-state encoder shares one ``prepare_injection_input`` check,
    so the production abstract-routed encoder covers all of them.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        append_14_site_patch_abstract_routed_encoder(
            QuantumCircuit(NUM_QUBITS_PATCH_14_SITES),
            patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
            add_barriers=False,
            **injection_arguments,
        )


# -------------------------
# Routed encoder properties
# -------------------------


@pytest.mark.parametrize(("patch_spec", "encoder"), PATCH_ROUTED_ENCODERS)
def test_patch_routed_encoder_two_qubit_gates_are_local(
    patch_spec: PatchTestSpec,
    encoder: str,
) -> None:
    """Every two-qubit gate in each routed encoder follows its patch geometry."""
    circuit = QuantumCircuit(patch_spec.num_qubits)
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    if patch_spec.name == "11-site":
        coordinates = PATCH_11_SITE_COORDS
    elif encoder == "hirano":
        coordinates = PATCH_14_SITE_HIRANO_ENCODER_COORDS
    elif encoder == "abstract-routed":
        coordinates = PATCH_14_SITE_COORDS
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    for instruction in circuit.data:
        if instruction.operation.num_qubits != 2:
            continue
        site_a = circuit.find_bit(instruction.qubits[0]).index
        site_b = circuit.find_bit(instruction.qubits[1]).index
        ax, ay = coordinates[site_a]
        bx, by = coordinates[site_b]
        assert (abs(ax - bx), abs(ay - by)) == (1, 1)
        if patch_spec.name == "11-site":
            assert {site_a in patch_spec.data_sites, site_b in patch_spec.data_sites} == {
                True,
                False,
            }
        elif patch_spec.name == "14-site":
            assert (abs(ax - bx), abs(ay - by)) == (1, 1)
        else:  # pragma: no cover - parametrization uses only supported patches.
            raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ROUTED_ENCODER_STATE_CASES)
def test_patch_routed_encoder_helper_expectation_values(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: InjectedStateName,
) -> None:
    """Routed-encoder helpers return to their expected product state."""
    circuit = QuantumCircuit(patch_spec.num_qubits)
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    statevector = Statevector(circuit)
    if patch_spec.name == "11-site":
        helper_sites = PATCH_11_SITE_HELPER_SITES
        observable = SparsePauliOp("Z")
    elif encoder == "hirano":
        helper_sites = PATCH_14_SITE_HELPER_SITES
        observable = SparsePauliOp("X")
    elif encoder == "abstract-routed":
        helper_sites = tuple(
            site for site in range(patch_spec.num_qubits) if site not in patch_spec.data_sites
        )
        observable = SparsePauliOp("Z")
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    for site in helper_sites:
        assert statevector.expectation_value(observable, [site]).real == pytest.approx(
            1.0, abs=TOLERANCE
        )


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ROUTED_ENCODER_STATE_CASES)
def test_patch_routed_encoder_helper_flags_are_zero(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: InjectedStateName,
) -> None:
    """Measured routed-encoder helpers return the all-zero flag outcome."""
    if patch_spec.name == "11-site":
        helper_sites = PATCH_11_SITE_HELPER_SITES
    elif encoder == "abstract-routed":
        helper_sites = PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES
    elif encoder == "hirano":
        helper_sites = PATCH_14_SITE_HELPER_SITES
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    circuit = QuantumCircuit(patch_spec.num_qubits, len(helper_sites))
    if patch_spec.name == "11-site":
        append_11_site_patch_rosenfeld_encoder(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            injected_state=state,
            flag_bits=list(range(len(helper_sites))),
            add_barriers=False,
        )
    elif encoder == "hirano":
        append_14_site_patch_hirano_encoder(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            injected_state=state,
            flag_bits=list(range(len(helper_sites))),
            add_barriers=False,
        )
    elif encoder == "abstract-routed":
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            patch_qubits=list(range(patch_spec.num_qubits)),
            injected_state=state,
            flag_bits=list(range(len(helper_sites))),
            add_barriers=False,
        )
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    shots = 1024
    assert AerSimulator().run(circuit, shots=shots).result().get_counts() == {
        "0" * len(helper_sites): shots
    }


@pytest.mark.parametrize(("patch_spec", "encoder"), PATCH_ROUTED_ENCODERS)
def test_patch_routed_encoder_two_qubit_depth_does_not_increase_after_transpile(
    patch_spec: PatchTestSpec,
    encoder: str,
) -> None:
    """Fixed-layout square-lattice transpilation adds no two-qubit depth."""
    circuit = QuantumCircuit(patch_spec.num_qubits)
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    if patch_spec.name == "11-site":
        coordinates = PATCH_11_SITE_COORDS
    elif encoder == "hirano":
        coordinates = PATCH_14_SITE_HIRANO_ENCODER_COORDS
    elif encoder == "abstract-routed":
        coordinates = PATCH_14_SITE_COORDS
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    coupling_map = [
        [site_a, site_b]
        for site_a, coordinate_a in enumerate(coordinates)
        for site_b, coordinate_b in enumerate(coordinates)
        if site_a != site_b
        and (
            abs(coordinate_a[0] - coordinate_b[0]),
            abs(coordinate_a[1] - coordinate_b[1]),
        )
        == (1, 1)
    ]
    circuit = RemoveBarriers()(circuit)
    original_depth = two_qubit_depth(circuit)
    transpiled = transpile(
        circuit,
        basis_gates=["cx", "id", "rz", "sx", "x"],
        coupling_map=coupling_map,
        initial_layout=list(range(patch_spec.num_qubits)),
        routing_method="none",
        optimization_level=3,
    )
    transpiled_depth = two_qubit_depth(transpiled)
    assert transpiled_depth <= original_depth
    if encoder == "abstract-routed":
        assert transpiled_depth == original_depth


# ------------------------------
# Specific to the 14-site patch
# ------------------------------


def test_14_site_abstract_encoder_uses_only_fixed_data_sites() -> None:
    """The mapped abstract encoder keeps D1 at ``(1, 5)`` and ignores helpers."""
    assert PATCH_14_SITE_COORDS[1] == (1, 5)
    circuit = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES)
    append_14_site_patch_abstract_encoder(
        circuit,
        patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
        injected_state="T",
        add_barriers=False,
    )

    used_two_qubit_sites = {
        circuit.find_bit(qubit).index
        for instruction in circuit.data
        if instruction.operation.num_qubits == 2
        for qubit in instruction.qubits
    }
    assert used_two_qubit_sites <= set(PATCH_14_SITE_DATA_SITES)


@pytest.mark.parametrize("state", HIRANO_STATES)
def test_14_site_hirano_encoder_matches_abstract_y_state(
    state: InjectedStateName,
) -> None:
    """Uncomputing Hirano with the abstract encoder returns every qubit to zero."""
    hirano = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES)
    append_14_site_patch_hirano_encoder(
        hirano,
        patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
        injected_state=state,
        add_barriers=False,
    )
    abstract = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES)
    append_14_site_patch_abstract_encoder(
        abstract,
        patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
        injected_state=state,
        add_barriers=False,
    )

    check = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES, NUM_QUBITS_PATCH_14_SITES)
    check.compose(hirano, inplace=True)
    for site in PATCH_14_SITE_HELPER_SITES:
        check.h(site)
    check.compose(abstract.inverse(), inplace=True)
    check.measure(range(NUM_QUBITS_PATCH_14_SITES), range(NUM_QUBITS_PATCH_14_SITES))

    counts = AerSimulator().run(check, shots=1).result().get_counts()
    assert counts == {"0" * NUM_QUBITS_PATCH_14_SITES: 1}


@pytest.mark.parametrize("state", ["0", "1", "+", "-", "T", "Tdg", "Ty"])
def test_14_site_hirano_encoder_rejects_non_y_states(
    state: InjectedStateName,
) -> None:
    """Hirano's single phase injection cannot encode a non-Y input."""
    with pytest.raises(ValueError, match="logical-Y eigenstate"):
        append_14_site_patch_hirano_encoder(
            QuantumCircuit(NUM_QUBITS_PATCH_14_SITES),
            patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
            injected_state=state,
        )


def test_14_site_encoder_two_qubit_depths() -> None:
    """The Hirano and abstract-routed encoders retain their optimized CX depths."""
    hirano = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES)
    append_14_site_patch_hirano_encoder(
        hirano,
        patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
        injected_state="S",
        add_barriers=False,
    )
    abstract_routed = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES)
    append_14_site_patch_abstract_routed_encoder(
        abstract_routed,
        patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
        injected_state="S",
        add_barriers=False,
    )

    assert two_qubit_depth(hirano) == 11
    assert two_qubit_depth(abstract_routed) == 9
    assert hirano.count_ops()["cx"] == 25
    assert abstract_routed.count_ops()["cx"] == 23


# ----------------------------------------------------
# Superdense stabilizer measurement for 14-site patch
# ----------------------------------------------------


def test_14_site_superdense_qed_matches_hirano_cx_depth_with_two_fewer_gates() -> None:
    """Both QED circuits have CX depth eight after collapsing disjoint gates."""
    hirano_qed = QuantumCircuit(15)
    # Remapped into our site labels from Hirano's
    # ``xz_syndrome_extraction_after_injection_generator`` in
    # ``references/hirano_code/src/steane_code.py``.
    hirano_qed_cx_layers = (
        ((14, 1), (7, 8), (11, 12), (9, 10), (13, 2)),
        ((8, 5), (1, 14), (11, 3), (12, 2), (9, 6), (10, 0)),
        ((8, 1), (11, 0), (9, 4), (2, 13)),
        ((7, 4), (8, 0), (11, 5), (10, 13)),
        ((4, 7), (0, 8), (5, 11), (13, 10)),
        ((5, 8), (0, 11), (6, 9), (13, 2)),
        ((1, 8), (3, 11), (2, 12), (4, 9), (0, 10)),
        ((7, 8), (11, 12), (9, 10), (2, 13)),
    )
    for layer in hirano_qed_cx_layers:
        for control, target in layer:
            hirano_qed.cx(control, target)

    fixed_patch_qed = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES, 7)
    append_14_site_patch_superdense_stabilizer_measurement(
        fixed_patch_qed,
        patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
        stabilizer_bits=list(range(6)),
        d2_copy_flag_bit=6,
        add_barriers=False,
    )

    assert two_qubit_depth(hirano_qed) == 8
    assert two_qubit_depth(fixed_patch_qed) == 8
    assert hirano_qed.count_ops()["cx"] == 36
    # The fixed-patch encoder already places D1 at its final site, saving the
    # two CXs that Hirano's QED uses to relocate D1 from its injection-stage site.
    assert fixed_patch_qed.count_ops()["cx"] == 34


@pytest.mark.parametrize("face_bases", tuple(product((None, "X", "Z", "BOTH"), repeat=3)))
def test_14_site_superdense_subsets_do_not_exceed_full_round_cx_depth(face_bases) -> None:
    """Exhaust all 64 subsets, including mixed faces and the empty selection."""
    selected_indices = tuple(index for index, basis in enumerate(face_bases) if basis is not None)
    selected_bases = tuple(face_bases[index] for index in selected_indices)
    full_round = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES, 7)
    append_14_site_patch_superdense_stabilizer_measurement(
        full_round,
        patch_qubits=list(range(14)),
        stabilizer_bits=list(range(6)),
        d2_copy_flag_bit=6,
        add_barriers=False,
    )
    selected_round = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES, 7)
    stabilizer_count = sum(2 if basis == "BOTH" else 1 for basis in selected_bases)
    partner_flag_count = sum(basis != "BOTH" for basis in selected_bases)
    if selected_indices:
        append_14_site_patch_superdense_stabilizer_measurement(
            selected_round,
            patch_qubits=list(range(14)),
            stabilizer_bits=list(range(stabilizer_count)),
            bell_partner_flag_bits=(
                list(range(stabilizer_count, stabilizer_count + partner_flag_count))
                if partner_flag_count
                else None
            ),
            d2_copy_flag_bit=6 if 2 in selected_indices else None,
            basis=selected_bases,
            stabilizer_indices=selected_indices,
            add_barriers=False,
        )
    assert two_qubit_depth(full_round) == 8
    assert two_qubit_depth(selected_round) <= two_qubit_depth(full_round)
    assert selected_round.count_ops().get("measure", 0) == (
        2 * len(selected_indices) + (2 in selected_indices)
    )


@pytest.mark.parametrize("face_bases", tuple(product((None, "X", "Z", "BOTH"), repeat=3)))
def test_14_site_superdense_subsets_prepare_and_read_each_bell_pair_once(face_bases) -> None:
    """Every subset creates, uncomputes, and reads each active face's Bell pair once.

    A single-basis face records its requested parity in ``stabilizer_bits`` and its
    partner ancilla in ``bell_partner_flag_bits``; a dual-basis face records both
    ancillas as stabilizers; an uninvolved face touches neither ancilla.
    """
    selected_indices = tuple(index for index, basis in enumerate(face_bases) if basis is not None)
    selected_bases = tuple(face_bases[index] for index in selected_indices)
    stabilizer_count = sum(2 if basis == "BOTH" else 1 for basis in selected_bases)
    partner_flag_count = sum(basis != "BOTH" for basis in selected_bases)
    circuit = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES, 7)
    if selected_indices:
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
            stabilizer_bits=list(range(stabilizer_count)),
            bell_partner_flag_bits=(
                list(range(stabilizer_count, stabilizer_count + partner_flag_count))
                if partner_flag_count
                else None
            ),
            d2_copy_flag_bit=6 if 2 in selected_indices else None,
            basis=selected_bases,
            stabilizer_indices=selected_indices,
            add_barriers=False,
        )

    # The (X-ancilla, Z-ancilla) CX appears only in Bell preparation and uncompute,
    # never in a coupling layer; roles follow the classical bit each readout targets.
    reset_sites = [
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.data
        if instruction.operation.name == "reset"
    ]
    ancilla_ancilla_cnots = [
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in circuit.data
        if instruction.operation.name == "cx"
    ]
    role_by_measured_site: dict[int, str] = {}
    measured_sites: list[int] = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        site = circuit.find_bit(instruction.qubits[0]).index
        bit_index = circuit.find_bit(instruction.clbits[0]).index
        measured_sites.append(site)
        if bit_index < stabilizer_count:
            role_by_measured_site[site] = "STABILIZER"
        elif bit_index < stabilizer_count + partner_flag_count:
            role_by_measured_site[site] = "BELL_PARTNER_FLAG"
        else:
            role_by_measured_site[site] = "D2_COPY_FLAG"
    touched_sites = {
        circuit.find_bit(qubit).index
        for instruction in circuit.data
        for qubit in instruction.qubits
    }

    for index, (x_ancilla, z_ancilla) in enumerate(PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS):
        face_basis = face_bases[index]
        if face_basis is None:
            # Uninvolved face: no operation of any kind (reset, H, CX, or measure)
            # touches either ancilla.
            assert x_ancilla not in touched_sites and z_ancilla not in touched_sites
            continue
        # Bell pair created once (each ancilla reset once) and uncomputed once (its
        # lone ancilla-ancilla CX appears once in preparation and once in uncompute).
        assert reset_sites.count(x_ancilla) == reset_sites.count(z_ancilla) == 1
        assert ancilla_ancilla_cnots.count((x_ancilla, z_ancilla)) == 2
        # Each ancilla is measured exactly once, into its role's classical bits.
        assert measured_sites.count(x_ancilla) == measured_sites.count(z_ancilla) == 1
        if face_basis == "BOTH":
            assert (
                role_by_measured_site[x_ancilla] == role_by_measured_site[z_ancilla] == "STABILIZER"
            )
        elif face_basis == "X":
            assert role_by_measured_site[x_ancilla] == "STABILIZER"
            assert role_by_measured_site[z_ancilla] == "BELL_PARTNER_FLAG"
        elif face_basis == "Z":
            assert role_by_measured_site[x_ancilla] == "BELL_PARTNER_FLAG"
            assert role_by_measured_site[z_ancilla] == "STABILIZER"
        else:
            raise AssertionError(f"Unexpected face basis {face_basis}.")


@pytest.mark.parametrize("face_bases", tuple(product((None, "X", "Z", "BOTH"), repeat=3)))
def test_14_site_superdense_subsets_read_d2_copy_only_with_face_0246(face_bases) -> None:
    """Read the D2' copy flag once exactly when face 0246 is selected."""
    selected_indices = tuple(index for index, basis in enumerate(face_bases) if basis is not None)
    selected_bases = tuple(face_bases[index] for index in selected_indices)
    stabilizer_count = sum(2 if basis == "BOTH" else 1 for basis in selected_bases)
    partner_flag_count = sum(basis != "BOTH" for basis in selected_bases)
    circuit = QuantumCircuit(NUM_QUBITS_PATCH_14_SITES, 7)
    if selected_indices:
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=list(range(NUM_QUBITS_PATCH_14_SITES)),
            stabilizer_bits=list(range(stabilizer_count)),
            bell_partner_flag_bits=(
                list(range(stabilizer_count, stabilizer_count + partner_flag_count))
                if partner_flag_count
                else None
            ),
            d2_copy_flag_bit=6 if 2 in selected_indices else None,
            basis=selected_bases,
            stabilizer_indices=selected_indices,
            add_barriers=False,
        )
    touched_sites = {
        circuit.find_bit(qubit).index
        for instruction in circuit.data
        for qubit in instruction.qubits
    }
    d2_copy_measurements = [
        circuit.find_bit(instruction.clbits[0]).index
        for instruction in circuit.data
        if instruction.operation.name == "measure"
        and circuit.find_bit(instruction.qubits[0]).index == 13
    ]
    if face_bases[2] is not None:
        # Read exactly once into the D2-copy flag bit (6) for X-only, Z-only, and
        # BOTH: the BOTH detour resets D2' between phases rather than measuring an
        # intermediate flag.
        assert d2_copy_measurements == [6]
    else:
        # Face 0246 absent: the D2' detour never runs, so site 13 is untouched.
        assert 13 not in touched_sites
