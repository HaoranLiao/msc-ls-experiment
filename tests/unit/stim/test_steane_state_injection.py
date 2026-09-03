"""Stim tests for the abstract, 11-site, and 14-site Steane encoders.

These parallel the Clifford-applicable Qiskit state-injection tests using exact
stabilizer simulation. Non-Clifford inputs, azimuth sweeps, and IBM transpilation
have no Stim equivalent.
"""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np
import pytest
import stim

from msc.noise.fault_injection_stim import append_pauli_faults
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
    PATCH_14_SITE_SUPERDENSE_STABILIZER_MEASUREMENT_LAYERS,
    PATCH_14_SITE_X_STABILIZERS,
    PATCH_14_SITE_Z_STABILIZERS,
    X_STABILIZERS,
    Z_STABILIZERS,
)
from msc.steane_code.state_injection_stim import (
    StimInjectedStateName,
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
from msc.utils.circuit_common import logical_y_parity_is_inverted
from msc.utils.normalize import normalize_pauli_char

NUM_DATA_QUBITS = 7

CLIFFORD_STATES: tuple[StimInjectedStateName, ...] = (
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
)
HIRANO_STATES: tuple[StimInjectedStateName, ...] = (
    "S",
    "i",
    "Y+",
    "Sdg",
    "-i",
    "Y-",
)
LOGICAL_EXPECTATION_VALUES: dict[StimInjectedStateName, tuple[int, int, int]] = {
    "0": (1, 0, 0),
    "1": (-1, 0, 0),
    "+": (0, 1, 0),
    "-": (0, -1, 0),
    "i": (0, 0, 1),
    "-i": (0, 0, -1),
    "Y+": (0, 0, 1),
    "Y-": (0, 0, -1),
    "S": (0, 0, 1),
    "Sdg": (0, 0, -1),
}
LOGICAL_EXPECTATION_INDEX = {"Z": 0, "X": 1, "Y": 2}


def _pauli_string(
    support: tuple[int, ...],
    pauli: str,
    n_qubits: int,
) -> stim.PauliString:
    """Return a physical Pauli string on the indexed support."""
    characters = ["_"] * n_qubits
    for site in support:
        characters[site] = pauli
    return stim.PauliString("".join(characters))


def _logical_pauli_observable(
    support: tuple[int, ...],
    pauli: str,
    n_qubits: int,
) -> stim.PauliString:
    """Return the correctly signed logical Pauli on a Steane support."""
    if pauli == "Y":
        return (
            1j
            * _pauli_string(support, "X", n_qubits)
            * _pauli_string(
                support,
                "Z",
                n_qubits,
            )
        )
    return _pauli_string(support, pauli, n_qubits)


def _peek(circuit: stim.Circuit, observable: stim.PauliString) -> int:
    """Return the exact expectation value of an observable."""
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)
    return simulator.peek_observable_expectation(observable)


def _sample(circuit: stim.Circuit, shots: int = 100) -> np.ndarray:
    """Sample a Stim circuit and return its measurement matrix."""
    return circuit.compile_sampler().sample(shots)


# =============================================================================
# Abstract
# =============================================================================


# --------------------------
# Logical expectation values
# --------------------------


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_logical_expectation(
    state: StimInjectedStateName,
    basis: str,
) -> None:
    """Verify every Clifford state, Pauli basis, and logical representative."""
    circuit = stim.Circuit()
    append_abstract_encoder(
        circuit,
        data_qubits=range(NUM_DATA_QUBITS),
        injected_state=state,
        add_ticks=False,
    )
    representatives = LOGICAL_Z_REPRESENTATIVES if basis == "Z" else LOGICAL_X_REPRESENTATIVES
    expected = LOGICAL_EXPECTATION_VALUES[state][LOGICAL_EXPECTATION_INDEX[basis]]
    for support in representatives:
        assert (
            _peek(
                circuit,
                _logical_pauli_observable(support, basis, NUM_DATA_QUBITS),
            )
            == expected
        )


# =============================================================================
# 11-site and 14-site patches
# =============================================================================


class PatchTestSpec(NamedTuple):
    """Geometry and operator supports shared by the parameterized patch tests."""

    name: str
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
    pytest.param(PATCH_14_SITE_TEST_SPEC, "abstract-routed", id="14-site-abstract-routed"),
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
    for state in CLIFFORD_STATES
] + [
    pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", state, id=f"14-site-hirano-{state}")
    for state in HIRANO_STATES
]
PATCH_FAULT_ENCODER_STATE_CASES = [
    pytest.param(spec, encoder, state, id=f"{spec.name}-{encoder}-{state}")
    for spec, encoder in ARBITRARY_STATE_PATCH_ENCODERS
    for state in ("0", "S")
] + [
    pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", state, id=f"14-site-hirano-{state}")
    for state in ("S", "Sdg")
]
PATCH_14_SITE_ENCODERS = [
    pytest.param("abstract", id="abstract"),
    pytest.param("abstract-routed", id="abstract-routed"),
    pytest.param("hirano", id="hirano"),
]
PATCH_ROUTED_ENCODER_STATE_CASES = (
    [
        pytest.param(
            PATCH_11_SITE_TEST_SPEC,
            "rosenfeld",
            state,
            id=f"11-site-rosenfeld-{state}",
        )
        for state in CLIFFORD_STATES
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
        for state in CLIFFORD_STATES
    ]
)


def _apply_patch_encoder(
    circuit: stim.Circuit,
    patch_spec: PatchTestSpec,
    encoder: str,
    injected_state: StimInjectedStateName,
) -> None:
    """Append one encoder covered by the shared patch tests."""
    if patch_spec.name == "11-site" and encoder == "abstract":
        append_11_site_patch_abstract_encoder(
            circuit,
            injected_state=injected_state,
            add_ticks=False,
        )
    elif patch_spec.name == "11-site" and encoder == "rosenfeld":
        append_11_site_patch_rosenfeld_encoder(
            circuit,
            injected_state=injected_state,
            measure_flags=False,
            add_ticks=False,
        )
    elif patch_spec.name == "14-site" and encoder == "abstract":
        append_14_site_patch_abstract_encoder(
            circuit,
            injected_state=injected_state,
            add_ticks=False,
        )
    elif patch_spec.name == "14-site" and encoder == "abstract-routed":
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            injected_state=injected_state,
            measure_flags=False,
            add_ticks=False,
        )
    elif patch_spec.name == "14-site" and encoder == "hirano":
        append_14_site_patch_hirano_encoder(
            circuit,
            injected_state=injected_state,
            measure_flags=False,
            add_ticks=False,
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
    state: StimInjectedStateName,
    basis: str,
) -> None:
    """Verify every encoded patch state against every logical representative."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    representatives = (
        patch_spec.logical_z_representatives
        if basis == "Z"
        else patch_spec.logical_x_representatives
    )
    expected = LOGICAL_EXPECTATION_VALUES[state][LOGICAL_EXPECTATION_INDEX[basis]]
    for support in representatives:
        assert (
            _peek(
                circuit,
                _logical_pauli_observable(support, basis, patch_spec.num_qubits),
            )
            == expected
        )


# -----------------------------------
# Non-destructive logical measurement
# -----------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ENCODER_STATE_CASES)
def test_patch_logical_via_measurement_circuit(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: StimInjectedStateName,
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
        circuit = stim.Circuit()
        _apply_patch_encoder(circuit, patch_spec, encoder, state)
        if patch_spec.name == "11-site":
            append_11_site_patch_logical_measurement(
                circuit,
                basis,
                support=support,
                ancilla_qubit=ancilla_qubit,
                destructive=False,
            )
        elif patch_spec.name == "14-site":
            append_14_site_patch_logical_measurement(
                circuit,
                basis,
                support=support,
                ancilla_qubit=ancilla_qubit,
                destructive=False,
            )
        else:  # pragma: no cover - parametrization uses only supported patches.
            raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")

        assert circuit[-1].name == {"X": "MX", "Y": "MY", "Z": "M"}[basis]
        measured = _peek(
            circuit[:-1],
            _pauli_string((ancilla_qubit,), basis, patch_spec.num_qubits + 1),
        )
        if basis == "Y" and logical_y_parity_is_inverted(support):
            measured = -measured
        assert measured == expected


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
    injected_state: StimInjectedStateName,
    faults: list[tuple[str, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode, inject faults, and return the measured X/Z syndromes."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, injected_state)
    append_pauli_faults(circuit, faults)
    measurement_start = circuit.num_measurements
    if patch_spec.name == "11-site":
        append_11_site_patch_stabilizer_measurement(circuit, add_ticks=False)
    elif patch_spec.name == "14-site":
        append_14_site_patch_superdense_stabilizer_measurement(circuit, add_ticks=False)
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")

    syndrome = _sample(circuit, shots=1)[0, measurement_start:]
    if patch_spec.name == "11-site":
        return (
            tuple(int(bit) for bit in syndrome[:3]),
            tuple(int(bit) for bit in syndrome[3:]),
        )
    return (
        tuple(int(syndrome[index]) for index in (0, 2, 4)),
        tuple(int(syndrome[index]) for index in (1, 3, 5)),
    )


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ENCODER_STATE_CASES)
def test_patch_all_stabilizers_via_ancilla_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: StimInjectedStateName,
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
    state: StimInjectedStateName,
) -> None:
    """Verify all six stabilizer operators have expectation +1."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    for basis, stabilizers in (
        ("X", patch_spec.x_stabilizers),
        ("Z", patch_spec.z_stabilizers),
    ):
        for support in stabilizers:
            assert (
                _peek(
                    circuit,
                    _pauli_string(support, basis, patch_spec.num_qubits),
                )
                == 1
            )


@pytest.mark.parametrize(("patch_spec", "encoder"), PATCH_ENCODERS)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_patch_single_basis_stabilizer_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    basis: str,
) -> None:
    """Verify a complete one-basis stabilizer round on either patch."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    if patch_spec.name == "11-site":
        append_11_site_patch_stabilizer_measurement(
            circuit,
            basis=basis,
            add_ticks=False,
        )
    elif patch_spec.name == "14-site":
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            basis=basis,
            add_ticks=False,
        )
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")
    expected_measurement_count = 7 if patch_spec.name == "14-site" else 3
    assert circuit.num_measurements == expected_measurement_count
    assert circuit.num_detectors == expected_measurement_count
    assert not _sample(circuit).any()


@pytest.mark.parametrize(("patch_spec", "encoder"), PATCH_ENCODERS)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_patch_selected_stabilizer_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    basis: str,
) -> None:
    """Verify either patch measures only the requested stabilizer subset."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    measurement_start = len(circuit)
    if patch_spec.name == "11-site":
        append_11_site_patch_stabilizer_measurement(
            circuit,
            basis=basis,
            stabilizer_indices=(0, 2),
            add_ticks=False,
        )
        unselected_ancilla_sites = {6}
    elif patch_spec.name == "14-site":
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            basis=basis,
            stabilizer_indices=(0, 2),
            add_ticks=False,
        )
        unselected_ancilla_sites = {11, 12}
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")

    touched_qubits = {
        target.value
        for instruction in circuit[measurement_start:]
        for target in instruction.targets_copy()
        if target.is_qubit_target
    }
    assert not unselected_ancilla_sites.intersection(touched_qubits)
    expected_measurement_count = 5 if patch_spec.name == "14-site" else 2
    assert circuit.num_measurements == expected_measurement_count
    assert circuit.num_detectors == expected_measurement_count
    assert not _sample(circuit).any()


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


@pytest.mark.slow
class TestPatchSyndromeExtraction:
    """Verify both patch measurements against expected fault syndromes."""

    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_no_fault_trivial(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: StimInjectedStateName,
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
        state: StimInjectedStateName,
        site_index: int,
        pauli: str,
    ) -> None:
        """Single-qubit faults produce the expected syndrome."""
        faults = [(pauli, patch_spec.data_sites[site_index])]
        assert _build_and_run_patch_syndrome(
            patch_spec,
            encoder,
            state,
            faults,
        ) == _expected_patch_syndrome(patch_spec, faults)

    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    @pytest.mark.parametrize(("first", "second"), _DATA_SITE_PAIRS)
    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_two_qubit_same_pauli(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: StimInjectedStateName,
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
            patch_spec,
            encoder,
            state,
            faults,
        ) == _expected_patch_syndrome(patch_spec, faults)

    @pytest.mark.parametrize("paulis", [("X", "Z"), ("X", "Y"), ("Z", "Y")])
    @pytest.mark.parametrize(("first", "second"), _DATA_SITE_PAIRS)
    @pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_FAULT_ENCODER_STATE_CASES)
    def test_two_qubit_mixed_pauli(
        self,
        patch_spec: PatchTestSpec,
        encoder: str,
        state: StimInjectedStateName,
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
            patch_spec,
            encoder,
            state,
            faults,
        ) == _expected_patch_syndrome(patch_spec, faults)


# -------------------------
# Routed encoder properties
# -------------------------


@pytest.mark.parametrize(
    ("patch_spec", "encoder"),
    [
        pytest.param(PATCH_11_SITE_TEST_SPEC, "rosenfeld", id="11-site-rosenfeld"),
        pytest.param(PATCH_14_SITE_TEST_SPEC, "hirano", id="14-site-hirano"),
        pytest.param(
            PATCH_14_SITE_TEST_SPEC,
            "abstract-routed",
            id="14-site-abstract-routed",
        ),
    ],
)
def test_patch_routed_encoder_two_qubit_gates_are_local(
    patch_spec: PatchTestSpec,
    encoder: str,
) -> None:
    """Every two-qubit gate in each routed encoder follows its patch geometry."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, "S")
    if patch_spec.name == "11-site":
        coordinates = PATCH_11_SITE_COORDS
    elif encoder == "hirano":
        coordinates = PATCH_14_SITE_HIRANO_ENCODER_COORDS
    elif encoder == "abstract-routed":
        coordinates = PATCH_14_SITE_COORDS
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    for instruction in circuit:
        if instruction.name != "CX":
            continue
        sites = [target.value for target in instruction.targets_copy()]
        for site_a, site_b in zip(sites[::2], sites[1::2], strict=True):
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
            else:  # pragma: no cover - parametrization uses supported patches.
                raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ROUTED_ENCODER_STATE_CASES)
def test_patch_routed_encoder_helper_expectation_values(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: StimInjectedStateName,
) -> None:
    """Routed-encoder helpers return to their expected product state."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    if patch_spec.name == "11-site":
        helper_sites = PATCH_11_SITE_HELPER_SITES
        helper_basis = "Z"
    elif encoder == "hirano":
        helper_sites = PATCH_14_SITE_HELPER_SITES
        helper_basis = "X"
    elif encoder == "abstract-routed":
        helper_sites = tuple(
            site for site in range(patch_spec.num_qubits) if site not in patch_spec.data_sites
        )
        helper_basis = "Z"
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    for site in helper_sites:
        assert (
            _peek(
                circuit,
                _pauli_string((site,), helper_basis, patch_spec.num_qubits),
            )
            == 1
        )


@pytest.mark.parametrize(("patch_spec", "encoder", "state"), PATCH_ROUTED_ENCODER_STATE_CASES)
def test_patch_routed_encoder_helper_flags_are_zero(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: StimInjectedStateName,
) -> None:
    """Measured routed-encoder helpers return all-zero detector outcomes."""
    if patch_spec.name == "11-site":
        helper_sites = PATCH_11_SITE_HELPER_SITES
    elif encoder == "abstract-routed":
        helper_sites = PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES
    elif encoder == "hirano":
        helper_sites = PATCH_14_SITE_HELPER_SITES
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    circuit = stim.Circuit()
    if patch_spec.name == "11-site":
        append_11_site_patch_rosenfeld_encoder(
            circuit,
            injected_state=state,
            measure_flags=True,
            add_ticks=False,
        )
    elif encoder == "hirano":
        append_14_site_patch_hirano_encoder(
            circuit,
            injected_state=state,
            measure_flags=True,
            add_ticks=False,
        )
    elif encoder == "abstract-routed":
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            injected_state=state,
            measure_flags=True,
            add_ticks=False,
        )
    else:  # pragma: no cover - parametrization uses supported encoder pairs.
        raise ValueError(f"Unsupported {patch_spec.name} encoder: {encoder!r}.")
    shots = _sample(circuit, shots=1024)
    detection_events = circuit.compile_detector_sampler(seed=1).sample(32)
    assert shots.shape == (1024, len(helper_sites))
    assert circuit.num_detectors == len(helper_sites)
    assert not shots.any()
    assert not detection_events.any()


# ------------------------------
# Specific to the 14-site patch
# ------------------------------


def test_14_site_abstract_encoder_uses_only_fixed_data_sites() -> None:
    """The mapped abstract encoder keeps D1 at ``(1, 5)`` and ignores helpers."""
    assert PATCH_14_SITE_COORDS[1] == (1, 5)
    circuit = stim.Circuit()
    append_14_site_patch_abstract_encoder(
        circuit,
        injected_state="S",
        add_ticks=False,
    )
    used_two_qubit_sites = {
        target.value
        for instruction in circuit
        if instruction.name == "CX"
        for target in instruction.targets_copy()
    }
    assert used_two_qubit_sites <= set(PATCH_14_SITE_DATA_SITES)


@pytest.mark.parametrize("state", HIRANO_STATES)
def test_14_site_hirano_encoder_matches_abstract_y_state(
    state: StimInjectedStateName,
) -> None:
    """Verify Hirano and abstract encoders prepare the same 14-qubit state."""
    hirano = stim.Circuit()
    append_14_site_patch_hirano_encoder(
        hirano,
        injected_state=state,
        measure_flags=False,
        add_ticks=False,
    )
    hirano.append("H", PATCH_14_SITE_HELPER_SITES)
    abstract = stim.Circuit()
    append_14_site_patch_abstract_encoder(
        abstract,
        injected_state=state,
        add_ticks=False,
    )

    hirano_simulator = stim.TableauSimulator()
    hirano_simulator.set_num_qubits(NUM_QUBITS_PATCH_14_SITES)
    hirano_simulator.do_circuit(hirano)
    abstract_simulator = stim.TableauSimulator()
    abstract_simulator.set_num_qubits(NUM_QUBITS_PATCH_14_SITES)
    abstract_simulator.do_circuit(abstract)
    assert hirano_simulator.canonical_stabilizers() == abstract_simulator.canonical_stabilizers()


@pytest.mark.parametrize("state", ["0", "1", "+", "-"])
def test_14_site_hirano_encoder_rejects_non_y_states(
    state: StimInjectedStateName,
) -> None:
    """Hirano's single phase injection cannot encode a non-Y input."""
    with pytest.raises(ValueError, match="logical-Y eigenstate"):
        append_14_site_patch_hirano_encoder(
            stim.Circuit(),
            injected_state=state,
        )


# ----------------
# Specific to Stim
# ----------------


@pytest.mark.parametrize(
    ("patch_spec", "encoder", "state", "basis", "expected_parity"),
    [
        pytest.param(spec, encoder, state, basis, expected, id=f"{spec.name}-{basis}{sign}")
        for spec, encoder in ARBITRARY_STATE_PATCH_ENCODERS
        for state, basis, expected, sign in (
            ("0", "Z", 0, "+"),
            ("1", "Z", 1, "-"),
            ("+", "X", 0, "+"),
            ("-", "X", 1, "-"),
            ("S", "Y", 0, "+"),
            ("Sdg", "Y", 1, "-"),
        )
    ]
    + [
        pytest.param(
            PATCH_14_SITE_TEST_SPEC,
            "hirano",
            "S",
            "Y",
            0,
            id="14-site-hirano-Y+",
        ),
        pytest.param(
            PATCH_14_SITE_TEST_SPEC,
            "hirano",
            "Sdg",
            "Y",
            1,
            id="14-site-hirano-Y-",
        ),
    ],
)
def test_patch_destructive_logical_measurement(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: StimInjectedStateName,
    basis: str,
    expected_parity: int,
) -> None:
    """Verify destructive logical readout, including logical-Y correction."""
    representatives = (
        patch_spec.logical_z_representatives
        if basis == "Z"
        else patch_spec.logical_x_representatives
    )
    for support in representatives:
        circuit = stim.Circuit()
        _apply_patch_encoder(circuit, patch_spec, encoder, state)
        if patch_spec.name == "11-site":
            append_11_site_patch_logical_measurement(
                circuit,
                basis,
                support=support,
                destructive=True,
            )
        elif patch_spec.name == "14-site":
            append_14_site_patch_logical_measurement(
                circuit,
                basis,
                support=support,
                destructive=True,
            )
        else:  # pragma: no cover - parametrization uses only supported patches.
            raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")
        parities = _sample(circuit)[:, -len(support) :].sum(axis=1) % 2
        assert (parities == expected_parity).all()


@pytest.mark.parametrize(
    ("patch_spec", "encoder", "state", "basis"),
    [
        pytest.param(spec, encoder, "+", "X", id=f"{spec.name}-{encoder}")
        for spec, encoder in ARBITRARY_STATE_PATCH_ENCODERS
    ]
    + [
        pytest.param(
            PATCH_14_SITE_TEST_SPEC,
            "hirano",
            "S",
            "Y",
            id="14-site-hirano",
        )
    ],
)
def test_patch_logical_measurement_can_include_stim_observable(
    patch_spec: PatchTestSpec,
    encoder: str,
    state: StimInjectedStateName,
    basis: str,
) -> None:
    """Verify a terminal logical measurement can define ``L0`` directly."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, patch_spec, encoder, state)
    if patch_spec.name == "11-site":
        append_11_site_patch_logical_measurement(
            circuit,
            basis,
            observable_index=0,
        )
    elif patch_spec.name == "14-site":
        append_14_site_patch_logical_measurement(
            circuit,
            basis,
            observable_index=0,
        )
    else:  # pragma: no cover - parametrization uses only supported patches.
        raise ValueError(f"Unsupported patch: {patch_spec.name!r}.")
    _, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
        32,
        separate_observables=True,
    )
    assert circuit.num_observables == 1
    assert not observable_flips.any()


def _build_14_site_superdense_circuit(
    encoder: str,
    state: StimInjectedStateName = "S",
) -> stim.Circuit:
    """Append a 14-site encoder followed by a complete superdense round."""
    circuit = stim.Circuit()
    _apply_patch_encoder(circuit, PATCH_14_SITE_TEST_SPEC, encoder, state)
    append_14_site_patch_superdense_stabilizer_measurement(
        circuit,
        add_ticks=False,
    )
    return circuit


@pytest.mark.parametrize("encoder", PATCH_14_SITE_ENCODERS)
def test_14_site_superdense_detectors_are_deterministic(encoder: str) -> None:
    """Verify superdense ancilla detectors never fire on a clean codeword."""
    circuit = _build_14_site_superdense_circuit(encoder)
    detection_events = circuit.compile_detector_sampler(seed=2).sample(64)
    assert circuit.num_detectors == 7
    assert circuit.num_measurements == 7
    assert not detection_events.any()


@pytest.mark.parametrize("state", HIRANO_STATES)
@pytest.mark.parametrize("encoder", PATCH_14_SITE_ENCODERS)
def test_14_site_superdense_round_preserves_code(
    encoder: str,
    state: StimInjectedStateName,
) -> None:
    """Verify all stabilizers and the logical state survive a superdense round."""
    circuit = _build_14_site_superdense_circuit(encoder, state)
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)
    for basis, stabilizers in (
        ("X", PATCH_14_SITE_X_STABILIZERS),
        ("Z", PATCH_14_SITE_Z_STABILIZERS),
    ):
        for support in stabilizers:
            assert (
                simulator.peek_observable_expectation(
                    _pauli_string(support, basis, NUM_QUBITS_PATCH_14_SITES)
                )
                == 1
            )

    expected_y = LOGICAL_EXPECTATION_VALUES[state][LOGICAL_EXPECTATION_INDEX["Y"]]
    for support in PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES:
        assert (
            simulator.peek_observable_expectation(
                _logical_pauli_observable(support, "Y", NUM_QUBITS_PATCH_14_SITES)
            )
            == expected_y
        )


def test_14_site_superdense_layers_are_local_collision_free_and_keep_cx_depth() -> None:
    """D1/H8 routing preserves all twelve collision-free superdense CX layers."""
    cx_layers = 0
    all_cnots: list[tuple[int, int]] = []
    for tick_layer in PATCH_14_SITE_SUPERDENSE_STABILIZER_MEASUREMENT_LAYERS:
        layer_cnots = [
            pair for gate_name, operands in tick_layer if gate_name == "CX" for pair in operands
        ]
        if not layer_cnots:
            continue
        cx_layers += 1
        used_sites = [site for pair in layer_cnots for site in pair]
        assert len(used_sites) == len(set(used_sites))
        for control, target in layer_cnots:
            cx, cy = PATCH_14_SITE_COORDS[control]
            tx, ty = PATCH_14_SITE_COORDS[target]
            assert (abs(cx - tx), abs(cy - ty)) == (1, 1)
        all_cnots.extend(layer_cnots)

    assert cx_layers == 12
    assert (1, 8) in all_cnots
    assert (8, 1) in all_cnots
    assert (1, 7) not in all_cnots
    assert (7, 1) not in all_cnots


@pytest.mark.parametrize("encoder", PATCH_14_SITE_ENCODERS)
def test_14_site_superdense_round_leaves_data_qubit_2_restored(
    encoder: str,
) -> None:
    """Verify the D2/D2' detour restores D2 and disentangles D2'."""
    circuit = _build_14_site_superdense_circuit(encoder)
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)
    assert (
        simulator.peek_observable_expectation(
            _pauli_string(
                PATCH_14_SITE_X_STABILIZERS[1],
                "X",
                NUM_QUBITS_PATCH_14_SITES,
            )
        )
        == 1
    )
    assert (
        simulator.peek_observable_expectation(_pauli_string((13,), "X", NUM_QUBITS_PATCH_14_SITES))
        == 1
    )
