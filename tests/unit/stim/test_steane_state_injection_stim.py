"""Parallel Stim tests for the 11-site patch-Crumble Steane encoder.

These mirror the patch-Crumble portions of the Qiskit state-injection suite
using exact Clifford simulation. Qiskit's non-Clifford azimuth-angle cases and
IBM-backend transpilation test have no Stim equivalent and are intentionally
excluded.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from magic_state_cultivation.noise.fault_injection_stim import append_pauli_faults
from magic_state_cultivation.steane_code.state_injection_qiskit import (
    NUM_PATCH_QUBITS,
    PATCH_DATA_SITES,
    PATCH_HELPER_SITES,
    PATCH_LOGICAL_X_REPRESENTATIVES,
    PATCH_LOGICAL_Z_REPRESENTATIVES,
    PATCH_STABILIZER_ANCILLA_SITES,
    PATCH_X_STABILIZERS,
    PATCH_Z_STABILIZERS,
)
from magic_state_cultivation.steane_code.state_injection_stim import (
    StimInjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from magic_state_cultivation.utils.circuit_stim import logical_y_parity_is_inverted

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

LOGICAL_EXPECTATIONS: dict[StimInjectedStateName, tuple[int, int, int]] = {
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

PATCH_LOGICAL_BASIS_CASES = (
    pytest.param("Z", PATCH_LOGICAL_Z_REPRESENTATIVES, id="Z"),
    pytest.param("X", PATCH_LOGICAL_X_REPRESENTATIVES, id="X"),
    pytest.param("Y", PATCH_LOGICAL_X_REPRESENTATIVES, id="Y"),
)

N_PATCH_STABILIZERS = len(PATCH_X_STABILIZERS)
N_PATCH_SYNDROME_BITS = 2 * N_PATCH_STABILIZERS


def _encoded_circuit(
    state: StimInjectedStateName,
    *,
    measure_helpers: bool = False,
) -> stim.Circuit:
    """Return a standalone patch-Crumble encoder circuit."""
    circuit = stim.Circuit()
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        injected_state=state,
        measure_helpers=measure_helpers,
        add_ticks=False,
    )
    return circuit


def _pauli_string(
    pauli: str,
    support: tuple[int, ...],
    *,
    n_qubits: int = NUM_PATCH_QUBITS,
) -> stim.PauliString:
    """Return a patch-sized physical Pauli string."""
    chars = ["_"] * n_qubits
    for site in support:
        chars[site] = pauli
    return stim.PauliString("".join(chars))


def _logical_observable(pauli: str, support: tuple[int, ...]) -> stim.PauliString:
    """Return the correctly signed logical observable on ``support``."""
    if pauli == "Y":
        return 1j * _pauli_string("X", support) * _pauli_string("Z", support)
    return _pauli_string(pauli, support)


def _peek(circuit: stim.Circuit, observable: stim.PauliString) -> int:
    """Return the exact expectation value of ``observable``."""
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)
    return simulator.peek_observable_expectation(observable)


def _sample(circuit: stim.Circuit, shots: int = 100) -> np.ndarray:
    """Sample a Stim circuit and return its measurement matrix."""
    return circuit.compile_sampler().sample(shots)


# ---------------------------------------------------------------------------
# Logical observables and measurement circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("basis", "representatives"), PATCH_LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_logical_expectation(
    state: StimInjectedStateName,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify every logical representative carries the injected state."""
    circuit = _encoded_circuit(state)
    expected = LOGICAL_EXPECTATIONS[state][LOGICAL_EXPECTATION_INDEX[basis]]

    for support in representatives:
        measured = _peek(circuit, _logical_observable(basis, support))
        assert measured == expected, (
            f"State {state!r}, support {support}: "
            f"<{basis}_L> expected {expected}, got {measured}"
        )


@pytest.mark.parametrize(("basis", "representatives"), PATCH_LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_logical_via_measurement_circuit(
    state: StimInjectedStateName,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify non-destructive ancilla readout for every representative."""
    expected = LOGICAL_EXPECTATIONS[state][LOGICAL_EXPECTATION_INDEX[basis]]
    ancilla_qubit = NUM_PATCH_QUBITS

    for support in representatives:
        circuit = _encoded_circuit(state)
        append_patch_logical_measurement(
            circuit,
            basis,
            support=support,
            ancilla_qubit=ancilla_qubit,
            destructive=False,
        )

        assert circuit[-1].name == "M"
        measured = _peek(
            circuit[:-1],
            _pauli_string(
                "Z",
                (ancilla_qubit,),
                n_qubits=NUM_PATCH_QUBITS + 1,
            ),
        )
        if basis == "Y" and logical_y_parity_is_inverted(support):
            measured = -measured
        assert measured == expected, (
            f"State {state!r}, support {support}: " f"<Z_anc> expected {expected}, got {measured}"
        )


@pytest.mark.parametrize(
    ("state", "basis", "expected_parity"),
    [
        pytest.param("0", "Z", 0, id="Z+"),
        pytest.param("1", "Z", 1, id="Z-"),
        pytest.param("+", "X", 0, id="X+"),
        pytest.param("-", "X", 1, id="X-"),
        pytest.param("S", "Y", 0, id="Y+"),
        pytest.param("Sdg", "Y", 1, id="Y-"),
    ],
)
def test_destructive_patch_logical_measurement(
    state: StimInjectedStateName,
    basis: str,
    expected_parity: int,
) -> None:
    """Verify destructive logical readout, including logical-Y sign correction."""
    representatives = (
        PATCH_LOGICAL_Z_REPRESENTATIVES if basis == "Z" else PATCH_LOGICAL_X_REPRESENTATIVES
    )
    for support in representatives:
        circuit = _encoded_circuit(state)
        append_patch_logical_measurement(
            circuit,
            basis,
            support=support,
            destructive=True,
        )

        shots = _sample(circuit)
        parities = shots[:, -len(support) :].sum(axis=1) % 2
        assert (parities == expected_parity).all()


# ---------------------------------------------------------------------------
# Stabilizer observables and measurement circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_stabilizer_expectation_values(state: StimInjectedStateName) -> None:
    """Verify all six stabilizer operators have expectation +1."""
    circuit = _encoded_circuit(state)

    for basis, stabilizers in (
        ("X", PATCH_X_STABILIZERS),
        ("Z", PATCH_Z_STABILIZERS),
    ):
        for index, support in enumerate(stabilizers):
            measured = _peek(circuit, _pauli_string(basis, support))
            assert measured == 1, (
                f"State {state!r}: {basis}-stabilizer {index} "
                f"on {support} expected +1, got {measured}"
            )


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_all_stabilizers_via_ancilla_measurement(
    state: StimInjectedStateName,
) -> None:
    """Verify the full six-bit syndrome is zero."""
    circuit = _encoded_circuit(state)
    append_patch_stabilizer_measurement(circuit, add_ticks=False)

    assert circuit.num_measurements == N_PATCH_SYNDROME_BITS
    assert not _sample(circuit).any()


@pytest.mark.parametrize("basis", ["x", "z"])
def test_patch_single_basis_stabilizer_measurement(basis: str) -> None:
    """Verify a complete one-basis stabilizer round."""
    circuit = _encoded_circuit("S")
    append_patch_stabilizer_measurement(
        circuit,
        basis=basis,
        add_ticks=False,
    )

    assert circuit.num_measurements == N_PATCH_STABILIZERS
    assert not _sample(circuit).any()


@pytest.mark.parametrize("basis", ["x", "z"])
def test_patch_selected_stabilizer_measurement(basis: str) -> None:
    """Verify ordered stabilizer subsets use only their assigned ancillas."""
    selected_indices = (0, 2)
    circuit = _encoded_circuit("S")
    measurement_start = len(circuit)
    append_patch_stabilizer_measurement(
        circuit,
        basis=basis,
        stabilizer_indices=selected_indices,
        add_ticks=False,
    )

    unselected_ancilla = PATCH_STABILIZER_ANCILLA_SITES[1]
    appended_instructions = circuit[measurement_start:]
    touched_qubits = {
        target.value
        for instruction in appended_instructions
        for target in instruction.targets_copy()
        if target.is_qubit_target
    }
    assert unselected_ancilla not in touched_qubits
    assert circuit.num_measurements == len(selected_indices)
    assert not _sample(circuit).any()


# ---------------------------------------------------------------------------
# Stabilizer syndromes with injected faults
# ---------------------------------------------------------------------------


def _expected_patch_syndrome(
    faults: list[tuple[str, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the expected X- and Z-stabilizer syndromes."""
    x_syndrome = [0] * N_PATCH_STABILIZERS
    z_syndrome = [0] * N_PATCH_STABILIZERS

    for pauli, site in faults:
        for index, support in enumerate(PATCH_X_STABILIZERS):
            if site in support and pauli in {"Z", "Y"}:
                x_syndrome[index] ^= 1
        for index, support in enumerate(PATCH_Z_STABILIZERS):
            if site in support and pauli in {"X", "Y"}:
                z_syndrome[index] ^= 1

    return tuple(x_syndrome), tuple(z_syndrome)


def _build_and_run_patch_syndrome(
    faults: list[tuple[str, int]],
    *,
    injected_state: StimInjectedStateName,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Inject faults, measure all stabilizers, and return both syndromes."""
    circuit = _encoded_circuit(injected_state)
    append_pauli_faults(circuit, faults)
    append_patch_stabilizer_measurement(circuit, add_ticks=False)

    syndrome = _sample(circuit, shots=1)[0, -N_PATCH_SYNDROME_BITS:]
    return (
        tuple(int(bit) for bit in syndrome[:N_PATCH_STABILIZERS]),
        tuple(int(bit) for bit in syndrome[N_PATCH_STABILIZERS:]),
    )


_SINGLE_FAULTS = [[(pauli, site)] for pauli in ("X", "Y", "Z") for site in PATCH_DATA_SITES]
_SINGLE_IDS = [f"{pauli}_{site}" for pauli in ("X", "Y", "Z") for site in PATCH_DATA_SITES]

_TWO_SAME_FAULTS: list[list[tuple[str, int]]] = []
_TWO_SAME_IDS: list[str] = []
for _pauli in ("X", "Y", "Z"):
    for _index, _site_1 in enumerate(PATCH_DATA_SITES):
        for _site_2 in PATCH_DATA_SITES[_index + 1 :]:
            _TWO_SAME_FAULTS.append([(_pauli, _site_1), (_pauli, _site_2)])
            _TWO_SAME_IDS.append(f"{_pauli}{_pauli}_{_site_1}_{_site_2}")

_TWO_MIXED_FAULTS: list[list[tuple[str, int]]] = []
_TWO_MIXED_IDS: list[str] = []
for _pauli_1, _pauli_2 in (("X", "Z"), ("X", "Y"), ("Z", "Y")):
    for _index, _site_1 in enumerate(PATCH_DATA_SITES):
        for _site_2 in PATCH_DATA_SITES[_index + 1 :]:
            _TWO_MIXED_FAULTS.append([(_pauli_1, _site_1), (_pauli_2, _site_2)])
            _TWO_MIXED_IDS.append(f"{_pauli_1}{_pauli_2}_{_site_1}_{_site_2}")


class TestPatchSyndromeExtraction:
    """Verify measured syndromes after injected Pauli faults."""

    @pytest.mark.parametrize("state", CLIFFORD_STATES)
    def test_no_fault_trivial(self, state: StimInjectedStateName) -> None:
        measured = _build_and_run_patch_syndrome([], injected_state=state)
        assert measured == ((0, 0, 0), (0, 0, 0))

    @pytest.mark.parametrize("state", ["0", "S"])
    @pytest.mark.parametrize("faults", _SINGLE_FAULTS, ids=_SINGLE_IDS)
    def test_single_fault(
        self,
        state: StimInjectedStateName,
        faults: list[tuple[str, int]],
    ) -> None:
        assert _build_and_run_patch_syndrome(
            faults,
            injected_state=state,
        ) == _expected_patch_syndrome(faults)

    @pytest.mark.parametrize("state", ["0", "S"])
    @pytest.mark.parametrize("faults", _TWO_SAME_FAULTS, ids=_TWO_SAME_IDS)
    def test_two_qubit_same_pauli(
        self,
        state: StimInjectedStateName,
        faults: list[tuple[str, int]],
    ) -> None:
        assert _build_and_run_patch_syndrome(
            faults,
            injected_state=state,
        ) == _expected_patch_syndrome(faults)

    @pytest.mark.parametrize("state", ["0", "S"])
    @pytest.mark.parametrize("faults", _TWO_MIXED_FAULTS, ids=_TWO_MIXED_IDS)
    def test_two_qubit_mixed_pauli(
        self,
        state: StimInjectedStateName,
        faults: list[tuple[str, int]],
    ) -> None:
        assert _build_and_run_patch_syndrome(
            faults,
            injected_state=state,
        ) == _expected_patch_syndrome(faults)


# ---------------------------------------------------------------------------
# Crumble helper sites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_crumble_helper_sites_expectation_value(
    state: StimInjectedStateName,
) -> None:
    """Verify every helper finishes in |0>."""
    circuit = _encoded_circuit(state)
    for site in PATCH_HELPER_SITES:
        assert _peek(circuit, _pauli_string("Z", (site,))) == 1


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_crumble_helper_sites_bit_value(
    state: StimInjectedStateName,
) -> None:
    """Verify measured helper sites always return the all-zero outcome."""
    circuit = _encoded_circuit(state, measure_helpers=True)
    shots = _sample(circuit, shots=1024)

    assert shots.shape == (1024, len(PATCH_HELPER_SITES))
    assert not shots.any()
