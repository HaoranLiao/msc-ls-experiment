"""Parallel Stim tests for the 11-site Rosenfeld et al. Steane encoder.

These mirror the Rosenfeld et al. portions of the Qiskit state-injection suite
using exact Clifford simulation. Qiskit's non-Clifford azimuth-angle cases and
IBM-backend transpilation test have no Stim equivalent and are intentionally
excluded.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from msc.noise.fault_injection_stim import append_pauli_faults
from msc.steane_code.state_injection_common import (
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_11_SITE_DATA_SITES,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES,
    PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES,
    PATCH_11_SITE_STABILIZER_ANCILLA_SITES,
    PATCH_11_SITE_X_STABILIZERS,
    PATCH_11_SITE_Z_STABILIZERS,
    PATCH_14_SITE_COORDS,
    PATCH_14_SITE_D2_PRIME_SITE,
    PATCH_14_SITE_DATA_SITES,
    PATCH_14_SITE_HELPER_SITES,
    PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES,
    PATCH_14_SITE_X_STABILIZERS,
    PATCH_14_SITE_Z_STABILIZERS,
)
from msc.steane_code.state_injection_stim import (
    StimInjectedStateName,
    append_11_site_patch_logical_measurement,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_hirano_encoder,
    append_14_site_patch_superdense_stabilizer_measurement,
)
from msc.utils.circuit_common import logical_y_parity_is_inverted

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

PATCH_11_SITE_LOGICAL_BASIS_CASES = (
    pytest.param("Z", PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES, id="Z"),
    pytest.param("X", PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES, id="X"),
    pytest.param("Y", PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES, id="Y"),
)

N_PATCH_STABILIZERS = len(PATCH_11_SITE_X_STABILIZERS)
N_PATCH_SYNDROME_BITS = 2 * N_PATCH_STABILIZERS


def _encoded_circuit(
    state: StimInjectedStateName,
    *,
    measure_flags: bool = False,
) -> stim.Circuit:
    """Return a standalone Rosenfeld et al. encoder circuit."""
    circuit = stim.Circuit()
    append_11_site_patch_rosenfeld_encoder(
        circuit,
        injected_state=state,
        measure_flags=measure_flags,
        add_ticks=False,
    )
    return circuit


def _pauli_string(
    pauli: str,
    support: tuple[int, ...],
    *,
    n_qubits: int = NUM_QUBITS_PATCH_11_SITES,
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


# ===========================================================================
# 11-site patch (Rosenfeld injection encoder)
# ===========================================================================


# ---------------------------------------------------------------------------
# Flag measurements
# ---------------------------------------------------------------------------


def test_encoder_flags_are_stim_detectors() -> None:
    """Verify measured Rosenfeld-encoder helper flags carry detector annotations."""
    circuit = _encoded_circuit("S", measure_flags=True)
    detection_events = circuit.compile_detector_sampler(seed=1).sample(32)

    assert circuit.num_detectors == len(PATCH_11_SITE_HELPER_SITES)
    assert not detection_events.any()


# ---------------------------------------------------------------------------
# Logical observables and measurement circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("basis", "representatives"), PATCH_11_SITE_LOGICAL_BASIS_CASES)
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


@pytest.mark.parametrize(("basis", "representatives"), PATCH_11_SITE_LOGICAL_BASIS_CASES)
@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_logical_via_measurement_circuit(
    state: StimInjectedStateName,
    basis: str,
    representatives: tuple[tuple[int, ...], ...],
) -> None:
    """Verify non-destructive ancilla readout for every representative."""
    expected = LOGICAL_EXPECTATIONS[state][LOGICAL_EXPECTATION_INDEX[basis]]
    ancilla_qubit = NUM_QUBITS_PATCH_11_SITES

    for support in representatives:
        circuit = _encoded_circuit(state)
        append_11_site_patch_logical_measurement(
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
                n_qubits=NUM_QUBITS_PATCH_11_SITES + 1,
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
        PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES
        if basis == "Z"
        else PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES
    )
    for support in representatives:
        circuit = _encoded_circuit(state)
        append_11_site_patch_logical_measurement(
            circuit,
            basis,
            support=support,
            destructive=True,
        )

        shots = _sample(circuit)
        parities = shots[:, -len(support) :].sum(axis=1) % 2
        assert (parities == expected_parity).all()


def test_patch_logical_measurement_can_include_stim_observable() -> None:
    """Verify a terminal logical measurement can define ``L0`` directly."""
    circuit = _encoded_circuit("+")
    append_11_site_patch_logical_measurement(
        circuit,
        "X",
        observable_index=0,
    )
    _, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
        32,
        separate_observables=True,
    )

    assert circuit.num_observables == 1
    assert not observable_flips.any()


# ---------------------------------------------------------------------------
# Stabilizer observables and measurement circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_stabilizer_expectation_values(state: StimInjectedStateName) -> None:
    """Verify all six stabilizer operators have expectation +1."""
    circuit = _encoded_circuit(state)

    for basis, stabilizers in (
        ("X", PATCH_11_SITE_X_STABILIZERS),
        ("Z", PATCH_11_SITE_Z_STABILIZERS),
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
    append_11_site_patch_stabilizer_measurement(circuit, add_ticks=False)

    assert circuit.num_measurements == N_PATCH_SYNDROME_BITS
    assert circuit.num_detectors == N_PATCH_SYNDROME_BITS
    assert not _sample(circuit).any()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_patch_single_basis_stabilizer_measurement(basis: str) -> None:
    """Verify a complete one-basis stabilizer round."""
    circuit = _encoded_circuit("S")
    append_11_site_patch_stabilizer_measurement(
        circuit,
        basis=basis,
        add_ticks=False,
    )

    assert circuit.num_measurements == N_PATCH_STABILIZERS
    assert not _sample(circuit).any()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_patch_selected_stabilizer_measurement(basis: str) -> None:
    """Verify ordered stabilizer subsets use only their assigned ancillas."""
    selected_indices = (0, 2)
    circuit = _encoded_circuit("S")
    measurement_start = len(circuit)
    append_11_site_patch_stabilizer_measurement(
        circuit,
        basis=basis,
        stabilizer_indices=selected_indices,
        add_ticks=False,
    )

    unselected_ancilla = PATCH_11_SITE_STABILIZER_ANCILLA_SITES[1]
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
        for index, support in enumerate(PATCH_11_SITE_X_STABILIZERS):
            if site in support and pauli in {"Z", "Y"}:
                x_syndrome[index] ^= 1
        for index, support in enumerate(PATCH_11_SITE_Z_STABILIZERS):
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
    append_11_site_patch_stabilizer_measurement(circuit, add_ticks=False)

    syndrome = _sample(circuit, shots=1)[0, -N_PATCH_SYNDROME_BITS:]
    return (
        tuple(int(bit) for bit in syndrome[:N_PATCH_STABILIZERS]),
        tuple(int(bit) for bit in syndrome[N_PATCH_STABILIZERS:]),
    )


_SINGLE_FAULTS = [[(pauli, site)] for pauli in ("X", "Y", "Z") for site in PATCH_11_SITE_DATA_SITES]
_SINGLE_IDS = [f"{pauli}_{site}" for pauli in ("X", "Y", "Z") for site in PATCH_11_SITE_DATA_SITES]

_TWO_SAME_FAULTS: list[list[tuple[str, int]]] = []
_TWO_SAME_IDS: list[str] = []
for _pauli in ("X", "Y", "Z"):
    for _index, _site_1 in enumerate(PATCH_11_SITE_DATA_SITES):
        for _site_2 in PATCH_11_SITE_DATA_SITES[_index + 1 :]:
            _TWO_SAME_FAULTS.append([(_pauli, _site_1), (_pauli, _site_2)])
            _TWO_SAME_IDS.append(f"{_pauli}{_pauli}_{_site_1}_{_site_2}")

_TWO_MIXED_FAULTS: list[list[tuple[str, int]]] = []
_TWO_MIXED_IDS: list[str] = []
for _pauli_1, _pauli_2 in (("X", "Z"), ("X", "Y"), ("Z", "Y")):
    for _index, _site_1 in enumerate(PATCH_11_SITE_DATA_SITES):
        for _site_2 in PATCH_11_SITE_DATA_SITES[_index + 1 :]:
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
# Rosenfeld-encoder helper sites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_rosenfeld_helper_sites_expectation_value(
    state: StimInjectedStateName,
) -> None:
    """Verify every helper finishes in |0>."""
    circuit = _encoded_circuit(state)
    for site in PATCH_11_SITE_HELPER_SITES:
        assert _peek(circuit, _pauli_string("Z", (site,))) == 1


@pytest.mark.parametrize("state", CLIFFORD_STATES)
def test_patch_rosenfeld_flag_qubits_bit_value(
    state: StimInjectedStateName,
) -> None:
    """Verify measured flag qubits always return the all-zero outcome."""
    circuit = _encoded_circuit(state, measure_flags=True)
    shots = _sample(circuit, shots=1024)

    assert shots.shape == (1024, len(PATCH_11_SITE_HELPER_SITES))
    assert not shots.any()


# ===========================================================================
# 14-site patch (Hirano injection encoder)
# ===========================================================================


# States injectable by the 14-site Hirano encoder: the logical-Y eigenstate
# family only (S/i/Y+ -> logical Y=+1, Sdg/-i/Y- -> logical Y=-1); every other
# Clifford state raises. This is the 14-site analogue of the 11-site
# CLIFFORD_STATES sweep and reuses the shared LOGICAL_EXPECTATIONS values, so the
# sweep also checks that the equivalent injection names agree.
PATCH_14_SITE_INJECTION_STATES: tuple[StimInjectedStateName, ...] = (
    "i",
    "-i",
    "Y+",
    "Y-",
    "S",
    "Sdg",
)

# Superdense ancilla measurement-bit layout: [X0145, Z0145, X0235, Z0235, X0246,
# Z0246] — one X- and one Z-ancilla bit per weight-4 Steane face.
SUPERDENSE_NUM_DETECTORS = 6
SUPERDENSE_X_STAB_BIT = {0: 0, 1: 2, 2: 4}
SUPERDENSE_Z_STAB_BIT = {0: 1, 1: 3, 2: 5}


def _encoded_14_site_circuit(
    state: StimInjectedStateName,
    *,
    measure_flags: bool = False,
) -> stim.Circuit:
    """Return a standalone Hirano 14-site injection encoder circuit."""
    circuit = stim.Circuit()
    append_14_site_patch_hirano_encoder(
        circuit,
        injected_state=state,
        measure_flags=measure_flags,
        add_ticks=False,
    )
    return circuit


def _logical_observable_14_site(pauli: str, support: tuple[int, ...]) -> stim.PauliString:
    """Return the correctly signed 14-site logical observable on ``support``."""
    if pauli == "Y":
        return (
            1j
            * _pauli_string("X", support, n_qubits=NUM_QUBITS_PATCH_14_SITES)
            * _pauli_string("Z", support, n_qubits=NUM_QUBITS_PATCH_14_SITES)
        )
    return _pauli_string(pauli, support, n_qubits=NUM_QUBITS_PATCH_14_SITES)


def _superdense_circuit(
    *,
    injected_state: StimInjectedStateName = "S",
    faults: list[tuple[str, int]] | None = None,
) -> tuple[stim.Circuit, int]:
    """Return an injection + superdense-round circuit and its measurement start.

    The encoder is built without flag measurements, so the superdense round's six
    detectors and measurements are isolated at the end of the record.
    """
    circuit = _encoded_14_site_circuit(injected_state)
    if faults:
        append_pauli_faults(circuit, faults)
    measurement_start = circuit.num_measurements
    append_14_site_patch_superdense_stabilizer_measurement(circuit, add_ticks=False)
    return circuit, measurement_start


# ---------------------------------------------------------------------------
# Flag measurements
# ---------------------------------------------------------------------------


def test_14_site_encoder_flags_are_stim_detectors() -> None:
    """Verify measured Hirano-encoder helper flags carry detector annotations."""
    circuit = stim.Circuit()
    append_14_site_patch_hirano_encoder(circuit, measure_flags=True, add_ticks=False)
    detection_events = circuit.compile_detector_sampler(seed=1).sample(32)

    assert circuit.num_measurements == len(PATCH_14_SITE_HELPER_SITES)
    assert circuit.num_detectors == len(PATCH_14_SITE_HELPER_SITES)
    assert not detection_events.any()


# ---------------------------------------------------------------------------
# Logical observables and measurement circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["Z", "X", "Y"])
@pytest.mark.parametrize("state", PATCH_14_SITE_INJECTION_STATES)
def test_14_site_encoder_logical_expectation(
    state: StimInjectedStateName,
    basis: str,
) -> None:
    """Verify every logical representative carries the injected logical-Y state."""
    circuit = _encoded_14_site_circuit(state)
    expected = LOGICAL_EXPECTATIONS[state][LOGICAL_EXPECTATION_INDEX[basis]]

    for support in PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES:
        measured = _peek(circuit, _logical_observable_14_site(basis, support))
        assert measured == expected, (
            f"State {state!r}, support {support}: "
            f"<{basis}_L> expected {expected}, got {measured}"
        )


# ---------------------------------------------------------------------------
# Stabilizer observables and measurement circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", PATCH_14_SITE_INJECTION_STATES)
def test_14_site_encoder_stabilizers_are_plus_one(
    state: StimInjectedStateName,
) -> None:
    """Verify all six Steane stabilizers project to +1 after injection."""
    circuit = _encoded_14_site_circuit(state)

    for basis, stabilizers in (
        ("X", PATCH_14_SITE_X_STABILIZERS),
        ("Z", PATCH_14_SITE_Z_STABILIZERS),
    ):
        for index, support in enumerate(stabilizers):
            measured = _peek(
                circuit,
                _pauli_string(basis, support, n_qubits=NUM_QUBITS_PATCH_14_SITES),
            )
            assert measured == 1, (
                f"State {state!r}: {basis}-stabilizer {index} on {support} "
                f"expected +1, got {measured}"
            )


def test_superdense_detectors_are_deterministic() -> None:
    """Verify the superdense ancilla detectors never fire on a clean codeword."""
    circuit, _ = _superdense_circuit()
    detection_events = circuit.compile_detector_sampler(seed=2).sample(64)

    assert circuit.num_detectors == SUPERDENSE_NUM_DETECTORS
    assert circuit.num_measurements == SUPERDENSE_NUM_DETECTORS
    assert not detection_events.any()


@pytest.mark.parametrize("state", PATCH_14_SITE_INJECTION_STATES)
def test_superdense_round_preserves_code(
    state: StimInjectedStateName,
) -> None:
    """Verify all six stabilizers stay +1 and the logical survives the round."""
    circuit, _ = _superdense_circuit(injected_state=state)
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)

    for basis, stabilizers in (
        ("X", PATCH_14_SITE_X_STABILIZERS),
        ("Z", PATCH_14_SITE_Z_STABILIZERS),
    ):
        for support in stabilizers:
            assert (
                simulator.peek_observable_expectation(
                    _pauli_string(basis, support, n_qubits=NUM_QUBITS_PATCH_14_SITES)
                )
                == 1
            )

    expected_y = LOGICAL_EXPECTATIONS[state][LOGICAL_EXPECTATION_INDEX["Y"]]
    for support in PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES:
        assert (
            simulator.peek_observable_expectation(_logical_observable_14_site("Y", support))
            == expected_y
        )


# ---------------------------------------------------------------------------
# Stabilizer syndromes with injected faults
# ---------------------------------------------------------------------------


def _expected_superdense_syndrome(faults: list[tuple[str, int]]) -> tuple[int, ...]:
    """Return the expected ancilla-bit vector for the given data faults."""
    bits = [0] * SUPERDENSE_NUM_DETECTORS
    for pauli, site in faults:
        for index, support in enumerate(PATCH_14_SITE_X_STABILIZERS):
            if site in support and pauli in {"Z", "Y"}:
                bits[SUPERDENSE_X_STAB_BIT[index]] ^= 1
        for index, support in enumerate(PATCH_14_SITE_Z_STABILIZERS):
            if site in support and pauli in {"X", "Y"}:
                bits[SUPERDENSE_Z_STAB_BIT[index]] ^= 1
    return tuple(bits)


def _build_and_run_superdense_syndrome(
    faults: list[tuple[str, int]],
    *,
    injected_state: StimInjectedStateName,
) -> tuple[int, ...]:
    """Inject faults, run the superdense round, and return the ancilla syndrome."""
    circuit, measurement_start = _superdense_circuit(
        injected_state=injected_state,
        faults=faults,
    )
    syndrome = _sample(circuit, shots=1)[0, measurement_start:]
    return tuple(int(bit) for bit in syndrome)


_SUPERDENSE_SINGLE_FAULTS = [
    [(pauli, site)] for pauli in ("X", "Y", "Z") for site in PATCH_14_SITE_DATA_SITES
]
_SUPERDENSE_SINGLE_IDS = [
    f"{pauli}_{site}" for pauli in ("X", "Y", "Z") for site in PATCH_14_SITE_DATA_SITES
]

_SUPERDENSE_TWO_SAME_FAULTS: list[list[tuple[str, int]]] = []
_SUPERDENSE_TWO_SAME_IDS: list[str] = []
for _pauli in ("X", "Y", "Z"):
    for _index, _site_1 in enumerate(PATCH_14_SITE_DATA_SITES):
        for _site_2 in PATCH_14_SITE_DATA_SITES[_index + 1 :]:
            _SUPERDENSE_TWO_SAME_FAULTS.append([(_pauli, _site_1), (_pauli, _site_2)])
            _SUPERDENSE_TWO_SAME_IDS.append(f"{_pauli}{_pauli}_{_site_1}_{_site_2}")

_SUPERDENSE_TWO_MIXED_FAULTS: list[list[tuple[str, int]]] = []
_SUPERDENSE_TWO_MIXED_IDS: list[str] = []
for _pauli_1, _pauli_2 in (("X", "Z"), ("X", "Y"), ("Z", "Y")):
    for _index, _site_1 in enumerate(PATCH_14_SITE_DATA_SITES):
        for _site_2 in PATCH_14_SITE_DATA_SITES[_index + 1 :]:
            _SUPERDENSE_TWO_MIXED_FAULTS.append([(_pauli_1, _site_1), (_pauli_2, _site_2)])
            _SUPERDENSE_TWO_MIXED_IDS.append(f"{_pauli_1}{_pauli_2}_{_site_1}_{_site_2}")


class TestSuperdenseSyndromeExtraction:
    """Verify superdense ancilla syndromes after injected Pauli faults."""

    @pytest.mark.parametrize("state", PATCH_14_SITE_INJECTION_STATES)
    def test_no_fault_trivial(self, state: StimInjectedStateName) -> None:
        measured = _build_and_run_superdense_syndrome([], injected_state=state)
        assert measured == (0,) * SUPERDENSE_NUM_DETECTORS

    @pytest.mark.parametrize("state", ["S", "Sdg"])
    @pytest.mark.parametrize("faults", _SUPERDENSE_SINGLE_FAULTS, ids=_SUPERDENSE_SINGLE_IDS)
    def test_single_fault(
        self,
        state: StimInjectedStateName,
        faults: list[tuple[str, int]],
    ) -> None:
        assert _build_and_run_superdense_syndrome(
            faults,
            injected_state=state,
        ) == _expected_superdense_syndrome(faults)

    @pytest.mark.parametrize("state", ["S", "Sdg"])
    @pytest.mark.parametrize("faults", _SUPERDENSE_TWO_SAME_FAULTS, ids=_SUPERDENSE_TWO_SAME_IDS)
    def test_two_qubit_same_pauli(
        self,
        state: StimInjectedStateName,
        faults: list[tuple[str, int]],
    ) -> None:
        assert _build_and_run_superdense_syndrome(
            faults,
            injected_state=state,
        ) == _expected_superdense_syndrome(faults)

    @pytest.mark.parametrize("state", ["S", "Sdg"])
    @pytest.mark.parametrize("faults", _SUPERDENSE_TWO_MIXED_FAULTS, ids=_SUPERDENSE_TWO_MIXED_IDS)
    def test_two_qubit_mixed_pauli(
        self,
        state: StimInjectedStateName,
        faults: list[tuple[str, int]],
    ) -> None:
        assert _build_and_run_superdense_syndrome(
            faults,
            injected_state=state,
        ) == _expected_superdense_syndrome(faults)


# ---------------------------------------------------------------------------
# Hirano-encoder helper sites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", PATCH_14_SITE_INJECTION_STATES)
def test_14_site_encoder_helpers_return_disentangled(
    state: StimInjectedStateName,
) -> None:
    """Verify every helper site finishes disentangled in |+>."""
    circuit = _encoded_14_site_circuit(state)
    for site in PATCH_14_SITE_HELPER_SITES:
        assert (
            _peek(
                circuit,
                _pauli_string("X", (site,), n_qubits=NUM_QUBITS_PATCH_14_SITES),
            )
            == 1
        )


@pytest.mark.parametrize("state", PATCH_14_SITE_INJECTION_STATES)
def test_14_site_encoder_flag_qubits_bit_value(
    state: StimInjectedStateName,
) -> None:
    """Verify measured helper flags always return the all-zero X-basis outcome."""
    circuit = stim.Circuit()
    append_14_site_patch_hirano_encoder(
        circuit,
        injected_state=state,
        measure_flags=True,
        add_ticks=False,
    )
    shots = _sample(circuit, shots=1024)

    assert shots.shape == (1024, len(PATCH_14_SITE_HELPER_SITES))
    assert not shots.any()


# ---------------------------------------------------------------------------
# 14-site–specific structure
# ---------------------------------------------------------------------------


def test_14_site_encoder_cnots_are_local() -> None:
    """Verify every encoder CX acts on diagonally adjacent ``(±1, ±1)`` sites."""
    circuit = _encoded_14_site_circuit("S")

    for instruction in circuit:
        if instruction.name != "CX":
            continue
        qubit_targets = [target.value for target in instruction.targets_copy()]
        for control, target in zip(qubit_targets[::2], qubit_targets[1::2], strict=True):
            (cx, cy), (tx, ty) = PATCH_14_SITE_COORDS[control], PATCH_14_SITE_COORDS[target]
            assert (abs(cx - tx), abs(cy - ty)) == (
                1,
                1,
            ), f"CX {control}->{target} is not a diagonal neighbour"


def test_14_site_encoder_accepts_y_eigenstate_family_and_rejects_others() -> None:
    """Verify equivalent S/S_DAG names are accepted and other states raise."""
    for state in ("S", "i", "Y+", "Sdg", "-i", "Y-"):
        append_14_site_patch_hirano_encoder(stim.Circuit(), injected_state=state)
    for state in ("0", "1", "+", "-"):
        with pytest.raises(ValueError, match="can inject only the logical-Y eigenstate family"):
            append_14_site_patch_hirano_encoder(stim.Circuit(), injected_state=state)


def test_superdense_round_leaves_data_qubit_2_restored() -> None:
    """Verify the D2/D2' repetition-code detour restores qubit 2 and clears D2'."""
    circuit, _ = _superdense_circuit()
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)

    # Face {0,2,3,5} X-stabilizer support includes data qubit 2 at its home site;
    # it stays +1 (qubit 2 is not relocated onto D2').
    home_face = _pauli_string(
        "X", PATCH_14_SITE_X_STABILIZERS[1], n_qubits=NUM_QUBITS_PATCH_14_SITES
    )
    assert simulator.peek_observable_expectation(home_face) == 1
    # D2' returns disentangled to |+> after the X-rep code is undone.
    d2_prime = _pauli_string(
        "X", (PATCH_14_SITE_D2_PRIME_SITE,), n_qubits=NUM_QUBITS_PATCH_14_SITES
    )
    assert simulator.peek_observable_expectation(d2_prime) == 1
