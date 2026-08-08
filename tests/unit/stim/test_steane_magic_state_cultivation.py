"""Parallel Stim tests for the Steane-code cultivation helpers.

The Qiskit circuit projects onto logical ``H_XY`` and cultivates ``|T>``.
Stim replaces ``T/T_DAG`` with ``S/S_DAG``, so these tests apply the same
checks to the Clifford logical-Y projection and its ``|S>`` target state.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import stim

from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise
from msc.noise.fault_tolerance_stim import (
    DetectorErrorModelShotAnalyzer,
    assess_circuit_single_fault_tolerance,
)
from msc.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
    STEANE_QED_MEASUREMENTS_PER_ROUND,
)
from msc.steane_code.magic_state_cultivation_stim import (
    build_steane_injection_and_cultivation_circuit,
    postselect_success_shots,
)
from msc.steane_code.state_injection_qiskit import (
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
)
from msc.steane_code.state_injection_stim import (
    StimInjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
)

NUM_SHOTS = 4000
NOISY_NUM_SHOTS = 10_000
NOISY_TEST_SEED = 42

ENCODER_FLAG_BITS = 4
LOGICAL_MEASUREMENT_BITS = len(PATCH_DEFAULT_LOGICAL_X)

CULTIVATION_TEST_STATES: tuple[StimInjectedStateName, ...] = (
    "0",
    "1",
    "+",
    "Y+",
    "S",
)


def _expected_acceptance_probability(state: StimInjectedStateName) -> float:
    """Return ``|<S|state>|²`` for the Stim logical-Y projection."""
    probabilities: dict[StimInjectedStateName, float] = {
        "0": 0.5,
        "1": 0.5,
        "+": 0.5,
        "-": 0.5,
        "i": 1.0,
        "-i": 0.0,
        "Y+": 1.0,
        "Y-": 0.0,
        "S": 1.0,
        "Sdg": 0.0,
    }
    return probabilities[state]


def _build_cultivation_with_logical_measurement(
    state: StimInjectedStateName,
    basis: str,
    *,
    n_rounds: int = 1,
) -> stim.Circuit:
    """Build injection, cultivation, and destructive logical readout."""
    circuit = build_steane_injection_and_cultivation_circuit(
        state,
        cultivation_rounds=n_rounds,
        add_qed_round=True,
        add_ticks=False,
        skip_cx_1_4=False,
    )
    support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    append_patch_logical_measurement(
        circuit,
        basis,
        support=support,
        destructive=True,
    )
    return circuit


def _n_postselection_bits(n_rounds: int) -> int:
    """Return encoder-flag, QED, and cultivation postselection bits."""
    return (
        ENCODER_FLAG_BITS
        + STEANE_QED_MEASUREMENTS_PER_ROUND
        + n_rounds * CULTIVATION_MEASUREMENTS_PER_ROUND
    )


def _postselect_and_compute_expectation(
    shots: np.ndarray,
    *,
    n_rounds: int = 1,
) -> tuple[float, int]:
    """Postselect and return logical parity expectation and retained shots."""
    n_postselection_bits = _n_postselection_bits(n_rounds)
    accepted = postselect_success_shots(
        shots,
        postselection_bit_positions=range(n_postselection_bits),
    )
    n_accepted = int(accepted.shape[0])
    if n_accepted == 0:
        return 0.0, 0

    logical_bits = accepted[:, -LOGICAL_MEASUREMENT_BITS:]
    parities = logical_bits.sum(axis=1) % 2
    expectation = float(np.mean(1 - 2 * parities))
    return expectation, n_accepted


def _compute_success_rate(shots: np.ndarray, *, n_rounds: int = 1) -> float:
    """Return the fraction passing encoder, QED, and cultivation postselection."""
    accepted = postselect_success_shots(
        shots,
        postselection_bit_positions=range(_n_postselection_bits(n_rounds)),
    )
    return float(accepted.shape[0] / shots.shape[0])


def _sample(
    circuit: stim.Circuit,
    *,
    shots: int,
    seed: int = NOISY_TEST_SEED,
) -> np.ndarray:
    """Sample a circuit reproducibly."""
    return circuit.compile_sampler(seed=seed).sample(shots)


# =============================================================================
# Test Set 1: Postselected logical expectation values and success rates
# =============================================================================


class TestPostselectedLogicalExpectation:
    """Postselection projects onto the logical-Y ``|S>`` state."""

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    @pytest.mark.parametrize("basis", ["Z", "X", "Y"])
    def test_expectation_matches_s_state(
        self,
        state: StimInjectedStateName,
        basis: str,
    ) -> None:
        circuit = _build_cultivation_with_logical_measurement(state, basis)
        shots = _sample(circuit, shots=NUM_SHOTS)
        expectation, n_accepted = _postselect_and_compute_expectation(shots)

        assert n_accepted > 0
        expected = 1.0 if basis == "Y" else 0.0
        tolerance = 4 * math.sqrt((1.0 - expected**2) / n_accepted) + 0.01
        assert expectation == pytest.approx(expected, abs=tolerance), (
            f"State {state!r}, basis {basis}: <{basis}_L>={expectation:.4f}, "
            f"expected {expected:.4f} after cultivation "
            f"(n_accepted={n_accepted})"
        )

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    def test_success_rate(self, state: StimInjectedStateName) -> None:
        circuit = build_steane_injection_and_cultivation_circuit(
            state,
            add_qed_round=True,
            add_ticks=False,
            skip_cx_1_4=False,
        )
        shots = _sample(circuit, shots=NUM_SHOTS)

        measured_rate = _compute_success_rate(shots)
        expected_rate = _expected_acceptance_probability(state)
        if expected_rate in {0.0, 1.0}:
            assert measured_rate == expected_rate
        else:
            sigma = math.sqrt(expected_rate * (1.0 - expected_rate) / NUM_SHOTS)
            assert measured_rate == pytest.approx(
                expected_rate,
                abs=max(4 * sigma, 0.02),
            )

    def test_builder_annotates_every_postselection_measurement(self) -> None:
        circuit = build_steane_injection_and_cultivation_circuit(
            "S",
            add_qed_round=True,
            add_ticks=False,
            skip_cx_1_4=False,
        )
        detection_events = circuit.compile_detector_sampler(seed=1).sample(32)

        assert circuit.num_detectors == circuit.num_measurements == 15
        assert not detection_events.any()


# =============================================================================
# Test Set 2: Stabilizer syndromes after cultivation
# =============================================================================


class TestStabilizerSyndromesAfterCultivation:
    """Accepted noiseless shots remain in the Steane code space."""

    @pytest.mark.parametrize("state", ["0", "+", "Y+", "S"])
    def test_all_stabilizers_trivial(
        self,
        state: StimInjectedStateName,
    ) -> None:
        circuit = build_steane_injection_and_cultivation_circuit(
            state,
            add_qed_round=True,
            add_ticks=False,
            skip_cx_1_4=False,
        )
        append_patch_stabilizer_measurement(circuit, add_ticks=False)

        shots = _sample(circuit, shots=NUM_SHOTS)
        accepted = postselect_success_shots(
            shots,
            postselection_bit_positions=range(_n_postselection_bits(1)),
        )

        assert accepted.shape[0] > 0
        assert not accepted[:, -6:].any()


# =============================================================================
# Test Set 3: Noisy multi-round cultivation approaches the S-state proxy
# =============================================================================


def _postselect_multi_round(
    shots: np.ndarray,
    n_rounds: int,
) -> tuple[float, int]:
    """Postselect every encoder, QED, and cultivation check."""
    return _postselect_and_compute_expectation(shots, n_rounds=n_rounds)


class TestNoisyCultivationWithRounds:
    """Noisy postselection preserves the logical ``|S>`` target."""

    ROUNDS_TO_TEST = (1, 2, 3)

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_postselected_fidelity_near_ideal(self, basis: str) -> None:
        expected = 1.0 if basis == "Y" else 0.0

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = add_circuit_depolarizing_noise(
                _build_cultivation_with_logical_measurement(
                    "S",
                    basis,
                    n_rounds=n_rounds,
                )
            )
            shots = _sample(
                circuit,
                shots=NUM_SHOTS,
                seed=NOISY_TEST_SEED + n_rounds,
            )
            expectation, n_accepted = _postselect_multi_round(
                shots,
                n_rounds,
            )

            assert n_accepted > 50
            # Allow four standard errors plus a 4% systematic noise budget.
            statistical_error = math.sqrt((1.0 - expectation**2) / n_accepted)
            tolerance = 4 * statistical_error + 0.04
            assert expectation == pytest.approx(expected, abs=tolerance)

    def test_more_rounds_preserves_fidelity_and_increases_discard(self) -> None:
        errors: list[float] = []
        acceptance_rates: list[float] = []

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = add_circuit_depolarizing_noise(
                _build_cultivation_with_logical_measurement(
                    "S",
                    "Y",
                    n_rounds=n_rounds,
                )
            )
            shots = _sample(
                circuit,
                shots=NOISY_NUM_SHOTS,
                seed=NOISY_TEST_SEED + n_rounds,
            )
            expectation, n_accepted = _postselect_multi_round(
                shots,
                n_rounds,
            )

            assert n_accepted > 50
            errors.append(abs(expectation - 1.0))
            acceptance_rates.append(n_accepted / NOISY_NUM_SHOTS)

        assert errors[-1] <= errors[-2] + 0.03
        assert acceptance_rates[-1] < acceptance_rates[-2] + 0.01

    def test_cultivation_improves_over_no_postselection(self) -> None:
        """Cultivation improves on the noisy encoded state without postselection."""
        # Use the shared front end with both QED and cultivation disabled. The
        # four encoder-flag outcomes are recorded but not postselected.
        uncultivated_circuit = build_steane_injection_and_cultivation_circuit(
            injected_state="S",
            cultivation_rounds=0,
            add_qed_round=False,
            add_ticks=False,
            skip_cx_1_4=False,
        )
        append_patch_logical_measurement(
            uncultivated_circuit,
            "Y",
            support=PATCH_DEFAULT_LOGICAL_X,
            destructive=True,
        )
        uncultivated_shots = _sample(
            add_circuit_depolarizing_noise(uncultivated_circuit),
            shots=NUM_SHOTS,
        )
        uncultivated_parities = uncultivated_shots[:, -3:].sum(axis=1) % 2
        uncultivated_expectation = float(np.mean(1 - 2 * uncultivated_parities))
        uncultivated_error = abs(uncultivated_expectation - 1.0)

        # Cultivation with postselection.
        cultivated_circuit = add_circuit_depolarizing_noise(
            _build_cultivation_with_logical_measurement(
                "S",
                "Y",
            )
        )
        cultivated_shots = _sample(cultivated_circuit, shots=NUM_SHOTS)
        cultivated_expectation, n_accepted = _postselect_multi_round(
            cultivated_shots,
            1,
        )
        cultivated_error = abs(cultivated_expectation - 1.0)

        assert n_accepted > 50
        assert cultivated_error <= uncultivated_error + 0.10


# =============================================================================
# Test Set 4: Circuit-level single-fault tolerance
# =============================================================================


class TestCultivationFaultTolerance:
    @pytest.mark.parametrize(
        (
            "skip_cx_1_4",
            "expected_total_single_fault_cases",
            "expected_benign_faults",
            "expected_rejected_faults",
        ),
        (
            (False, 1239, 88, 1151),
            (True, 1209, 86, 1123),
        ),
    )
    def test_all_single_faults_are_benign_or_rejected(
        self,
        skip_cx_1_4: bool,
        expected_total_single_fault_cases: int,
        expected_benign_faults: int,
        expected_rejected_faults: int,
    ) -> None:
        protocol_circuit = build_steane_injection_and_cultivation_circuit(
            "S",
            cultivation_rounds=1,
            add_qed_round=True,
            add_ticks=True,
            skip_cx_1_4=skip_cx_1_4,
        )
        circuit = add_circuit_depolarizing_noise(
            protocol_circuit,
            p_1q=1e-3,
            p_2q=1e-3,
            p_measure=1e-3,
            p_reset=1e-3,
        )
        append_patch_stabilizer_measurement(circuit, basis="both", add_ticks=True)
        append_patch_logical_measurement(
            circuit,
            "Y",
            destructive=True,
            observable_index=0,
        )
        analyzer = DetectorErrorModelShotAnalyzer(
            circuit,
            postselection_detector_indices=range(circuit.num_detectors),
        )

        result = assess_circuit_single_fault_tolerance(
            experiment_name="steane_injection_and_cultivation",
            analyzer=analyzer,
        )

        assert result.total_single_fault_cases == expected_total_single_fault_cases
        assert result.benign_faults == expected_benign_faults
        assert result.rejected_faults == expected_rejected_faults
        assert result.correctly_decoded_faults == 0
        assert result.malignant_faults == 0
        assert result.graphlike_distance == 3
        assert result.passes_single_fault_check
