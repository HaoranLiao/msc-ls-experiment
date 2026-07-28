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

from magic_state_cultivation.noise.fault_injection_stim import add_circuit_depolarizing_noise
from magic_state_cultivation.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
)
from magic_state_cultivation.steane_code.magic_state_cultivation_stim import (
    append_steane_injection_and_cultivation,
    postselect_success_shots,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import (
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
)
from magic_state_cultivation.steane_code.state_injection_stim import (
    StimInjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)

NUM_SHOTS = 4000
NOISY_NUM_SHOTS = 10_000
NOISY_TEST_SEED = 42

ENCODER_HELPER_BITS = 4
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


def _make_cultivation_circuit(
    injected_state: StimInjectedStateName = "S",
    n_rounds: int = 1,
    *,
    add_ticks: bool = False,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> stim.Circuit:
    """Build patch-Crumble injection followed by cultivation rounds."""
    circuit = stim.Circuit()
    append_steane_injection_and_cultivation(
        circuit,
        injected_state=injected_state,
        cultivation_rounds=n_rounds,
        measure_encoder_helpers=True,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )
    return circuit


def _build_cultivation_with_logical_measurement(
    state: StimInjectedStateName,
    basis: str,
    *,
    n_rounds: int = 1,
) -> stim.Circuit:
    """Build injection, cultivation, and destructive logical readout."""
    circuit = _make_cultivation_circuit(state, n_rounds)
    support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    append_patch_logical_measurement(
        circuit,
        basis,
        support=support,
        destructive=True,
    )
    return circuit


def _n_postselection_bits(n_rounds: int) -> int:
    """Return encoder-helper plus cultivation postselection bits."""
    return ENCODER_HELPER_BITS + n_rounds * CULTIVATION_MEASUREMENTS_PER_ROUND


def _postselect_and_compute_expectation(
    shots: np.ndarray,
    *,
    n_rounds: int = 1,
) -> tuple[float, int]:
    """Postselect and return logical parity expectation and retained shots."""
    n_postselection_bits = _n_postselection_bits(n_rounds)
    accepted = postselect_success_shots(
        shots,
        cult_bit_positions=range(n_postselection_bits),
    )
    n_accepted = int(accepted.shape[0])
    if n_accepted == 0:
        return 0.0, 0

    logical_bits = accepted[:, -LOGICAL_MEASUREMENT_BITS:]
    parities = logical_bits.sum(axis=1) % 2
    expectation = float(np.mean(1 - 2 * parities))
    return expectation, n_accepted


def _compute_success_rate(shots: np.ndarray, *, n_rounds: int = 1) -> float:
    """Return the fraction passing encoder and cultivation postselection."""
    accepted = postselect_success_shots(
        shots,
        cult_bit_positions=range(_n_postselection_bits(n_rounds)),
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
        circuit = _make_cultivation_circuit(state)
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
        circuit = _make_cultivation_circuit(state)
        append_patch_stabilizer_measurement(circuit, add_ticks=False)

        shots = _sample(circuit, shots=NUM_SHOTS)
        accepted = postselect_success_shots(
            shots,
            cult_bit_positions=range(_n_postselection_bits(1)),
        )

        assert accepted.shape[0] > 0
        assert not accepted[:, -6:].any()


# =============================================================================
# Test Set 3: Noisy multi-round cultivation approaches the S-state proxy
# =============================================================================


NOISE_P_1Q = 0.0005
NOISE_P_2Q = 0.005
NOISE_MEASURE = 0.005
NOISE_RESET = 0.005


def _add_test_noise(circuit: stim.Circuit) -> stim.Circuit:
    """Apply rates parallel to the Qiskit cultivation noise model."""
    return add_circuit_depolarizing_noise(
        circuit,
        p_1q=NOISE_P_1Q,
        p_2q=NOISE_P_2Q,
        p_measure=NOISE_MEASURE,
        p_reset=NOISE_RESET,
    )


def _postselect_multi_round(
    shots: np.ndarray,
    n_rounds: int,
) -> tuple[float, int]:
    """Postselect every encoder/cultivation check and compute expectation."""
    return _postselect_and_compute_expectation(shots, n_rounds=n_rounds)


class TestNoisyCultivationWithRounds:
    """Noisy postselection preserves the logical ``|S>`` target."""

    ROUNDS_TO_TEST = (1, 2, 3)

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_postselected_fidelity_near_ideal(self, basis: str) -> None:
        expected = 1.0 if basis == "Y" else 0.0

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _add_test_noise(
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
            circuit = _add_test_noise(
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
        raw_circuit = stim.Circuit()
        append_patch_steane_state_injection_encoder_crumble_sequences(
            raw_circuit,
            injected_state="S",
            add_ticks=False,
        )
        append_patch_logical_measurement(
            raw_circuit,
            "Y",
            support=PATCH_DEFAULT_LOGICAL_X,
            destructive=True,
        )
        raw_shots = _sample(
            _add_test_noise(raw_circuit),
            shots=NUM_SHOTS,
        )
        raw_parities = raw_shots.sum(axis=1) % 2
        raw_expectation = float(np.mean(1 - 2 * raw_parities))

        cultivated = _add_test_noise(
            _build_cultivation_with_logical_measurement(
                "S",
                "Y",
            )
        )
        cultivated_shots = _sample(cultivated, shots=NUM_SHOTS)
        cultivated_expectation, n_accepted = _postselect_multi_round(
            cultivated_shots,
            1,
        )

        assert n_accepted > 50
        assert abs(cultivated_expectation - 1.0) <= (abs(raw_expectation - 1.0) + 0.10)
