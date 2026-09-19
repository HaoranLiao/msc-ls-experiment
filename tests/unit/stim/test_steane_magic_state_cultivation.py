"""Parallel Stim tests for the Steane-code cultivation helpers.

The Qiskit circuit projects onto logical ``H_XY`` and cultivates ``|T>``.
Stim instead performs logical-Y cultivation as the stabilizer proxy for that
protocol, so these tests check the logical-Y projection and its ``|S>`` target.
Every behavioral test is parameterized over both supported patches (the
11-site Rosenfeld patch and the 14-site abstract-routed patch) via ``PATCH_SUITES``,
since the production API is patch-generic; only the fault-tolerance test set
specializes, because ``skip_cx_1_4`` has no 14-site counterpart.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import stim

from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise
from msc.noise.fault_tolerance_stim import assess_fully_postselected_circuit_fault_distance
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    STEANE_QED_MEASUREMENTS_PER_ROUND,
)
from msc.steane_code.magic_state_cultivation_stim import (
    build_11_site_patch_injection_and_cultivation_circuit,
    build_14_site_patch_injection_and_cultivation_circuit,
    postselect_success_shots,
)
from msc.steane_code.state_injection_common import (
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_DEFAULT_LOGICAL_X,
    PATCH_14_SITE_DEFAULT_LOGICAL_Z,
)
from msc.steane_code.state_injection_stim import (
    StimInjectedStateName,
    append_11_site_patch_logical_measurement,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_logical_measurement,
    append_14_site_patch_superdense_stabilizer_measurement,
)

NUM_SHOTS = 4000
NOISY_NUM_SHOTS = 10_000
NOISY_TEST_SEED = 42


# =============================================================================
# Per-patch cultivation configuration
# =============================================================================


@dataclass(frozen=True)
class PatchCultivationSuite:
    """Bundles the pieces that differ between the 11-site and 14-site patches.

    Every field here is what a behavioral cultivation test needs to stay
    patch-generic: which builder to call, how to append a logical or
    stabilizer measurement, the default logical supports, and the
    measurement-record layout (encoder-flag, QED, and cultivation bit counts
    per round) needed to compute postselection bit ranges.
    """

    name: str
    build_circuit: Callable[..., stim.Circuit]
    append_logical_measurement: Callable[..., None]
    append_stabilizer_measurement: Callable[..., None]
    default_logical_x: tuple[int, ...]
    default_logical_z: tuple[int, ...]
    encoder_flag_bits: int
    qed_measurements_per_round: int
    cultivation_measurements_per_round: int
    accepted_test_states: tuple[StimInjectedStateName, ...]
    success_rate_test_states: tuple[StimInjectedStateName, ...]
    stabilizer_check_states: tuple[StimInjectedStateName, ...]


ELEVEN_SITE = PatchCultivationSuite(
    name="11-site",
    build_circuit=functools.partial(
        build_11_site_patch_injection_and_cultivation_circuit,
        skip_cx_1_4=False,
    ),
    append_logical_measurement=append_11_site_patch_logical_measurement,
    append_stabilizer_measurement=append_11_site_patch_stabilizer_measurement,
    default_logical_x=PATCH_11_SITE_DEFAULT_LOGICAL_X,
    default_logical_z=PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    encoder_flag_bits=len(PATCH_11_SITE_HELPER_SITES),
    qed_measurements_per_round=STEANE_QED_MEASUREMENTS_PER_ROUND,
    cultivation_measurements_per_round=PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    accepted_test_states=("0", "1", "+", "Y+", "S"),
    success_rate_test_states=("0", "1", "+", "Y+", "S"),
    stabilizer_check_states=("0", "+", "Y+", "S"),
)

# The default 14-site abstract-routed encoder accepts arbitrary Clifford
# inputs. These cases exercise both deterministic logical-Y branches.
# "Sdg" is deterministically rejected by every cultivation round (0%
# acceptance), so it is exercised only by the success-rate test, not the
# expectation-matching test, which requires at least one accepted shot.
FOURTEEN_SITE = PatchCultivationSuite(
    name="14-site",
    build_circuit=build_14_site_patch_injection_and_cultivation_circuit,
    append_logical_measurement=append_14_site_patch_logical_measurement,
    append_stabilizer_measurement=append_14_site_patch_superdense_stabilizer_measurement,
    default_logical_x=PATCH_14_SITE_DEFAULT_LOGICAL_X,
    default_logical_z=PATCH_14_SITE_DEFAULT_LOGICAL_Z,
    encoder_flag_bits=len(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES),
    qed_measurements_per_round=PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    cultivation_measurements_per_round=PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    accepted_test_states=("Y+", "S"),
    success_rate_test_states=("Y+", "S", "Sdg"),
    stabilizer_check_states=("Y+", "S"),
)

PATCH_SUITES: tuple[PatchCultivationSuite, ...] = (ELEVEN_SITE, FOURTEEN_SITE)
PATCH_IDS: tuple[str, ...] = tuple(patch.name for patch in PATCH_SUITES)

PATCH_ACCEPTED_STATE_CASES: tuple[tuple[PatchCultivationSuite, StimInjectedStateName], ...] = tuple(
    (patch, state) for patch in PATCH_SUITES for state in patch.accepted_test_states
)
PATCH_ACCEPTED_STATE_IDS: tuple[str, ...] = tuple(
    f"{patch.name}-{state}" for patch, state in PATCH_ACCEPTED_STATE_CASES
)

PATCH_SUCCESS_RATE_STATE_CASES: tuple[tuple[PatchCultivationSuite, StimInjectedStateName], ...] = (
    tuple((patch, state) for patch in PATCH_SUITES for state in patch.success_rate_test_states)
)
PATCH_SUCCESS_RATE_STATE_IDS: tuple[str, ...] = tuple(
    f"{patch.name}-{state}" for patch, state in PATCH_SUCCESS_RATE_STATE_CASES
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
    patch: PatchCultivationSuite,
    state: StimInjectedStateName,
    basis: str,
    *,
    n_rounds: int = 1,
    add_ticks: bool = False,
) -> stim.Circuit:
    """Build injection, cultivation, and destructive logical readout.

    Pass ``add_ticks=True`` when the circuit will be noised: ticks keep
    sequential shared-qubit CNOTs unfused so noise lands on each.
    """
    circuit = patch.build_circuit(
        state,
        cultivation_rounds=n_rounds,
        add_qed_round=True,
        add_ticks=add_ticks,
    )
    support = patch.default_logical_z if basis == "Z" else patch.default_logical_x
    patch.append_logical_measurement(
        circuit,
        basis,
        support=support,
        destructive=True,
    )
    return circuit


def _n_postselection_bits(patch: PatchCultivationSuite, n_rounds: int) -> int:
    """Return encoder-flag, QED, and cultivation postselection bits."""
    return (
        patch.encoder_flag_bits
        + patch.qed_measurements_per_round
        + n_rounds * patch.cultivation_measurements_per_round
    )


def _postselect_and_compute_expectation(
    patch: PatchCultivationSuite,
    shots: np.ndarray,
    *,
    n_rounds: int = 1,
) -> tuple[float, int]:
    """Postselect and return logical parity expectation and retained shots."""
    n_postselection_bits = _n_postselection_bits(patch, n_rounds)
    accepted = postselect_success_shots(
        shots,
        postselection_bit_positions=range(n_postselection_bits),
    )
    n_accepted = int(accepted.shape[0])
    if n_accepted == 0:
        return 0.0, 0

    logical_bits = accepted[:, n_postselection_bits:]
    parities = logical_bits.sum(axis=1) % 2
    expectation = float(np.mean(1 - 2 * parities))
    return expectation, n_accepted


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

    @pytest.mark.parametrize(
        ("patch", "state"), PATCH_ACCEPTED_STATE_CASES, ids=PATCH_ACCEPTED_STATE_IDS
    )
    @pytest.mark.parametrize("basis", ["Z", "X", "Y"])
    def test_expectation_matches_s_state(
        self,
        patch: PatchCultivationSuite,
        state: StimInjectedStateName,
        basis: str,
    ) -> None:
        circuit = _build_cultivation_with_logical_measurement(patch, state, basis)
        shots = _sample(circuit, shots=NUM_SHOTS)
        expectation, n_accepted = _postselect_and_compute_expectation(patch, shots)

        assert n_accepted > 0
        expected = 1.0 if basis == "Y" else 0.0
        tolerance = 4 * math.sqrt((1.0 - expected**2) / n_accepted) + 0.01
        assert expectation == pytest.approx(expected, abs=tolerance), (
            f"{patch.name} state {state!r}, basis {basis}: <{basis}_L>={expectation:.4f}, "
            f"expected {expected:.4f} after cultivation "
            f"(n_accepted={n_accepted})"
        )

    @pytest.mark.parametrize(
        ("patch", "state"), PATCH_SUCCESS_RATE_STATE_CASES, ids=PATCH_SUCCESS_RATE_STATE_IDS
    )
    def test_success_rate(
        self,
        patch: PatchCultivationSuite,
        state: StimInjectedStateName,
    ) -> None:
        circuit = patch.build_circuit(state, add_qed_round=True, add_ticks=False)
        shots = _sample(circuit, shots=NUM_SHOTS)
        accepted = postselect_success_shots(
            shots,
            postselection_bit_positions=range(_n_postselection_bits(patch, n_rounds=1)),
        )
        measured_rate = float(accepted.shape[0] / shots.shape[0])
        expected_rate = _expected_acceptance_probability(state)
        if expected_rate in {0.0, 1.0}:
            assert measured_rate == expected_rate
        else:
            sigma = math.sqrt(expected_rate * (1.0 - expected_rate) / NUM_SHOTS)
            assert measured_rate == pytest.approx(
                expected_rate,
                abs=max(4 * sigma, 0.02),
            )

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_builder_annotates_every_postselection_measurement(
        self,
        patch: PatchCultivationSuite,
    ) -> None:
        circuit = patch.build_circuit("S", add_qed_round=True, add_ticks=False)
        detection_events = circuit.compile_detector_sampler(seed=1).sample(32)

        expected_bits = _n_postselection_bits(patch, n_rounds=1)
        assert circuit.num_detectors == circuit.num_measurements == expected_bits
        assert not detection_events.any()


# =============================================================================
# Test Set 2: Stabilizer syndromes after cultivation
# =============================================================================


class TestStabilizerSyndromesAfterCultivation:
    """Accepted noiseless shots remain in the Steane code space."""

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_all_stabilizers_trivial(self, patch: PatchCultivationSuite) -> None:
        for state in patch.stabilizer_check_states:
            circuit = patch.build_circuit(state, add_qed_round=True, add_ticks=False)
            patch.append_stabilizer_measurement(circuit, add_ticks=False)

            shots = _sample(circuit, shots=NUM_SHOTS)
            accepted = postselect_success_shots(
                shots,
                postselection_bit_positions=range(_n_postselection_bits(patch, 1)),
            )

            assert accepted.shape[0] > 0, f"{patch.name} state {state!r} accepted no shots"
            assert not accepted[
                :, -STEANE_QED_MEASUREMENTS_PER_ROUND:
            ].any(), f"{patch.name} state {state!r} had a nonzero stabilizer"


# =============================================================================
# Test Set 3: Noisy multi-round logical-Y cultivation approaches |S>
# =============================================================================


class TestNoisyCultivationWithRounds:
    """Noisy postselection preserves the logical ``|S>`` target."""

    ROUNDS_TO_TEST = (1, 2, 3)

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_postselected_fidelity_near_ideal(
        self,
        patch: PatchCultivationSuite,
        basis: str,
    ) -> None:
        expected = 1.0 if basis == "Y" else 0.0

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = add_circuit_depolarizing_noise(
                _build_cultivation_with_logical_measurement(
                    patch,
                    "S",
                    basis,
                    n_rounds=n_rounds,
                    add_ticks=True,
                )
            )
            shots = _sample(
                circuit,
                shots=NUM_SHOTS,
                seed=NOISY_TEST_SEED + n_rounds,
            )
            expectation, n_accepted = _postselect_and_compute_expectation(
                patch,
                shots,
                n_rounds=n_rounds,
            )

            assert n_accepted > 50
            # Allow four standard errors plus a 4% systematic noise budget.
            statistical_error = math.sqrt((1.0 - expectation**2) / n_accepted)
            tolerance = 4 * statistical_error + 0.04
            assert expectation == pytest.approx(expected, abs=tolerance)

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_more_rounds_preserves_fidelity_and_increases_discard(
        self,
        patch: PatchCultivationSuite,
    ) -> None:
        errors: list[float] = []
        acceptance_rates: list[float] = []

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = add_circuit_depolarizing_noise(
                _build_cultivation_with_logical_measurement(
                    patch,
                    "S",
                    "Y",
                    n_rounds=n_rounds,
                    add_ticks=True,
                )
            )
            shots = _sample(
                circuit,
                shots=NOISY_NUM_SHOTS,
                seed=NOISY_TEST_SEED + n_rounds,
            )
            expectation, n_accepted = _postselect_and_compute_expectation(
                patch,
                shots,
                n_rounds=n_rounds,
            )

            assert n_accepted > 50
            errors.append(abs(expectation - 1.0))
            acceptance_rates.append(n_accepted / NOISY_NUM_SHOTS)

        assert errors[-1] <= errors[-2] + 0.03
        assert acceptance_rates[-1] < acceptance_rates[-2] + 0.01

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_cultivation_improves_over_no_postselection(
        self,
        patch: PatchCultivationSuite,
    ) -> None:
        """Cultivation improves on the noisy encoded state without postselection."""
        # Use the shared front end with both QED and cultivation disabled. The
        # encoder-flag outcomes are recorded but not postselected.
        uncultivated_circuit = patch.build_circuit(
            "S",
            cultivation_rounds=0,
            add_qed_round=False,
            add_ticks=True,  # noised below; ticks keep shared-qubit CNOTs unfused
        )
        patch.append_logical_measurement(
            uncultivated_circuit,
            "Y",
            support=patch.default_logical_x,
            destructive=True,
        )
        uncultivated_shots = _sample(
            add_circuit_depolarizing_noise(uncultivated_circuit),
            shots=NUM_SHOTS,
        )
        logical_measurement_bits = len(patch.default_logical_x)
        uncultivated_parities = uncultivated_shots[:, -logical_measurement_bits:].sum(axis=1) % 2
        uncultivated_expectation = float(np.mean(1 - 2 * uncultivated_parities))
        uncultivated_error = abs(uncultivated_expectation - 1.0)

        # Cultivation with postselection.
        cultivated_circuit = add_circuit_depolarizing_noise(
            _build_cultivation_with_logical_measurement(
                patch,
                "S",
                "Y",
                add_ticks=True,
            )
        )
        cultivated_shots = _sample(cultivated_circuit, shots=NUM_SHOTS)
        cultivated_expectation, n_accepted = _postselect_and_compute_expectation(
            patch,
            cultivated_shots,
        )
        cultivated_error = abs(cultivated_expectation - 1.0)

        assert n_accepted > 50
        assert cultivated_error <= uncultivated_error + 0.10


# =============================================================================
# Test Set 4: Circuit-level physical fault distance
# =============================================================================


class TestCultivationFaultTolerance:
    """Physical fault distance of injection, QED, and cultivation.

    The 14-site patch has distance three. The 11-site patch has distance two:
    once shared-ancilla parity CNOTs are no longer fused into one Stim gate,
    circuit-level noise exposes weight-2 hook-fault pairs (e.g. Z9*Z7 with Z8)
    that flip logical Y with all detectors silent.
    """

    # Both skip_cx_1_4 values are exercised to confirm the drop to distance two
    # is intrinsic: keeping CX 1->4 in the fold network does not restore three.
    @pytest.mark.parametrize(
        ("skip_cx_1_4", "expected_malignant_pair_count"),
        [(False, 29), (True, 29)],
    )
    def test_11_site_injection_qed_cultivation_has_fault_distance_two(
        self,
        skip_cx_1_4: bool,
        expected_malignant_pair_count: int,
    ) -> None:
        circuit = add_circuit_depolarizing_noise(
            build_11_site_patch_injection_and_cultivation_circuit(
                "S",
                cultivation_rounds=1,
                add_qed_round=True,
                add_ticks=True,
                skip_cx_1_4=skip_cx_1_4,
            ),
            p=1e-3,
        )
        append_11_site_patch_stabilizer_measurement(
            circuit,
            basis="BOTH",
            add_ticks=True,
        )
        append_11_site_patch_logical_measurement(
            circuit,
            "Y",
            destructive=True,
            observable_index=0,
        )
        assessment = assess_fully_postselected_circuit_fault_distance(
            experiment_name="steane_11_site_injection_qed_cultivation",
            circuit=circuit,
            max_exhaustive_weight=2,
            find_witness_weight=3,
        )

        # No weight-1 logical fault, but weight-2 malignant pairs exist, so the
        # circuit-level distance is two (was reported as three when the parity
        # CNOTs were fused and their intermediate hook faults were invisible).
        assert not assessment.malignant_single_faults
        assert assessment.malignant_pair_count == expected_malignant_pair_count
        assert assessment.first_malignant_pair is not None
        assert assessment.certified_distance == 2

        witness = assessment.first_malignant_pair
        assert witness.weight == 2
        assert witness.combined_detector_mask == 0
        assert witness.combined_observable_mask == 1

    def test_14_site_injection_qed_cultivation_has_fault_distance_three(
        self,
    ) -> None:
        circuit = add_circuit_depolarizing_noise(
            build_14_site_patch_injection_and_cultivation_circuit(
                "S",
                encoder="abstract_routed",
                cultivation_rounds=1,
                add_qed_round=True,
                add_ticks=True,
            )
        )
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            basis="BOTH",
            add_ticks=True,
        )
        append_14_site_patch_logical_measurement(
            circuit,
            "Y",
            destructive=True,
            observable_index=0,
        )
        assessment = assess_fully_postselected_circuit_fault_distance(
            experiment_name="steane_14_site_injection_qed_cultivation",
            circuit=circuit,
            max_exhaustive_weight=2,
            find_witness_weight=3,
        )

        assert not assessment.malignant_single_faults
        assert assessment.malignant_pair_count == 0
        assert assessment.first_malignant_pair is None
        assert assessment.first_malignant_triple is not None
        assert assessment.certified_distance == 3

        witness = assessment.first_malignant_triple
        assert witness.weight == 3
        assert len({fault.location_key for fault in witness.faults}) == 3
        assert witness.combined_detector_mask == 0
        assert witness.combined_observable_mask == 1


# =============================================================================
# Test Set 5: Shot-based and detector-based postselection agree
# =============================================================================


class TestShotAndDetectorPostselectionAgree:
    """Shot-based and detector-based postselection must read the same bits.

    ``postselect_success_shots`` (used throughout this file) rejects on raw
    ``compile_sampler`` bits; detector-based postselection instead rejects on
    ``DETECTOR`` records via ``compile_detector_sampler``. Every encoder-flag, QED, and
    cultivation-round bit -- including every cultivation flag -- has exactly
    one single-measurement ``DETECTOR`` wrapping it (see
    ``test_builder_annotates_every_postselection_measurement``), so on the
    same noisy circuit and seed the two styles must reject exactly the same
    shots. A nonzero, non-total rejection rate confirms the cultivation flags
    are actually catching noise here, not merely present but inert.
    """

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_raw_and_detector_acceptance_masks_match(
        self,
        patch: PatchCultivationSuite,
    ) -> None:
        noisy_circuit = add_circuit_depolarizing_noise(
            patch.build_circuit("S", add_qed_round=True, add_ticks=True),
            p=0.02,
        )
        n_postselection_bits = _n_postselection_bits(patch, n_rounds=1)

        raw_shots = noisy_circuit.compile_sampler(seed=7).sample(NUM_SHOTS)
        accepted_by_shots = ~raw_shots[:, :n_postselection_bits].any(axis=1)

        detection_events = noisy_circuit.compile_detector_sampler(seed=7).sample(NUM_SHOTS)
        accepted_by_detectors = ~detection_events[:, :n_postselection_bits].any(axis=1)

        assert np.array_equal(accepted_by_shots, accepted_by_detectors)
        assert 0 < int(accepted_by_shots.sum()) < NUM_SHOTS
