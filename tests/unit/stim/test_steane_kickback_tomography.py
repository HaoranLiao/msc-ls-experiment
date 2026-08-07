"""Tests for Steane-code kickback tomography helpers in Stim."""

from __future__ import annotations

import math

import numpy as np
import pytest

from msc.steane_code.kickback_tomography_qiskit import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_FULL_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographySummary,
)
from msc.steane_code.kickback_tomography_stim import (
    KickbackTomographyExperiment,
    build_kickback_tomography_experiment,
    kickback_tomography_shot_is_retained,
    summarize_kickback_tomography_shots,
)
from msc.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
)
from msc.steane_code.state_injection_stim import StimInjectedStateName

NUM_SHOTS = 2000


def _measurement_record_order(
    experiment: KickbackTomographyExperiment,
    qed_cycle_indices: tuple[int, ...],
) -> list[str]:
    """Return measurement-stage names in Stim record order."""
    layout = experiment.layout
    label_by_position: dict[int, str] = {}

    def label_positions(label: str, positions: tuple[int, ...]) -> None:
        for position in positions:
            label_by_position[position] = label

    if layout.preparation_round is not None:
        label_positions("prep_cult", layout.preparation_round.all_positions)
    label_positions(
        "kt_measure",
        layout.tomography_measure_round.all_positions,
    )
    label_positions(
        "kt_check",
        layout.tomography_check_round.all_positions,
    )
    for cycle_index, positions in zip(
        qed_cycle_indices,
        layout.qed_cycles,
        strict=True,
    ):
        label_positions(f"qed_{cycle_index}", positions)
    label_positions("final_stab", layout.final_stabilizers)

    order: list[str] = []
    for position in range(experiment.circuit.num_measurements):
        label = label_by_position[position]
        if not order or order[-1] != label:
            order.append(label)
    return order


def _expected_y_axis_error_probability(state: StimInjectedStateName) -> float:
    """Return the probability of the logical-Y ``-1`` outcome."""
    y_plus_probabilities: dict[StimInjectedStateName, float] = {
        "0": 0.5,
        "1": 0.5,
        "+": 0.5,
        "-": 0.5,
        "i": 1.0,
        "Y+": 1.0,
        "S": 1.0,
        "-i": 0.0,
        "Y-": 0.0,
        "Sdg": 0.0,
    }
    return 1.0 - y_plus_probabilities[state]


def _sample(
    experiment: KickbackTomographyExperiment,
    shots: int = NUM_SHOTS,
) -> np.ndarray:
    """Sample a Stim KT experiment."""
    return experiment.circuit.compile_sampler().sample(shots)


# =============================================================================
# Test Set 1: Circuit construction and layout
# =============================================================================


class TestKickbackTomographyCircuitConstruction:
    """KT builders expose the expected Stim measurement-record layout."""

    @pytest.mark.parametrize(
        ("requested_indices", "expected_indices", "expected_measurement_order"),
        [
            (
                None,
                KT_DEFAULT_QED_CYCLE_INDICES,
                [
                    "qed_1",
                    "prep_cult",
                    "kt_measure",
                    "qed_3",
                    "kt_check",
                    "final_stab",
                ],
            ),
            (
                KT_FULL_QED_CYCLE_INDICES,
                KT_FULL_QED_CYCLE_INDICES,
                [
                    "qed_1",
                    "prep_cult",
                    "qed_2",
                    "kt_measure",
                    "qed_3",
                    "kt_check",
                    "qed_4",
                    "final_stab",
                ],
            ),
            (
                (4, 2),
                (2, 4),
                [
                    "prep_cult",
                    "qed_2",
                    "kt_measure",
                    "kt_check",
                    "qed_4",
                    "final_stab",
                ],
            ),
        ],
    )
    def test_qed_schedules_follow_paper_cycle_order(
        self,
        requested_indices: tuple[int, ...] | None,
        expected_indices: tuple[int, ...],
        expected_measurement_order: list[str],
    ) -> None:
        kwargs = {} if requested_indices is None else {"qed_cycle_indices": requested_indices}
        experiment = build_kickback_tomography_experiment(
            add_ticks=False,
            **kwargs,
        )
        circuit = experiment.circuit
        layout = experiment.layout

        assert circuit.num_qubits == 11
        assert circuit.num_measurements == (
            3 * CULTIVATION_MEASUREMENTS_PER_ROUND
            + len(expected_indices) * KT_STABILIZER_BITS_PER_CYCLE
            + KT_STABILIZER_BITS_PER_CYCLE // 2
        )
        assert layout.preparation_round is not None
        assert len(layout.qed_cycles) == len(expected_indices)
        assert all(len(cycle) == KT_STABILIZER_BITS_PER_CYCLE for cycle in layout.qed_cycles)
        assert len(layout.final_stabilizers) == (KT_STABILIZER_BITS_PER_CYCLE // 2)
        assert _measurement_record_order(experiment, expected_indices) == expected_measurement_order

    @pytest.mark.parametrize(
        ("basis", "expected_bits"),
        [("x", 3), ("z", 3), ("both", 6), ("none", 0)],
    )
    def test_final_stabilizer_basis_controls_detector_layout(
        self,
        basis: str,
        expected_bits: int,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis=basis,
            add_ticks=False,
        )

        assert len(experiment.layout.final_stabilizers) == expected_bits

    def test_circuit_contains_detector_and_observable_annotations(self) -> None:
        experiment = build_kickback_tomography_experiment(add_ticks=False)
        circuit = experiment.circuit
        detection_events, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
            32, separate_observables=True
        )

        assert circuit.num_detectors == len(experiment.layout.detector_positions) + 2
        assert circuit.num_observables == 1
        assert not detection_events.any()
        assert not observable_flips.any()


# =============================================================================
# Test Set 2: Sampled-shot parsing and postselection semantics
# =============================================================================


class TestKickbackTomographyShotAnalysis:
    """Shot analysis separates detector rejection and KT root outcomes."""

    def test_summary_classifies_retained_root_outcomes(self) -> None:
        experiment = build_kickback_tomography_experiment(
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="none",
            add_ticks=False,
        )
        layout = experiment.layout
        n_bits = experiment.circuit.num_measurements

        def shot_with(*positions: int) -> np.ndarray:
            shot = np.zeros(n_bits, dtype=bool)
            shot[list(positions)] = True
            return shot

        measurement_root = layout.tomography_measure_round.folded_root
        measurement_root_double_check = layout.tomography_measure_round.folded_root_double_check
        check_root = layout.tomography_check_round.folded_root
        check_root_double_check = layout.tomography_check_round.folded_root_double_check
        cultivation_flag = layout.tomography_measure_round.flag_positions[0]

        good_success = shot_with()
        good_error = shot_with(
            measurement_root,
            measurement_root_double_check,
            check_root,
            check_root_double_check,
        )
        measurement_only_disagreement = shot_with(
            measurement_root,
            measurement_root_double_check,
        )
        check_only_disagreement = shot_with(
            check_root,
            check_root_double_check,
        )
        rejected_detector = shot_with(cultivation_flag)

        shots = np.concatenate(
            [
                np.repeat(good_success[np.newaxis, :], 7, axis=0),
                np.repeat(good_error[np.newaxis, :], 3, axis=0),
                np.repeat(
                    measurement_only_disagreement[np.newaxis, :],
                    5,
                    axis=0,
                ),
                np.repeat(
                    check_only_disagreement[np.newaxis, :],
                    4,
                    axis=0,
                ),
                np.repeat(
                    rejected_detector[np.newaxis, :],
                    11,
                    axis=0,
                ),
            ],
        )

        summary = summarize_kickback_tomography_shots(shots, layout)

        assert kickback_tomography_shot_is_retained(good_success, layout)
        assert kickback_tomography_shot_is_retained(good_error, layout)
        assert kickback_tomography_shot_is_retained(
            measurement_only_disagreement,
            layout,
        )
        assert kickback_tomography_shot_is_retained(
            check_only_disagreement,
            layout,
        )
        assert not kickback_tomography_shot_is_retained(
            rejected_detector,
            layout,
        )
        assert summary == KickbackTomographySummary(
            total_shots=30,
            retained_shots=19,
            rejected_shots=11,
            logical_success_shots=7,
            logical_error_shots=3,
            measure_check_disagreement_shots=9,
        )
        assert summary.logical_error_probability == pytest.approx(3 / 19)
        assert summary.fidelity_lower_bound == pytest.approx(16 / 19)

        agreement_only_summary = summarize_kickback_tomography_shots(
            shots,
            layout,
            retain_measure_check_disagreement=False,
        )
        assert not kickback_tomography_shot_is_retained(
            measurement_only_disagreement,
            layout,
            retain_measure_check_disagreement=False,
        )
        assert not kickback_tomography_shot_is_retained(
            check_only_disagreement,
            layout,
            retain_measure_check_disagreement=False,
        )
        assert agreement_only_summary == KickbackTomographySummary(
            total_shots=30,
            retained_shots=10,
            rejected_shots=20,
            logical_success_shots=7,
            logical_error_shots=3,
            measure_check_disagreement_shots=0,
        )


# =============================================================================
# Test Set 3: Noiseless KT behavior
# =============================================================================


class TestNoiselessKickbackTomography:
    """Noiseless KT matches logical-Y projective-measurement probabilities."""

    def test_cultivated_s_state_has_zero_kt_logical_error(self) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state="S",
            add_ticks=False,
        )
        summary = summarize_kickback_tomography_shots(
            _sample(experiment, shots=10),
            experiment.layout,
        )

        assert summary.retained_shots == 10
        assert summary.logical_error_shots == 0
        assert summary.retainment_rate == 1.0
        assert summary.logical_error_probability == 0.0

    @pytest.mark.parametrize("state", ["0", "+", "1", "Y+", "S"])
    def test_injection_only_kt_error_probability(
        self,
        state: StimInjectedStateName,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state=state,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="none",
            add_ticks=False,
        )
        summary = summarize_kickback_tomography_shots(
            _sample(experiment),
            experiment.layout,
        )

        expected_error = _expected_y_axis_error_probability(state)
        sigma = math.sqrt(expected_error * (1.0 - expected_error) / NUM_SHOTS)
        tolerance = max(4 * sigma, 0.02)

        assert summary.retained_shots == NUM_SHOTS
        assert summary.logical_error_probability == pytest.approx(
            expected_error,
            abs=tolerance,
        )

    def test_preparation_cultivation_projects_before_kt(self) -> None:
        """
        With preparation cultivation, the state is projected to the +1 Y-axis
        state and hence has no logical error in KT.
        """
        experiment = build_kickback_tomography_experiment(
            injected_state="1",
            include_preparation_cultivation=True,
            qed_cycle_indices=(),
            final_stabilizer_basis="none",
            add_ticks=False,
        )
        summary = summarize_kickback_tomography_shots(
            _sample(experiment),
            experiment.layout,
        )

        assert summary.retained_shots > 0
        assert summary.logical_error_probability == 0.0
