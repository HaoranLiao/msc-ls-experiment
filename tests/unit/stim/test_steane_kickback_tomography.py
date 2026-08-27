"""Tests for Steane-code kickback tomography helpers in Stim."""

from __future__ import annotations

import math

import numpy as np
import pytest

from msc.steane_code.kickback_tomography_common import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_FULL_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographySummary,
)
from msc.steane_code.kickback_tomography_stim import (
    KickbackTomographyExperiment,
    build_kickback_tomography_experiment,
    summarize_kickback_tomography_shots,
)
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
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
    label_positions("final_bell_partner_flags", layout.final_bell_partner_flags)

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

    def test_default_patch_has_14_sites(self) -> None:
        experiment = build_kickback_tomography_experiment(add_ticks=False)
        assert experiment.circuit.num_qubits == 14

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
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_qed_schedules_follow_paper_cycle_order(
        self,
        requested_indices: tuple[int, ...] | None,
        expected_indices: tuple[int, ...],
        expected_measurement_order: list[str],
        patch_sites: int,
    ) -> None:
        kwargs = {} if requested_indices is None else {"qed_cycle_indices": requested_indices}
        experiment = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_ticks=False,
            **kwargs,
        )
        circuit = experiment.circuit
        layout = experiment.layout

        measurements_per_round = {
            11: PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            14: PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
        }[patch_sites]
        assert circuit.num_qubits == patch_sites
        assert circuit.num_measurements == (
            3 * measurements_per_round
            + len(expected_indices) * KT_STABILIZER_BITS_PER_CYCLE
            + KT_STABILIZER_BITS_PER_CYCLE // 2
            + (KT_STABILIZER_BITS_PER_CYCLE // 2 if patch_sites == 14 else 0)
        )
        assert layout.preparation_round is not None
        assert len(layout.qed_cycles) == len(expected_indices)
        assert all(len(cycle) == KT_STABILIZER_BITS_PER_CYCLE for cycle in layout.qed_cycles)
        assert layout.qed_cycle_indices == expected_indices
        assert len(layout.final_stabilizers) == (KT_STABILIZER_BITS_PER_CYCLE // 2)
        assert len(layout.final_bell_partner_flags) == (3 if patch_sites == 14 else 0)
        if patch_sites == 14:
            expected_measurement_order = [
                *expected_measurement_order[:-1],
                *(["final_stab", "final_bell_partner_flags"] * 3),
            ]
        assert _measurement_record_order(experiment, expected_indices) == expected_measurement_order

    @pytest.mark.parametrize(
        ("basis", "expected_bits"),
        [("X", 3), ("Z", 3), ("BOTH", 6), ("NONE", 0)],
    )
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_final_stabilizer_basis_controls_detector_layout(
        self,
        basis: str,
        expected_bits: int,
        patch_sites: int,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis=basis,
            add_ticks=False,
        )
        assert len(experiment.layout.final_stabilizers) == expected_bits
        expected_bell_partner_bits = (
            KT_STABILIZER_BITS_PER_CYCLE // 2 if patch_sites == 14 and basis in ("X", "Z") else 0
        )
        assert len(experiment.layout.final_bell_partner_flags) == expected_bell_partner_bits

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_circuit_contains_detector_and_observable_annotations(self, patch_sites: int) -> None:
        experiment = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_ticks=False,
        )
        circuit = experiment.circuit
        detection_events, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
            32, separate_observables=True
        )

        root_check_detector_count = 2 if patch_sites == 11 else 0
        assert circuit.num_detectors == (
            len(experiment.layout.detector_positions) + root_check_detector_count
        )
        assert circuit.num_observables == 1
        assert not detection_events.any()
        assert not observable_flips.any()

    def test_unsupported_patch_sites_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="patch_sites must be 11 or 14"):
            build_kickback_tomography_experiment(patch_sites=13)


# =============================================================================
# Test Set 2: Sampled-shot parsing and postselection semantics
# =============================================================================


class TestKickbackTomographyShotAnalysis:
    """Shot analysis separates detector rejection and KT root outcomes."""

    @pytest.mark.parametrize(
        ("patch_sites", "encoder_flag_count", "round_bit_count", "round_flag_count"),
        [(11, 4, 5, 3), (14, 5, 7, 6)],
    )
    def test_measurement_record_partitions_every_output(
        self,
        patch_sites: int,
        encoder_flag_count: int,
        round_bit_count: int,
        round_flag_count: int,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            "S",
            patch_sites=patch_sites,
            measure_encoder_flags=True,
            qed_cycle_indices=KT_FULL_QED_CYCLE_INDICES,
            final_stabilizer_basis="BOTH",
            add_ticks=False,
        )
        layout = experiment.layout
        assert layout.preparation_round is not None

        qed_by_index = dict(zip(layout.qed_cycle_indices, layout.qed_cycles, strict=True))
        output_partition = (
            *layout.encoder_flags,
            *qed_by_index[1],
            *layout.preparation_round.all_positions,
            *qed_by_index[2],
            *layout.tomography_measure_round.all_positions,
            *qed_by_index[3],
            *layout.tomography_check_round.all_positions,
            *qed_by_index[4],
            *layout.final_stabilizers,
            *layout.final_bell_partner_flags,
        )
        expected_detector_positions = (
            *layout.encoder_flags,
            *layout.preparation_round.all_positions,
            *layout.tomography_measure_round.flag_positions,
            *layout.tomography_check_round.flag_positions,
            *(position for cycle in layout.qed_cycles for position in cycle),
            *layout.final_stabilizers,
            *layout.final_bell_partner_flags,
        )

        assert output_partition == tuple(range(experiment.circuit.num_measurements))
        assert len(layout.encoder_flags) == encoder_flag_count
        assert len(layout.preparation_round.all_positions) == round_bit_count
        assert len(layout.tomography_measure_round.all_positions) == round_bit_count
        assert len(layout.tomography_check_round.all_positions) == round_bit_count
        assert len(layout.tomography_measure_round.flag_positions) == round_flag_count
        if patch_sites == 14:
            assert layout.tomography_measure_round.folded_root_check is None
            assert (
                layout.tomography_measure_round.all_positions[4]
                == layout.tomography_measure_round.flag_positions[3]
            )
        assert layout.qed_cycle_indices == KT_FULL_QED_CYCLE_INDICES
        assert all(len(cycle) == KT_STABILIZER_BITS_PER_CYCLE for cycle in layout.qed_cycles)
        assert len(layout.final_stabilizers) == KT_STABILIZER_BITS_PER_CYCLE
        assert layout.detector_positions == expected_detector_positions

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_summary_correctly_postselects_and_classifies_kt_root_outcomes(
        self, patch_sites: int
    ) -> None:
        """Postselect synthetic shots and classify the two KT folded roots.

        Kickback tomography measures the logical H_XY axis twice — a "measure"
        round and a "check" round — each producing a folded root bit. The
        summary function:

        - Rejects any shot with a nonzero detector bit (encoder flags, prep,
          QED syndromes, KT flags, final stabilizer).
        - (11-site only) Rejects shots where a folded root disagrees with its
          repeated site-7 root-check bit. The 14-site patch has no root check.
        - Of the survivors: if the two roots disagree, it's a measure/check
          disagreement (retained-but-diagnostic by default, or rejected if
          ``retain_measure_check_disagreement=False``); if they agree at 0 →
          logical success; agree at 1 → logical error.
        """
        experiment = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            add_ticks=False,
        )
        layout = experiment.layout
        n_bits = experiment.circuit.num_measurements

        def shot_with(*positions: int) -> np.ndarray:
            shot = np.zeros(n_bits, dtype=bool)
            shot[list(positions)] = True
            return shot

        measurement_root = layout.tomography_measure_round.folded_root
        measurement_root_check = layout.tomography_measure_round.folded_root_check
        check_root = layout.tomography_check_round.folded_root
        check_root_check = layout.tomography_check_round.folded_root_check
        cultivation_flag = layout.tomography_measure_round.flag_positions[0]

        good_success = shot_with()
        if measurement_root_check is None:
            assert check_root_check is None
            measurement_check_for_root_one = ()
            check_check_for_root_one = ()
            rejected_round_check = shot_with(layout.tomography_measure_round.flag_positions[3])
        else:
            assert check_root_check is not None
            measurement_check_for_root_one = (measurement_root_check,)
            check_check_for_root_one = (check_root_check,)
            rejected_round_check = shot_with(measurement_root)

        good_error = shot_with(
            measurement_root,
            check_root,
            *measurement_check_for_root_one,
            *check_check_for_root_one,
        )
        measurement_only_disagreement = shot_with(
            measurement_root,
            *measurement_check_for_root_one,
        )
        check_only_disagreement = shot_with(
            check_root,
            *check_check_for_root_one,
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
                np.repeat(rejected_round_check[np.newaxis, :], 2, axis=0),
                np.repeat(
                    rejected_detector[np.newaxis, :],
                    11,
                    axis=0,
                ),
            ],
        )

        summary = summarize_kickback_tomography_shots(shots, layout)

        assert summary == KickbackTomographySummary(
            total_shots=32,
            retained_shots=19,
            rejected_shots=13,
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
        assert agreement_only_summary == KickbackTomographySummary(
            total_shots=32,
            retained_shots=10,
            rejected_shots=22,
            logical_success_shots=7,
            logical_error_shots=3,
            measure_check_disagreement_shots=0,
        )


# =============================================================================
# Test Set 3: Noiseless KT behavior
# =============================================================================


class TestNoiselessKickbackTomography:
    """Noiseless KT matches logical-Y projective-measurement probabilities."""

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_cultivated_s_state_has_zero_kt_logical_error(self, patch_sites: int) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state="S",
            patch_sites=patch_sites,
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
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_injection_only_kt_error_probability(
        self,
        state: StimInjectedStateName,
        patch_sites: int,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state=state,
            patch_sites=patch_sites,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
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

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_preparation_cultivation_projects_before_kt(self, patch_sites: int) -> None:
        """
        With preparation cultivation, the state is projected to the +1 Y-axis
        state and hence has no logical error in KT.
        """
        experiment = build_kickback_tomography_experiment(
            injected_state="1",
            patch_sites=patch_sites,
            include_preparation_cultivation=True,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            add_ticks=False,
        )
        summary = summarize_kickback_tomography_shots(
            _sample(experiment),
            experiment.layout,
        )

        assert summary.retained_shots > 0
        assert summary.logical_error_probability == 0.0
