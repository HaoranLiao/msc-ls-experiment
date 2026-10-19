"""Tests for Steane-code kickback tomography helpers in Stim."""

from __future__ import annotations

import math

import numpy as np
import pytest
import stim

from msc.steane_code.kickback_tomography_common import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_FULL_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographyBitLayout,
    KickbackTomographySummary,
    resolve_kickback_tomography_patch_spec,
)
from msc.steane_code.kickback_tomography_stim import (
    analyze_kickback_tomography_shots,
    build_kickback_tomography_experiment,
)
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
)
from msc.steane_code.state_injection_common import (
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_14_SITE_DEFAULT_LOGICAL_X,
)
from msc.steane_code.state_injection_stim import StimInjectedStateName

NUM_SHOTS = 2000


def _measurement_record_order(
    circuit: stim.Circuit,
    bit_layout: KickbackTomographyBitLayout,
    qed_cycle_indices: tuple[int, ...],
) -> list[str]:
    """Return measurement-stage names in Stim record order."""
    label_by_position: dict[int, str] = {}

    def label_positions(label: str, positions: tuple[int, ...]) -> None:
        for position in positions:
            label_by_position[position] = label

    if bit_layout.preparation_round is not None:
        label_positions("prep_cult", bit_layout.preparation_round.all_positions)
    label_positions(
        "kt_measure",
        bit_layout.tomography_measure_round.all_positions,
    )
    label_positions(
        "kt_check",
        bit_layout.tomography_check_round.all_positions,
    )
    for cycle_index, positions in zip(
        qed_cycle_indices,
        bit_layout.qed_cycles,
        strict=True,
    ):
        label_positions(f"qed_{cycle_index}", positions)
    label_positions(
        "final_readout",
        tuple({position for face in bit_layout.final_readout_faces for position in face}),
    )

    order: list[str] = []
    for position in range(circuit.num_measurements):
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
    circuit: stim.Circuit,
    shots: int = NUM_SHOTS,
) -> np.ndarray:
    """Sample a Stim KT circuit."""
    return circuit.compile_sampler().sample(shots)


# =============================================================================
# Test Set 1: Circuit construction and bit_layout
# =============================================================================


class TestKickbackTomographyCircuitConstruction:
    """KT builders expose the expected Stim measurement-record bit_layout."""

    def test_default_patch_has_14_sites(self) -> None:
        circuit, _ = build_kickback_tomography_experiment(add_ticks=False)
        assert circuit.num_qubits == 14

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_s_is_the_default_cultivated_state(self, patch_sites: int) -> None:
        default_build = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_ticks=False,
        )
        s_build = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            cultivated_state="S",
            add_ticks=False,
        )
        assert default_build == s_build

    def test_non_s_cultivated_state_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cultivated_state must be 'S' in Stim"):
            build_kickback_tomography_experiment(cultivated_state="T", add_ticks=False)

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
                    "qed_4",
                    "final_readout",
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
                    "final_readout",
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
                    "final_readout",
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
        circuit, bit_layout = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_ticks=False,
            **kwargs,
        )

        measurements_per_round = {
            11: PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            14: PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
        }[patch_sites]
        assert circuit.num_qubits == patch_sites
        qed_measurement_count = (
            PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND
            if patch_sites == 14
            else KT_STABILIZER_BITS_PER_CYCLE
        )
        final_measurement_count = 7
        assert circuit.num_measurements == (
            3 * measurements_per_round
            + len(expected_indices) * qed_measurement_count
            + final_measurement_count
        )
        assert bit_layout.preparation_round is not None
        assert len(bit_layout.qed_cycles) == len(expected_indices)
        assert all(len(cycle) == qed_measurement_count for cycle in bit_layout.qed_cycles)
        assert bit_layout.qed_cycle_indices == expected_indices
        assert len(bit_layout.final_readout_faces) == 3
        assert (
            _measurement_record_order(circuit, bit_layout, expected_indices)
            == expected_measurement_order
        )

    @pytest.mark.parametrize("basis", ["X", "Z"])
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_final_readout_checks_every_stabilizer_face(
        self,
        basis: str,
        patch_sites: int,
    ) -> None:
        _, bit_layout = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_readout_basis=basis,
            add_ticks=False,
        )
        assert len(bit_layout.final_readout_faces) == 3
        assert all(len(face) == 4 for face in bit_layout.final_readout_faces)

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_circuit_contains_detector_and_observable_annotations(self, patch_sites: int) -> None:
        circuit, bit_layout = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_ticks=False,
        )
        detection_events, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
            32, separate_observables=True
        )

        root_check_detector_count = 2
        assert circuit.num_detectors == (
            len(bit_layout.detector_positions)
            + len(bit_layout.final_readout_faces)
            + root_check_detector_count
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
        (
            "patch_sites",
            "encoder_flag_count",
            "round_bit_count",
            "round_flag_count",
            "root_check_index",
        ),
        [(11, 4, 5, 3, 4), (14, 5, 6, 4, 3)],
    )
    def test_measurement_record_partitions_every_output(
        self,
        patch_sites: int,
        encoder_flag_count: int,
        round_bit_count: int,
        round_flag_count: int,
        root_check_index: int,
    ) -> None:
        circuit, bit_layout = build_kickback_tomography_experiment(
            "S",
            patch_sites=patch_sites,
            measure_encoder_flags=True,
            qed_cycle_indices=KT_FULL_QED_CYCLE_INDICES,
            final_readout_basis="X",
            add_ticks=False,
        )
        assert bit_layout.preparation_round is not None

        qed_by_index = dict(zip(bit_layout.qed_cycle_indices, bit_layout.qed_cycles, strict=True))
        final_readout_positions = sorted(
            {position for face in bit_layout.final_readout_faces for position in face}
        )
        output_partition = (
            *bit_layout.encoder_flags,
            *qed_by_index[1],
            *bit_layout.preparation_round.all_positions,
            *qed_by_index[2],
            *bit_layout.tomography_measure_round.all_positions,
            *qed_by_index[3],
            *bit_layout.tomography_check_round.all_positions,
            *qed_by_index[4],
            *final_readout_positions,
        )
        expected_detector_positions = (
            *bit_layout.encoder_flags,
            *bit_layout.preparation_round.all_positions,
            *bit_layout.tomography_measure_round.flag_positions,
            *bit_layout.tomography_check_round.flag_positions,
            *(position for cycle in bit_layout.qed_cycles for position in cycle),
        )

        assert output_partition == tuple(range(circuit.num_measurements))
        assert len(bit_layout.encoder_flags) == encoder_flag_count
        assert len(bit_layout.preparation_round.all_positions) == round_bit_count
        assert len(bit_layout.tomography_measure_round.all_positions) == round_bit_count
        assert len(bit_layout.tomography_check_round.all_positions) == round_bit_count
        assert len(bit_layout.tomography_measure_round.flag_positions) == round_flag_count
        assert (
            bit_layout.tomography_measure_round.all_positions[root_check_index]
            == bit_layout.tomography_measure_round.folded_root_check
        )
        assert bit_layout.qed_cycle_indices == KT_FULL_QED_CYCLE_INDICES
        qed_measurement_count = (
            PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND
            if patch_sites == 14
            else KT_STABILIZER_BITS_PER_CYCLE
        )
        assert all(len(cycle) == qed_measurement_count for cycle in bit_layout.qed_cycles)
        assert len(final_readout_positions) == 7
        assert len(bit_layout.final_readout_faces) == 3
        assert bit_layout.detector_positions == expected_detector_positions

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_summary_correctly_postselects_and_classifies_kt_root_outcomes(
        self, patch_sites: int
    ) -> None:
        """Postselect synthetic shots and classify the two KT folded roots.

        Kickback tomography measures the logical H_XY axis twice — a "measure"
        round and a "check" round — each producing a folded root bit. The
        summary function:

        - Rejects any shot with a nonzero detector bit (encoder flags, prep,
          QED syndromes, KT flags) or an odd final readout face parity.
        - Rejects shots where a folded root disagrees with its repeated
          helper-root check.
        - Of the survivors: if the two roots disagree, it's a measure/check
          disagreement, rejected and counted separately; if they agree at 0 →
          logical success; agree at 1 → logical error.
        """
        circuit, bit_layout = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_readout_basis="X",
            add_ticks=False,
        )
        n_bits = circuit.num_measurements

        def shot_with(*positions: int) -> np.ndarray:
            shot = np.zeros(n_bits, dtype=bool)
            shot[list(positions)] = True
            return shot

        measurement_root = bit_layout.tomography_measure_round.folded_root
        measurement_root_check = bit_layout.tomography_measure_round.folded_root_check
        check_root = bit_layout.tomography_check_round.folded_root
        check_root_check = bit_layout.tomography_check_round.folded_root_check
        cultivation_flag = bit_layout.tomography_measure_round.flag_positions[0]

        good_success = shot_with()
        if measurement_root_check is None:
            assert check_root_check is None
            measurement_check_for_root_one = ()
            check_check_for_root_one = ()
            rejected_round_check = shot_with(bit_layout.tomography_measure_round.flag_positions[3])
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

        # The final readout records the data sites in order. One flipped data bit
        # makes its faces odd; flipping a logical support keeps every face even.
        data_sites = resolve_kickback_tomography_patch_spec(patch_sites).data_sites
        position_by_site = dict(
            zip(
                data_sites,
                sorted({position for face in bit_layout.final_readout_faces for position in face}),
                strict=True,
            )
        )
        logical_support = {
            11: PATCH_11_SITE_DEFAULT_LOGICAL_X,
            14: PATCH_14_SITE_DEFAULT_LOGICAL_X,
        }[patch_sites]
        odd_face_parity = shot_with(position_by_site[logical_support[0]])
        even_face_parities = shot_with(*(position_by_site[site] for site in logical_support))

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
                np.repeat(odd_face_parity[np.newaxis, :], 6, axis=0),
                np.repeat(even_face_parities[np.newaxis, :], 2, axis=0),
            ],
        )

        summary = analyze_kickback_tomography_shots(shots, bit_layout)

        assert summary == KickbackTomographySummary(
            total_shots=40,
            retained_shots=12,
            rejected_shots=28,
            logical_success_shots=9,
            logical_error_shots=3,
            measure_check_disagreement_shots=9,
        )
        assert summary.logical_error_probability == pytest.approx(3 / 12)
        assert summary.fidelity_lower_bound == pytest.approx(9 / 12)


# =============================================================================
# Test Set 3: Noiseless KT behavior
# =============================================================================


class TestNoiselessKickbackTomography:
    """Noiseless KT matches logical-Y projective-measurement probabilities."""

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_cultivated_s_state_has_zero_kt_logical_error(self, patch_sites: int) -> None:
        circuit, bit_layout = build_kickback_tomography_experiment(
            injected_state="S",
            patch_sites=patch_sites,
            add_ticks=False,
        )
        summary = analyze_kickback_tomography_shots(
            _sample(circuit, shots=10),
            bit_layout,
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
        circuit, bit_layout = build_kickback_tomography_experiment(
            injected_state=state,
            patch_sites=patch_sites,
            include_preparation_cultivation=False,
            qed_cycle_indices=(),
            final_readout_basis="X",
            add_ticks=False,
        )
        summary = analyze_kickback_tomography_shots(
            _sample(circuit),
            bit_layout,
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
    def test_preparation_cultivation_projects_before_kt(
        self,
        patch_sites: int,
    ) -> None:
        """
        With preparation cultivation, the state is projected to the +1 Y-axis
        state and hence has no logical error in KT.
        """
        circuit, bit_layout = build_kickback_tomography_experiment(
            injected_state="1",
            patch_sites=patch_sites,
            include_preparation_cultivation=True,
            qed_cycle_indices=(),
            final_readout_basis="X",
            add_ticks=False,
        )
        summary = analyze_kickback_tomography_shots(
            _sample(circuit),
            bit_layout,
        )

        assert summary.retained_shots > 0
        assert summary.logical_error_probability == 0.0
