"""Tests for Steane-code kickback tomography helpers."""

from __future__ import annotations

import math

import pytest
from qiskit_aer import AerSimulator

from msc.steane_code.kickback_tomography_common import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_FULL_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographySummary,
)
from msc.steane_code.kickback_tomography_qiskit import (
    KickbackTomographyExperiment,
    analyze_kickback_tomography_counts,
    build_kickback_tomography_experiment,
)
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    CultivatedStateName,
)
from msc.steane_code.state_injection_qiskit import InjectedStateName

NUM_SHOTS = 2000


def _measurement_register_order(experiment: KickbackTomographyExperiment) -> list[str]:
    """Return classical-register names in circuit measurement order."""

    order: list[str] = []
    for instruction in experiment.circuit.data:
        if instruction.operation.name != "measure":
            continue
        register = experiment.circuit.find_bit(instruction.clbits[0]).registers[0][0]
        if not order or order[-1] != register.name:
            order.append(register.name)
    return order


# =============================================================================
# Test Set 1: Circuit construction and layout
# =============================================================================


class TestKickbackTomographyCircuitConstruction:
    """KT builders expose the expected Qiskit registers and bit layout."""

    def test_default_patch_has_14_sites(self) -> None:
        experiment = build_kickback_tomography_experiment(add_barriers=False)
        assert experiment.circuit.num_qubits == 14

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_ty_is_the_default_cultivated_state(self, patch_sites: int) -> None:
        default_experiment = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_barriers=False,
        )
        ty_experiment = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            cultivated_state="Ty",
            add_barriers=False,
        )
        assert default_experiment == ty_experiment

    def test_unsupported_cultivated_state_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cultivated_state must be 'T' or 'Ty'"):
            build_kickback_tomography_experiment(
                cultivated_state="S",
                add_barriers=False,
            )

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
            add_barriers=False,
            **kwargs,
        )
        circuit = experiment.circuit
        layout = experiment.layout
        metadata = circuit.metadata["kickback_tomography_bit_layout"]

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
        final_measurement_count = 7 if patch_sites == 14 else 3
        assert circuit.num_clbits == (
            3 * measurements_per_round
            + len(expected_indices) * qed_measurement_count
            + final_measurement_count
        )
        assert layout.preparation_round is not None
        assert len(layout.qed_cycles) == len(expected_indices)
        assert all(len(cycle) == qed_measurement_count for cycle in layout.qed_cycles)
        assert layout.qed_cycle_indices == expected_indices
        assert len(layout.final_stabilizers) == 3
        assert len(layout.final_bell_partner_flags) == (3 if patch_sites == 14 else 0)
        assert metadata["qed_cycle_indices"] == expected_indices
        assert metadata["final_d2_copy_flag"] == layout.final_d2_copy_flag
        if patch_sites == 14:
            expected_measurement_order = [
                *expected_measurement_order[:-1],
                *(["final_stab", "final_bell_partner_flags"] * 3),
                "final_d2_copy_flag",
            ]
        assert _measurement_register_order(experiment) == expected_measurement_order

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
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis=basis,
            add_barriers=False,
        )
        assert len(experiment.layout.final_stabilizers) == expected_bits
        expected_bell_partner_bits = (
            KT_STABILIZER_BITS_PER_CYCLE // 2 if patch_sites == 14 and basis in ("X", "Z") else 0
        )
        assert len(experiment.layout.final_bell_partner_flags) == expected_bell_partner_bits

    def test_unsupported_patch_sites_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="patch_sites must be 11 or 14"):
            build_kickback_tomography_experiment(patch_sites=13)


# =============================================================================
# Test Set 2: Counts parsing and postselection semantics
# =============================================================================


class TestKickbackTomographyCountsAnalysis:
    """Counts analysis separates detector rejection and KT root outcomes."""

    @pytest.mark.parametrize(
        ("patch_sites", "encoder_flag_count", "round_bit_count", "round_flag_count"),
        [(11, 4, 5, 3), (14, 5, 7, 6)],
    )
    def test_bitstring_layout_partitions_every_output(
        self,
        patch_sites: int,
        encoder_flag_count: int,
        round_bit_count: int,
        round_flag_count: int,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            "T",
            patch_sites=patch_sites,
            measure_encoder_flags=True,
            qed_cycle_indices=KT_FULL_QED_CYCLE_INDICES,
            final_stabilizer_basis="BOTH",
            add_barriers=False,
        )
        layout = experiment.layout
        assert layout.preparation_round is not None

        output_partition = (
            *layout.encoder_flags,
            *layout.preparation_round.all_positions,
            *(position for cycle in layout.qed_cycles for position in cycle),
            *layout.tomography_measure_round.all_positions,
            *layout.tomography_check_round.all_positions,
            *layout.final_stabilizers,
            *layout.final_bell_partner_flags,
            *((layout.final_d2_copy_flag,) if layout.final_d2_copy_flag is not None else ()),
        )
        expected_detector_positions = (
            *layout.encoder_flags,
            *layout.preparation_round.all_positions,
            *layout.tomography_measure_round.flag_positions,
            *layout.tomography_check_round.flag_positions,
            *(position for cycle in layout.qed_cycles for position in cycle),
            *layout.final_stabilizers,
            *layout.final_bell_partner_flags,
            *((layout.final_d2_copy_flag,) if layout.final_d2_copy_flag is not None else ()),
        )

        assert sorted(output_partition) == list(range(experiment.circuit.num_clbits))
        assert len(set(output_partition)) == len(output_partition)
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
        qed_measurement_count = (
            PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND
            if patch_sites == 14
            else KT_STABILIZER_BITS_PER_CYCLE
        )
        assert all(len(cycle) == qed_measurement_count for cycle in layout.qed_cycles)
        assert len(layout.final_stabilizers) == KT_STABILIZER_BITS_PER_CYCLE
        assert layout.detector_positions == expected_detector_positions

    @pytest.mark.parametrize("patch_sites", [11, 14])
    @pytest.mark.parametrize("no_reset", [False, True])
    def test_summary_correctly_postselects_and_classifies_kt_root_outcomes(
        self,
        patch_sites: int,
        no_reset: bool,
    ) -> None:
        """Postselect synthetic counts and classify the two KT folded roots.

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
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            no_reset=no_reset,
            add_barriers=False,
        )
        layout = experiment.layout

        n_bits = experiment.circuit.num_clbits

        def bitstring_with(*positions: int) -> str:
            bits = ["0"] * n_bits
            for position in positions:
                bits[n_bits - 1 - position] = "1"
            return "".join(bits)

        measurement_root = layout.tomography_measure_round.folded_root
        measurement_root_check = layout.tomography_measure_round.folded_root_check
        check_root = layout.tomography_check_round.folded_root
        check_root_check = layout.tomography_check_round.folded_root_check
        cultivation_flag = layout.tomography_measure_round.flag_positions[0]

        good_success = bitstring_with()
        if measurement_root_check is None:
            assert check_root_check is None
            measurement_check_for_root_one = ()
            check_check_for_root_one = ()
            rejected_round_check = bitstring_with(layout.tomography_measure_round.flag_positions[3])
        else:
            assert check_root_check is not None
            measurement_check_for_root_one = (
                (measurement_root_check,)
                if layout.tomography_measure_round.reset_before_root_check
                else ()
            )
            check_check_for_root_one = (
                (check_root_check,) if layout.tomography_check_round.reset_before_root_check else ()
            )
            rejected_round_check = bitstring_with(
                measurement_root,
                *(
                    ()
                    if layout.tomography_measure_round.reset_before_root_check
                    else (measurement_root_check,)
                ),
            )

        good_error = bitstring_with(
            measurement_root,
            check_root,
            *measurement_check_for_root_one,
            *check_check_for_root_one,
        )
        measurement_only_disagreement = bitstring_with(
            measurement_root,
            *measurement_check_for_root_one,
        )
        check_only_disagreement = bitstring_with(
            check_root,
            *check_check_for_root_one,
        )
        rejected_detector = bitstring_with(cultivation_flag)

        counts = {
            good_success: 7,
            good_error: 3,
            measurement_only_disagreement: 5,
            check_only_disagreement: 4,
            rejected_round_check: 2,
            rejected_detector: 11,
        }

        summary = analyze_kickback_tomography_counts(counts, layout)

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

        agreement_only_summary = analyze_kickback_tomography_counts(
            counts,
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
    """Noiseless KT matches the selected magic-state-axis measurements."""

    @pytest.mark.parametrize("cultivated_state", ["T", "Ty"])
    @pytest.mark.parametrize("patch_sites", [11, 14])
    @pytest.mark.parametrize("no_reset", [False, True])
    def test_cultivated_magic_state_has_zero_kt_logical_error(
        self,
        cultivated_state: CultivatedStateName,
        patch_sites: int,
        no_reset: bool,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            cultivated_state=cultivated_state,
            patch_sites=patch_sites,
            no_reset=no_reset,
            add_barriers=False,
        )
        shots = 100 if patch_sites == 14 and no_reset else 10
        counts = (
            AerSimulator()
            .run(experiment.circuit, shots=shots, seed_simulator=42)
            .result()
            .get_counts()
        )
        summary = analyze_kickback_tomography_counts(counts, experiment.layout)

        if patch_sites == 14 and no_reset:
            assert summary.retained_shots > 0
        else:
            assert summary.retained_shots == shots
        assert summary.logical_error_shots == 0
        assert summary.logical_error_probability == 0.0

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("cultivated_state", "injection_argument", "injection", "expected_error"),
        [
            ("T", "injected_state", "Ty", 0.25),
            ("T", "injected_state", "+", (1.0 - 1.0 / math.sqrt(2.0)) / 2.0),
            ("T", "injected_state", "1", 0.5),
            ("T", "injected_state", "Y+", (1.0 - 1.0 / math.sqrt(2.0)) / 2.0),
            ("Ty", "injected_state", "T", 0.25),
            ("Ty", "injected_state", "+", (1.0 - 1.0 / math.sqrt(2.0)) / 2.0),
            ("Ty", "injected_state", "1", (1.0 + 1.0 / math.sqrt(2.0)) / 2.0),
            ("Ty", "injected_state", "Y+", 0.5),
            ("T", "injection_azimuth_angle", math.pi / 4, 0.0),
            ("T", "injection_azimuth_angle", 3 * math.pi / 4, 0.5),
            ("T", "injection_azimuth_angle", 5 * math.pi / 4, 1.0),
            ("Ty", "injection_polar_angle", math.pi / 4, 0.0),
            ("Ty", "injection_polar_angle", 3 * math.pi / 4, 0.5),
            ("Ty", "injection_polar_angle", 5 * math.pi / 4, 1.0),
        ],
    )
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_injection_only_kt_error_tracks_the_cultivated_axis_overlap(
        self,
        cultivated_state: CultivatedStateName,
        injection_argument: str,
        injection: InjectedStateName | float,
        expected_error: float,
        patch_sites: int,
    ) -> None:
        """Report one minus the injected state's overlap with the cultivated axis."""
        experiment = build_kickback_tomography_experiment(
            cultivated_state=cultivated_state,
            patch_sites=patch_sites,
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            add_barriers=False,
            **{injection_argument: injection},
        )
        counts = (
            AerSimulator()
            .run(experiment.circuit, shots=NUM_SHOTS, seed_simulator=42)
            .result()
            .get_counts()
        )
        summary = analyze_kickback_tomography_counts(counts, experiment.layout)

        sigma = math.sqrt(expected_error * (1.0 - expected_error) / NUM_SHOTS)

        assert summary.retained_shots == NUM_SHOTS
        assert summary.logical_error_probability == pytest.approx(
            expected_error, abs=max(4 * sigma, 0.02)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("cultivated_state", ["T", "Ty"])
    @pytest.mark.parametrize("patch_sites", [11, 14])
    @pytest.mark.parametrize("no_reset", [False, True])
    def test_preparation_cultivation_projects_before_kt(
        self,
        cultivated_state: CultivatedStateName,
        patch_sites: int,
        no_reset: bool,
    ) -> None:
        """
        With preparation cultivation, the state is projected to the selected
        +1 magic-state axis and hence has no logical error in KT.
        """
        experiment = build_kickback_tomography_experiment(
            injected_state="1",
            cultivated_state=cultivated_state,
            patch_sites=patch_sites,
            include_prep_cultivation=True,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            no_reset=no_reset,
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=NUM_SHOTS).result().get_counts()
        summary = analyze_kickback_tomography_counts(counts, experiment.layout)

        assert summary.retained_shots > 0
        assert summary.logical_error_probability == 0.0
