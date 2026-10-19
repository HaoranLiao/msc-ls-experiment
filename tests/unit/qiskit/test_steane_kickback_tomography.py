"""Tests for Steane-code kickback tomography helpers."""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from msc.steane_code.kickback_tomography_common import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_FULL_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographySummary,
    resolve_kickback_tomography_patch_spec,
)
from msc.steane_code.kickback_tomography_qiskit import (
    analyze_kickback_tomography_counts,
    build_kickback_tomography_experiment,
)
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    CultivatedStateName,
)
from msc.steane_code.state_injection_common import (
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_14_SITE_DEFAULT_LOGICAL_X,
)
from msc.steane_code.state_injection_qiskit import InjectedStateName

NUM_SHOTS = 2000


def _measurement_register_order(circuit: QuantumCircuit) -> list[str]:
    """Return classical-register names in circuit measurement order."""

    order: list[str] = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        register = circuit.find_bit(instruction.clbits[0]).registers[0][0]
        if not order or order[-1] != register.name:
            order.append(register.name)
    return order


# =============================================================================
# Test Set 1: Circuit construction and bit_layout
# =============================================================================


class TestKickbackTomographyCircuitConstruction:
    """KT builders expose the expected Qiskit registers and bit layout."""

    def test_default_patch_has_14_sites(self) -> None:
        circuit, _ = build_kickback_tomography_experiment(add_barriers=False)
        assert circuit.num_qubits == 14

    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_ty_is_the_default_cultivated_state(self, patch_sites: int) -> None:
        default_build = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            add_barriers=False,
        )
        ty_build = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            cultivated_state="Ty",
            add_barriers=False,
        )
        assert default_build == ty_build

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
            final_readout_basis="X",
            add_barriers=False,
            **kwargs,
        )
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
        final_measurement_count = 7
        assert circuit.num_clbits == (
            3 * measurements_per_round
            + len(expected_indices) * qed_measurement_count
            + final_measurement_count
        )
        assert bit_layout.preparation_round is not None
        assert len(bit_layout.qed_cycles) == len(expected_indices)
        assert all(len(cycle) == qed_measurement_count for cycle in bit_layout.qed_cycles)
        assert bit_layout.qed_cycle_indices == expected_indices
        assert len(bit_layout.final_readout_faces) == 3
        assert metadata["qed_cycle_indices"] == expected_indices
        assert metadata["final_readout_faces"] == bit_layout.final_readout_faces
        assert _measurement_register_order(circuit) == expected_measurement_order

    @pytest.mark.parametrize("basis", ["X", "Z"])
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_final_readout_checks_every_stabilizer_face(
        self,
        basis: str,
        patch_sites: int,
    ) -> None:
        _, bit_layout = build_kickback_tomography_experiment(
            patch_sites=patch_sites,
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_readout_basis=basis,
            add_barriers=False,
        )
        assert len(bit_layout.final_readout_faces) == 3
        assert all(len(face) == 4 for face in bit_layout.final_readout_faces)

    def test_unsupported_patch_sites_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="patch_sites must be 11 or 14"):
            build_kickback_tomography_experiment(patch_sites=13)


# =============================================================================
# Test Set 2: Counts parsing and postselection semantics
# =============================================================================


class TestKickbackTomographyCountsAnalysis:
    """Counts analysis separates detector rejection and KT root outcomes."""

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
    def test_bitstring_layout_partitions_every_output(
        self,
        patch_sites: int,
        encoder_flag_count: int,
        round_bit_count: int,
        round_flag_count: int,
        root_check_index: int,
    ) -> None:
        circuit, bit_layout = build_kickback_tomography_experiment(
            "T",
            patch_sites=patch_sites,
            measure_encoder_flags=True,
            qed_cycle_indices=KT_FULL_QED_CYCLE_INDICES,
            final_readout_basis="X",
            add_barriers=False,
        )
        assert bit_layout.preparation_round is not None
        final_readout_positions = sorted(
            {position for face in bit_layout.final_readout_faces for position in face}
        )

        output_partition = (
            *bit_layout.encoder_flags,
            *bit_layout.preparation_round.all_positions,
            *(position for cycle in bit_layout.qed_cycles for position in cycle),
            *bit_layout.tomography_measure_round.all_positions,
            *bit_layout.tomography_check_round.all_positions,
            *final_readout_positions,
        )
        expected_detector_positions = (
            *bit_layout.encoder_flags,
            *bit_layout.preparation_round.all_positions,
            *bit_layout.tomography_measure_round.flag_positions,
            *bit_layout.tomography_check_round.flag_positions,
            *(position for cycle in bit_layout.qed_cycles for position in cycle),
        )

        assert sorted(output_partition) == list(range(circuit.num_clbits))
        assert len(set(output_partition)) == len(output_partition)
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
        self,
        patch_sites: int,
    ) -> None:
        """Postselect synthetic counts and classify the two KT folded roots.

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
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_readout_basis="X",
            add_barriers=False,
        )

        n_bits = circuit.num_clbits

        def bitstring_with(*positions: int) -> str:
            bits = ["0"] * n_bits
            for position in positions:
                bits[n_bits - 1 - position] = "1"
            return "".join(bits)

        measurement_root = bit_layout.tomography_measure_round.folded_root
        measurement_root_check = bit_layout.tomography_measure_round.folded_root_check
        check_root = bit_layout.tomography_check_round.folded_root
        check_root_check = bit_layout.tomography_check_round.folded_root_check
        cultivation_flag = bit_layout.tomography_measure_round.flag_positions[0]

        good_success = bitstring_with()
        if measurement_root_check is None:
            assert check_root_check is None
            measurement_check_for_root_one = ()
            check_check_for_root_one = ()
            rejected_round_check = bitstring_with(
                bit_layout.tomography_measure_round.flag_positions[3]
            )
        else:
            assert check_root_check is not None
            measurement_check_for_root_one = (measurement_root_check,)
            check_check_for_root_one = (check_root_check,)
            rejected_round_check = bitstring_with(measurement_root)

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
        odd_face_parity = bitstring_with(position_by_site[logical_support[0]])
        even_face_parities = bitstring_with(*(position_by_site[site] for site in logical_support))

        counts = {
            good_success: 7,
            good_error: 3,
            measurement_only_disagreement: 5,
            check_only_disagreement: 4,
            rejected_round_check: 2,
            rejected_detector: 11,
            odd_face_parity: 6,
            even_face_parities: 2,
        }

        summary = analyze_kickback_tomography_counts(counts, bit_layout)

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
    """Noiseless KT matches the selected magic-state-axis measurements."""

    @pytest.mark.parametrize("cultivated_state", ["T", "Ty"])
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_cultivated_magic_state_has_zero_kt_logical_error(
        self,
        cultivated_state: CultivatedStateName,
        patch_sites: int,
    ) -> None:
        circuit, bit_layout = build_kickback_tomography_experiment(
            cultivated_state=cultivated_state,
            patch_sites=patch_sites,
            add_barriers=False,
        )
        shots = 10
        counts = AerSimulator().run(circuit, shots=shots, seed_simulator=42).result().get_counts()
        summary = analyze_kickback_tomography_counts(counts, bit_layout)

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
        circuit, bit_layout = build_kickback_tomography_experiment(
            cultivated_state=cultivated_state,
            patch_sites=patch_sites,
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_readout_basis="X",
            add_barriers=False,
            **{injection_argument: injection},
        )
        counts = (
            AerSimulator().run(circuit, shots=NUM_SHOTS, seed_simulator=42).result().get_counts()
        )
        summary = analyze_kickback_tomography_counts(counts, bit_layout)

        sigma = math.sqrt(expected_error * (1.0 - expected_error) / NUM_SHOTS)

        assert summary.retained_shots == NUM_SHOTS
        assert summary.logical_error_probability == pytest.approx(
            expected_error, abs=max(4 * sigma, 0.02)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("cultivated_state", ["T", "Ty"])
    @pytest.mark.parametrize("patch_sites", [11, 14])
    def test_preparation_cultivation_projects_before_kt(
        self,
        cultivated_state: CultivatedStateName,
        patch_sites: int,
    ) -> None:
        """
        With preparation cultivation, the state is projected to the selected
        +1 magic-state axis and hence has no logical error in KT.
        """
        circuit, bit_layout = build_kickback_tomography_experiment(
            injected_state="1",
            cultivated_state=cultivated_state,
            patch_sites=patch_sites,
            include_prep_cultivation=True,
            qed_cycle_indices=(),
            final_readout_basis="X",
            add_barriers=False,
        )
        counts = AerSimulator().run(circuit, shots=NUM_SHOTS).result().get_counts()
        summary = analyze_kickback_tomography_counts(counts, bit_layout)

        assert summary.retained_shots > 0
        assert summary.logical_error_probability == 0.0
