"""Tests for Steane-code kickback tomography helpers."""

from __future__ import annotations

import math

import pytest
from qiskit_aer import AerSimulator

from msc.steane_code.kickback_tomography_qiskit import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_FULL_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographyExperiment,
    KickbackTomographySummary,
    build_kickback_tomography_experiment,
    kickback_tomography_shot_is_retained,
    summarize_kickback_tomography_counts,
)
from msc.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
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


def _expected_h_xy_axis_error_probability(state: InjectedStateName) -> float:
    """Return probability of the -1 H_XY-axis outcome for a named input state."""

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    h_xy_plus_probs: dict[InjectedStateName, float] = {
        "0": 0.5,
        "1": 0.5,
        "+": (1.0 + inv_sqrt2) / 2.0,
        "-": (1.0 - inv_sqrt2) / 2.0,
        "i": (1.0 + inv_sqrt2) / 2.0,
        "Y+": (1.0 + inv_sqrt2) / 2.0,
        "S": (1.0 + inv_sqrt2) / 2.0,
        "-i": (1.0 - inv_sqrt2) / 2.0,
        "Y-": (1.0 - inv_sqrt2) / 2.0,
        "T": 1.0,
        "Tdg": 0.5,
        "Ty": 0.75,
    }
    return 1.0 - h_xy_plus_probs[state]


# =============================================================================
# Test Set 1: Circuit construction and layout
# =============================================================================


class TestKickbackTomographyCircuitConstruction:
    """KT builders expose the expected Qiskit registers and bit layout."""

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
            add_barriers=False,
            **kwargs,
        )
        circuit = experiment.circuit
        layout = experiment.layout
        metadata = circuit.metadata["kickback_tomography_bit_layout"]

        assert circuit.num_qubits == 11
        assert circuit.num_clbits == (
            3 * CULTIVATION_MEASUREMENTS_PER_ROUND
            + len(expected_indices) * KT_STABILIZER_BITS_PER_CYCLE
            + KT_STABILIZER_BITS_PER_CYCLE // 2
        )
        assert layout.preparation_round is not None
        assert len(layout.qed_cycles) == len(expected_indices)
        assert all(len(cycle) == 6 for cycle in layout.qed_cycles)
        assert layout.qed_cycle_indices == expected_indices
        assert len(layout.final_stabilizers) == 3
        assert metadata["qed_cycle_indices"] == expected_indices
        assert _measurement_register_order(experiment) == expected_measurement_order

    @pytest.mark.parametrize(
        ("basis", "expected_bits"),
        [("X", 3), ("Z", 3), ("BOTH", 6), ("NONE", 0)],
    )
    def test_final_stabilizer_basis_controls_detector_layout(
        self,
        basis: str,
        expected_bits: int,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis=basis,
            add_barriers=False,
        )

        assert len(experiment.layout.final_stabilizers) == expected_bits


# =============================================================================
# Test Set 2: Counts parsing and postselection semantics
# =============================================================================


class TestKickbackTomographyCountsAnalysis:
    """Counts analysis separates detector rejection and KT root outcomes."""

    @pytest.mark.parametrize("no_reset", [False, True])
    def test_summary_classifies_retained_root_outcomes(self, no_reset: bool) -> None:
        experiment = build_kickback_tomography_experiment(
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
        measurement_root_double_check = layout.tomography_measure_round.folded_root_double_check
        check_root = layout.tomography_check_round.folded_root
        check_root_double_check = layout.tomography_check_round.folded_root_double_check
        cultivation_flag = layout.tomography_measure_round.flag_positions[0]

        good_success = bitstring_with()
        expected_double_checks = (
            () if no_reset else (measurement_root_double_check, check_root_double_check)
        )
        good_error = bitstring_with(measurement_root, check_root, *expected_double_checks)
        measurement_only_disagreement = bitstring_with(
            measurement_root,
            *((measurement_root_double_check,) if not no_reset else ()),
        )
        check_only_disagreement = bitstring_with(
            check_root,
            *((check_root_double_check,) if not no_reset else ()),
        )
        rejected_double_check = bitstring_with(
            measurement_root,
            *((measurement_root_double_check,) if no_reset else ()),
        )
        rejected_detector = bitstring_with(cultivation_flag)

        counts = {
            good_success: 7,
            good_error: 3,
            measurement_only_disagreement: 5,
            check_only_disagreement: 4,
            rejected_detector: 11,
        }

        summary = summarize_kickback_tomography_counts(counts, layout)

        assert kickback_tomography_shot_is_retained(good_success, layout)
        assert kickback_tomography_shot_is_retained(good_error, layout)
        assert kickback_tomography_shot_is_retained(measurement_only_disagreement, layout)
        assert kickback_tomography_shot_is_retained(check_only_disagreement, layout)
        assert not kickback_tomography_shot_is_retained(rejected_double_check, layout)
        assert not kickback_tomography_shot_is_retained(rejected_detector, layout)
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

        agreement_only_summary = summarize_kickback_tomography_counts(
            counts,
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
    """Noiseless KT matches H_XY projective-measurement probabilities."""

    @pytest.mark.parametrize("no_reset", [False, True])
    def test_cultivated_t_state_has_zero_kt_logical_error(self, no_reset: bool) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state="T",
            no_reset=no_reset,
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=10).result().get_counts()
        summary = summarize_kickback_tomography_counts(counts, experiment.layout)

        assert summary.retained_shots == 10
        assert summary.logical_error_shots == 0
        assert summary.retainment_rate == 1.0
        assert summary.logical_error_probability == 0.0

    @pytest.mark.parametrize("state", ["Ty", "+", "1", "Y+", "T"])
    @pytest.mark.parametrize("no_reset", [False, True])
    def test_injection_only_kt_error_probability(
        self,
        state: InjectedStateName,
        no_reset: bool,
    ) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state=state,
            include_prep_cultivation=False,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            no_reset=no_reset,
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=NUM_SHOTS).result().get_counts()
        summary = summarize_kickback_tomography_counts(counts, experiment.layout)

        expected_error = _expected_h_xy_axis_error_probability(state)
        sigma = math.sqrt(expected_error * (1.0 - expected_error) / NUM_SHOTS)
        tolerance = max(4 * sigma, 0.02)

        assert summary.retained_shots == NUM_SHOTS
        assert summary.logical_error_probability == pytest.approx(expected_error, abs=tolerance)

    @pytest.mark.parametrize("no_reset", [False, True])
    def test_preparation_cultivation_projects_before_kt(self, no_reset: bool) -> None:
        """
        With preparation cultivation, the state is projected to the +1 H_XY
        state and hence has no logical error in KT.
        """
        experiment = build_kickback_tomography_experiment(
            injected_state="1",
            include_prep_cultivation=True,
            qed_cycle_indices=(),
            final_stabilizer_basis="NONE",
            no_reset=no_reset,
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=NUM_SHOTS).result().get_counts()
        summary = summarize_kickback_tomography_counts(counts, experiment.layout)

        assert summary.retained_shots > 0
        assert summary.logical_error_probability == 0.0
