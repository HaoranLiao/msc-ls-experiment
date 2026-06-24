"""Tests for Steane-code kickback tomography helpers."""

from __future__ import annotations

import math

import pytest
from qiskit_aer import AerSimulator

from research_qec_haoran.steane_code.kickback_tomography_qiskit import (
    KT_CULTIVATION_BITS_PER_ROUND,
    KT_STABILIZER_BITS_PER_CYCLE,
    KickbackTomographySummary,
    bit_value,
    build_kickback_tomography_experiment,
    postselect_kickback_tomography_counts,
    summarize_kickback_tomography_counts,
)
from research_qec_haoran.steane_code.state_injection_qiskit import InjectedStateName

NUM_SHOTS = 2000


def _expected_h_axis_error_probability(state: InjectedStateName) -> float:
    """Return probability of the -1 H-axis outcome for a named input state."""

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    h_plus_probs: dict[InjectedStateName, float] = {
        "H": 1.0,
        "0": (1.0 + inv_sqrt2) / 2.0,
        "1": (1.0 - inv_sqrt2) / 2.0,
        "+": (1.0 + inv_sqrt2) / 2.0,
        "-": (1.0 - inv_sqrt2) / 2.0,
        "Y+": 0.5,
        "Y-": 0.5,
        "i": 0.5,
        "-i": 0.5,
        "S": 0.5,
        "T": 0.75,
        "Tdg": 0.75,
    }
    return 1.0 - h_plus_probs[state]


# =============================================================================
# Test Set 1: Circuit construction and layout
# =============================================================================


class TestKickbackTomographyCircuitConstruction:
    """KT builders expose the expected Qiskit registers and bit layout."""

    def test_full_kt_circuit_layout(self) -> None:
        experiment = build_kickback_tomography_experiment(add_barriers=False)
        circuit = experiment.circuit
        layout = experiment.layout

        assert circuit.num_qubits == 17
        assert circuit.num_clbits == (
            3 * KT_CULTIVATION_BITS_PER_ROUND
            + 3 * KT_STABILIZER_BITS_PER_CYCLE
            + KT_STABILIZER_BITS_PER_CYCLE
        )

        assert layout.preparation is not None
        assert len(layout.preparation.all_positions) == 5
        assert len(layout.tomography_measurement.all_positions) == 5
        assert len(layout.tomography_check.all_positions) == 5
        assert len(layout.qec_cycles) == 3
        assert all(len(cycle) == 6 for cycle in layout.qec_cycles)
        assert len(layout.final_stabilizers) == 6

    def test_injection_only_without_qec_layout(self) -> None:
        experiment = build_kickback_tomography_experiment(
            include_preparation_cultivation=False,
            qec_cycle_indices=(),
            final_stabilizer_basis="none",
            add_barriers=False,
        )
        layout = experiment.layout

        assert layout.preparation is None
        assert layout.qec_cycles == ()
        assert layout.final_stabilizers == ()
        assert experiment.circuit.num_clbits == 2 * KT_CULTIVATION_BITS_PER_ROUND

    def test_invalid_qec_cycle_indices_raise(self) -> None:
        with pytest.raises(ValueError, match="0, 1, and 2"):
            build_kickback_tomography_experiment(qec_cycle_indices=(3,))

    def test_duplicate_qec_cycle_indices_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            build_kickback_tomography_experiment(qec_cycle_indices=(0, 0))

    def test_qec_cycle_registers_follow_protocol_order(self) -> None:
        experiment = build_kickback_tomography_experiment(
            include_preparation_cultivation=False,
            qec_cycle_indices=(2, 0),
            final_stabilizer_basis="none",
            add_barriers=False,
        )

        assert [register.name for register in experiment.circuit.cregs] == [
            "qec_0",
            "qec_2",
            "kt_measure",
            "kt_check",
        ]


# =============================================================================
# Test Set 2: Counts parsing and postselection semantics
# =============================================================================


class TestKickbackTomographyCountsParsing:
    """Counts helpers use Qiskit's rightmost-bit convention."""

    def test_bit_value_uses_qiskit_position_convention(self) -> None:
        assert bit_value("101", 0) == 1
        assert bit_value("101", 1) == 0
        assert bit_value("101", 2) == 1
        assert bit_value("1 01", 2) == 1

    def test_postselection_keeps_measurement_root_as_logical_outcome(self) -> None:
        experiment = build_kickback_tomography_experiment(
            include_preparation_cultivation=False,
            qec_cycle_indices=(),
            final_stabilizer_basis="none",
            add_barriers=False,
        )
        layout = experiment.layout

        n_bits = experiment.circuit.num_clbits

        def bitstring_with(*positions: int) -> str:
            bits = ["0"] * n_bits
            for position in positions:
                bits[n_bits - 1 - position] = "1"
            return "".join(bits)

        measure = layout.tomography_measurement.folded_observable
        measure_site_check = layout.tomography_measurement.folded_site_helper_check
        check = layout.tomography_check.folded_observable
        check_site_check = layout.tomography_check.folded_site_helper_check
        helper = layout.tomography_measurement.helper_checks[0]

        good_success = bitstring_with()
        good_error = bitstring_with(measure, measure_site_check, check, check_site_check)
        disagreement_error = bitstring_with(measure, measure_site_check)
        rejected_detector = bitstring_with(helper)

        counts = {
            good_success: 7,
            good_error: 3,
            disagreement_error: 5,
            rejected_detector: 11,
        }

        accepted = postselect_kickback_tomography_counts(counts, layout)
        summary = summarize_kickback_tomography_counts(counts, layout)

        assert accepted == {good_success: 7, good_error: 3, disagreement_error: 5}
        assert summary == KickbackTomographySummary(
            total_shots=26,
            accepted_shots=15,
            rejected_shots=11,
            logical_success_shots=7,
            logical_error_shots=8,
        )
        assert summary.logical_error_probability == pytest.approx(8 / 15)


# =============================================================================
# Test Set 3: Noiseless KT behavior
# =============================================================================


class TestNoiselessKickbackTomography:
    """Noiseless KT matches H-axis projective-measurement probabilities."""

    def test_cultivated_h_state_has_zero_kt_logical_error(self) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state="H",
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=10).result().get_counts()
        summary = summarize_kickback_tomography_counts(counts, experiment.layout)

        assert summary.accepted_shots == 10
        assert summary.logical_error_shots == 0
        assert summary.retainment_rate == 1.0
        assert summary.logical_error_probability == 0.0

    @pytest.mark.parametrize("state", ["H", "+", "1", "Y+", "T"])
    def test_injection_only_kt_error_probability(self, state: InjectedStateName) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state=state,
            include_preparation_cultivation=False,
            qec_cycle_indices=(),
            final_stabilizer_basis="none",
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=NUM_SHOTS).result().get_counts()
        summary = summarize_kickback_tomography_counts(counts, experiment.layout)

        expected_error = _expected_h_axis_error_probability(state)
        sigma = math.sqrt(expected_error * (1.0 - expected_error) / NUM_SHOTS)
        tolerance = max(4 * sigma, 0.02)

        assert summary.accepted_shots == NUM_SHOTS
        assert summary.logical_error_probability == pytest.approx(expected_error, abs=tolerance)

    @pytest.mark.parametrize("state", ["+", "1", "Y+"])
    def test_preparation_cultivation_projects_before_kt(self, state: InjectedStateName) -> None:
        experiment = build_kickback_tomography_experiment(
            injected_state=state,
            qec_cycle_indices=(),
            final_stabilizer_basis="none",
            add_barriers=False,
        )
        counts = AerSimulator().run(experiment.circuit, shots=NUM_SHOTS).result().get_counts()
        summary = summarize_kickback_tomography_counts(counts, experiment.layout)

        assert summary.accepted_shots > 0
        assert summary.logical_error_probability == 0.0
