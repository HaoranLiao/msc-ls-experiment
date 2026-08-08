"""Tests for the Stim Steane-to-rotated-d5 lattice-surgery transfer."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
import stim
from state_transfer_test_utils import (
    assert_demolition_measurement_groups,
    assert_merge_round_measurement_groups,
    assert_optional_post_transfer_continuation,
    assert_transfer_detection_policy,
)

from msc.lattice_surgery import (
    ls_common as common,
)
from msc.lattice_surgery import (
    ls_helper_stim as helper,
)
from msc.lattice_surgery import (
    ls_to_d5_rotated_stim as d5t,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _one_parsed_shot(**circuit_kwargs) -> dict[str, object]:
    build_kwargs = {"include_cultivation": False, **circuit_kwargs}
    parse_kwargs = {
        "include_cultivation": build_kwargs.get("include_cultivation", False),
        "include_post_transfer_stabilizer_rounds": build_kwargs.get(
            "include_post_transfer_stabilizer_rounds",
            False,
        ),
        "surface_readout_basis": build_kwargs.get("surface_readout_basis"),
    }
    circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
        add_ticks=False,
        **build_kwargs,
    )
    shot = circuit.compile_sampler(seed=1).sample(1)[0]
    return d5t.parse_transfer_measurements(shot, **parse_kwargs)


def _corrected_logical_counts(
    *,
    injected_state: str,
    basis: str,
    shots: int = 512,
    include_cultivation: bool = False,
) -> Counter[int]:
    circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        surface_readout_basis=basis,
        add_ticks=False,
    )
    sampled_shots = circuit.compile_sampler(seed=42).sample(shots)
    logical_counts: Counter[int] = Counter()
    for shot in sampled_shots:
        measurements = d5t.parse_transfer_measurements(
            shot,
            include_cultivation=include_cultivation,
            surface_readout_basis=basis,
        )
        if common.transfer_acceptance_succeeds(measurements):
            logical_counts[common.corrected_surface_logical_readout(measurements, basis=basis)] += 1
    return logical_counts


# ===========================================================================
# SET 1: Target configuration and optional continuation
# ===========================================================================


class TestTransferConfiguration:
    def test_target_geometry_and_boundary_observables(self) -> None:
        spec = d5t.TRANSFER_SPEC
        assert spec.distance == 5
        assert d5t.MERGED_BOUNDARY_CHECKS == (
            ("Z1a", ("1",), ("a",)),
            ("Z35bc", ("3", "5"), ("b", "c")),
            ("Zde", (), ("d", "e")),
        )
        assert {
            label
            for _, steane_support, _ in spec.merged_boundary_checks
            for label in steane_support
        } == set(common.HIRANO_STEANE_LOGICAL_Z)
        assert {
            label
            for _, _, surface_support in spec.merged_boundary_checks
            for label in surface_support
        } == set(spec.logical_z)
        assert spec.top_x_recovery_labels == ("X_ab", "X_cd")
        assert set(spec.merged_x_labels) == set(spec.x_labels) - {"X_ab", "X_cd"}

    def test_transfer_layout_matches_circuit_record(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            include_cultivation=False,
            add_ticks=False,
        )
        layout = helper.transfer_measurement_layout(
            d5t.TRANSFER_SPEC,
            include_cultivation=False,
            surface_readout_basis=None,
        )
        assert circuit.num_qubits == 63
        assert circuit.num_measurements == layout.num_measurements == 114
        assert circuit.num_detectors == 93
        assert circuit.num_observables == 0
        assert (
            layout.record("merged_surface_z").positions[0] < layout.record("steane_x").positions[0]
        )

    def test_optional_post_transfer_continuation_adds_five_rounds(self) -> None:
        assert_optional_post_transfer_continuation(spec=d5t.TRANSFER_SPEC)
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            include_post_transfer_stabilizer_rounds=True,
            surface_readout_basis="X",
            add_ticks=False,
        )
        measurements = d5t.parse_transfer_measurements(
            circuit.compile_sampler(seed=2).sample(1)[0],
            include_cultivation=False,
            include_post_transfer_stabilizer_rounds=True,
            surface_readout_basis="X",
        )
        assert circuit.num_measurements == 114 + 5 * 24 + 5
        assert circuit.num_detectors == 93 + 5 * 24
        assert circuit.num_observables == 1
        assert any(
            isinstance(instruction, stim.CircuitRepeatBlock)
            and instruction.repeat_count == d5t.TRANSFER_SPEC.distance
            for instruction in circuit
        )
        assert len(measurements["post_transfer_surface_syndrome_delta_history"]) == 5
        assert all(
            bit == 0
            for layer in measurements["post_transfer_surface_syndrome_delta_history"]
            for sector in layer.values()
            for bit in sector.values()
        )


# ===========================================================================
# SET 2: Simultaneous merge-schedule actions
# ===========================================================================


class TestSimultaneousMergeSchedule:
    @pytest.mark.parametrize(
        ("round_number", "steane_checks"),
        (
            (1, ("Z0145", "Z0235", "X0235", "Z0246", "X0246")),
            (2, ("Z0145", "Z0235", "Z0246")),
            (3, ()),
        ),
    )
    def test_merge_round_contains_exact_measurement_groups(
        self,
        round_number: int,
        steane_checks: tuple[str, ...],
    ) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_ticks=False,
        )
        assert_merge_round_measurement_groups(
            circuit,
            spec=d5t.TRANSFER_SPEC,
            round_number=round_number,
            expected_steane_checks=steane_checks,
        )

    def test_demolition_measures_steane_then_recovers_top_edge(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_ticks=False,
        )
        measurements = _one_parsed_shot(injected_state="0")
        assert_demolition_measurement_groups(
            circuit,
            spec=d5t.TRANSFER_SPEC,
            measurements=measurements,
        )


# ===========================================================================
# SET 3: Transfer analysis and detector policy
# ===========================================================================


class TestTransferAnalysis:
    def test_noiseless_transfer_accepts_all_hard_checks(self) -> None:
        measurements = _one_parsed_shot(injected_state="0")
        assert measurements["steane_verification_rounds"] == [
            {"Z0145": 0, "Z0235": 0, "X0235": 0, "Z0246": 0, "X0246": 0},
            {"Z0145": 0, "Z0235": 0, "Z0246": 0},
        ]
        assert common.merged_boundary_checks_stable(measurements)
        assert common.post_injection_qed_succeeds(measurements)
        assert common.steane_verification_succeeds(measurements)
        assert common.x_verification_succeeds(measurements)
        assert common.transfer_acceptance_succeeds(measurements)

    def test_detection_policy_postselects_boundary_and_steane_not_surface(self) -> None:
        assert_transfer_detection_policy(_one_parsed_shot(injected_state="0"))

    def test_surface_detector_history_completes_post_transfer_frame(self) -> None:
        measurements = _one_parsed_shot(injected_state="0")
        delta_history = measurements["merged_surface_syndrome_delta_history"]
        frame = measurements["post_transfer_surface_stabilizer_frame"]
        assert len(delta_history) == common.LATTICE_SURGERY_ROUNDS
        assert sum(len(layer[sector]) for layer in delta_history for sector in ("z", "x")) == 66
        assert all(
            bit == 0
            for layer in delta_history
            for sector in ("z", "x")
            for bit in layer[sector].values()
        )
        assert set(frame.as_dict()) == set(d5t.TRANSFER_SPEC.z_labels) | set(
            d5t.TRANSFER_SPEC.x_labels
        )

    def test_x_cd_recovery_is_reference_not_hard_postselection(self) -> None:
        measurements = _one_parsed_shot(injected_state="0")
        measurements["surface_x_recovery"]["X_cd"] ^= 1
        assert common.transfer_acceptance_succeeds(measurements)

    def test_parser_rejects_wrong_record_width(self) -> None:
        with pytest.raises(ValueError, match="shot width"):
            d5t.parse_transfer_measurements(
                np.zeros(10, dtype=bool),
                include_cultivation=False,
            )


# ===========================================================================
# SET 4: Logical-state transfer correctness
# ===========================================================================


class TestLogicalTransfer:
    @pytest.mark.parametrize(
        ("injected_state", "basis", "expected_bit"),
        (("0", "Z", 0), ("1", "Z", 1), ("+", "X", 0), ("-", "X", 1)),
    )
    def test_transfer_preserves_logical_basis_states_without_cultivation(
        self,
        injected_state: str,
        basis: str,
        expected_bit: int,
    ) -> None:
        logical_counts = _corrected_logical_counts(
            injected_state=injected_state,
            basis=basis,
        )
        assert logical_counts[expected_bit] == sum(logical_counts.values()) == 512

    @pytest.mark.parametrize("basis", ("X", "Y", "Z"))
    def test_s_state_expectation_after_transfer_without_cultivation(self, basis: str) -> None:
        expected = 1.0 if basis == "Y" else 0.0
        logical_counts = _corrected_logical_counts(injected_state="S", basis=basis)
        num_accepted = sum(logical_counts.values())
        expectation = common.logical_expectation_from_logical_counts(logical_counts)
        tolerance = 4 * math.sqrt((1 - expected**2) / num_accepted) + 0.02
        assert num_accepted == 512
        assert expectation == pytest.approx(expected, abs=tolerance)

    def test_s_state_cultivation_proxy_accepts_and_transfers_y_plus(self) -> None:
        logical_counts = _corrected_logical_counts(
            injected_state="S",
            basis="Y",
            shots=64,
            include_cultivation=True,
        )
        assert logical_counts == Counter({0: 64})
