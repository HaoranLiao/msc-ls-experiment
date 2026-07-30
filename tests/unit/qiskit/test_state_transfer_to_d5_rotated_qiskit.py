"""Tests for Steane-to-rotated-d5 lattice-surgery transfer helpers.

Organized into four protocol-focused sets:
1. Target configuration and optional fixed-patch continuation
2. Simultaneous actions in each merge-schedule row
3. Transfer acceptance, detector history, and frame interpretation
4. Logical-state transfer correctness
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
from qiskit_aer import AerSimulator

from magic_state_cultivation.state_transfer_by_lattice_surgery import (
    state_transfer_analysis_qiskit as analysis,
)
from magic_state_cultivation.state_transfer_by_lattice_surgery import (
    state_transfer_common_qiskit as common,
)
from magic_state_cultivation.state_transfer_by_lattice_surgery import (
    state_transfer_to_d5_rotated_qiskit as d5t,
)
from state_transfer_test_utils import (
    assert_demolition_measurement_groups,
    assert_merge_round_measurement_groups,
    assert_optional_post_transfer_continuation,
    assert_transfer_detection_policy,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _one_parsed_shot(**circuit_kwargs):
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
        add_barriers=False,
        **build_kwargs,
    )
    bitstring = next(
        iter(AerSimulator(method="stabilizer").run(circuit, shots=1).result().get_counts())
    )
    return d5t.parse_transfer_measurements(bitstring, **parse_kwargs)


def _corrected_logical_counts(
    *,
    injected_state: str,
    basis: str,
    shots: int = 512,
) -> Counter[int]:
    circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
        injected_state=injected_state,
        include_cultivation=False,
        surface_readout_basis=basis,
        add_barriers=False,
    )
    counts = (
        AerSimulator(method="stabilizer")
        .run(circuit, shots=shots, seed_simulator=42)
        .result()
        .get_counts()
    )

    logical_counts: Counter[int] = Counter()
    for bitstring, count in counts.items():
        measurements = d5t.parse_transfer_measurements(
            bitstring,
            include_cultivation=False,
            surface_readout_basis=basis,
        )
        if not analysis.transfer_acceptance_succeeds(measurements):
            continue
        logical_bit = analysis.corrected_surface_logical_readout(
            measurements,
            basis=basis,
        )
        logical_counts[logical_bit] += count
    return logical_counts


# ===========================================================================
# SET 1: Target configuration and optional continuation
# ===========================================================================


class TestTransferConfiguration:
    """Verify the paper-facing d5 geometry is bound to the shared protocol."""

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
        assert set(spec.merged_x_labels) == set(spec.x_labels) - {
            "X_ab",
            "X_cd",
        }

    def test_optional_post_transfer_continuation_adds_distance_rounds_before_readout(
        self,
    ) -> None:
        transfer_only = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )
        assert "post_transfer_surface_z" not in {
            register.name for register in transfer_only.cregs
        }
        assert "post_transfer_surface_x" not in {
            register.name for register in transfer_only.cregs
        }

        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            include_post_transfer_stabilizer_rounds=True,
            surface_readout_basis="X",
            add_barriers=False,
        )
        assert_optional_post_transfer_continuation(
            circuit,
            spec=d5t.TRANSFER_SPEC,
            logical_register_name="surface_logical_x",
            logical_support=d5t.TRANSFER_SPEC.logical_x,
        )


# ===========================================================================
# SET 2: Simultaneous merge-schedule actions
# ===========================================================================


class TestSimultaneousMergeSchedule:
    """Verify every logical row of the interleaved merge-schedule table."""

    @pytest.mark.parametrize(
        ("round_number", "steane_checks"),
        (
            pytest.param(
                1,
                ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
                id="round-1",
            ),
            pytest.param(2, ("Z0145", "Z0235", "Z0246"), id="round-2"),
            pytest.param(3, (), id="round-3"),
        ),
    )
    def test_merge_round_contains_exact_interface_surface_and_steane_groups(
        self,
        round_number: int,
        steane_checks: tuple[str, ...],
    ) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )

        assert_merge_round_measurement_groups(
            circuit,
            spec=d5t.TRANSFER_SPEC,
            round_number=round_number,
            expected_steane_checks=steane_checks,
        )

    def test_demolition_measures_steane_in_x_then_recovers_top_edge(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )
        assert_demolition_measurement_groups(
            circuit,
            spec=d5t.TRANSFER_SPEC,
            measurements=measurements,
        )


# ===========================================================================
# SET 3: Transfer analysis and detector policy
# ===========================================================================


class TestTransferAnalysis:
    """Verify postselection, temporal records, and post-transfer frame state."""

    def test_noiseless_transfer_accepts_all_hard_checks(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        assert measurements["steane_verification_rounds"] == [
            {"Z0145": 0, "Z0235": 0, "X0235": 0, "Z0246": 0, "X0246": 0},
            {"Z0145": 0, "Z0235": 0, "Z0246": 0},
        ]
        assert set(measurements["final_x_verification_checks"]) == {
            "X0145ab",
            "X0235",
            "X0246",
        }
        assert analysis.merged_boundary_checks_stable(measurements)
        assert analysis.post_injection_qed_succeeds(measurements)
        assert analysis.steane_verification_succeeds(measurements)
        assert analysis.x_verification_succeeds(measurements)
        assert analysis.transfer_acceptance_succeeds(measurements)

    def test_detection_policy_postselects_boundary_and_steane_not_surface(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )
        assert_transfer_detection_policy(measurements)

    def test_surface_detector_history_completes_post_transfer_frame(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )
        delta_history = measurements["merged_surface_syndrome_delta_history"]
        frame = measurements["post_transfer_surface_stabilizer_frame"]

        assert len(delta_history) == common.LATTICE_SURGERY_ROUNDS
        assert all(
            bit == 0
            for delta_layer in delta_history
            for basis in ("z", "x")
            for bit in delta_layer[basis].values()
        )
        assert sum(
            len(layer[basis]) for layer in delta_history for basis in ("z", "x")
        ) == 66
        assert set(frame.as_dict()) == set(d5t.TRANSFER_SPEC.z_labels) | set(
            d5t.TRANSFER_SPEC.x_labels
        )
        assert all(
            frame.current(label) == measurements["merged_surface_z_rounds"][-1][label]
            for label in d5t.TRANSFER_SPEC.z_labels
        )
        assert all(
            frame.current(label) == measurements["merged_surface_x_rounds"][-1][label]
            for label in d5t.TRANSFER_SPEC.merged_x_labels
        )
        assert all(
            frame.current(label) == measurements["surface_x_recovery"][label]
            for label in d5t.TRANSFER_SPEC.top_x_recovery_labels
        )

    def test_x_cd_recovery_is_reference_not_hard_postselection(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )
        measurements["surface_x_recovery"]["X_cd"] ^= 1

        assert analysis.transfer_acceptance_succeeds(measurements)


# ===========================================================================
# SET 4: Logical-state transfer correctness
# ===========================================================================


class TestLogicalTransfer:
    """Verify corrected target readout matches the transferred logical state."""

    @pytest.mark.parametrize(
        ("injected_state", "basis", "expected_bit"),
        (
            pytest.param("0", "Z", 0, id="zero"),
            pytest.param("1", "Z", 1, id="one"),
            pytest.param("+", "X", 0, id="plus"),
            pytest.param("-", "X", 1, id="minus"),
        ),
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

        assert logical_counts[expected_bit] == sum(logical_counts.values())

    @pytest.mark.parametrize(
        ("basis", "expected"),
        (
            pytest.param("X", 0.0, id="X"),
            pytest.param("Y", 1.0, id="Y"),
            pytest.param("Z", 0.0, id="Z"),
        ),
    )
    def test_s_state_expectation_after_transfer_without_cultivation(
        self,
        basis: str,
        expected: float,
    ) -> None:
        logical_counts = _corrected_logical_counts(
            injected_state="S",
            basis=basis,
        )
        num_accepted = sum(logical_counts.values())
        expectation = analysis.logical_expectation_from_logical_counts(logical_counts)
        tolerance = 4 * math.sqrt((1 - expected**2) / num_accepted) + 0.02

        assert expectation == pytest.approx(expected, abs=tolerance)
