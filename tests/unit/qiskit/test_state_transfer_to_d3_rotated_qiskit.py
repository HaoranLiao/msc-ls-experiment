"""Tests for Steane-to-rotated-d3 lattice-surgery transfer helpers.

Organized into three test sets:
1. Circuit labels and paper-facing d=3 adaptation check schedule
2. Measurement parsing, postselection predicates, and merged-code syndrome data
3. Logical-state transfer correctness after Pauli-frame interpretation
"""

from __future__ import annotations

from collections import Counter

import pytest
from qiskit.circuit import CircuitInstruction, Gate
from qiskit.circuit.library import XGate, ZGate
from qiskit_aer import AerSimulator

from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_analysis_qiskit as analysis,
)
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_common_qiskit as common,
)
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_to_d3_rotated_qiskit as d3t,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _one_parsed_shot(**circuit_kwargs):
    parse_kwargs = {
        "include_cultivation": circuit_kwargs.get("include_cultivation", True),
        "surface_readout_basis": circuit_kwargs.get("surface_readout_basis"),
        "surface_readout_mode": circuit_kwargs.get("surface_readout_mode", "data"),
    }
    circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
        add_barriers=False,
        **circuit_kwargs,
    )
    bitstring = next(iter(AerSimulator().run(circuit, shots=1).result().get_counts()))
    return d3t.parse_transfer_measurements(bitstring, **parse_kwargs)


def _corrected_logical_counts(
    *,
    injected_state: str,
    include_cultivation: bool,
    basis: str,
    shots: int = 512,
) -> Counter[int]:
    circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        surface_readout_basis=basis,
        add_barriers=False,
    )
    counts = AerSimulator().run(circuit, shots=shots).result().get_counts()

    logical_counts: Counter[int] = Counter()
    for bitstring, count in counts.items():
        measurements = d3t.parse_transfer_measurements(
            bitstring,
            include_cultivation=include_cultivation,
            surface_readout_basis=basis,
        )
        logical_bit = analysis.corrected_surface_logical_readout(
            measurements,
            basis=basis,
            spec=d3t.TRANSFER_SPEC,
        )
        logical_counts[logical_bit] += count
    return logical_counts


def _parsed_surface_fault_shots(
    *,
    injected_state: str,
    fault: Gate,
    surface_site: str,
    basis: str,
    shots: int = 32,
) -> list[dict[str, object]]:
    """Run the main d3 transfer after one fault following surface initialization."""
    circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
        injected_state=injected_state,
        include_cultivation=False,
        surface_readout_basis=basis,
        surface_readout_mode="logical_pauli",
        add_barriers=True,
    )
    initialization_end = next(
        index
        for index, instruction in enumerate(circuit.data)
        if instruction.operation.name == "barrier"
        and instruction.operation.label == "surface init parallel Z checks"
    )
    surface_qubit = next(register for register in circuit.qregs if register.name == surface_site)[0]
    circuit.data.insert(
        initialization_end + 1,
        CircuitInstruction(fault, (surface_qubit,), ()),
    )

    counts = AerSimulator(method="stabilizer").run(circuit, shots=shots).result().get_counts()
    return [
        d3t.parse_transfer_measurements(
            bitstring,
            include_cultivation=False,
            surface_readout_basis=basis,
            surface_readout_mode="logical_pauli",
        )
        for bitstring in counts
    ]


# ===========================================================================
# SET 1: Circuit labels and paper-facing d=3 adaptation check schedule
# ===========================================================================


class TestTransferCircuitLayout:
    """Verify the transfer circuit exposes the intended paper-style interface."""

    def test_circuit_exposes_paper_labeled_data_registers(self) -> None:
        circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
            include_cultivation=True,
            surface_readout_basis=None,
            add_barriers=False,
        )
        register_names = {register.name for register in circuit.qregs}

        for label in common.HIRANO_STEANE_DATA_ORDER:
            assert label in register_names

        for label in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
            assert label in register_names

    def test_merged_boundary_checks_match_d3_adaptation(self) -> None:
        circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
            include_cultivation=True,
            surface_readout_basis=None,
            add_barriers=False,
        )

        assert d3t.MERGED_BOUNDARY_CHECKS == (
            ("Z1a", ("1",), ("a",)),
            ("Z35bc", ("3", "5"), ("b", "c")),
        )
        assert {register.name for register in circuit.qregs if register.name.startswith("zz_")} == {
            "zz_1a",
            "zz_35bc",
        }
        assert circuit.cregs[3].name == "ls_zz"
        # Three rounds, two merged-boundary checks per round.
        assert circuit.cregs[3].size == 6
        assert {
            label for _, steane_support, _ in d3t.MERGED_BOUNDARY_CHECKS for label in steane_support
        } == set(common.HIRANO_STEANE_LOGICAL_Z)
        assert {
            label
            for _, _, surface_support in d3t.MERGED_BOUNDARY_CHECKS
            for label in surface_support
        } == set(d3t.TRANSFER_SPEC.logical_z)

    def test_paper_style_steane_checks_and_logical_x_are_declared(self) -> None:
        assert common.HIRANO_STEANE_LOGICAL_X == ("1", "4", "6")
        assert common.STEANE_LS_VERIFICATION_ROUNDS == (
            ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
            ("Z0145", "Z0235", "Z0246"),
        )
        assert common.STEANE_LS_VERIFICATION_LABELS == (
            "round1_Z0145",
            "round1_Z0235",
            "round1_X0235",
            "round1_Z0246",
            "round1_X0246",
            "round2_Z0145",
            "round2_Z0235",
            "round2_Z0246",
        )

    def test_steane_check_groups_are_inside_merged_rounds_one_and_two(self) -> None:
        circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )
        register_bits = {register.name: set(register) for register in circuit.cregs}

        def measurement_positions(register_name: str) -> list[int]:
            return [
                position
                for position, instruction in enumerate(circuit.data)
                if instruction.operation.name == "measure"
                and instruction.clbits[0] in register_bits[register_name]
            ]

        merged = measurement_positions("ls_zz")
        steane = measurement_positions("steane_verify")
        demolition = measurement_positions("steane_x")

        assert max(merged[:2]) < min(steane[:5]) < max(steane[:5]) < min(merged[2:4])
        assert max(merged[2:4]) < min(steane[5:]) < max(steane[5:]) < min(merged[4:])
        assert max(merged[4:]) < min(demolition)

    def test_paper_facing_default_does_not_append_post_transfer_qec(self) -> None:
        circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )

        assert "surface_post_z" not in {register.name for register in circuit.cregs}
        assert "surface_post_x" not in {register.name for register in circuit.cregs}

    def test_top_x_recovery_is_unconditional(self) -> None:
        circuit = d3t.build_steane_to_rotated_d3_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )

        assert "surface_x_recovery" in {register.name for register in circuit.cregs}


# ===========================================================================
# SET 2: Measurement parsing, postselection predicates, and syndrome data
# ===========================================================================


class TestTransferMeasurementParsing:
    """Verify parsed measurement records expose each protocol stage clearly."""

    def test_parser_exposes_readable_register_chunks_for_transfer_core(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis="Z",
            surface_readout_mode="logical_pauli",
        )

        assert set(measurements["registers"]) == {
            "enc_helpers",
            "init_z",
            "ls_zz",
            "merged_surface_z",
            "merged_surface_x",
            "steane_verify",
            "steane_x",
            "surface_x_recovery",
            "surface_logical_z",
        }
        assert measurements["registers"]["surface_logical_z"]["labels"] == ("a", "b", "c")
        assert (
            measurements["registers"]["steane_verify"]["labels"]
            == common.STEANE_LS_VERIFICATION_LABELS
        )
        assert measurements["steane_verification_rounds"] == [
            {"Z0145": 0, "Z0235": 0, "X0235": 0, "Z0246": 0, "X0246": 0},
            {"Z0145": 0, "Z0235": 0, "Z0246": 0},
        ]

    def test_transfer_acceptance_predicates_accept_noiseless_basis_state(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis=None,
        )

        assert set(measurements["final_x_verification_checks"]) == {
            "X0145ab",
            "X0235",
            "X0246",
        }
        assert analysis.merged_boundary_checks_stable(measurements)
        assert analysis.steane_verification_succeeds(measurements)
        assert analysis.x_verification_succeeds(measurements)
        assert measurements["x_verification_succeeds"]
        assert analysis.transfer_acceptance_succeeds(measurements)

    def test_merged_interval_surface_syndrome_rounds_have_noiseless_deltas(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis=None,
        )

        assert len(measurements["merged_surface_z_rounds"]) == 3
        assert len(measurements["merged_surface_x_rounds"]) == 3
        assert len(measurements["merged_surface_syndrome_deltas"]) == 3
        assert all(
            bit == 0
            for deltas in measurements["merged_surface_syndrome_deltas"]
            for basis in ("z", "x")
            for bit in deltas[basis].values()
        )

    def test_merged_interval_surface_syndrome_schedule_is_explicit(self) -> None:
        assert d3t.TRANSFER_SPEC.z_labels == ("Z_abde", "Z_cf", "Z_dg", "Z_efhi")
        assert d3t.TRANSFER_SPEC.merged_x_labels == ("X_bcef", "X_degh", "X_hi")

    def test_lookup_table_early_rejection_is_separate_from_steane_postselection(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis=None,
        )

        signature = analysis.early_rejection_syndrome_signature(measurements)
        items = analysis.early_rejection_syndrome_items(measurements)
        table = analysis.LookupTableEarlyRejection.from_signatures([signature])

        assert all(bit == 0 for bit in signature)
        assert items
        assert all(name.startswith("merged_surface:") for name, _ in items)
        assert analysis.steane_verification_succeeds(measurements)
        assert analysis.x_verification_succeeds(measurements)
        assert analysis.lookup_table_early_rejection_accepts(measurements)
        assert not analysis.lookup_table_early_rejection_accepts(
            measurements,
            lookup_table=table,
        )
        assert not analysis.transfer_acceptance_succeeds(measurements, lookup_table=table)

    def test_lookup_table_remove_all_nontrivial_rejects_syndrome_signature(self) -> None:
        table = analysis.LookupTableEarlyRejection(remove_all_nontrivial=True)

        assert not table.rejects((0, 0, 0))
        assert table.rejects((0, 1, 0))

    def test_parser_rejects_register_chunks_with_wrong_width(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            d3t.parse_transfer_measurements(
                "0000 0000000 0 0000",
                include_cultivation=False,
            )


# ===========================================================================
# SET 3: Logical-state transfer correctness
# ===========================================================================


class TestLogicalTransferCorrectness:
    """Verify corrected target readout matches the transferred logical state."""

    def test_transfer_preserves_logical_z_basis_states_without_cultivation(self) -> None:
        zero_counts = _corrected_logical_counts(
            injected_state="0",
            include_cultivation=False,
            basis="Z",
        )
        one_counts = _corrected_logical_counts(
            injected_state="1",
            include_cultivation=False,
            basis="Z",
        )

        assert zero_counts[0] == sum(zero_counts.values())
        assert one_counts[1] == sum(one_counts.values())

    def test_transfer_preserves_logical_x_basis_states_without_cultivation(self) -> None:
        plus_counts = _corrected_logical_counts(
            injected_state="+",
            include_cultivation=False,
            basis="X",
        )
        minus_counts = _corrected_logical_counts(
            injected_state="-",
            include_cultivation=False,
            basis="X",
        )

        assert plus_counts[0] == sum(plus_counts.values())
        assert minus_counts[1] == sum(minus_counts.values())

    def test_pre_merge_surface_x_fault_is_not_double_counted(self) -> None:
        measurements_by_outcome = _parsed_surface_fault_shots(
            injected_state="0",
            fault=XGate(),
            surface_site="a",
            basis="Z",
        )

        assert all(analysis.transfer_acceptance_succeeds(item) for item in measurements_by_outcome)
        assert all(
            item["merged_surface_qec_recoveries"][0].x_error_sites == {"a"}
            for item in measurements_by_outcome
        )
        assert all(
            analysis.corrected_surface_logical_readout(
                item,
                basis="Z",
                spec=d3t.TRANSFER_SPEC,
            )
            == 0
            for item in measurements_by_outcome
        )

    def test_merged_surface_x_checks_correct_phase_fault(self) -> None:
        measurements_by_outcome = _parsed_surface_fault_shots(
            injected_state="+",
            fault=ZGate(),
            surface_site="d",
            basis="X",
        )

        assert all(analysis.transfer_acceptance_succeeds(item) for item in measurements_by_outcome)
        assert all(
            item["merged_surface_qec_recoveries"][0].z_error_sites == {"d"}
            for item in measurements_by_outcome
        )
        assert all(
            analysis.corrected_surface_logical_readout(
                item,
                basis="X",
                spec=d3t.TRANSFER_SPEC,
            )
            == 0
            for item in measurements_by_outcome
        )

    def test_direct_logical_pauli_readout_preserves_basis_states_without_cultivation(self) -> None:
        zero_measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis="Z",
            surface_readout_mode="logical_pauli",
        )
        plus_measurements = _one_parsed_shot(
            injected_state="+",
            include_cultivation=False,
            surface_readout_basis="X",
            surface_readout_mode="logical_pauli",
        )

        assert (
            analysis.corrected_surface_logical_readout(
                zero_measurements,
                basis="Z",
                spec=d3t.TRANSFER_SPEC,
            )
            == 0
        )
        assert (
            analysis.corrected_surface_logical_readout(
                plus_measurements,
                basis="X",
                spec=d3t.TRANSFER_SPEC,
            )
            == 0
        )
