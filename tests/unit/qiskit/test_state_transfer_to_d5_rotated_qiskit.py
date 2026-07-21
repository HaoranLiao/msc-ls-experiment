"""Tests for Steane-to-rotated-d5 lattice-surgery transfer helpers.

Organized into three test sets:
1. Figure 6 circuit geometry and measurement ordering
2. Parsing, postselection, and syndrome references
3. Corrected logical-state transfer on complementary Pauli bases
"""

from __future__ import annotations

from collections import Counter

from qiskit_aer import AerSimulator

from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_analysis_qiskit as analysis,
)
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_common_qiskit as common,
)
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_to_d5_rotated_qiskit as d5t,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _one_parsed_shot(**circuit_kwargs):
    build_kwargs = {"include_cultivation": False, **circuit_kwargs}
    parse_kwargs = {
        "include_cultivation": build_kwargs.get("include_cultivation", False),
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
    shots: int = 128,
) -> Counter[int]:
    circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
        injected_state=injected_state,
        include_cultivation=False,
        surface_readout_basis=basis,
        add_barriers=False,
    )
    counts = AerSimulator(method="stabilizer").run(circuit, shots=shots).result().get_counts()

    logical_counts: Counter[int] = Counter()
    for bitstring, count in counts.items():
        measurements = d5t.parse_transfer_measurements(
            bitstring,
            include_cultivation=False,
            surface_readout_basis=basis,
        )
        logical_bit = analysis.corrected_surface_logical_readout(
            measurements,
            basis=basis,
        )
        logical_counts[logical_bit] += count
    return logical_counts


# ===========================================================================
# SET 1: Circuit labels and paper-facing d=5 transfer-core check schedule
# ===========================================================================


class TestD5TransferCircuitLayout:
    """Verify the d=5 transfer circuit exposes the paper-style interface."""

    def test_circuit_exposes_steane_and_surface_data_registers(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis=None,
            add_barriers=False,
        )
        register_names = {register.name for register in circuit.qregs}

        for label in common.HIRANO_STEANE_DATA_ORDER:
            assert label in register_names
        for label in d5t.TRANSFER_SPEC.surface_order:
            assert label in register_names

    def test_merged_boundary_checks_match_paper_d5_schedule(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis=None,
            add_barriers=False,
        )

        assert d5t.MERGED_BOUNDARY_CHECKS == (
            ("Z1a", ("1",), ("a",)),
            ("Z35bc", ("3", "5"), ("b", "c")),
            ("Zde", (), ("d", "e")),
        )
        assert {register.name for register in circuit.qregs if register.name.startswith("zz_")} == {
            "zz_1a",
            "zz_35bc",
            "zz_de",
        }
        ls_zz_register = next(register for register in circuit.cregs if register.name == "ls_zz")
        assert ls_zz_register.size == 9
        assert {
            label for _, steane_support, _ in d5t.MERGED_BOUNDARY_CHECKS for label in steane_support
        } == set(common.HIRANO_STEANE_LOGICAL_Z)
        assert {
            label
            for _, _, surface_support in d5t.MERGED_BOUNDARY_CHECKS
            for label in surface_support
        } == set(d5t.TRANSFER_SPEC.logical_z)

    def test_top_edge_recovery_labels_are_declared(self) -> None:
        spec = d5t.TRANSFER_SPEC

        assert spec.top_x_recovery_labels == ("X_ab", "X_cd")
        assert "X_ab" not in spec.merged_x_labels
        assert "X_cd" not in spec.merged_x_labels
        assert set(spec.z_labels) == set(spec.z_stabilizers)
        assert set(spec.merged_x_labels) == set(spec.x_stabilizers) - {
            "X_ab",
            "X_cd",
        }

    def test_steane_check_groups_are_inside_merged_rounds_one_and_two(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
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

        surface_init_z = measurement_positions("surface_init_z")
        merged = measurement_positions("ls_zz")
        steane = measurement_positions("steane_verify")
        demolition = measurement_positions("steane_x")

        assert max(surface_init_z) < min(merged)
        assert max(merged[:3]) < min(steane[:5]) < max(steane[:5]) < min(merged[3:6])
        assert max(merged[3:6]) < min(steane[5:]) < max(steane[5:]) < min(merged[6:])
        assert max(merged[6:]) < min(demolition)

    def test_paper_facing_default_does_not_append_post_transfer_qec(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )

        assert "surface_post_z" not in {register.name for register in circuit.cregs}
        assert "surface_post_x" not in {register.name for register in circuit.cregs}

    def test_top_x_recovery_is_unconditional(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )

        recovery_register = next(
            register for register in circuit.cregs if register.name == "surface_x_recovery"
        )
        assert recovery_register.size == 2

    def test_top_x_recovery_checks_are_parallel_scheduled(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            add_barriers=False,
        )
        ancillas = {
            label: next(register for register in circuit.qregs if register.name == label.lower())[0]
            for label in d5t.TRANSFER_SPEC.top_x_recovery_labels
        }
        cx_positions = {
            label: [
                position
                for position, instruction in enumerate(circuit.data)
                if instruction.operation.name == "cx" and ancilla in instruction.qubits
            ]
            for label, ancilla in ancillas.items()
        }
        recovery_bits = set(
            next(register for register in circuit.cregs if register.name == "surface_x_recovery")
        )
        measurement_positions = [
            position
            for position, instruction in enumerate(circuit.data)
            if instruction.operation.name == "measure" and instruction.clbits[0] in recovery_bits
        ]

        assert cx_positions["X_ab"][-2] < cx_positions["X_cd"][-2]
        assert cx_positions["X_cd"][-2] < cx_positions["X_ab"][-1]
        assert cx_positions["X_ab"][-1] < cx_positions["X_cd"][-1]
        assert measurement_positions[1] == measurement_positions[0] + 1


# ===========================================================================
# SET 2: Measurement parsing, postselection predicates, and syndrome data
# ===========================================================================


class TestD5TransferMeasurementParsing:
    """Verify parsed d=5 measurement records expose each protocol stage."""

    def test_parser_exposes_transfer_core_registers(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis="Z",
        )

        assert set(measurements["registers"]) == {
            "enc_helpers",
            "surface_init_z",
            "ls_zz",
            "merged_surface_z",
            "merged_surface_x",
            "steane_verify",
            "steane_x",
            "surface_x_recovery",
            "surface_logical_z",
        }
        assert measurements["registers"]["surface_x_recovery"]["labels"] == ("X_ab", "X_cd")
        assert measurements["registers"]["surface_init_z"]["labels"] == d5t.TRANSFER_SPEC.z_labels
        assert measurements["registers"]["surface_logical_z"]["labels"] == (
            "a",
            "b",
            "c",
            "d",
            "e",
        )
        assert all("Zde" in round_bits for round_bits in measurements["merged_zz_rounds"])

    def test_transfer_acceptance_predicates_accept_noiseless_basis_state(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        assert set(measurements["surface_x_recovery"]) == {"X_ab", "X_cd"}
        assert set(measurements["final_x_verification_checks"]) == {
            "X0145ab",
            "X0235",
            "X0246",
        }
        assert analysis.merged_boundary_checks_stable(measurements)
        assert analysis.steane_verification_succeeds(measurements)
        assert analysis.x_verification_succeeds(measurements)
        assert analysis.transfer_acceptance_succeeds(measurements)

    def test_first_x_cd_recovery_outcome_is_not_hard_postselected(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        measurements["surface_x_recovery"]["X_cd"] ^= 1

        assert analysis.transfer_acceptance_succeeds(measurements)

    def test_merged_interval_surface_syndrome_rounds_have_noiseless_deltas(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        assert len(measurements["merged_surface_z_rounds"]) == 3
        assert len(measurements["merged_surface_x_rounds"]) == 3
        assert len(measurements["merged_surface_syndrome_deltas"]) == 3
        assert set(measurements["merged_surface_syndrome_deltas"][0]["z"]) == set(
            d5t.TRANSFER_SPEC.z_labels
        )
        assert all(
            bit == 0
            for deltas in measurements["merged_surface_syndrome_deltas"]
            for basis in ("z", "x")
            for bit in deltas[basis].values()
        )

    def test_surface_delta_history_has_papers_66_merge_detectors(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        deltas = measurements["merged_surface_syndrome_deltas"]
        z_delta_count = sum(len(layer["z"]) for layer in deltas)
        x_delta_count = sum(len(layer["x"]) for layer in deltas)

        # The pre-merge |+_L> initialization supplies the Z reference, so all
        # three merged rounds contribute both compatible-X and Z detectors.
        assert x_delta_count == 3 * len(d5t.TRANSFER_SPEC.merged_x_labels)
        assert z_delta_count == 3 * len(d5t.TRANSFER_SPEC.z_labels)
        assert x_delta_count + z_delta_count == 66
        assert set(deltas[0]["z"]) == set(d5t.TRANSFER_SPEC.z_labels)

    def test_joint_decoder_uses_d5_merge_record_to_locate_final_round_x_fault(
        self,
    ) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        # X_a immediately after the final parity measurement changes the last
        # Z_abfg result but none of the already-recorded boundary outcomes.
        measurements["merged_surface_z_rounds"][2]["Z_abfg"] ^= 1
        measurements["merged_surface_syndrome_deltas"] = (
            analysis.merged_surface_syndrome_deltas(
                measurements,
                merged_surface_z_labels=d5t.TRANSFER_SPEC.z_labels,
                merged_surface_x_labels=d5t.TRANSFER_SPEC.merged_x_labels,
            )
        )
        after_parity = analysis.joint_transfer_decode(
            measurements,
            spec=d5t.TRANSFER_SPEC,
        )

        # Moving the same inferred X_a fault to before the final parity adds
        # the corresponding Z1a change detector and removes the output frame
        # correction, because that error is absorbed into teleportation.
        measurements["merged_zz_rounds"][2]["Z1a"] ^= 1
        before_parity = analysis.joint_transfer_decode(
            measurements,
            spec=d5t.TRANSFER_SPEC,
        )

        assert after_parity.surface_qec_logical_x_byproduct == 1
        assert after_parity.merged_boundary_delta_history[1]["Z1a"] == 0
        assert before_parity.surface_qec_logical_x_byproduct == 0
        assert before_parity.merged_boundary_delta_history[1]["Z1a"] == 1


# ===========================================================================
# SET 3: Logical-state transfer correctness
# ===========================================================================


class TestD5LogicalTransferCorrectness:
    """Verify corrected target readout matches transferred logical basis states."""

    def test_transfer_preserves_logical_z_basis_states_without_cultivation(self) -> None:
        zero_counts = _corrected_logical_counts(injected_state="0", basis="Z")
        one_counts = _corrected_logical_counts(injected_state="1", basis="Z")

        assert zero_counts[0] == sum(zero_counts.values())
        assert one_counts[1] == sum(one_counts.values())

    def test_transfer_preserves_logical_x_basis_states_without_cultivation(self) -> None:
        plus_counts = _corrected_logical_counts(injected_state="+", basis="X")
        minus_counts = _corrected_logical_counts(injected_state="-", basis="X")

        assert plus_counts[0] == sum(plus_counts.values())
        assert minus_counts[1] == sum(minus_counts.values())
