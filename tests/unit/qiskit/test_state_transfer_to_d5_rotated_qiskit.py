"""Tests for Steane-to-rotated-d5 lattice-surgery transfer helpers.

Organized into three test sets:
1. Figure 6 circuit geometry and measurement ordering
2. Parsing, postselection, syndrome references, and early rejection
3. Corrected logical-state transfer on complementary Pauli bases
"""

from __future__ import annotations

from collections import Counter

from qiskit_aer import AerSimulator

from research_qec_haoran.state_transfer_by_lattice_surgery import (
    LookupTableEarlyRejection,
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
        "merged_rounds": build_kwargs.get("merged_rounds", 3),
        "include_merged_surface_syndromes": build_kwargs.get(
            "include_merged_surface_syndromes",
            True,
        ),
        "include_steane_verification": build_kwargs.get("include_steane_verification", True),
        "include_surface_qec": build_kwargs.get("include_surface_qec", False),
        "surface_qec_rounds": build_kwargs.get("surface_qec_rounds", 3),
        "surface_readout_basis": build_kwargs.get("surface_readout_basis"),
        "surface_readout_mode": build_kwargs.get("surface_readout_mode", "data"),
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
        surface_readout_mode="logical_pauli",
        add_barriers=False,
    )
    counts = AerSimulator(method="stabilizer").run(circuit, shots=shots).result().get_counts()

    logical_counts: Counter[int] = Counter()
    for bitstring, count in counts.items():
        measurements = d5t.parse_transfer_measurements(
            bitstring,
            include_cultivation=False,
            surface_readout_basis=basis,
            surface_readout_mode="logical_pauli",
        )
        logical_bit = d5t.corrected_surface_logical_readout(measurements, basis=basis)
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

        for label in d5t.HIRANO_STEANE_DATA_ORDER:
            assert label in register_names
        for label in d5t.SURFACE_ORDER:
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
        assert circuit.cregs[2].name == "ls_zz"
        assert circuit.cregs[2].size == 9

    def test_top_edge_recovery_labels_are_declared(self) -> None:
        assert d5t.SURFACE_TOP_X_RECOVERY_LABELS == ("X_ab", "X_cd")
        assert d5t.MERGED_INTERVAL_SURFACE_SYNDROME_SCHEDULE == (
            ("merged_surface_z", d5t.SURFACE_MERGED_Z_LABELS),
            ("merged_surface_x", d5t.SURFACE_MERGED_X_LABELS),
        )
        assert "X_ab" not in d5t.SURFACE_MERGED_X_LABELS
        assert "X_cd" not in d5t.SURFACE_MERGED_X_LABELS

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

        merged = measurement_positions("ls_zz")
        steane = measurement_positions("steane_verify")
        demolition = measurement_positions("steane_x")

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

    def test_top_x_recovery_is_independent_of_steane_verification_ablation(self) -> None:
        circuit = d5t.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state="0",
            include_cultivation=False,
            include_steane_verification=False,
            add_barriers=False,
        )

        recovery_register = next(
            register for register in circuit.cregs if register.name == "surface_x_recovery"
        )
        assert recovery_register.size == 2


# ===========================================================================
# SET 2: Measurement parsing, postselection predicates, and syndrome data
# ===========================================================================


class TestD5TransferMeasurementParsing:
    """Verify parsed d=5 measurement records expose each protocol stage."""

    def test_parser_exposes_transfer_core_registers(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
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
        assert measurements["registers"]["surface_x_recovery"]["labels"] == ("X_ab", "X_cd")
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
        assert set(measurements["local_x_verification_checks"]) == {
            "X0145ab",
            "X0235",
            "X0246",
        }
        assert d5t.merged_boundary_checks_stable(measurements)
        assert d5t.steane_verification_succeeds(measurements)
        assert d5t.x_verification_succeeds(measurements)
        assert d5t.transfer_acceptance_succeeds(measurements)

    def test_merged_interval_surface_syndrome_rounds_have_noiseless_deltas(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
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

    def test_optional_post_transfer_surface_qec_rounds_are_available(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_surface_qec=True,
            surface_readout_basis=None,
        )
        schedule = d5t.post_transfer_surface_syndrome_schedule(2)

        assert len(measurements["surface_post_z_rounds"]) == 3
        assert len(measurements["surface_post_x_rounds"]) == 3
        assert measurements["surface_qec_syndrome_is_trivial"]
        assert [name for name, _ in schedule] == [
            "surface_init_z",
            "surface_top_x_recovery",
            "round1_surface_post_z",
            "round1_surface_post_x",
            "round2_surface_post_z",
            "round2_surface_post_x",
        ]
        assert schedule[1][1] == ("X_ab", "X_cd")
        assert "Z_abfg" in schedule[2][1]
        assert "X_ab" in schedule[3][1]
        assert "X_cd" in schedule[3][1]

    def test_lookup_table_early_rejection_can_reject_pre_expansion_signature(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            surface_readout_basis=None,
        )

        signature = d5t.early_rejection_syndrome_signature(measurements)
        table = LookupTableEarlyRejection.from_signatures([signature])

        assert all(bit == 0 for bit in signature)
        assert d5t.early_rejection_syndrome_items(measurements)
        assert d5t.lookup_table_early_rejection_accepts(measurements)
        assert not d5t.lookup_table_early_rejection_accepts(measurements, lookup_table=table)
        assert not d5t.transfer_acceptance_succeeds(measurements, lookup_table=table)


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
