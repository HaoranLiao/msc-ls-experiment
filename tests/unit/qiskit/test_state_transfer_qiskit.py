"""Tests for Steane-to-rotated-d3 lattice-surgery transfer helpers.

Organized into three test sets:
1. Circuit labels and paper-aligned d=3 check schedule
2. Measurement parsing, postselection predicates, and post-transfer QEC data
3. Logical-state transfer correctness after Pauli-frame interpretation
"""

from __future__ import annotations

from collections import Counter

from qiskit_aer import AerSimulator

from research_qec_haoran.msc_by_lattice_surgery import (
    HIRANO_STEANE_DATA_ORDER,
    HIRANO_STEANE_LOGICAL_X,
    MERGED_BOUNDARY_CHECKS,
    STEANE_LS_VERIFICATION_LABELS,
    build_steane_to_rotated_d3_lattice_surgery_circuit,
    corrected_surface_logical_readout,
    merged_boundary_checks_stable,
    parse_transfer_measurements,
    steane_verification_succeeds,
    transfer_acceptance_succeeds,
    x_verification_succeeds,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _one_parsed_shot(**circuit_kwargs):
    parse_kwargs = {
        "include_cultivation": circuit_kwargs.get("include_cultivation", True),
        "merged_rounds": circuit_kwargs.get("merged_rounds", 3),
        "noisy": circuit_kwargs.get("noisy", False),
        "include_steane_verification": circuit_kwargs.get("include_steane_verification", True),
        "include_surface_qec": circuit_kwargs.get("include_surface_qec", True),
        "surface_qec_rounds": circuit_kwargs.get("surface_qec_rounds", 3),
        "surface_readout_basis": circuit_kwargs.get("surface_readout_basis"),
        "surface_readout_mode": circuit_kwargs.get("surface_readout_mode", "data"),
    }
    circuit = build_steane_to_rotated_d3_lattice_surgery_circuit(
        add_barriers=False,
        **circuit_kwargs,
    )
    bitstring = next(iter(AerSimulator().run(circuit, shots=1).result().get_counts()))
    return parse_transfer_measurements(bitstring, **parse_kwargs)


def _corrected_logical_counts(
    *,
    injected_state: str,
    include_cultivation: bool,
    basis: str,
    shots: int = 512,
) -> Counter[int]:
    circuit = build_steane_to_rotated_d3_lattice_surgery_circuit(
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        surface_readout_basis=basis,
        add_barriers=False,
    )
    counts = AerSimulator().run(circuit, shots=shots).result().get_counts()

    logical_counts: Counter[int] = Counter()
    for bitstring, count in counts.items():
        measurements = parse_transfer_measurements(
            bitstring,
            include_cultivation=include_cultivation,
            surface_readout_basis=basis,
        )
        logical_bit = corrected_surface_logical_readout(measurements, basis=basis)
        logical_counts[logical_bit] += count
    return logical_counts


# ===========================================================================
# SET 1: Circuit labels and paper-aligned d=3 check schedule
# ===========================================================================


class TestTransferCircuitLayout:
    """Verify the transfer circuit exposes the intended paper-style interface."""

    def test_circuit_exposes_paper_labeled_data_registers(self) -> None:
        circuit = build_steane_to_rotated_d3_lattice_surgery_circuit(
            include_cultivation=True,
            surface_readout_basis=None,
            add_barriers=False,
        )
        register_names = {register.name for register in circuit.qregs}

        for label in HIRANO_STEANE_DATA_ORDER:
            assert label in register_names

        for label in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
            assert label in register_names

    def test_merged_boundary_checks_match_d3_adaptation(self) -> None:
        circuit = build_steane_to_rotated_d3_lattice_surgery_circuit(
            include_cultivation=True,
            surface_readout_basis=None,
            add_barriers=False,
        )

        assert MERGED_BOUNDARY_CHECKS == (
            ("Z1a", ("1",), ("a",)),
            ("Z35bc", ("3", "5"), ("b", "c")),
        )
        assert {register.name for register in circuit.qregs if register.name.startswith("zz_")} == {
            "zz_1a",
            "zz_35bc",
        }
        assert circuit.cregs[2].name == "ls_zz"
        # Three rounds, two merged-boundary checks per round.
        assert circuit.cregs[2].size == 6

    def test_paper_style_final_x_checks_and_logical_x_are_declared(self) -> None:
        assert HIRANO_STEANE_LOGICAL_X == ("1", "4", "6")
        assert STEANE_LS_VERIFICATION_LABELS == (
            "Z0145",
            "Z0235",
            "X0235",
            "Z0246",
            "X0246",
        )


# ===========================================================================
# SET 2: Measurement parsing, postselection predicates, and QEC data
# ===========================================================================


class TestTransferMeasurementParsing:
    """Verify parsed measurement records expose each protocol stage clearly."""

    def test_parser_exposes_readable_register_chunks_for_complete_transfer(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            noisy=True,
            surface_readout_basis="Z",
            surface_readout_mode="logical_pauli",
        )

        assert set(measurements["registers"]) == {
            "init_z",
            "ls_zz",
            "steane_verify",
            "steane_x",
            "surface_x_recovery",
            "surface_post_z",
            "surface_post_x",
            "surface_logical_z",
        }
        assert measurements["registers"]["surface_logical_z"]["labels"] == ("a", "b", "c")

    def test_transfer_acceptance_predicates_accept_noiseless_basis_state(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            noisy=True,
            surface_readout_basis=None,
        )

        assert set(measurements["reconstructed_x_checks"]) == {"X0235", "X0246", "X0145ab"}
        assert set(measurements["local_x_verification_checks"]) == {
            "X0145ab",
            "X0235",
            "X0246",
        }
        assert merged_boundary_checks_stable(measurements)
        assert steane_verification_succeeds(measurements)
        assert x_verification_succeeds(measurements)
        assert measurements["x_verification_succeeds"]
        assert transfer_acceptance_succeeds(measurements)

    def test_post_transfer_surface_qec_rounds_have_trivial_noiseless_deltas(self) -> None:
        measurements = _one_parsed_shot(
            injected_state="0",
            include_cultivation=False,
            surface_readout_basis=None,
        )

        assert len(measurements["surface_post_z_rounds"]) == 3
        assert len(measurements["surface_post_x_rounds"]) == 3
        assert len(measurements["surface_qec_deltas"]) == 3
        assert measurements["surface_qec_syndrome_is_trivial"]


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

        assert corrected_surface_logical_readout(zero_measurements, basis="Z") == 0
        assert corrected_surface_logical_readout(plus_measurements, basis="X") == 0
