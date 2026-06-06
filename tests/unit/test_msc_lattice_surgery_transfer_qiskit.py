from __future__ import annotations

from collections import Counter

from qiskit_aer import AerSimulator

from research_qec_haoran.msc_by_lattice_surgery import (
    MERGED_BOUNDARY_CHECKS,
    PAPER_STEANE_DATA_ORDER,
    build_steane_to_rotated_d3_lattice_surgery_circuit,
    corrected_surface_logical_readout,
    parse_transfer_measurements,
)


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
        merged_rounds=1,
        surface_readout_basis=basis,
        add_barriers=False,
    )
    counts = AerSimulator().run(circuit, shots=shots).result().get_counts()

    logical_counts: Counter[int] = Counter()
    for bitstring, count in counts.items():
        measurements = parse_transfer_measurements(
            bitstring,
            include_cultivation=include_cultivation,
            merged_rounds=1,
            surface_readout_basis=basis,
        )
        logical_bit = corrected_surface_logical_readout(measurements, basis=basis)
        logical_counts[logical_bit] += count
    return logical_counts


def test_transfer_circuit_exposes_paper_labeled_data_registers() -> None:
    circuit = build_steane_to_rotated_d3_lattice_surgery_circuit(
        include_cultivation=True,
        surface_readout_basis=None,
        add_barriers=False,
    )
    register_names = {register.name for register in circuit.qregs}

    for label in PAPER_STEANE_DATA_ORDER:
        assert label in register_names

    for label in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        assert label in register_names


def test_transfer_measures_paper_style_merged_boundary_checks() -> None:
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
    assert circuit.cregs[2].size == 2


def test_transfer_preserves_logical_z_basis_states_without_cultivation() -> None:
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


def test_transfer_preserves_logical_x_basis_states_without_cultivation() -> None:
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
