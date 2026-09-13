"""Configuration-to-circuit contracts for Qiskit Ty tomography and LER."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from msc.experiment.ty_tomography_diagnostics import ty_tomography_retainment_breakdown
from msc.state_transfer import common
from msc.state_transfer.ler_qiskit import (
    analyze_ty_logical_error_rate_counts,
    build_ty_logical_error_rate_circuits,
)
from msc.state_transfer.tomography_qiskit import (
    TomographyTargetDistance,
    analyze_ty_tomography_counts,
    build_ty_tomography_circuits,
)

TOMOGRAPHY_CONFIG: dict[str, object] = {
    "target_distance": 5,
    "patch_sites": 14,
    "full_post_selection": False,
    "persistent_surface_x_post_selection": True,
    "post_transfer_stabilizer_rounds": 2,
    "add_barriers": False,
}


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({**TOMOGRAPHY_CONFIG, **overrides}, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _bitstring(positions: tuple[int, ...], width: int) -> str:
    bits = ["0"] * width
    for position in positions:
        bits[width - 1 - position] = "1"
    return "".join(bits)


@pytest.mark.parametrize(
    ("target_distance", "patch_sites", "rounds", "add_barriers", "encoder_flags"),
    ((3, 11, 2, False, 4), (5, 14, 4, True, 5)),
)
def test_tomography_yaml_matches_constructed_circuits_and_analysis(
    tmp_path: Path,
    target_distance: TomographyTargetDistance,
    patch_sites: common.SteanePatchSites,
    rounds: int,
    add_barriers: bool,
    encoder_flags: int,
) -> None:
    config_path = _write_config(
        tmp_path,
        target_distance=target_distance,
        patch_sites=patch_sites,
        post_transfer_stabilizer_rounds=rounds,
        add_barriers=add_barriers,
    )

    circuits = build_ty_tomography_circuits(config_path=config_path)

    expected_data_qubits = target_distance**2
    expected_stabilizers_per_basis = (expected_data_qubits - 1) // 2
    for basis, circuit in circuits.items():
        registers = {register.name: register for register in circuit.cregs}
        assert len(registers["enc_flags"]) == encoder_flags
        assert len(registers["post_transfer_surface_z"]) == (
            rounds * expected_stabilizers_per_basis
        )
        assert len(registers["post_transfer_surface_x"]) == (
            rounds * expected_stabilizers_per_basis
        )
        assert len(registers[f"surface_data_{basis.lower()}"]) == expected_data_qubits
        assert bool(circuit.count_ops().get("barrier")) is add_barriers

    summary = analyze_ty_tomography_counts(
        {
            "X": {"0" * circuits["X"].num_clbits: 4},
            "Z": {"0" * circuits["Z"].num_clbits: 4},
        },
        config_path=config_path,
    )
    assert summary.accepted_x_shots == summary.accepted_z_shots == 4


@pytest.mark.parametrize(
    ("full_post_selection", "persistent_surface_x_post_selection", "retained_shots"),
    ((False, True, 70), (False, False, 80), (True, False, 70)),
)
def test_postselection_yaml_controls_acceptance(
    tmp_path: Path,
    full_post_selection: bool,
    persistent_surface_x_post_selection: bool,
    retained_shots: int,
) -> None:
    config_path = _write_config(
        tmp_path,
        full_post_selection=full_post_selection,
        persistent_surface_x_post_selection=persistent_surface_x_post_selection,
    )
    circuit = build_ty_tomography_circuits(config_path=config_path)["X"]
    registers = {register.name: register for register in circuit.cregs}
    encoder_flag = circuit.find_bit(registers["enc_flags"][0]).index
    persistent_surface_x = circuit.find_bit(registers["post_transfer_surface_x"][2]).index
    counts = {
        _bitstring((), circuit.num_clbits): 70,
        _bitstring((encoder_flag,), circuit.num_clbits): 20,
        _bitstring((persistent_surface_x,), circuit.num_clbits): 10,
    }

    breakdown = ty_tomography_retainment_breakdown(
        counts,
        basis="X",
        config_path=config_path,
    )

    assert breakdown.retained_shots == retained_shots


def test_ler_yaml_matches_sweep_circuits_and_analysis(tmp_path: Path) -> None:
    rounds = (1, 2, 4)
    config_path = _write_config(
        tmp_path,
        target_distance=3,
        patch_sites=11,
        persistent_surface_x_post_selection=False,
        post_transfer_stabilizer_rounds=list(rounds),
    )

    circuits_by_round = build_ty_logical_error_rate_circuits(config_path=config_path)

    assert tuple(circuits_by_round) == rounds
    counts_by_round: dict[
        int,
        dict[common.SurfaceReadoutBasis, dict[str, int]],
    ] = {}
    for round_count, circuits in circuits_by_round.items():
        x_circuit = circuits["X"]
        x_registers = {register.name: register for register in x_circuit.cregs}
        persistent_surface_x = x_circuit.find_bit(x_registers["post_transfer_surface_x"][2]).index
        counts_by_round[round_count] = {
            "X": {
                _bitstring((), x_circuit.num_clbits): 9,
                _bitstring((persistent_surface_x,), x_circuit.num_clbits): 1,
            },
            "Z": {"0" * circuits["Z"].num_clbits: 10},
        }
        for basis, circuit in circuits.items():
            registers = {register.name: register for register in circuit.cregs}
            assert len(registers["enc_flags"]) == 4
            assert len(registers["post_transfer_surface_z"]) == 4 * round_count
            assert len(registers["post_transfer_surface_x"]) == 4 * round_count
            assert len(registers[f"surface_data_{basis.lower()}"]) == 9
            assert "barrier" not in circuit.count_ops()

    summary = analyze_ty_logical_error_rate_counts(
        counts_by_round,
        config_path=config_path,
    )
    assert tuple(point.post_transfer_stabilizer_rounds for point in summary.points) == rounds
    assert all(point.tomography.accepted_x_shots == 10 for point in summary.points)
    assert all(point.tomography.accepted_z_shots == 10 for point in summary.points)
    assert summary.fit.logical_error_rate_per_round == pytest.approx(0.0, abs=1e-8)
