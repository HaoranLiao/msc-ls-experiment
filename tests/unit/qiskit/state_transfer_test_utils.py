"""Shared assertions for rotated-surface lattice-surgery transfer tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from qiskit import QuantumCircuit

from msc.state_transfer_by_lattice_surgery import (
    state_transfer_common as common,
)
from msc.surface_code import rotated_common


def labeled_measurement_positions(
    circuit: QuantumCircuit,
    *,
    register_name: str,
    labels: Sequence[str],
) -> dict[str, int]:
    """Map semantic labels to their measurement-instruction positions."""
    register = next(register for register in circuit.cregs if register.name == register_name)
    assert len(register) == len(labels)
    register_bits = set(register)

    position_by_bit = {
        instruction.clbits[0]: position
        for position, instruction in enumerate(circuit.data)
        if instruction.operation.name == "measure" and instruction.clbits[0] in register_bits
    }
    assert set(position_by_bit) == register_bits
    return {
        label: position_by_bit[classical_bit]
        for label, classical_bit in zip(labels, register, strict=True)
    }


def assert_merge_round_measurement_groups(
    circuit: QuantumCircuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    round_number: int,
    expected_steane_checks: tuple[str, ...],
) -> None:
    """Check one table row's interface, surface, and color-code measurements."""
    boundary_names = tuple(name for name, _, _ in spec.merged_boundary_checks)
    boundary_positions = labeled_measurement_positions(
        circuit,
        register_name="ls_zz",
        labels=common.round_labels(boundary_names),
    )
    surface_z_positions = labeled_measurement_positions(
        circuit,
        register_name="merged_surface_z",
        labels=common.round_labels(spec.z_labels),
    )
    surface_x_positions = labeled_measurement_positions(
        circuit,
        register_name="merged_surface_x",
        labels=common.round_labels(spec.merged_x_labels),
    )
    steane_positions = labeled_measurement_positions(
        circuit,
        register_name="steane_verify",
        labels=common.STEANE_LS_VERIFICATION_LABELS,
    )

    prefix = f"round{round_number}_"
    selected_boundary_positions = [boundary_positions[f"{prefix}{name}"] for name in boundary_names]
    selected_surface_positions = [
        surface_z_positions[f"{prefix}{label}"] for label in spec.z_labels
    ] + [surface_x_positions[f"{prefix}{label}"] for label in spec.merged_x_labels]
    selected_steane_labels = tuple(
        label.removeprefix(prefix)
        for label in common.STEANE_LS_VERIFICATION_LABELS
        if label.startswith(prefix)
    )
    selected_steane_positions = [
        steane_positions[f"{prefix}{label}"] for label in expected_steane_checks
    ]

    assert selected_steane_labels == expected_steane_checks
    assert set(spec.z_labels) == set(spec.z_stabilizers)
    assert set(spec.merged_x_labels) == set(spec.x_labels) - set(spec.top_x_recovery_labels)
    assert max(selected_boundary_positions) < min(selected_surface_positions)

    if selected_steane_positions:
        assert max(selected_surface_positions) < min(selected_steane_positions)
        round_end = max(selected_steane_positions)
    else:
        round_end = max(selected_surface_positions)

    if round_number < common.LATTICE_SURGERY_ROUNDS:
        next_prefix = f"round{round_number + 1}_"
        next_boundary_start = min(
            boundary_positions[f"{next_prefix}{name}"] for name in boundary_names
        )
        assert round_end < next_boundary_start
    else:
        demolition_positions = labeled_measurement_positions(
            circuit,
            register_name="steane_x",
            labels=common.HIRANO_STEANE_DATA_ORDER,
        )
        assert round_end < min(demolition_positions.values())


def assert_optional_post_transfer_continuation(
    circuit: QuantumCircuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    logical_register_name: str,
    logical_support: tuple[str, ...],
) -> None:
    """Check that distance full rounds precede the terminal logical readout."""
    registers = {register.name: register for register in circuit.cregs}
    post_positions = labeled_measurement_positions(
        circuit,
        register_name="post_transfer_surface_z",
        labels=rotated_common.repeated_round_labels(
            spec.z_labels,
            rounds=spec.distance,
        ),
    ) | labeled_measurement_positions(
        circuit,
        register_name="post_transfer_surface_x",
        labels=rotated_common.repeated_round_labels(
            spec.x_labels,
            rounds=spec.distance,
        ),
    )
    logical_positions = labeled_measurement_positions(
        circuit,
        register_name=logical_register_name,
        labels=logical_support,
    )

    assert len(registers["post_transfer_surface_z"]) == (spec.distance * len(spec.z_labels))
    assert len(registers["post_transfer_surface_x"]) == (spec.distance * len(spec.x_labels))
    assert max(post_positions.values()) < min(logical_positions.values())


def assert_demolition_measurement_groups(
    circuit: QuantumCircuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    measurements: dict[str, object],
) -> None:
    """Check demolition, top-edge recovery, and reconstructed final X checks."""
    surface_z_positions = labeled_measurement_positions(
        circuit,
        register_name="merged_surface_z",
        labels=common.round_labels(spec.z_labels),
    )
    surface_x_positions = labeled_measurement_positions(
        circuit,
        register_name="merged_surface_x",
        labels=common.round_labels(spec.merged_x_labels),
    )
    demolition_positions = labeled_measurement_positions(
        circuit,
        register_name="steane_x",
        labels=common.HIRANO_STEANE_DATA_ORDER,
    )
    recovery_positions = labeled_measurement_positions(
        circuit,
        register_name="surface_x_recovery",
        labels=spec.top_x_recovery_labels,
    )

    round_three_positions = [surface_z_positions[f"round3_{label}"] for label in spec.z_labels] + [
        surface_x_positions[f"round3_{label}"] for label in spec.merged_x_labels
    ]
    round_three_end = max(round_three_positions)
    assert round_three_end < min(demolition_positions.values())
    assert max(demolition_positions.values()) < min(recovery_positions.values())

    for measurement_position in demolition_positions.values():
        measured_qubit = circuit.data[measurement_position].qubits[0]
        assert any(
            instruction.operation.name == "h" and measured_qubit in instruction.qubits
            for instruction in circuit.data[round_three_end + 1 : measurement_position]
        )

    steane_x = measurements["steane_x"]
    recovered_x_ab = measurements["surface_x_recovery"]["X_ab"]
    assert measurements["final_x_verification_checks"] == {
        "X0235": common.parity(steane_x, ("0", "2", "3", "5")),
        "X0246": common.parity(steane_x, ("0", "2", "4", "6")),
        "X0145ab": (common.parity(steane_x, ("0", "1", "4", "5")) ^ recovered_x_ab),
    }


def assert_transfer_detection_policy(measurements: dict[str, object]) -> None:
    """Check hard boundary/color rejection versus decoded surface outcomes."""
    common_values = measurements["merged_zz_rounds"][-1]
    assert common.merged_boundary_logical_zz_outcome(measurements) == (
        sum(common_values.values()) % 2
    )

    boundary_names = tuple(measurements["merged_zz_rounds"][0])
    for boundary_name in boundary_names[:2]:
        measurements["merged_zz_rounds"][1][boundary_name] ^= 1

    assert not common.merged_boundary_checks_stable(measurements)
    assert not common.transfer_acceptance_succeeds(measurements)
    with pytest.raises(ValueError, match="individually stable"):
        common.merged_boundary_logical_zz_outcome(measurements)

    for boundary_name in boundary_names[:2]:
        measurements["merged_zz_rounds"][1][boundary_name] ^= 1
    measurements["steane_verification"]["round1_Z0145"] ^= 1
    assert not common.transfer_acceptance_succeeds(measurements)

    measurements["steane_verification"]["round1_Z0145"] ^= 1
    measurements["merged_surface_logical_x_byproduct"] ^= 1
    assert common.transfer_acceptance_succeeds(measurements)
