"""Shared assertions for Stim lattice-surgery transfer tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import stim

from msc.lattice_surgery import (
    ls_common as common,
)
from msc.lattice_surgery import (
    ls_helper_stim as helper,
)
from msc.steane_code.state_injection_stim import (
    PATCH_STABILIZER_ANCILLA_SITES,
)


def measured_qubits(circuit: stim.Circuit) -> tuple[int, ...]:
    """Return measured qubits in Stim record order for this circuit family."""
    return tuple(
        target.qubit_value
        for instruction in circuit
        if instruction.name in {"M", "MX", "MY"}
        for target in instruction.targets_copy()
    )


def assert_merge_round_measurement_groups(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    round_number: int,
    expected_steane_checks: tuple[str, ...],
) -> None:
    """Check one merged round's boundary, surface, and Steane groups."""
    layout = helper.transfer_measurement_layout(
        spec,
        include_cultivation=False,
        surface_readout_basis=None,
    )
    qregs = helper.build_transfer_qregs(spec)
    measured = measured_qubits(circuit)
    prefix = f"round{round_number}_"

    def positions(family: str, selected_labels: Sequence[str]) -> tuple[int, ...]:
        record = layout.record(family)
        position_by_label = dict(zip(record.labels, record.positions, strict=True))
        return tuple(position_by_label[label] for label in selected_labels)

    boundary_names = tuple(name for name, _, _ in spec.merged_boundary_checks)
    boundary_labels = tuple(f"{prefix}{name}" for name in boundary_names)
    z_labels = tuple(f"{prefix}{name}" for name in spec.z_labels)
    x_labels = tuple(f"{prefix}{name}" for name in spec.merged_x_labels)
    steane_labels = tuple(f"{prefix}{name}" for name in expected_steane_checks)
    boundary_positions = positions("ls_zz", boundary_labels)
    z_positions = positions("merged_surface_z", z_labels)
    x_positions = positions("merged_surface_x", x_labels)
    steane_positions = positions("steane_verify", steane_labels) if steane_labels else ()

    assert tuple(measured[position] for position in boundary_positions) == tuple(
        qregs.merge_ancillas[f"zz_{name[1:].lower()}"] for name in boundary_names
    )
    assert tuple(measured[position] for position in z_positions) == tuple(
        qregs.surface_z_ancillas[name.lower()] for name in spec.z_labels
    )
    assert tuple(measured[position] for position in x_positions) == tuple(
        qregs.surface_x_ancillas[name.lower()] for name in spec.merged_x_labels
    )
    if steane_positions:
        patch_qubits = helper.hirano_labeled_patch_qubits(qregs)
        assert tuple(measured[position] for position in steane_positions) == tuple(
            patch_qubits[
                PATCH_STABILIZER_ANCILLA_SITES[
                    common.STEANE_LS_VERIFICATION_STABILIZER_INDICES[check_name]
                ]
            ]
            for check_name in expected_steane_checks
        )
    assert max(boundary_positions) < min(z_positions + x_positions)
    if steane_positions:
        assert max(z_positions + x_positions) < min(steane_positions)


def assert_optional_post_transfer_continuation(
    *,
    spec: common.SurfaceCodeTransferSpec,
) -> None:
    """Check that distance full rounds precede terminal logical readout."""
    layout = helper.transfer_measurement_layout(
        spec,
        include_cultivation=False,
        include_post_transfer_stabilizer_rounds=True,
        surface_readout_basis="X",
    )
    post_z = layout.record("post_transfer_surface_z")
    post_x = layout.record("post_transfer_surface_x")
    logical = layout.record("surface_logical_x")
    assert len(post_z.positions) == spec.distance * len(spec.z_labels)
    assert len(post_x.positions) == spec.distance * len(spec.x_labels)
    assert max(post_z.positions + post_x.positions) < min(logical.positions)


def assert_demolition_measurement_groups(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    measurements: dict[str, object],
) -> None:
    """Check Steane demolition, top-edge recovery, and final X checks."""
    layout = helper.transfer_measurement_layout(
        spec,
        include_cultivation=False,
        surface_readout_basis=None,
    )
    qregs = helper.build_transfer_qregs(spec)
    measured = measured_qubits(circuit)
    demolition = layout.record("steane_x")
    recovery = layout.record("surface_x_recovery")
    assert tuple(measured[position] for position in demolition.positions) == tuple(
        qregs.steane_data[label] for label in common.HIRANO_STEANE_DATA_ORDER
    )
    assert tuple(measured[position] for position in recovery.positions) == tuple(
        qregs.surface_x_ancillas[label.lower()] for label in spec.top_x_recovery_labels
    )
    assert max(demolition.positions) < min(recovery.positions)

    steane_x = measurements["steane_x"]
    recovered_x_ab = measurements["surface_x_recovery"]["X_ab"]
    assert measurements["final_x_verification_checks"] == {
        "X0235": common.parity(steane_x, ("0", "2", "3", "5")),
        "X0246": common.parity(steane_x, ("0", "2", "4", "6")),
        "X0145ab": common.parity(steane_x, ("0", "1", "4", "5")) ^ recovered_x_ab,
    }


def assert_transfer_detection_policy(measurements: dict[str, object]) -> None:
    """Check hard boundary/Steane rejection versus decoded surface data."""
    common_values = measurements["merged_zz_rounds"][-1]
    assert common.merged_boundary_logical_zz_outcome(measurements) == (
        sum(common_values.values()) % 2
    )
    boundary_name = next(iter(measurements["merged_zz_rounds"][0]))
    measurements["merged_zz_rounds"][1][boundary_name] ^= 1
    assert not common.merged_boundary_checks_stable(measurements)
    assert not common.transfer_acceptance_succeeds(measurements)
    with pytest.raises(ValueError, match="individually stable"):
        common.merged_boundary_logical_zz_outcome(measurements)

    measurements["merged_zz_rounds"][1][boundary_name] ^= 1
    measurements["steane_verification"]["round1_Z0145"] ^= 1
    assert not common.transfer_acceptance_succeeds(measurements)

    measurements["steane_verification"]["round1_Z0145"] ^= 1
    measurements["merged_surface_logical_x_byproduct"] ^= 1
    assert common.transfer_acceptance_succeeds(measurements)
