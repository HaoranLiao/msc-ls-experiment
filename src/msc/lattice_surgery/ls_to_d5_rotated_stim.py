"""Stim Clifford-proxy Steane-to-rotated-d5 transfer by lattice surgery."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import stim

from msc.lattice_surgery import (
    ls_analysis_stim as analysis,
)
from msc.lattice_surgery import (
    ls_common as common,
)
from msc.lattice_surgery import (
    ls_helper_stim as helper,
)
from msc.rotated_surface_code import d5_stim as d5
from msc.rotated_surface_code import helper_stim as surface_helper
from msc.rotated_surface_code.post_transfer import post_transfer_stim
from msc.steane_code.state_injection_stim import (
    DEFAULT_STIM_INJECTED_STATE,
    StimInjectedStateName,
)
from msc.utils.normalize import normalize_pauli_char

IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY = "ideal_post_transfer_surface_z"
IDEAL_POST_TRANSFER_SURFACE_X_FAMILY = "ideal_post_transfer_surface_x"

MERGED_BOUNDARY_CHECKS: tuple[common.MergedBoundaryCheck, ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
    ("Zde", (), ("d", "e")),
)

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d5",
    distance=5,
    surface_order=tuple(d5.SURFACE_ORDER),
    logical_x=tuple(d5.DEFAULT_LOGICAL_X),
    logical_z=tuple(d5.DEFAULT_LOGICAL_Z),
    z_stabilizers=d5.Z_STABILIZERS,
    x_stabilizers=d5.X_STABILIZERS,
    merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
    top_x_recovery_labels=("X_ab", "X_cd"),
    surface_code_operations=common.SurfaceCodeOperations(
        prepare_product_plus=d5.prepare_product_plus,
        append_scheduled_stabilizer_round=d5.append_scheduled_stabilizer_round,
        append_logical_measurement=d5.append_logical_measurement,
    ),
)


def build_steane_to_rotated_d5_lattice_surgery_circuit(
    *,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build the d=5 transfer using Stim's Clifford ``S``-state proxy."""
    return helper.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        add_ticks=add_ticks,
    )


def append_ideal_verification_and_destructive_readout(
    circuit: stim.Circuit,
    *,
    transfer_layout: helper.TransferMeasurementLayout,
    readout_basis: common.SurfaceReadoutBasis,
    add_ticks: bool = True,
) -> helper.TransferMeasurementLayout:
    """Append the ideal closing round, all-data readout, and annotations.

    The transfer circuit must already contain its protocol and merge-surface
    detectors. This continuation closes the restored surface reference with
    one noise-free full round, closes matching-basis stabilizers against a
    destructive all-data readout, and includes the transfer-corrected logical
    observable used by lattice-surgery-only error-rate experiments.
    """
    readout_basis = normalize_pauli_char(readout_basis)
    if readout_basis == "Y":
        raise ValueError("ideal all-data verification requires an X or Z basis.")

    registers = helper.build_transfer_qregs(TRANSFER_SPEC)
    surface_qubits = helper.surface_data_qubits(registers, TRANSFER_SPEC.surface_order)
    merged_z_reference = transfer_layout.record("merged_surface_z").round_positions(
        rounds=common.LATTICE_SURGERY_ROUNDS
    )[-1]
    post_transfer_stim.append_rotated_post_transfer(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[
            registers.surface_z_ancillas[label.lower()] for label in TRANSFER_SPEC.z_labels
        ],
        x_ancillas=[
            registers.surface_x_ancillas[label.lower()] for label in TRANSFER_SPEC.x_labels
        ],
        z_stabilizer_names=TRANSFER_SPEC.z_labels,
        x_stabilizer_names=TRANSFER_SPEC.x_labels,
        stabilizer_rounds=1,
        append_scheduled_stabilizer_round=(
            TRANSFER_SPEC.surface_code_operations.append_scheduled_stabilizer_round
        ),
        logical_readout_basis=None,
        append_logical_measurement=(
            TRANSFER_SPEC.surface_code_operations.append_logical_measurement
        ),
        z_reference_measurement_positions=merged_z_reference,
        x_reference_measurement_positions=helper.restored_surface_x_reference_positions(
            TRANSFER_SPEC, transfer_layout
        ),
        add_ticks=add_ticks,
    )
    surface_helper.append_all_data_readout(
        circuit,
        data_qubits=surface_qubits,
        basis=readout_basis,
        add_ticks=add_ticks,
    )

    surface_data_family = f"ideal_surface_data_{readout_basis.lower()}"
    layout = transfer_layout.with_appended_records(
        (
            (IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY, TRANSFER_SPEC.z_labels),
            (IDEAL_POST_TRANSFER_SURFACE_X_FAMILY, TRANSFER_SPEC.x_labels),
            (surface_data_family, TRANSFER_SPEC.surface_order),
        )
    )
    if circuit.num_measurements != layout.num_measurements:
        raise RuntimeError("ideal verification layout does not match the circuit.")

    final_stabilizer_record_name = (
        IDEAL_POST_TRANSFER_SURFACE_X_FAMILY
        if readout_basis == "X"
        else IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY
    )
    d5.append_all_data_readout_detectors_and_observable(
        circuit,
        readout_basis,
        final_stabilizer_measurement_positions=layout.record(
            final_stabilizer_record_name
        ).positions_by_label(),
        data_measurement_positions=layout.record(surface_data_family).positions_by_label(),
        logical_frame_measurement_positions=helper.transfer_logical_frame_positions(
            transfer_layout, readout_basis
        ),
    )
    return layout


def parse_transfer_measurements(
    shot: Sequence[bool | int] | np.ndarray,
    *,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    """Parse one d=5 Stim record row into labeled protocol fields."""
    return analysis.parse_transfer_measurements(
        shot,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
