from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import stim

from msc.rotated_surface_code import d5_stim as d5
from msc.rotated_surface_code import helper_stim as surface_helper
from msc.rotated_surface_code.post_transfer import post_transfer_stim
from msc.state_transfer import analysis_stim as analysis
from msc.state_transfer import common
from msc.state_transfer import helper_stim as helper
from msc.steane_code.state_injection_stim import (
    DEFAULT_STIM_INJECTED_STATE,
    StimInjectedStateName,
)
from msc.utils.circuit_common import resolve_logical_measurement_terms
from msc.utils.normalize import normalize_pauli_char

IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY = "ideal_post_transfer_surface_z"
IDEAL_POST_TRANSFER_SURFACE_X_FAMILY = "ideal_post_transfer_surface_x"


TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d5",
    distance=5,
    surface_order=tuple(d5.SURFACE_ORDER),
    logical_x=tuple(d5.DEFAULT_LOGICAL_X),
    logical_z=tuple(d5.DEFAULT_LOGICAL_Z),
    z_stabilizers=d5.Z_STABILIZERS,
    x_stabilizers=d5.X_STABILIZERS,
    merged_boundary_checks=(
        ("Z1a", ("1",), ("a",)),
        ("Z35bc", ("3", "5"), ("b", "c")),
        ("Zde", (), ("d", "e")),
    ),
    top_x_recovery_labels=("X_ab", "X_cd"),
    surface_code_operations=common.SurfaceCodeOperations(
        d5.prepare_product_plus,
        d5.append_scheduled_stabilizer_round,
        d5.append_logical_measurement,
    ),
)
MERGED_BOUNDARY_CHECKS = TRANSFER_SPEC.merged_boundary_checks


# =============================================================================
# Transfer and ideal closing boundary
# =============================================================================


def build_steane_to_rotated_d5_lattice_surgery_circuit(
    *,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build the detector-annotated d=5 transfer using Stim's Clifford proxy.

    Stim replaces logical-H_XY cultivation with logical-Y cultivation and the
    physical T input with ``S|+>``. The transfer schedule, postselection
    detectors, spacetime surface detectors, optional fixed-patch continuation,
    and transfer-corrected logical observable otherwise follow the shared spec.
    """
    return helper.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        add_ticks=add_ticks,
    )


def append_ideal_verification_and_readout(
    circuit: stim.Circuit,
    *,
    transfer_layout: helper.TransferMeasurementLayout,
    readout_basis: common.SurfaceReadoutBasis,
    add_ticks: bool = True,
) -> helper.TransferMeasurementLayout:
    """Append an ideal closing round and transfer-corrected readout.

    Call structure: close the restored surface-check frame with one full round
    -> read out the requested logical basis -> include the transfer-frame-
    corrected observable. X/Z use all-data readout and final-boundary
    detectors; Y uses its logical support because no single-basis all-data
    measurement can close both CSS sectors.
    """
    readout_basis = normalize_pauli_char(readout_basis)

    qubits = helper.transfer_qubits(TRANSFER_SPEC)
    surface_qubits = [qubits[label] for label in TRANSFER_SPEC.surface_order]
    operations = TRANSFER_SPEC.surface_code_operations

    # Close the noisy transfer against a noise-free full surface round.
    post_transfer_stim.append_rotated_post_transfer(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.z_labels],
        x_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.x_labels],
        z_stabilizer_names=TRANSFER_SPEC.z_labels,
        x_stabilizer_names=TRANSFER_SPEC.x_labels,
        stabilizer_rounds=1,
        append_scheduled_stabilizer_round=operations.append_scheduled_stabilizer_round,
        logical_readout_basis=None,
        append_logical_measurement=operations.append_logical_measurement,
        z_reference_measurement_positions=transfer_layout.rounds("merged_surface_z")[-1],
        x_reference_measurement_positions=helper.restored_surface_x_reference_positions(
            TRANSFER_SPEC, transfer_layout
        ),
        add_ticks=add_ticks,
    )

    ideal_round_records = (
        (IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY, TRANSFER_SPEC.z_labels),
        (IDEAL_POST_TRANSFER_SURFACE_X_FAMILY, TRANSFER_SPEC.x_labels),
    )
    if readout_basis == "Y":
        measurement_terms, _ = resolve_logical_measurement_terms(
            "Y",
            logical_x_support=TRANSFER_SPEC.logical_x,
            logical_z_support=TRANSFER_SPEC.logical_z,
        )
        d5.append_logical_measurement(
            circuit,
            "Y",
            surface_qubits=surface_qubits,
            destructive=True,
        )
        if add_ticks:
            circuit.append("TICK", [])
        layout = transfer_layout.appended(
            ideal_round_records
            + (("surface_logical_y", tuple(label for label, _ in measurement_terms)),)
        )
        helper.append_transfer_logical_observable(circuit, layout=layout, basis="Y")
    else:
        # X/Z all-data readout also closes matching-basis stabilizers.
        surface_helper.append_all_data_readout(
            circuit,
            data_qubits=surface_qubits,
            basis=readout_basis,
            add_ticks=add_ticks,
        )
        data_family = f"ideal_surface_data_{readout_basis.lower()}"
        layout = transfer_layout.appended(
            ideal_round_records + ((data_family, TRANSFER_SPEC.surface_order),)
        )
        final_family = (
            IDEAL_POST_TRANSFER_SURFACE_X_FAMILY
            if readout_basis == "X"
            else IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY
        )
        d5.append_all_data_readout_detectors_and_observable(
            circuit,
            readout_basis,
            final_stabilizer_measurement_positions=layout.by_label(final_family),
            data_measurement_positions=layout.by_label(data_family),
            logical_frame_measurement_positions=helper.transfer_logical_frame_positions(
                transfer_layout, readout_basis
            ),
        )
    if circuit.num_measurements != layout.num_measurements:
        raise RuntimeError("ideal verification layout does not match the circuit.")
    return layout


# =============================================================================
# Parsing
# =============================================================================


def parse_transfer_measurements(
    shot: Sequence[bool | int] | np.ndarray,
    *,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    return analysis.parse_transfer_measurements(
        shot,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
