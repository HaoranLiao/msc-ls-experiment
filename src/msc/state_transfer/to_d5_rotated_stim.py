"""Stim transfer from a Steane patch to the rotated d=5 surface code.

Public circuit paths::

    build_steane_to_rotated_d5_lattice_surgery_circuit(...)
    └── append_post_transfer_and_readout(...)
        ├── append_post_transfer_stabilization(...)
        │   └── post_transfer_stim.append_rotated_full_stabilizer_rounds(...)
        └── d5.append_full_data_readout(...)

    end-to-end logical-Y experiment
    ├── append_post_transfer_stabilization(...)
    ├── apply noise through that stabilization phase
    └── append_ideal_y_verification(...)
        ├── post_transfer_stim.append_rotated_full_stabilizer_rounds(...)
        └── destructive logical-Y measurement
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import stim

from msc.rotated_surface_code import d5_stim as d5
from msc.rotated_surface_code import helper_stim as surface_helper
from msc.rotated_surface_code.post_transfer.post_transfer_stim import (
    append_rotated_full_stabilizer_rounds,
)
from msc.state_transfer import analysis_stim as analysis
from msc.state_transfer import common
from msc.state_transfer import helper_stim as helper
from msc.steane_code.state_injection_stim import StimInjectedStateName
from msc.utils.circuit_common import resolve_logical_measurement_terms
from msc.utils.normalize import normalize_pauli_char

POST_TRANSFER_SURFACE_Z_FAMILY = "post_transfer_surface_z"
POST_TRANSFER_SURFACE_X_FAMILY = "post_transfer_surface_x"


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
    withheld_surface_stabilizer_labels=("X_ab", "X_cd"),
    surface_code_operations=common.SurfaceCodeOperations(
        d5.prepare_product_plus,
        d5.append_scheduled_stabilizer_round,
        d5.append_logical_measurement,
    ),
)
# Hirano sets post_selection=True on these two bulk-X measurement objects. The
# persistent flag rejects the initial known-+1 detector and all three merge-
# round detectors in each stream; other noisy surface streams remain decodable.
HIRANO_POSTSELECTED_SURFACE_X_LABELS = common.PERSISTENT_POSTSELECTED_SURFACE_X_LABELS


# =============================================================================
# Transfer and Hirano-style post-transfer readout
# =============================================================================


def build_steane_to_rotated_d5_lattice_surgery_circuit(
    *,
    patch_sites: common.SteanePatchSites = 14,
    injected_state: StimInjectedStateName = "S",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    merge_schedule: common.MergeSchedule = "hirano",
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    full_post_selection: bool = False,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build the detector-annotated d=5 transfer using Stim's Clifford proxy.

    Stim replaces logical-H_XY cultivation with logical-Y cultivation and the
    physical T input with ``S|+>``. If ``surface_readout_basis`` is given,
    ``full_post_selection=False`` appends five complete surface rounds before
    full-data readout; ``True`` appends one. ``surface_readout_basis=None`` ends
    at ``m_ab``. Hirano's full-data boundary supports X or Z readout, not Y.

    ``minimal_z_only`` requires the 14-site feeder: verify Z0145/Z0235 in
    merge round 1 and all three source Z checks in round 2, retaining Bell
    and copy flags. Both schedules demolish all seven source qubits at the end.
    """
    circuit = helper.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        patch_sites=patch_sites,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        merge_schedule=merge_schedule,
        add_ticks=add_ticks,
    )
    if surface_readout_basis is None:
        return circuit
    surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    if surface_readout_basis == "Y":
        raise ValueError("Hirano full-data readout basis must be X or Z.")
    transfer_layout = helper.transfer_measurement_layout(
        TRANSFER_SPEC,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        merge_schedule=merge_schedule,
    )
    append_post_transfer_and_readout(
        circuit,
        transfer_layout=transfer_layout,
        readout_basis=surface_readout_basis,
        full_post_selection=full_post_selection,
        add_ticks=add_ticks,
    )
    return circuit


def append_post_transfer_and_readout(
    circuit: stim.Circuit,
    *,
    transfer_layout: helper.TransferMeasurementLayout,
    readout_basis: common.SurfaceReadoutBasis,
    full_post_selection: bool = False,
    add_ticks: bool = True,
) -> helper.TransferMeasurementLayout:
    """Append Hirano's post-transfer branch and destructive all-data readout.

    ``full_post_selection=True`` appends one full surface-stabilizer round;
    ``False`` appends five rounds for error correction. The enclosing
    logical-error experiment applies noise to either branch and uses the same
    option to choose its detector postselection mask.

    Full round 1 compares every available transfer reference. ``m_ab`` supplies
    ``X_ab``, but no standalone measurement restores ``X_cd``; its round-1
    result instead starts the new stream without emitting a detector. Later
    rounds compare every X and Z check normally.

    The final X- or Z-basis measurement reads all 25 surface data qubits,
    reconstructs the compatible stabilizers against the final round, and
    includes the logical-support parity together with the transfer-frame
    records. Noise and detector postselection policy are owned by the enclosing
    logical-error experiment.

    Stabilization is a separate protocol phase because the end-to-end
    experiment applies noise through those rounds and then appends a noiseless
    logical-Y verification. This X/Z path continues directly from the shared
    stabilization boundary to destructive readout.
    """
    # Select the post-transfer branch and append its complete stabilizer rounds.
    readout_basis = normalize_pauli_char(readout_basis)
    stabilizer_rounds = 1 if full_post_selection else TRANSFER_SPEC.distance
    post_transfer_layout = append_post_transfer_stabilization(
        circuit,
        transfer_layout=transfer_layout,
        stabilizer_rounds=stabilizer_rounds,
        add_ticks=add_ticks,
    )

    # Read every data qubit and close the compatible syndrome boundary.
    qubits = helper.transfer_qubits(
        TRANSFER_SPEC,
        patch_sites=transfer_layout.patch_sites,
    )
    surface_qubits = [qubits[label] for label in TRANSFER_SPEC.surface_order]
    if readout_basis == "X":
        compatible_family = POST_TRANSFER_SURFACE_X_FAMILY
    elif readout_basis == "Z":
        compatible_family = POST_TRANSFER_SURFACE_Z_FAMILY
    else:
        raise ValueError("post-transfer destructive readout supports only X or Z.")
    compatible_stabilizer_positions = post_transfer_layout.rounds(
        compatible_family,
        stabilizer_rounds,
    )[-1]
    data_readout_positions = d5.append_full_data_readout(
        circuit,
        readout_basis,
        surface_qubits=surface_qubits,
        previous_stabilizer_measurement_positions=compatible_stabilizer_positions,
    )
    if add_ticks:
        circuit.append("TICK", [])
    data_readout_family = f"surface_data_{readout_basis.lower()}"
    measurement_labels = dict(post_transfer_layout.labels)
    measurement_positions = dict(post_transfer_layout.positions)
    measurement_labels[data_readout_family] = TRANSFER_SPEC.surface_order
    measurement_positions[data_readout_family] = data_readout_positions

    # Select the logical support within the all-data record.
    measurement_terms, _ = resolve_logical_measurement_terms(
        readout_basis,
        logical_x_support=TRANSFER_SPEC.logical_x,
        logical_z_support=TRANSFER_SPEC.logical_z,
    )
    logical_readout_family = f"surface_logical_{readout_basis.lower()}"
    measurement_labels[logical_readout_family] = tuple(label for label, _ in measurement_terms)
    data_readout_by_label = dict(
        zip(TRANSFER_SPEC.surface_order, data_readout_positions, strict=True)
    )
    measurement_positions[logical_readout_family] = tuple(
        data_readout_by_label[label] for label, _ in measurement_terms
    )

    # Combine that support with the lattice-surgery Pauli-frame records.
    readout_layout = helper.TransferMeasurementLayout(
        patch_sites=transfer_layout.patch_sites,
        labels=measurement_labels,
        positions=measurement_positions,
        num_measurements=circuit.num_measurements,
    )
    helper.append_transfer_logical_observable(
        circuit,
        layout=readout_layout,
        basis=readout_basis,
    )
    if circuit.num_measurements != readout_layout.num_measurements:
        raise RuntimeError("post-transfer readout layout does not match the circuit.")
    return readout_layout


def append_post_transfer_stabilization(
    circuit: stim.Circuit,
    *,
    transfer_layout: helper.TransferMeasurementLayout,
    stabilizer_rounds: int,
    add_ticks: bool = True,
) -> helper.TransferMeasurementLayout:
    """Append the d=5 transfer-specific stabilization boundary.

    The shared surface-code helper appends the repeated full-round schedule.
    This wrapper remains in the state-transfer module because it anchors the
    first Z round to merge round 3, anchors compatible X checks to merge round
    3, anchors ``X_ab`` to ``m_ab``, and leaves the previously withheld
    ``X_cd`` without a detector in round 1. It also extends the transfer
    measurement layout. These are lattice-surgery boundary semantics, not
    generic rotated-patch behavior.

    The returned layout marks the noise boundary used by the end-to-end
    experiment: its sampling circuit is noisy through this phase, while the
    later verification is ideal.
    """
    if stabilizer_rounds <= 0:
        raise ValueError("stabilizer_rounds must be positive.")
    qubits = helper.transfer_qubits(
        TRANSFER_SPEC,
        patch_sites=transfer_layout.patch_sites,
    )
    surface_qubits = [qubits[label] for label in TRANSFER_SPEC.surface_order]
    # Append one or more complete surface-code rounds without a final readout.
    z_measurement_rounds, x_measurement_rounds = append_rotated_full_stabilizer_rounds(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.z_labels],
        x_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.x_labels],
        z_stabilizer_names=TRANSFER_SPEC.z_labels,
        x_stabilizer_names=TRANSFER_SPEC.x_labels,
        stabilizer_rounds=stabilizer_rounds,
        append_scheduled_stabilizer_round=(
            TRANSFER_SPEC.surface_code_operations.append_scheduled_stabilizer_round
        ),
        add_ticks=add_ticks,
    )

    # Anchor the first Z round to the final merge round, then compare in time.
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=z_measurement_rounds,
        first_round_reference=transfer_layout.rounds("merged_surface_z")[-1],
    )

    # Anchor compatible X checks to merge round 3 and X_ab to m_ab. X_cd has no
    # post-merge reference, so its first full-round result starts a new stream.
    x_reference_by_label = dict(
        zip(
            TRANSFER_SPEC.merged_x_labels,
            transfer_layout.rounds("merged_surface_x")[-1],
            strict=True,
        )
    )
    x_reference_by_label.update(transfer_layout.by_label("surface_x_recovery"))
    first_x_round_by_label = dict(zip(TRANSFER_SPEC.x_labels, x_measurement_rounds[0], strict=True))
    referenced_x_labels = tuple(
        label for label in TRANSFER_SPEC.x_labels if label in x_reference_by_label
    )
    referenced_first_x_round = tuple(first_x_round_by_label[label] for label in referenced_x_labels)
    first_x_round_reference = tuple(x_reference_by_label[label] for label in referenced_x_labels)
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=(referenced_first_x_round,),
        first_round_reference=first_x_round_reference,
    )
    if len(x_measurement_rounds) > 1:
        surface_helper.append_stabilizer_detectors(
            circuit,
            rounds=x_measurement_rounds[1:],
            first_round_reference=x_measurement_rounds[0],
        )

    # Record every post-transfer result by round and stabilizer label.
    measurement_labels = dict(transfer_layout.labels)
    measurement_positions = dict(transfer_layout.positions)
    measurement_labels[POST_TRANSFER_SURFACE_Z_FAMILY] = tuple(
        f"round{round_index + 1}_{label}"
        for round_index in range(stabilizer_rounds)
        for label in TRANSFER_SPEC.z_labels
    )
    measurement_positions[POST_TRANSFER_SURFACE_Z_FAMILY] = tuple(
        position for one_round in z_measurement_rounds for position in one_round
    )
    measurement_labels[POST_TRANSFER_SURFACE_X_FAMILY] = tuple(
        f"round{round_index + 1}_{label}"
        for round_index in range(stabilizer_rounds)
        for label in TRANSFER_SPEC.x_labels
    )
    measurement_positions[POST_TRANSFER_SURFACE_X_FAMILY] = tuple(
        position for one_round in x_measurement_rounds for position in one_round
    )
    post_transfer_layout = helper.TransferMeasurementLayout(
        patch_sites=transfer_layout.patch_sites,
        labels=measurement_labels,
        positions=measurement_positions,
        num_measurements=circuit.num_measurements,
    )
    if circuit.num_measurements != post_transfer_layout.num_measurements:
        raise RuntimeError("post-transfer stabilization layout does not match the circuit.")
    return post_transfer_layout


def append_ideal_y_verification(
    circuit: stim.Circuit,
    *,
    post_transfer_layout: helper.TransferMeasurementLayout,
    stabilizer_rounds: int,
    add_ticks: bool = True,
) -> helper.TransferMeasurementLayout:
    """Append an ideal closing syndrome round and logical-Y verification.

    This follows Hirano's ``SteanePlusSurfaceCode.run``: after the noisy
    stabilization and epilogue rounds, the authors enter ``SuppressNoise`` and
    perform one complete syndrome-extraction round before the perfect logical
    Pauli measurement. The ideal round compares against the final noisy round,
    turning residual output faults into terminal X- and Z-detector events. The
    logical-Y measurement then supplies the observable; by itself it would not
    close both stabilizer histories.

    Thus the entire suffix is diagnostic, not another noisy QEC phase. It
    measures the logical-Y eigenvalue of Stim's cultivated ``S|+>`` proxy.
    """
    if stabilizer_rounds <= 0:
        raise ValueError("stabilizer_rounds must be positive.")
    qubits = helper.transfer_qubits(
        TRANSFER_SPEC,
        patch_sites=post_transfer_layout.patch_sites,
    )
    surface_qubits = [qubits[label] for label in TRANSFER_SPEC.surface_order]
    # One ideal round closes both noisy temporal syndrome histories.
    verification_z_rounds, verification_x_rounds = append_rotated_full_stabilizer_rounds(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.z_labels],
        x_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.x_labels],
        z_stabilizer_names=TRANSFER_SPEC.z_labels,
        x_stabilizer_names=TRANSFER_SPEC.x_labels,
        stabilizer_rounds=1,
        append_scheduled_stabilizer_round=(
            TRANSFER_SPEC.surface_code_operations.append_scheduled_stabilizer_round
        ),
        add_ticks=add_ticks,
    )
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=verification_z_rounds,
        first_round_reference=post_transfer_layout.rounds(
            POST_TRANSFER_SURFACE_Z_FAMILY,
            stabilizer_rounds,
        )[-1],
    )
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=verification_x_rounds,
        first_round_reference=post_transfer_layout.rounds(
            POST_TRANSFER_SURFACE_X_FAMILY,
            stabilizer_rounds,
        )[-1],
    )

    measurement_labels = dict(post_transfer_layout.labels)
    measurement_positions = dict(post_transfer_layout.positions)
    measurement_labels["surface_verify_z"] = TRANSFER_SPEC.z_labels
    measurement_positions["surface_verify_z"] = verification_z_rounds[0]
    measurement_labels["surface_verify_x"] = TRANSFER_SPEC.x_labels
    measurement_positions["surface_verify_x"] = verification_x_rounds[0]

    # Destructively measure logical Y after the closing syndrome round.
    first_logical_measurement_position = circuit.num_measurements
    TRANSFER_SPEC.surface_code_operations.append_logical_measurement(
        circuit,
        "Y",
        surface_qubits=surface_qubits,
        destructive=True,
    )
    if add_ticks:
        circuit.append("TICK", [])
    measurement_terms, _ = resolve_logical_measurement_terms(
        "Y",
        logical_x_support=TRANSFER_SPEC.logical_x,
        logical_z_support=TRANSFER_SPEC.logical_z,
    )
    logical_family = "surface_logical_y"
    measurement_labels[logical_family] = tuple(label for label, _ in measurement_terms)
    measurement_positions[logical_family] = tuple(
        range(first_logical_measurement_position, circuit.num_measurements)
    )

    verification_layout = helper.TransferMeasurementLayout(
        patch_sites=post_transfer_layout.patch_sites,
        labels=measurement_labels,
        positions=measurement_positions,
        num_measurements=circuit.num_measurements,
    )
    helper.append_transfer_logical_observable(
        circuit,
        layout=verification_layout,
        basis="Y",
    )
    if circuit.num_measurements != verification_layout.num_measurements:
        raise RuntimeError("logical-Y verification layout does not match the circuit.")
    return verification_layout


# =============================================================================
# Parsing
# =============================================================================


def parse_transfer_measurements(
    shot: Sequence[bool | int] | np.ndarray,
    *,
    patch_sites: common.SteanePatchSites = 14,
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    merge_schedule: common.MergeSchedule = "hirano",
) -> dict[str, object]:
    """Parse a circuit ending at the ``m_ab`` transfer boundary."""
    return analysis.parse_transfer_measurements(
        shot,
        spec=TRANSFER_SPEC,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        merge_schedule=merge_schedule,
    )
