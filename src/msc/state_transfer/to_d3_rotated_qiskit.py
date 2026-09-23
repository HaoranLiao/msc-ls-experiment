from __future__ import annotations

from collections.abc import Mapping, Sequence

from qiskit import ClassicalRegister, QuantumCircuit

from msc.rotated_surface_code import d3_qiskit as d3
from msc.rotated_surface_code.post_transfer import post_transfer_qiskit
from msc.state_transfer import analysis_qiskit as analysis
from msc.state_transfer import common
from msc.state_transfer import helper_qiskit as helper
from msc.steane_code.magic_state_cultivation_common import CultivatedStateName
from msc.steane_code.state_injection_qiskit import InjectedStateName
from msc.utils.normalize import normalize_pauli_char

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d3",
    distance=3,
    surface_order=tuple(d3.SURFACE_ORDER),
    logical_x=tuple(d3.DEFAULT_LOGICAL_X),
    logical_z=tuple(d3.DEFAULT_LOGICAL_Z),
    z_stabilizers=d3.Z_STABILIZERS,
    x_stabilizers=d3.X_STABILIZERS,
    merged_boundary_checks=(
        ("Z1a", ("1",), ("a",)),
        ("Z35bc", ("3", "5"), ("b", "c")),
    ),
    withheld_surface_stabilizer_labels=("X_ab",),
    surface_code_operations=common.SurfaceCodeOperations(
        d3.prepare_product_plus,
        d3.append_scheduled_stabilizer_round,
        d3.append_logical_measurement,
    ),
)
MERGED_BOUNDARY_CHECKS = TRANSFER_SPEC.merged_boundary_checks


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    injected_state: InjectedStateName = "Ty",
    cultivated_state: CultivatedStateName = "Ty",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = False,
    post_injection_qed_checks: Sequence[str] | None = None,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    full_post_selection: bool = False,
    post_transfer_stabilizer_rounds: int | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the authoritative register-owning d3 transfer circuit.

    The target has nine surface data qubits. Each merged round measures
    ``Z1a`` and ``Z35bc`` together with every compatible d3 stabilizer.
    Anticommuting ``X_ab`` is withheld until destructive Steane-X demolition
    and then recovered unconditionally.

    The circuit includes injection on the selected Steane patch,
    post-injection QED, optional logical-H_XZ cultivation by default (or H_XY
    when ``cultivated_state="T"``), both Steane
    verification groups, and every raw measurement needed for classical
    spacetime decoding. An X/Z ``surface_readout_basis`` appends the requested
    complete surface rounds before all-data readout. The reconstructed final
    stabilizers close the corresponding decoder history.
    ``include_post_transfer_stabilizer_rounds`` retains a
    continuation-only option when no readout is requested. ``None`` otherwise
    ends at the transfer boundary; full-data Y readout is unsupported.

    This d3 geometry is a compact simulation adaptation, not the paper's
    depicted target. This wrapper owns its continuation and readout boundary.
    """
    circuit = helper.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        injected_state=injected_state,
        cultivated_state=cultivated_state,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
        add_barriers=add_barriers,
    )
    if surface_readout_basis is None and not include_post_transfer_stabilizer_rounds:
        return circuit

    stabilizer_rounds = common.resolve_post_transfer_stabilizer_rounds(
        distance=TRANSFER_SPEC.distance,
        full_post_selection=full_post_selection and surface_readout_basis is not None,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
    )
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
        if surface_readout_basis == "Y":
            raise ValueError("full-data surface readout basis must be X or Z.")
    post_transfer_z_register = ClassicalRegister(
        stabilizer_rounds * len(TRANSFER_SPEC.z_labels),
        "post_transfer_surface_z",
    )
    post_transfer_x_register = ClassicalRegister(
        stabilizer_rounds * len(TRANSFER_SPEC.x_labels),
        "post_transfer_surface_x",
    )
    circuit.add_register(post_transfer_z_register, post_transfer_x_register)

    qubits = {register.name: register[0] for register in circuit.qregs}
    surface_qubits = [qubits[label] for label in TRANSFER_SPEC.surface_order]
    post_transfer_qiskit.append_rotated_full_stabilizer_rounds(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.z_labels],
        x_ancillas=[qubits[label.lower()] for label in TRANSFER_SPEC.x_labels],
        z_stabilizer_names=TRANSFER_SPEC.z_labels,
        x_stabilizer_names=TRANSFER_SPEC.x_labels,
        z_bits=post_transfer_z_register[:],
        x_bits=post_transfer_x_register[:],
        stabilizer_rounds=stabilizer_rounds,
        append_scheduled_stabilizer_round=(
            TRANSFER_SPEC.surface_code_operations.append_scheduled_stabilizer_round
        ),
        add_barriers=add_barriers,
    )
    if surface_readout_basis is None:
        return circuit

    surface_data_register = ClassicalRegister(
        len(TRANSFER_SPEC.surface_order),
        f"surface_data_{surface_readout_basis.lower()}",
    )
    circuit.add_register(surface_data_register)
    d3.append_full_data_readout(
        circuit,
        surface_readout_basis,
        surface_qubits=surface_qubits,
        bits=surface_data_register[:],
    )
    if add_barriers:
        circuit.barrier(label=f"destructive full-data {surface_readout_basis}")
    return circuit


def parse_transfer_measurements(
    bitstring: str,
    *,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = False,
    post_injection_qed_checks: Sequence[str] | None = None,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    full_post_selection: bool = False,
    persistent_surface_x_post_selection: bool = True,
    post_transfer_stabilizer_rounds: int | None = None,
    measurement_labels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    resolved_stabilizer_rounds = 0
    if surface_readout_basis is not None or include_post_transfer_stabilizer_rounds:
        resolved_stabilizer_rounds = common.resolve_post_transfer_stabilizer_rounds(
            distance=TRANSFER_SPEC.distance,
            full_post_selection=full_post_selection and surface_readout_basis is not None,
            post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        )
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
        post_transfer_stabilizer_rounds=resolved_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        full_data_readout=surface_readout_basis is not None,
        full_post_selection=full_post_selection,
        persistent_surface_x_post_selection=persistent_surface_x_post_selection,
        measurement_labels=measurement_labels,
    )
