from __future__ import annotations

from collections.abc import Sequence

from qiskit import ClassicalRegister, QuantumCircuit

from msc.rotated_surface_code import d5_qiskit as d5
from msc.rotated_surface_code.post_transfer import post_transfer_qiskit
from msc.state_transfer import analysis_qiskit as analysis
from msc.state_transfer import common
from msc.state_transfer import helper_qiskit as helper
from msc.steane_code.magic_state_cultivation_common import CultivatedStateName
from msc.steane_code.state_injection_qiskit import InjectedStateName
from msc.utils.normalize import normalize_pauli_char

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


def build_steane_to_rotated_d5_lattice_surgery_circuit(
    *,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    injected_state: InjectedStateName = "Ty",
    cultivated_state: CultivatedStateName = "Ty",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    full_post_selection: bool = False,
    post_transfer_stabilizer_rounds: int | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the authoritative register-owning d5 transfer circuit.

    The 25-data-qubit target is the rotated d=5 patch from Figure 6. Each
    merged round measures ``Z1a``, ``Z35bc``, and ``Zde`` with every compatible
    surface stabilizer. Anticommuting ``X_ab`` and ``X_cd`` are withheld during
    the merge. After destructive Steane-X demolition, only ``X_ab`` is measured
    as Hirano's ``m_ab``; ``X_cd`` waits for a later complete surface round.

    The circuit includes injection on the selected Steane patch,
    post-injection QED, optional logical-H_XZ cultivation by default (or H_XY
    when ``cultivated_state="T"``), both paper-defined
    Steane verification groups, and every raw measurement needed for classical
    spacetime decoding. If ``surface_readout_basis`` is X or Z, the requested
    complete surface rounds precede destructive all-data readout. ``None`` ends
    at ``m_ab``.
    """
    circuit = helper.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        injected_state=injected_state,
        cultivated_state=cultivated_state,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        add_barriers=add_barriers,
    )
    if surface_readout_basis is None:
        return circuit
    surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    if surface_readout_basis == "Y":
        raise ValueError("full-data surface readout basis must be X or Z.")
    stabilizer_rounds = common.resolve_post_transfer_stabilizer_rounds(
        distance=TRANSFER_SPEC.distance,
        full_post_selection=full_post_selection,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
    )
    post_transfer_z_register = ClassicalRegister(
        stabilizer_rounds * len(TRANSFER_SPEC.z_labels),
        "post_transfer_surface_z",
    )
    post_transfer_x_register = ClassicalRegister(
        stabilizer_rounds * len(TRANSFER_SPEC.x_labels),
        "post_transfer_surface_x",
    )
    surface_data_register = ClassicalRegister(
        len(TRANSFER_SPEC.surface_order),
        f"surface_data_{surface_readout_basis.lower()}",
    )
    circuit.add_register(
        post_transfer_z_register,
        post_transfer_x_register,
        surface_data_register,
    )
    append_post_transfer_and_readout(
        circuit,
        surface_readout_basis=surface_readout_basis,
        full_post_selection=full_post_selection,
        post_transfer_stabilizer_rounds=stabilizer_rounds,
        post_transfer_z_bits=post_transfer_z_register,
        post_transfer_x_bits=post_transfer_x_register,
        surface_data_bits=surface_data_register,
        add_barriers=add_barriers,
    )
    return circuit


def append_post_transfer_and_readout(
    circuit: QuantumCircuit,
    *,
    surface_readout_basis: common.SurfaceReadoutBasis,
    full_post_selection: bool = False,
    post_transfer_stabilizer_rounds: int | None = None,
    post_transfer_z_bits: Sequence[object],
    post_transfer_x_bits: Sequence[object],
    surface_data_bits: Sequence[object],
    add_barriers: bool = True,
) -> None:
    """Append full d=5 rounds and destructive all-data readout.

    The associated parser change-detects these rounds from the transfer
    references, reconstructs the compatible final syndrome from all 25 data
    bits, and includes that boundary in spacetime decoding.

    The two post-transfer bit sequences store round-major Z and X syndromes;
    ``surface_data_bits`` stores the final data in canonical
    ``a..y`` order. The caller owns these classical registers.
    """
    surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    if surface_readout_basis == "Y":
        raise ValueError("full-data surface readout basis must be X or Z.")
    stabilizer_rounds = common.resolve_post_transfer_stabilizer_rounds(
        distance=TRANSFER_SPEC.distance,
        full_post_selection=full_post_selection,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
    )
    qubits = {register.name: register[0] for register in circuit.qregs}
    surface_qubits = [qubits[label] for label in TRANSFER_SPEC.surface_order]
    z_ancillas = [qubits[label.lower()] for label in TRANSFER_SPEC.z_labels]
    x_ancillas = [qubits[label.lower()] for label in TRANSFER_SPEC.x_labels]

    post_transfer_qiskit.append_rotated_full_stabilizer_rounds(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=z_ancillas,
        x_ancillas=x_ancillas,
        z_stabilizer_names=TRANSFER_SPEC.z_labels,
        x_stabilizer_names=TRANSFER_SPEC.x_labels,
        z_bits=post_transfer_z_bits,
        x_bits=post_transfer_x_bits,
        stabilizer_rounds=stabilizer_rounds,
        append_scheduled_stabilizer_round=(
            TRANSFER_SPEC.surface_code_operations.append_scheduled_stabilizer_round
        ),
        add_barriers=add_barriers,
    )
    d5.append_full_data_readout(
        circuit,
        surface_readout_basis,
        surface_qubits=surface_qubits,
        bits=surface_data_bits,
    )
    if add_barriers:
        circuit.barrier(label=f"destructive full-data {surface_readout_basis}")


def parse_transfer_measurements(
    bitstring: str,
    *,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    full_post_selection: bool = False,
    persistent_surface_x_post_selection: bool = True,
    post_transfer_stabilizer_rounds: int | None = None,
) -> dict[str, object]:
    """Parse and decode the transfer, continuation, and full-data boundary."""
    resolved_stabilizer_rounds = 0
    if surface_readout_basis is not None:
        resolved_stabilizer_rounds = common.resolve_post_transfer_stabilizer_rounds(
            distance=TRANSFER_SPEC.distance,
            full_post_selection=full_post_selection,
            post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        )
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        post_transfer_stabilizer_rounds=resolved_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        full_data_readout=surface_readout_basis is not None,
        full_post_selection=full_post_selection,
        persistent_surface_x_post_selection=persistent_surface_x_post_selection,
    )
