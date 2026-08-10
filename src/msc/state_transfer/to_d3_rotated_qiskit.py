from __future__ import annotations

from qiskit import QuantumCircuit

from msc.rotated_surface_code import d3_qiskit as d3
from msc.state_transfer import analysis_qiskit as analysis
from msc.state_transfer import common, helper_qiskit as helper
from msc.steane_code.state_injection_qiskit import InjectedStateName

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
    top_x_recovery_labels=("X_ab",),
    surface_code_operations=common.SurfaceCodeOperations(
        d3.prepare_product_plus,
        d3.append_scheduled_stabilizer_round,
        d3.append_logical_measurement,
    ),
)
MERGED_BOUNDARY_CHECKS = TRANSFER_SPEC.merged_boundary_checks


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the authoritative register-owning d3 transfer circuit.

    The target has nine surface data qubits. Each merged round measures
    ``Z1a`` and ``Z35bc`` together with every compatible d3 stabilizer.
    Anticommuting ``X_ab`` is withheld until destructive Steane-X demolition
    and then recovered unconditionally.

    The circuit includes Crumble injection, post-injection Steane QED,
    optional logical-H_XY cultivation, both Steane verification groups, and
    every raw measurement needed for classical spacetime decoding. Setting
    ``include_post_transfer_stabilizer_rounds`` appends three full d3 rounds.
    ``surface_readout_basis`` optionally appends destructive X/Y/Z analysis;
    ``None`` ends at the transfer boundary.

    This d3 geometry is a compact simulation adaptation, not the paper's
    depicted target. The shared helper owns the full protocol-stage semantics.
    """
    return helper.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        add_barriers=add_barriers,
    )


def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
