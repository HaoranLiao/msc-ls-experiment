from __future__ import annotations

from qiskit import QuantumCircuit

from msc.rotated_surface_code import d5_qiskit as d5
from msc.state_transfer import analysis_qiskit as analysis
from msc.state_transfer import common
from msc.state_transfer import helper_qiskit as helper
from msc.steane_code.state_injection_qiskit import InjectedStateName

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


def build_steane_to_rotated_d5_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the authoritative register-owning d5 transfer circuit.

    The 25-data-qubit target is the rotated d=5 patch from Figure 6. Each
    merged round measures ``Z1a``, ``Z35bc``, and ``Zde`` with every compatible
    surface stabilizer. Anticommuting ``X_ab`` and ``X_cd`` are withheld until
    destructive Steane-X demolition and then recovered unconditionally; the
    first ``X_cd`` recovery is not hard-postselected.

    The circuit includes Crumble injection, post-injection Steane QED,
    optional logical-H_XY cultivation, both paper-defined Steane verification
    groups, and every raw measurement needed for classical spacetime decoding.
    Setting ``include_post_transfer_stabilizer_rounds`` appends five full d5
    rounds. ``surface_readout_basis`` optionally appends destructive X/Y/Z
    analysis; ``None`` ends at the Figure 6 transfer boundary.

    The optional fixed-patch continuation and terminal readout are analysis
    aids, not part of Figure 6; the paper instead continues after code expansion.
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
