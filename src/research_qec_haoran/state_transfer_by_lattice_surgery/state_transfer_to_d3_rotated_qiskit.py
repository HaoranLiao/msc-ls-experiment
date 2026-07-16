"""Steane-to-rotated-d3 state transfer by lattice surgery.

The paper draws the distance-5 construction. This module applies the same
three-round protocol to the shorter distance-3 top boundary: its merged logical
``Z_L Z_L`` measurement is the product of ``Z1a`` and ``Z35bc``. All compatible
surface checks and both paper-defined Steane check groups are still measured.

Circuit construction lives in ``state_transfer_common_qiskit``; parsing,
acceptance, decoding, and Pauli-frame interpretation live in
``state_transfer_analysis_qiskit``. This module only binds those shared
operations to the rotated-d3 geometry.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_analysis_qiskit as analysis,
)
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_common_qiskit as common,
)
from research_qec_haoran.state_transfer_by_lattice_surgery.state_transfer_common_qiskit import (
    SurfaceReadoutBasis,
    SurfaceReadoutMode,
)
from research_qec_haoran.steane_code.state_injection_qiskit import InjectedStateName
from research_qec_haoran.surface_code import rotated_d3_qiskit as d3

MERGED_BOUNDARY_CHECKS: tuple[common.MergedBoundaryCheck, ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
)

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d3",
    surface_order=tuple(d3.SURFACE_ORDER),
    logical_x=tuple(d3.DEFAULT_LOGICAL_X),
    logical_z=tuple(d3.DEFAULT_LOGICAL_Z),
    z_stabilizers=d3.Z_STABILIZERS,
    x_stabilizers=d3.X_STABILIZERS,
    merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
    top_x_recovery_labels=("X_ab",),
    prepare_plus=d3.prepare_plus,
    append_z_stabilizer_measurements=d3.append_z_stabilizer_measurements,
    append_x_stabilizer_measurements=d3.append_x_stabilizer_measurements,
    append_logical_measurement=d3.append_logical_measurement,
)


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the complete stabilizer-level d3 transfer adaptation.

    The three merged rounds, compatible surface checks, Steane verification,
    demolition, and ``X_ab`` recovery are unconditional protocol operations.
    ``surface_readout_basis`` adds an analysis measurement after transfer.
    """
    return common.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
        add_barriers=add_barriers,
    )


def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
) -> dict[str, object]:
    """Parse one d3 transfer count key into labeled protocol fields."""
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
