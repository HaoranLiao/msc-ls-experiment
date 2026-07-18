"""Paper-aligned Steane-to-rotated-d5 state transfer by lattice surgery.

This module binds the Figure 6 distance-5 geometry to the shared transfer
implementation. The protocol prepares the target patch in ``|+_L>``, performs
three merged rounds with ``Z1a``, ``Z35bc``, and ``Zde`` plus every compatible
surface check, measures the two specified Steane check groups, demolishes the
Steane block in X, and recovers ``X_ab`` and ``X_cd``. Acceptance and logical
feedforward are interpreted classically after measurement.

The checks are ideal parity-measurement abstractions of the paper observables;
they do not reproduce the hardware-specific superdense tick schedule or the
paper's circuit-level detector error model.
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
from research_qec_haoran.surface_code import rotated_d5_qiskit as d5

MERGED_BOUNDARY_CHECKS: tuple[common.MergedBoundaryCheck, ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
    ("Zde", (), ("d", "e")),
)

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d5",
    surface_order=tuple(d5.SURFACE_ORDER),
    logical_x=tuple(d5.DEFAULT_LOGICAL_X),
    logical_z=tuple(d5.DEFAULT_LOGICAL_Z),
    z_stabilizers=d5.Z_STABILIZERS,
    x_stabilizers=d5.X_STABILIZERS,
    merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
    top_x_recovery_labels=("X_ab", "X_cd"),
    surface_code_operations=common.SurfaceCodeOperations(
        prepare_plus=d5.prepare_plus,
        append_z_stabilizer_measurements=d5.append_z_stabilizer_measurements,
        append_x_stabilizer_measurements=d5.append_x_stabilizer_measurements,
        append_logical_measurement=d5.append_logical_measurement,
    ),
)


def build_steane_to_rotated_d5_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the complete Figure 6 stabilizer-level transfer.

    All three merged rounds and all paper-required verification and recovery
    checks are unconditional. ``surface_readout_basis`` adds an analysis
    measurement after the transfer; it is not part of Figure 6.
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
    """Parse one d5 transfer count key into labeled protocol fields."""
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
