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

from magic_state_cultivation.state_transfer_by_lattice_surgery import (
    state_transfer_analysis_qiskit as analysis,
)
from magic_state_cultivation.state_transfer_by_lattice_surgery import (
    state_transfer_common_qiskit as common,
)
from magic_state_cultivation.state_transfer_by_lattice_surgery.state_transfer_common_qiskit import (
    SurfaceReadoutBasis,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import InjectedStateName
from magic_state_cultivation.surface_code import rotated_d3_qiskit as d3

MERGED_BOUNDARY_CHECKS: tuple[common.MergedBoundaryCheck, ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
)

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d3",
    distance=3,
    surface_order=tuple(d3.SURFACE_ORDER),
    logical_x=tuple(d3.DEFAULT_LOGICAL_X),
    logical_z=tuple(d3.DEFAULT_LOGICAL_Z),
    z_stabilizers=d3.Z_STABILIZERS,
    x_stabilizers=d3.X_STABILIZERS,
    merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
    top_x_recovery_labels=("X_ab",),
    surface_code_operations=common.SurfaceCodeOperations(
        prepare_product_plus=d3.prepare_product_plus,
        append_z_stabilizer_measurements=d3.append_z_stabilizer_measurements,
        append_x_stabilizer_measurements=d3.append_x_stabilizer_measurements,
        append_stabilizer_round=d3.append_stabilizer_round,
        append_logical_measurement=d3.append_logical_measurement,
    ),
)


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the Steane-to-rotated-d3 lattice-surgery transfer circuit.

    This is the compact distance-3 adaptation of the paper's distance-5
    transfer geometry. The target contains nine surface data qubits. Its
    merged logical-Z boundary is measured through ``Z1a`` and ``Z35bc`` in
    each of three merged rounds. Every compatible d3 surface stabilizer is
    measured during those rounds; the anticommuting top-edge check ``X_ab`` is
    withheld until Steane demolition and then recovered unconditionally.

    The builder also includes the Crumble Steane encoder, optional logical-H_XY
    cultivation, both paper-defined Steane verification groups, destructive
    Steane-X demolition, and all classical registers needed by
    ``parse_transfer_measurements``. Temporal surface decoding is classical
    post-processing and is not executed while constructing the circuit.

    Parameters
    ----------
    injected_state
        Named physical state or amplitude pair supplied to the Steane Crumble
        encoder. The default ``"T"`` is the H_XY-axis magic-state input.
    include_cultivation
        Whether to append the logical-H_XY cultivation projection and its
        postselection measurements. Disable this only for transfer tests or an
        input that has already been cultivated.
    include_post_transfer_stabilizer_rounds
        Whether to append three full d3 surface-stabilizer rounds after the
        transfer and before any logical readout.
    surface_readout_basis
        Optional ``"X"`` or ``"Z"`` basis for a destructive analysis
        measurement of the standard d3 logical support after top-edge
        recovery. ``None`` ends the circuit at the paper-facing transfer
        boundary.
    add_barriers
        Whether to add labeled barriers between conceptual protocol stages.

    Returns
    -------
    QuantumCircuit
        A register-owning d3 transfer circuit with an optional terminal
        analysis readout.

    Notes
    -----
    The d3 boundary is an adaptation for compact simulation and testing; the
    paper explicitly depicts the d5 target. Full protocol-stage semantics,
    including the three temporal syndrome-extraction rounds and intentional
    optional fixed-patch continuation, are documented by
    ``common.append_lattice_surgery_circuit``.
    """
    return common.append_lattice_surgery_circuit(
        TRANSFER_SPEC,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=(include_post_transfer_stabilizer_rounds),
        surface_readout_basis=surface_readout_basis,
        add_barriers=add_barriers,
    )


def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    """Parse one d3 transfer count key into labeled protocol fields."""
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=(include_post_transfer_stabilizer_rounds),
        surface_readout_basis=surface_readout_basis,
    )
