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
from magic_state_cultivation.surface_code import rotated_d5_qiskit as d5

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
    injected_state: InjectedStateName = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """This is the authoritative d=5 transfer allocation and appending example.

    The target is the 25-data-qubit rotated distance-5 patch used by Figure 6.
    Each of the three merged rounds measures ``Z1a``, ``Z35bc``, and ``Zde``
    together with every compatible d5 surface stabilizer. The anticommuting
    top-edge checks ``X_ab`` and ``X_cd`` are withheld during the merge and
    recovered unconditionally after destructive Steane-X demolition. The
    first ``X_cd`` recovery outcome is not hard-postselected.

    The builder also includes the Crumble Steane encoder, post-injection
    Steane QED, optional logical-H_XY cultivation, and the two paper-defined
    Steane verification groups assigned to merged rounds 1 and 2. It records
    all raw measurements needed for temporal surface decoding by
    ``parse_transfer_measurements``; decoding and Pauli-frame interpretation
    occur classically after circuit execution.

    Parameters
    ----------
    injected_state
        Named physical state supplied to the Steane Crumble encoder. The
        default ``"T"`` is the H_XY-axis magic-state input.
    include_cultivation
        Whether to append the logical-H_XY cultivation projection and its
        postselection measurements. Disable this only for transfer tests or an
        input that has already been cultivated.
    include_post_transfer_stabilizer_rounds
        Whether to append five full d5 surface-stabilizer rounds after the
        transfer and before any logical readout.
    surface_readout_basis
        Optional ``"X"`` or ``"Z"`` basis for a destructive analysis
        measurement of the standard d5 logical support after top-edge
        recovery. ``None`` ends the circuit at the Figure 6 transfer boundary.
    add_barriers
        Whether to add labeled barriers between conceptual protocol stages.

    Returns
    -------
    QuantumCircuit
        A register-owning d5 transfer circuit with an optional terminal
        analysis readout.

    Notes
    -----
    The optional fixed-patch stabilizer rounds and terminal readout are analysis
    aids and are not part of Figure 6. The paper instead performs later
    stabilizing rounds after code expansion. Full shared protocol semantics are
    documented by ``common.append_lattice_surgery_circuit``.
    """
    return common.build_lattice_surgery_circuit(
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
    """Parse one d5 transfer count key into labeled protocol fields."""
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=(include_post_transfer_stabilizer_rounds),
        surface_readout_basis=surface_readout_basis,
    )
