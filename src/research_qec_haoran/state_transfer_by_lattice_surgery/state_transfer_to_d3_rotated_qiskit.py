"""Steane-to-rotated-d3 state transfer by lattice surgery.

The paper's explicit construction targets a distance-5 rotated patch.  This
module is the distance-3 adaptation: it keeps the same Steane checks, transfer
byproducts, and three-round workflow, but its shorter top boundary supports
only ``Z1a`` and ``Z35bc``.  The product of those checks is still the adapted
logical ``Z_L Z_L`` measurement.

Distance-independent circuit construction lives in
``state_transfer_common_qiskit``; parsing, acceptance, and frame interpretation
live in ``state_transfer_analysis_qiskit``. The default circuit performs:

1. Crumble state injection and optional H-axis cultivation on the Steane patch.
2. Rotated-d3 ``|+>`` preparation and an initial Z-stabilizer frame.
3. Three merged rounds.  Each contains the adapted boundary checks and all
   compatible surface stabilizers; Steane check groups 1 and 2 are included in
   merged rounds 1 and 2.
4. Destructive Steane X measurement and recovery of surface ``X_ab``.
5. Optional analysis-only surface QEC and logical readout.

The measured checks are ideal parity-measurement abstractions.  They reproduce
the paper's stabilizer observables and round order, not its hardware-specific
superdense gate schedule.  Known teleportation byproducts are interpreted with
a ``PauliFrame`` rather than physical correction gates.
"""

from __future__ import annotations

from collections.abc import Iterable

from qiskit import QuantumCircuit

from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_analysis_qiskit as analysis,
)
from research_qec_haoran.state_transfer_by_lattice_surgery import (
    state_transfer_common_qiskit as common,
)
from research_qec_haoran.state_transfer_by_lattice_surgery.state_transfer_analysis_qiskit import (
    LookupTableEarlyRejection,
    SyndromeItem,
    SyndromeSignature,
)
from research_qec_haoran.state_transfer_by_lattice_surgery.state_transfer_common_qiskit import (
    HIRANO_LABEL_TO_GOOGLE_SITE,
    HIRANO_STEANE_DATA_ORDER,
    HIRANO_STEANE_GEOMETRIC_ORDER,
    HIRANO_STEANE_LOGICAL_X,
    HIRANO_STEANE_LOGICAL_Z,
    LOCAL_STEANE_X_STABILIZER_CHECKS,
    LOCAL_X_AB_CORRELATION_CHECKS,
    STEANE_HELPER_LABEL_TO_GOOGLE_SITE,
    STEANE_LS_VERIFICATION_LABELS,
    STEANE_LS_VERIFICATION_ROUNDS,
    STEANE_X_RECONSTRUCTED_CHECKS,
    HiranoLabeledQRegs,
    SurfaceReadoutBasis,
    SurfaceReadoutMode,
)
from research_qec_haoran.steane_code.state_injection_qiskit import InjectedStateName
from research_qec_haoran.surface_code import rotated_d3_qiskit as d3
from research_qec_haoran.surface_code.decoding.rotated_decoder import RotatedDetectedErrors

SURFACE_ORDER: tuple[str, ...] = tuple(d3.SURFACE_ORDER)
DEFAULT_LOGICAL_X: tuple[str, ...] = tuple(d3.DEFAULT_LOGICAL_X)
DEFAULT_LOGICAL_Z: tuple[str, ...] = tuple(d3.DEFAULT_LOGICAL_Z)
X_STABILIZERS: dict[str, tuple[str, ...]] = d3.X_STABILIZERS
Z_STABILIZERS: dict[str, tuple[str, ...]] = d3.Z_STABILIZERS

TOP_BOUNDARY: tuple[str, ...] = ("a", "b", "c")
SURFACE_X_AB_LABELS: tuple[str, ...] = ("X_ab",)
MERGED_BOUNDARY_CHECKS: tuple[common.MergedBoundaryCheck, ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
)

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d3",
    surface_order=SURFACE_ORDER,
    logical_x=DEFAULT_LOGICAL_X,
    logical_z=DEFAULT_LOGICAL_Z,
    z_stabilizers=Z_STABILIZERS,
    x_stabilizers=X_STABILIZERS,
    merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
    top_x_recovery_checks=(("X_ab", ("a", "b")),),
    prepare_plus=d3.prepare_plus,
    append_z_stabilizer_measurements=d3.append_z_stabilizer_measurements,
    append_x_stabilizer_measurements=d3.append_x_stabilizer_measurements,
    append_stabilizer_round=d3.append_stabilizer_round,
    append_logical_measurement=d3.append_logical_measurement,
)

SURFACE_INIT_Z_LABELS = TRANSFER_SPEC.z_labels
SURFACE_POST_Z_LABELS = TRANSFER_SPEC.z_labels
SURFACE_POST_X_LABELS = TRANSFER_SPEC.x_labels
SURFACE_MERGED_Z_LABELS = TRANSFER_SPEC.z_labels
SURFACE_MERGED_X_LABELS = TRANSFER_SPEC.merged_x_labels
POST_TRANSFER_SURFACE_SYNDROME_SCHEDULE = (
    ("surface_init_z", SURFACE_INIT_Z_LABELS),
    ("surface_top_x_recovery", SURFACE_X_AB_LABELS),
    ("surface_post_z", SURFACE_POST_Z_LABELS),
    ("surface_post_x", SURFACE_POST_X_LABELS),
)
MERGED_INTERVAL_SURFACE_SYNDROME_SCHEDULE = (
    ("merged_surface_z", SURFACE_MERGED_Z_LABELS),
    ("merged_surface_x", SURFACE_MERGED_X_LABELS),
)

_SURFACE_CODE_KWARGS: dict[str, object] = {
    "surface_order": SURFACE_ORDER,
    "z_stabilizers": Z_STABILIZERS,
    "x_stabilizers": X_STABILIZERS,
    "logical_x": DEFAULT_LOGICAL_X,
    "logical_z": DEFAULT_LOGICAL_Z,
}
_SURFACE_PATCH_KWARGS: dict[str, object] = {
    **_SURFACE_CODE_KWARGS,
    "surface_post_z_labels": SURFACE_POST_Z_LABELS,
    "surface_post_x_labels": SURFACE_POST_X_LABELS,
}

merged_boundary_logical_zz_outcome = analysis.merged_boundary_logical_zz_outcome
steane_logical_x_outcome = analysis.steane_logical_x_outcome
reconstructed_x_checks = analysis.reconstructed_x_checks
local_x_verification_checks = analysis.local_x_verification_checks
x_verification_succeeds = analysis.x_verification_succeeds
steane_verification_succeeds = analysis.steane_verification_succeeds
cultivation_succeeds = analysis.cultivation_succeeds
encoder_helpers_succeed = analysis.encoder_helpers_succeed


def build_hirano_labeled_qregs(
    *,
    include_top_x_recovery: bool = True,
    include_merged_surface_syndromes: bool = False,
    include_surface_qec: bool = False,
    surface_logical_measurement_basis: SurfaceReadoutBasis | None = None,
) -> HiranoLabeledQRegs:
    """Create paper-labeled registers for the d3 transfer adaptation."""
    return common.build_transfer_qregs(
        TRANSFER_SPEC,
        include_x_recovery=include_top_x_recovery,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_surface_qec=include_surface_qec,
        surface_logical_measurement_basis=surface_logical_measurement_basis,
    )


def append_boundary_zz_checks(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    merged_rounds: int = 3,
    add_barriers: bool = True,
) -> None:
    """Append repeated adapted ``Z1a`` and ``Z35bc`` checks."""
    common.append_boundary_zz_checks(
        circuit,
        registers=registers,
        classical_bits=classical_bits,
        merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
        merged_rounds=merged_rounds,
        add_barriers=add_barriers,
    )


def append_merged_interval_surface_syndrome_round(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    surface_qubits: list,
    z_classical_bits: list,
    x_classical_bits: list,
    round_label: str,
    add_barriers: bool = True,
) -> None:
    """Measure compatible d3 surface stabilizers during one merged round."""
    common.append_merged_surface_syndrome_round(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        surface_qubits=surface_qubits,
        z_classical_bits=z_classical_bits,
        x_classical_bits=x_classical_bits,
        round_label=round_label,
        add_barriers=add_barriers,
    )


def append_surface_logical_readout(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure every d3 surface data qubit in the requested basis."""
    common.append_surface_logical_readout(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bits=classical_bits,
        basis=basis,
        add_barriers=add_barriers,
    )


def append_surface_x_ab_recovery_measurement(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bit,
    add_barriers: bool = True,
) -> None:
    """Recover top-boundary ``X_ab`` after Steane demolition."""
    common.append_surface_top_x_recovery_measurements(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bits=[classical_bit],
        add_barriers=add_barriers,
    )


def append_surface_logical_pauli_measurement(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bit,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure the standard d3 logical Pauli support indirectly."""
    common.append_surface_logical_pauli_measurement(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bit=classical_bit,
        basis=basis,
        destructive=False,
        add_barriers=add_barriers,
    )


def append_surface_logical_pauli_destructive(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Destructively measure the standard d3 logical Pauli support."""
    common.append_surface_logical_pauli_measurement(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bit=classical_bits,
        basis=basis,
        destructive=True,
        add_barriers=add_barriers,
    )


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    merged_rounds: int = 3,
    include_merged_surface_syndromes: bool = True,
    include_steane_verification: bool = True,
    include_surface_qec: bool = False,
    surface_qec_rounds: int = 3,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the complete stabilizer-level d3 transfer adaptation.

    Top-edge ``X_ab`` recovery is always included because it is part of the
    transfer itself. ``include_surface_qec`` appends an optional analysis stage
    and is false for the paper-facing transfer.
    """
    return common.build_lattice_surgery_circuit(
        TRANSFER_SPEC,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
        add_barriers=add_barriers,
    )


def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    merged_rounds: int = 3,
    include_merged_surface_syndromes: bool = True,
    include_steane_verification: bool = True,
    include_surface_qec: bool = False,
    surface_qec_rounds: int = 3,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
) -> dict[str, object]:
    """Parse a d3 transfer count key into labeled protocol fields."""
    return analysis.parse_transfer_measurements(
        bitstring,
        spec=TRANSFER_SPEC,
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )


def merged_boundary_checks_stable(measurements: dict[str, object]) -> bool:
    """Return whether every adapted boundary check is stable in time."""
    return analysis.merged_boundary_checks_stable(measurements, MERGED_BOUNDARY_CHECKS)


def surface_qec_deltas(measurements: dict[str, object]) -> list[dict[str, dict[str, int]]]:
    """Return optional post-transfer d3 syndrome deltas."""
    return analysis.surface_qec_deltas(
        measurements,
        surface_post_z_labels=SURFACE_POST_Z_LABELS,
        surface_post_x_labels=SURFACE_POST_X_LABELS,
    )


def merged_surface_syndrome_deltas(
    measurements: dict[str, object],
) -> list[dict[str, dict[str, int]]]:
    """Return d3 surface syndrome deltas measured during the merge."""
    return analysis.merged_surface_syndrome_deltas(
        measurements,
        merged_surface_z_labels=SURFACE_MERGED_Z_LABELS,
        merged_surface_x_labels=SURFACE_MERGED_X_LABELS,
    )


def merged_surface_qec_recoveries(
    measurements: dict[str, object],
) -> list[RotatedDetectedErrors]:
    """Decode every d3 surface syndrome round measured during the merge."""
    return analysis.merged_surface_qec_recoveries(measurements, **_SURFACE_CODE_KWARGS)


def merged_surface_qec_recovery(measurements: dict[str, object]) -> RotatedDetectedErrors:
    """Return the last decoded merged-interval d3 surface recovery."""
    return analysis.merged_surface_qec_recovery(measurements, **_SURFACE_CODE_KWARGS)


def surface_qec_recoveries(measurements: dict[str, object]) -> list[RotatedDetectedErrors]:
    """Decode all optional post-transfer d3 syndrome rounds."""
    return analysis.surface_qec_recoveries(measurements, **_SURFACE_PATCH_KWARGS)


def surface_qec_recovery(measurements: dict[str, object]) -> RotatedDetectedErrors:
    """Return the last optional post-transfer d3 recovery."""
    return analysis.surface_qec_recovery(measurements, **_SURFACE_PATCH_KWARGS)


def post_transfer_surface_syndrome_schedule(
    surface_qec_rounds: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the opt-in d3 surface-QEC schedule after transfer."""
    return analysis.post_transfer_surface_syndrome_schedule(
        surface_qec_rounds,
        surface_init_z_labels=SURFACE_INIT_Z_LABELS,
        surface_post_z_labels=SURFACE_POST_Z_LABELS,
        surface_post_x_labels=SURFACE_POST_X_LABELS,
        surface_x_recovery_labels=SURFACE_X_AB_LABELS,
    )


def early_rejection_syndrome_items(
    measurements: dict[str, object],
) -> tuple[SyndromeItem, ...]:
    """Return labeled pre-expansion syndrome bits for early rejection."""
    return analysis.early_rejection_syndrome_items(
        measurements,
        merged_surface_z_labels=SURFACE_MERGED_Z_LABELS,
        merged_surface_x_labels=SURFACE_MERGED_X_LABELS,
        surface_post_z_labels=SURFACE_POST_Z_LABELS,
        surface_post_x_labels=SURFACE_POST_X_LABELS,
    )


def early_rejection_syndrome_signature(
    measurements: dict[str, object],
) -> SyndromeSignature:
    """Return the ordered pre-expansion syndrome bit vector."""
    return analysis.early_rejection_syndrome_signature(early_rejection_syndrome_items(measurements))


def lookup_table_early_rejection_accepts(
    measurements: dict[str, object],
    *,
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None = None,
) -> bool:
    """Return whether the optional paper-style lookup table accepts."""
    return analysis.lookup_table_early_rejection_accepts(
        early_rejection_syndrome_signature(measurements),
        lookup_table=lookup_table,
    )


def transfer_acceptance_succeeds(
    measurements: dict[str, object],
    *,
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None = None,
) -> bool:
    """Return whether every fault-detecting d3 transfer check accepts."""
    return analysis.transfer_acceptance_succeeds(
        measurements,
        merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
        signature=early_rejection_syndrome_signature(measurements),
        lookup_table=lookup_table,
    )


def transfer_pauli_frame(
    measurements: dict[str, object],
    *,
    apply_merged_surface_phase_recovery: bool = True,
    apply_surface_qec: bool = True,
) -> PauliFrame:
    """Build the d3 target frame from teleportation and optional QEC."""
    return analysis.transfer_pauli_frame(
        measurements,
        **_SURFACE_PATCH_KWARGS,
        apply_merged_surface_phase_recovery=apply_merged_surface_phase_recovery,
        apply_surface_qec=apply_surface_qec,
    )


def raw_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the uncorrected d3 logical readout bit."""
    return analysis.raw_surface_logical_readout(
        measurements,
        basis=basis,
        logical_x=DEFAULT_LOGICAL_X,
        logical_z=DEFAULT_LOGICAL_Z,
    )


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the d3 logical readout interpreted through its Pauli frame."""
    return analysis.corrected_surface_logical_readout(
        measurements,
        basis=basis,
        frame=transfer_pauli_frame(measurements),
        logical_x=DEFAULT_LOGICAL_X,
        logical_z=DEFAULT_LOGICAL_Z,
    )


__all__ = [
    "DEFAULT_LOGICAL_X",
    "DEFAULT_LOGICAL_Z",
    "HIRANO_LABEL_TO_GOOGLE_SITE",
    "HIRANO_STEANE_DATA_ORDER",
    "HIRANO_STEANE_GEOMETRIC_ORDER",
    "HIRANO_STEANE_LOGICAL_X",
    "HIRANO_STEANE_LOGICAL_Z",
    "LOCAL_STEANE_X_STABILIZER_CHECKS",
    "LOCAL_X_AB_CORRELATION_CHECKS",
    "LookupTableEarlyRejection",
    "MERGED_BOUNDARY_CHECKS",
    "MERGED_INTERVAL_SURFACE_SYNDROME_SCHEDULE",
    "POST_TRANSFER_SURFACE_SYNDROME_SCHEDULE",
    "STEANE_HELPER_LABEL_TO_GOOGLE_SITE",
    "STEANE_LS_VERIFICATION_LABELS",
    "STEANE_LS_VERIFICATION_ROUNDS",
    "STEANE_X_RECONSTRUCTED_CHECKS",
    "SURFACE_MERGED_X_LABELS",
    "SURFACE_MERGED_Z_LABELS",
    "SURFACE_ORDER",
    "SURFACE_X_AB_LABELS",
    "TOP_BOUNDARY",
    "TRANSFER_SPEC",
    "X_STABILIZERS",
    "Z_STABILIZERS",
    "append_boundary_zz_checks",
    "append_merged_interval_surface_syndrome_round",
    "append_surface_logical_readout",
    "append_surface_x_ab_recovery_measurement",
    "build_hirano_labeled_qregs",
    "build_steane_to_rotated_d3_lattice_surgery_circuit",
    "corrected_surface_logical_readout",
    "cultivation_succeeds",
    "early_rejection_syndrome_items",
    "early_rejection_syndrome_signature",
    "encoder_helpers_succeed",
    "local_x_verification_checks",
    "lookup_table_early_rejection_accepts",
    "merged_boundary_checks_stable",
    "merged_boundary_logical_zz_outcome",
    "merged_surface_qec_recoveries",
    "merged_surface_qec_recovery",
    "merged_surface_syndrome_deltas",
    "parse_transfer_measurements",
    "post_transfer_surface_syndrome_schedule",
    "raw_surface_logical_readout",
    "reconstructed_x_checks",
    "steane_logical_x_outcome",
    "steane_verification_succeeds",
    "surface_qec_deltas",
    "surface_qec_recoveries",
    "surface_qec_recovery",
    "transfer_acceptance_succeeds",
    "transfer_pauli_frame",
    "x_verification_succeeds",
]
