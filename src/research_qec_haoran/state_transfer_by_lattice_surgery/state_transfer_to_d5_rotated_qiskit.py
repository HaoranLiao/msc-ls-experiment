"""Paper-aligned Steane-to-rotated-d5 state transfer by lattice surgery.

This module binds the distance-5 geometry from Figure 6 of Hirano et al. to the
shared transfer implementation.  The default circuit performs the full
stabilizer-level state-transfer sequence:

1. Prepare/cultivate the logical input on the Steane patch.
2. Initialize the d5 surface data in ``|+>`` and establish its Z frame.
3. Measure ``Z1a``, ``Z35bc``, and ``Zde`` for three merged rounds while also
   measuring every compatible stabilizer on both codes.  Steane check groups 1
   and 2 occur in merged rounds 1 and 2; ``X0145ab`` is withheld.
4. Destructively measure Steane data in X, recover surface ``X_ab``/``X_cd``,
   and reconstruct ``X0145ab``, ``X0235``, ``X0246``, and logical ``X146``.
5. Interpret the logical X/Z teleportation byproducts in a ``PauliFrame``.

The Qiskit checks are ideal parity-measurement abstractions.  They preserve the
paper's observables and round order but do not reproduce the authors' detailed
superdense, tick-level hardware schedule or detector error model.  Extra full
surface-QEC rounds are available only as an explicit analysis option and are
not part of the default transfer.

Distance-independent circuit construction lives in
``state_transfer_common_qiskit``; parsing, acceptance, and frame interpretation
live in ``state_transfer_analysis_qiskit``.
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
    HiranoLabeledQRegs,
    SurfaceReadoutBasis,
    SurfaceReadoutMode,
)
from research_qec_haoran.steane_code.state_injection_qiskit import InjectedStateName
from research_qec_haoran.surface_code import rotated_d5_qiskit as d5
from research_qec_haoran.surface_code.decoding.rotated_decoder import RotatedDetectedErrors

SURFACE_ORDER: tuple[str, ...] = tuple(d5.SURFACE_ORDER)
DEFAULT_LOGICAL_X: tuple[str, ...] = tuple(d5.DEFAULT_LOGICAL_X)
DEFAULT_LOGICAL_Z: tuple[str, ...] = tuple(d5.DEFAULT_LOGICAL_Z)
Z_STABILIZERS: dict[str, tuple[str, ...]] = d5.Z_STABILIZERS
X_STABILIZERS: dict[str, tuple[str, ...]] = d5.X_STABILIZERS

SURFACE_TOP_X_RECOVERY_LABELS: tuple[str, ...] = ("X_ab", "X_cd")
SURFACE_TOP_X_RECOVERY_SUPPORTS: dict[str, tuple[str, ...]] = {
    "X_ab": ("a", "b"),
    "X_cd": ("c", "d"),
}
MERGED_BOUNDARY_CHECKS: tuple[common.MergedBoundaryCheck, ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
    ("Zde", (), ("d", "e")),
)

TRANSFER_SPEC = common.SurfaceCodeTransferSpec(
    name="rotated d5",
    surface_order=SURFACE_ORDER,
    logical_x=DEFAULT_LOGICAL_X,
    logical_z=DEFAULT_LOGICAL_Z,
    z_stabilizers=Z_STABILIZERS,
    x_stabilizers=X_STABILIZERS,
    merged_boundary_checks=MERGED_BOUNDARY_CHECKS,
    top_x_recovery_checks=tuple(SURFACE_TOP_X_RECOVERY_SUPPORTS.items()),
    prepare_plus=d5.prepare_plus,
    append_z_stabilizer_measurements=d5.append_z_stabilizer_measurements,
    append_x_stabilizer_measurements=d5.append_x_stabilizer_measurements,
    append_stabilizer_round=d5.append_stabilizer_round,
    append_logical_measurement=d5.append_logical_measurement,
)

SURFACE_INIT_Z_LABELS = TRANSFER_SPEC.z_labels
SURFACE_POST_Z_LABELS = TRANSFER_SPEC.z_labels
SURFACE_POST_X_LABELS = TRANSFER_SPEC.x_labels
SURFACE_MERGED_Z_LABELS = TRANSFER_SPEC.z_labels
SURFACE_MERGED_X_LABELS = TRANSFER_SPEC.merged_x_labels
POST_TRANSFER_SURFACE_SYNDROME_SCHEDULE = (
    ("surface_init_z", SURFACE_INIT_Z_LABELS),
    ("surface_top_x_recovery", SURFACE_TOP_X_RECOVERY_LABELS),
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


def build_hirano_labeled_d5_qregs(
    *,
    include_top_x_recovery: bool = True,
    include_merged_surface_syndromes: bool = False,
    include_surface_qec: bool = False,
    surface_logical_measurement_basis: SurfaceReadoutBasis | None = None,
) -> HiranoLabeledQRegs:
    """Create paper-labeled registers for the d5 transfer."""
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
    """Append repeated ``Z1a``, ``Z35bc``, and ``Zde`` checks."""
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
    """Measure compatible d5 surface stabilizers during one merged round."""
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


def append_surface_top_x_recovery_measurements(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    add_barriers: bool = True,
) -> None:
    """Recover ``X_ab`` and ``X_cd`` after Steane demolition."""
    common.append_surface_top_x_recovery_measurements(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bits=classical_bits,
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
    """Measure every d5 surface data qubit in the requested basis."""
    common.append_surface_logical_readout(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bits=classical_bits,
        basis=basis,
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
    """Measure the standard d5 logical Pauli support indirectly."""
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
    """Destructively measure the standard d5 logical Pauli support."""
    common.append_surface_logical_pauli_measurement(
        circuit,
        spec=TRANSFER_SPEC,
        registers=registers,
        classical_bit=classical_bits,
        basis=basis,
        destructive=True,
        add_barriers=add_barriers,
    )


def build_steane_to_rotated_d5_lattice_surgery_circuit(
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
    """Build the complete Figure 6 stabilizer-level d5 transfer.

    Top-edge ``X_ab``/``X_cd`` recovery is always included because it is part of
    Figure 6. ``include_surface_qec`` is an opt-in analysis extension and is
    false for the paper-facing transfer.
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
    """Parse a d5 transfer count key into labeled protocol fields."""
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
    """Return whether every Figure 6 boundary check is stable in time."""
    return analysis.merged_boundary_checks_stable(measurements, MERGED_BOUNDARY_CHECKS)


def surface_qec_deltas(measurements: dict[str, object]) -> list[dict[str, dict[str, int]]]:
    """Return optional post-transfer d5 syndrome deltas."""
    return analysis.surface_qec_deltas(
        measurements,
        surface_post_z_labels=SURFACE_POST_Z_LABELS,
        surface_post_x_labels=SURFACE_POST_X_LABELS,
    )


def merged_surface_syndrome_deltas(
    measurements: dict[str, object],
) -> list[dict[str, dict[str, int]]]:
    """Return d5 surface syndrome deltas measured during the merge."""
    return analysis.merged_surface_syndrome_deltas(
        measurements,
        merged_surface_z_labels=SURFACE_MERGED_Z_LABELS,
        merged_surface_x_labels=SURFACE_MERGED_X_LABELS,
    )


def merged_surface_qec_recoveries(
    measurements: dict[str, object],
) -> list[RotatedDetectedErrors]:
    """Decode every d5 surface syndrome round measured during the merge."""
    return analysis.merged_surface_qec_recoveries(measurements, **_SURFACE_CODE_KWARGS)


def merged_surface_qec_recovery(measurements: dict[str, object]) -> RotatedDetectedErrors:
    """Return the last decoded merged-interval d5 surface recovery."""
    return analysis.merged_surface_qec_recovery(measurements, **_SURFACE_CODE_KWARGS)


def surface_qec_recoveries(measurements: dict[str, object]) -> list[RotatedDetectedErrors]:
    """Decode all optional post-transfer d5 syndrome rounds."""
    return analysis.surface_qec_recoveries(measurements, **_SURFACE_PATCH_KWARGS)


def surface_qec_recovery(measurements: dict[str, object]) -> RotatedDetectedErrors:
    """Return the last optional post-transfer d5 recovery."""
    return analysis.surface_qec_recovery(measurements, **_SURFACE_PATCH_KWARGS)


def post_transfer_surface_syndrome_schedule(
    surface_qec_rounds: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the opt-in d5 surface-QEC schedule after transfer."""
    return analysis.post_transfer_surface_syndrome_schedule(
        surface_qec_rounds,
        surface_init_z_labels=SURFACE_INIT_Z_LABELS,
        surface_post_z_labels=SURFACE_POST_Z_LABELS,
        surface_post_x_labels=SURFACE_POST_X_LABELS,
        surface_x_recovery_labels=SURFACE_TOP_X_RECOVERY_LABELS,
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
    """Return whether every fault-detecting Figure 6 check accepts."""
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
    """Build the d5 target frame from teleportation and optional QEC."""
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
    """Return the uncorrected d5 logical readout bit."""
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
    """Return the d5 logical readout interpreted through its Pauli frame."""
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
    "SURFACE_MERGED_X_LABELS",
    "SURFACE_MERGED_Z_LABELS",
    "SURFACE_ORDER",
    "SURFACE_TOP_X_RECOVERY_LABELS",
    "TRANSFER_SPEC",
    "X_STABILIZERS",
    "Z_STABILIZERS",
    "append_boundary_zz_checks",
    "append_merged_interval_surface_syndrome_round",
    "append_surface_logical_readout",
    "append_surface_top_x_recovery_measurements",
    "build_hirano_labeled_d5_qregs",
    "build_steane_to_rotated_d5_lattice_surgery_circuit",
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
