"""Framework-neutral Steane-to-surface lattice-surgery protocol semantics.

This module is the single source of truth for paper/Hirano labels, target
specifications, measurement-family labels, detector reconstruction, transfer
acceptance, spacetime surface decoding, and logical Pauli-frame interpretation.
Qiskit and Stim allocation, construction, and raw parsing live in the matching
``helper_<framework>`` and ``analysis_<framework>`` modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame
from msc.rotated_surface_code import common as surface_common
from msc.rotated_surface_code.decoding.phenomenological_decoder import SpaceTimeRotatedDecoder
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_14_SITE_CULTIVATION_ROOT_CHECK_INDEX,
    PATCH_14_SITE_FOLDED_MEASUREMENT_SITE,
)
from msc.steane_code.state_injection_common import (
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_HELPER_SITES,
    PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS,
    patch_stabilizer_labels,
    resolve_14_site_patch_stabilizer_checks,
    select_14_site_patch_superdense_measurement_layers,
)
from msc.utils.circuit_common import resolve_logical_measurement_terms
from msc.utils.normalize import normalize_pauli_char

SurfaceReadoutBasis = Literal["X", "Y", "Z"]
SteanePatchSites = Literal[11, 14]
MergeSchedule = Literal["hirano", "minimal_z_only"]
LATTICE_SURGERY_ROUNDS = 3
MergedBoundaryCheck = tuple[str, tuple[str, ...], tuple[str, ...]]
CircuitAppender = Callable[..., None]


# =============================================================================
# Protocol labels and target specification
# =============================================================================

HIRANO_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
HIRANO_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")
HIRANO_STEANE_LOGICAL_X: tuple[str, ...] = ("1", "4", "6")
HIRANO_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")
SURFACE_RECOVERY_X_LABELS: tuple[str, ...] = ("X_ab",)
PERSISTENT_POSTSELECTED_SURFACE_X_LABELS: tuple[str, ...] = ("X_bcgh", "X_fgkl")

STEANE_11_SITE_MERGE_VERIFICATION_STABILIZER_INDICES: dict[str, int] = {
    "Z0145": 1,
    "Z0235": 2,
    "X0235": 2,
    "Z0246": 0,
    "X0246": 0,
}
STEANE_14_SITE_MERGE_VERIFICATION_STABILIZER_INDICES: dict[str, int] = {
    "Z0145": 0,
    "Z0235": 1,
    "X0235": 1,
    "Z0246": 2,
    "X0246": 2,
}
# Hirano deliberately omits bare X0145 during the merge because it
# anticommutes with the active Z1a and Z35bc checks. Z0145 is still extracted;
# its X-basis Bell-partner readout is only a Bell-correlation flag because the
# X-parity data couplings are absent. After demolition, the accepted X check is
# the expanded X0145ab using the one-check m_ab extraction of X_ab.
STEANE_MERGE_VERIFICATION_ROUNDS: tuple[tuple[str, ...], ...] = (
    ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
    ("Z0145", "Z0235", "Z0246"),
    (),
)
STEANE_MERGE_VERIFICATION_LABELS: tuple[str, ...] = tuple(
    f"round{round_index + 1}_{check_name}"
    for round_index, round_checks in enumerate(STEANE_MERGE_VERIFICATION_ROUNDS)
    for check_name in round_checks
)
MINIMAL_Z_ONLY_MERGE_VERIFICATION_ROUNDS: tuple[tuple[str, ...], ...] = (
    ("Z0145", "Z0235"),
    # Z0246 detects X0X5, which commutes with Z0145 and Z0235. X0 can enter
    # during cultivation before its closing Z-check boundary, and X5 can enter
    # during merge; this round-2 check closes that otherwise undetected path.
    ("Z0145", "Z0235", "Z0246"),
    (),
)
STEANE_14_SITE_MERGE_BELL_PARTNER_FLAG_LABELS: tuple[str, ...] = (
    "round1_Z0145_bell_partner_flag",
    "round1_D2_copy_flag",
    "round2_Z0145_bell_partner_flag",
    "round2_Z0235_bell_partner_flag",
    "round2_Z0246_bell_partner_flag",
    "round2_D2_copy_flag",
)
STEANE_14_SITE_QED_D2_COPY_FLAG_LABEL = "D2_copy_flag"
FINAL_X_VERIFICATION_CHECKS: dict[str, tuple[str, ...]] = {
    "X0235": ("0", "2", "3", "5"),
    "X0246": ("0", "2", "4", "6"),
    "X0145ab": ("0", "1", "4", "5", "X_ab"),
}
GOOGLE_SITE_TO_HIRANO_LABEL: dict[int, str] = {
    0: "6",
    3: "4",
    4: "0",
    5: "2",
    8: "1",
    9: "5",
    10: "3",
}
HIRANO_LABEL_TO_GOOGLE_SITE: dict[str, int] = {
    label: site for site, label in GOOGLE_SITE_TO_HIRANO_LABEL.items()
}
STEANE_HELPER_LABEL_TO_GOOGLE_SITE: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h6": 6,
    "h7": 7,
}
HIRANO_PATCH_LABEL_BY_GOOGLE_SITE: dict[int, str] = {
    **GOOGLE_SITE_TO_HIRANO_LABEL,
    **{site: label for label, site in STEANE_HELPER_LABEL_TO_GOOGLE_SITE.items()},
}


@dataclass(frozen=True)
class SteaneTransferPatchSpec:
    """Patch-dependent Steane geometry and measurement layout for transfer."""

    site_labels: tuple[str, ...]
    data_sites_in_hirano_order: tuple[int, ...]
    encoder_flag_labels: tuple[str, ...]
    qed_labels: tuple[str, ...]
    cultivation_labels: tuple[str, ...]
    merge_verification_rounds: tuple[tuple[str, ...], ...]
    merge_verification_stabilizer_indices: Mapping[str, int]
    merge_bell_partner_flag_labels: tuple[str, ...]


def resolve_steane_transfer_patch_spec(
    patch_sites: SteanePatchSites,
    *,
    merge_schedule: MergeSchedule = "hirano",
) -> SteaneTransferPatchSpec:
    """Resolve patch geometry and the selected source-verification record contract."""
    if merge_schedule == "hirano":
        verification_rounds = STEANE_MERGE_VERIFICATION_ROUNDS
        bell_partner_flags = STEANE_14_SITE_MERGE_BELL_PARTNER_FLAG_LABELS
    elif merge_schedule == "minimal_z_only":
        verification_rounds = MINIMAL_Z_ONLY_MERGE_VERIFICATION_ROUNDS
        bell_partner_flags = tuple(
            label
            for round_index, checks in enumerate(verification_rounds, 1)
            for label in (
                *(f"round{round_index}_{check}_bell_partner_flag" for check in checks),
                *((f"round{round_index}_D2_copy_flag",) if "Z0246" in checks else ()),
            )
        )
    else:
        raise ValueError("merge_schedule must be 'hirano' or 'minimal_z_only'.")
    if patch_sites == NUM_QUBITS_PATCH_11_SITES:
        return SteaneTransferPatchSpec(
            site_labels=tuple(
                HIRANO_PATCH_LABEL_BY_GOOGLE_SITE[site] for site in range(NUM_QUBITS_PATCH_11_SITES)
            ),
            data_sites_in_hirano_order=tuple(
                HIRANO_LABEL_TO_GOOGLE_SITE[label] for label in HIRANO_STEANE_DATA_ORDER
            ),
            encoder_flag_labels=("enc_f1", "enc_f2", "enc_f6", "enc_f7"),
            qed_labels=patch_stabilizer_labels(NUM_QUBITS_PATCH_11_SITES),
            cultivation_labels=(
                "root_x",
                "cult_f1",
                "cult_f2",
                "cult_f6",
                "cult_f7",
            ),
            merge_verification_rounds=verification_rounds,
            merge_verification_stabilizer_indices=(
                STEANE_11_SITE_MERGE_VERIFICATION_STABILIZER_INDICES
            ),
            merge_bell_partner_flag_labels=(),
        )
    if patch_sites == NUM_QUBITS_PATCH_14_SITES:
        return SteaneTransferPatchSpec(
            site_labels=(
                *(str(site) for site in range(7)),
                *(f"h{site}" for site in PATCH_14_SITE_HELPER_SITES),
                "d2_prime",
            ),
            data_sites_in_hirano_order=tuple(range(7)),
            encoder_flag_labels=tuple(
                f"enc_h{site}" for site in PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES
            ),
            qed_labels=(
                *patch_stabilizer_labels(NUM_QUBITS_PATCH_14_SITES),
                STEANE_14_SITE_QED_D2_COPY_FLAG_LABEL,
            ),
            cultivation_labels=(
                "root_x",
                *(
                    (
                        "root_x_check"
                        if site == PATCH_14_SITE_FOLDED_MEASUREMENT_SITE
                        else f"cult_f{site}"
                    )
                    for site in PATCH_14_SITE_HELPER_SITES[1:]
                ),
            ),
            merge_verification_rounds=verification_rounds,
            merge_verification_stabilizer_indices=(
                STEANE_14_SITE_MERGE_VERIFICATION_STABILIZER_INDICES
            ),
            merge_bell_partner_flag_labels=bell_partner_flags,
        )
    raise ValueError("patch_sites must be 11 or 14.")


@dataclass(frozen=True)
class SurfaceCodeOperations:
    """Distance-specific surface-code operations used by transfer builders.

    Grouping these callables keeps ownership explicit without adding transfer-
    level wrappers around the d=3 and d=5 modules. Each framework supplies its
    own matching operation set.
    """

    prepare_product_plus: CircuitAppender
    append_scheduled_stabilizer_round: CircuitAppender
    append_logical_measurement: CircuitAppender


@dataclass(frozen=True)
class SurfaceCodeTransferSpec:
    """Distance-specific data and circuit operations for one transfer target.

    The d=3 and d=5 targets differ only in patch geometry, logical supports,
    merged-boundary checks, and top-edge X checks. Binding those values once
    avoids duplicate protocol builders. Distance- and framework-specific
    circuit callables are namespaced under ``surface_code_operations``.

    ``withheld_surface_stabilizer_labels`` lists the ordinary surface checks
    disabled by the merge. It is ``X_ab`` for d=3 and ``X_ab``, ``X_cd`` for
    d=5. Lattice-surgery completion nevertheless measures only ``X_ab``
    (Hirano's ``m_ab``). Canonical supports and schedules remain owned by the
    distance-specific surface-code modules.
    """

    name: str
    distance: int
    surface_order: tuple[str, ...]
    logical_x: tuple[str, ...]
    logical_z: tuple[str, ...]
    z_stabilizers: Mapping[str, tuple[str, ...]]
    x_stabilizers: Mapping[str, tuple[str, ...]]
    merged_boundary_checks: tuple[MergedBoundaryCheck, ...]
    withheld_surface_stabilizer_labels: tuple[str, ...]
    surface_code_operations: SurfaceCodeOperations

    @property
    def z_labels(self) -> tuple[str, ...]:
        """All ordinary surface Z-stabilizer labels in circuit order."""
        return tuple(self.z_stabilizers)

    @property
    def x_labels(self) -> tuple[str, ...]:
        """All ordinary surface X-stabilizer labels in circuit order."""
        return tuple(self.x_stabilizers)

    @property
    def merged_x_labels(self) -> tuple[str, ...]:
        """Surface X checks commuting with the active boundary-Z checks."""
        withheld = set(self.withheld_surface_stabilizer_labels)
        return tuple(label for label in self.x_stabilizers if label not in withheld)


# =============================================================================
# Measurement schema and shot interpretation
# =============================================================================


def resolve_post_transfer_stabilizer_rounds(
    *,
    distance: int,
    full_post_selection: bool,
    post_transfer_stabilizer_rounds: int | None,
) -> int:
    """Resolve and validate the number of surface-code continuation rounds."""
    rounds = post_transfer_stabilizer_rounds
    if rounds is None:
        rounds = 1 if full_post_selection else distance
    if rounds < 1:
        raise ValueError("post_transfer_stabilizer_rounds must be positive.")
    return rounds


def parity(bits: Mapping[str, int], labels: Sequence[str]) -> int:
    """Return the mod-2 parity over selected entries of a labeled bit mapping."""
    return sum(bits[label] for label in labels) % 2


def transfer_measurement_labels(
    spec: SurfaceCodeTransferSpec,
    *,
    patch_sites: SteanePatchSites = 14,
    include_cultivation: bool,
    merge_schedule: MergeSchedule = "hirano",
    surface_readout_basis: SurfaceReadoutBasis | None,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = False,
    post_injection_qed_checks: Sequence[str] | None = None,
    post_transfer_stabilizer_rounds: int = 0,
    full_data_readout: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return ordered measurement-family labels for construction and parsing.

    The result is authoritative for Qiskit register allocation and parsing and
    for Stim family labels. Stim separately maps these labels to global record
    positions because physical append order interleaves families within every
    merged round.

    ``measure_encoder_flags=False`` omits the ``enc_flags`` family so the
    encoder emits no postselection flags. ``measure_cultivation_flags=False``
    shrinks each ``cult`` round to its folded root and root check, dropping the
    four ancilla flags; this applies only to the 14-site patch and is ignored on
    the 11-site patch (whose cultivation round is unchanged).
    ``post_injection_qed_checks`` restricts the ``steane_qed`` family to a subset
    of the six 14-site checks (``None`` keeps all six; only the 14-site patch
    supports subsetting). All shift the record layout, so construction and
    parsing must pass the same values.
    """
    patch_spec = resolve_steane_transfer_patch_spec(patch_sites, merge_schedule=merge_schedule)
    if post_transfer_stabilizer_rounds < 0:
        raise ValueError("post_transfer_stabilizer_rounds must be nonnegative.")
    labels: dict[str, tuple[str, ...]] = {}
    if measure_encoder_flags:
        labels["enc_flags"] = patch_spec.encoder_flag_labels
    if post_injection_qed_checks is None:
        labels["steane_qed"] = patch_spec.qed_labels
    elif patch_sites == NUM_QUBITS_PATCH_14_SITES:
        stabilizer_bases, stabilizer_indices = resolve_14_site_patch_stabilizer_checks(
            post_injection_qed_checks
        )
        qed_labels = []
        if stabilizer_indices:
            _, readouts = select_14_site_patch_superdense_measurement_layers(
                stabilizer_bases, stabilizer_indices
            )
            for readout_basis, ancilla_site, readout_role in readouts:
                if readout_role == "D2_COPY_FLAG":
                    qed_labels.append(STEANE_14_SITE_QED_D2_COPY_FLAG_LABEL)
                    continue
                face_index = next(
                    index
                    for index, ancilla_pair in enumerate(PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS)
                    if ancilla_site in ancilla_pair
                )
                if readout_role == "STABILIZER":
                    qed_labels.append(
                        patch_spec.qed_labels[2 * face_index + (readout_basis == "Z")]
                    )
                elif readout_role == "BELL_PARTNER_FLAG":
                    check_name = patch_spec.qed_labels[2 * face_index + (readout_basis == "X")]
                    qed_labels.append(f"{check_name}_bell_partner_flag")
                else:
                    raise ValueError(f"Unsupported superdense readout role: {readout_role!r}.")
        labels["steane_qed"] = tuple(qed_labels)
    else:
        raise ValueError("post_injection_qed_checks subsetting requires the 14-site patch.")
    if include_cultivation:
        if not measure_cultivation_flags and patch_sites == NUM_QUBITS_PATCH_14_SITES:
            # 14-site only: keep the folded root and its root check (H10 read twice).
            labels["cult"] = (
                patch_spec.cultivation_labels[0],
                patch_spec.cultivation_labels[PATCH_14_SITE_CULTIVATION_ROOT_CHECK_INDEX],
            )
        else:
            labels["cult"] = patch_spec.cultivation_labels
    labels.update(
        surface_init_z=spec.z_labels,
        surface_init_x=spec.x_labels,
        interface_z=surface_common.repeated_round_labels(
            tuple(name for name, _, _ in spec.merged_boundary_checks),
            rounds=LATTICE_SURGERY_ROUNDS,
        ),
        merged_surface_z=surface_common.repeated_round_labels(
            spec.z_labels, rounds=LATTICE_SURGERY_ROUNDS
        ),
        merged_surface_x=surface_common.repeated_round_labels(
            spec.merged_x_labels, rounds=LATTICE_SURGERY_ROUNDS
        ),
        steane_verify=tuple(
            f"round{round_index}_{check}"
            for round_index, checks in enumerate(patch_spec.merge_verification_rounds, 1)
            for check in checks
        ),
        steane_x=HIRANO_STEANE_DATA_ORDER,
        surface_x_recovery=SURFACE_RECOVERY_X_LABELS,
    )
    if patch_spec.merge_bell_partner_flag_labels:
        labels["steane_bell_partner_flags"] = patch_spec.merge_bell_partner_flag_labels
    if post_transfer_stabilizer_rounds:
        labels.update(
            post_transfer_surface_z=surface_common.repeated_round_labels(
                spec.z_labels, rounds=post_transfer_stabilizer_rounds
            ),
            post_transfer_surface_x=surface_common.repeated_round_labels(
                spec.x_labels, rounds=post_transfer_stabilizer_rounds
            ),
        )
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
        if full_data_readout:
            if surface_readout_basis == "Y":
                raise ValueError("full-data surface readout basis must be X or Z.")
            labels[f"surface_data_{surface_readout_basis.lower()}"] = spec.surface_order
        else:
            measurement_terms, _ = resolve_logical_measurement_terms(
                surface_readout_basis,
                logical_x_support=spec.logical_x,
                logical_z_support=spec.logical_z,
            )
            labels[f"surface_logical_{surface_readout_basis.lower()}"] = tuple(
                support_item for support_item, _ in measurement_terms
            )
    return labels


def interpret_transfer_measurements(
    bits: Mapping[str, Mapping[str, int]],
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    post_transfer_stabilizer_rounds: int = 0,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    full_data_readout: bool = False,
    full_post_selection: bool = False,
    persistent_surface_x_post_selection: bool = True,
) -> dict[str, object]:
    """Interpret one labeled shot and derive all shared protocol state.

    Call structure:: label raw families -> split repeated rounds -> reconstruct
    detector histories and the restored surface frame -> decode the merge and
    post-transfer histories -> record logical byproducts. Full-data readout
    contributes both the logical support and the compatible final syndrome.
    """
    interface_z_rounds = _round_bits(
        bits["interface_z"],
        tuple(name for name, _, _ in spec.merged_boundary_checks),
        rounds=LATTICE_SURGERY_ROUNDS,
    )
    merged_surface_z_rounds = _round_bits(
        bits["merged_surface_z"], spec.z_labels, rounds=LATTICE_SURGERY_ROUNDS
    )
    merged_surface_x_rounds = _round_bits(
        bits["merged_surface_x"], spec.merged_x_labels, rounds=LATTICE_SURGERY_ROUNDS
    )
    parsed: dict[str, object] = {
        "post_injection_qed": bits["steane_qed"],
        "surface_init_z": bits["surface_init_z"],
        "surface_init_x": bits["surface_init_x"],
        "surface_x_recovery": bits["surface_x_recovery"],
        "steane_x": bits["steane_x"],
        "steane_verification": bits["steane_verify"],
        "interface_z_rounds": interface_z_rounds,
        "merged_surface_z_rounds": merged_surface_z_rounds,
        "merged_surface_x_rounds": merged_surface_x_rounds,
        "full_post_selection": full_post_selection,
        "persistent_surface_x_post_selection": persistent_surface_x_post_selection,
    }
    if measure_encoder_flags:
        parsed["enc_flags"] = bits["enc_flags"]
    if "steane_bell_partner_flags" in bits:
        parsed["steane_bell_partner_flags"] = bits["steane_bell_partner_flags"]
    if include_cultivation:
        parsed["cultivation"] = bits["cult"]
    if post_transfer_stabilizer_rounds:
        parsed["post_transfer_surface_z_rounds"] = _round_bits(
            bits["post_transfer_surface_z"],
            spec.z_labels,
            rounds=post_transfer_stabilizer_rounds,
        )
        parsed["post_transfer_surface_x_rounds"] = _round_bits(
            bits["post_transfer_surface_x"],
            spec.x_labels,
            rounds=post_transfer_stabilizer_rounds,
        )
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
        if full_data_readout:
            if surface_readout_basis == "Y":
                raise ValueError("full-data surface readout basis must be X or Z.")
            surface_data_readout = bits[f"surface_data_{surface_readout_basis.lower()}"]
            compatible_stabilizers = (
                spec.x_stabilizers if surface_readout_basis == "X" else spec.z_stabilizers
            )
            parsed["surface_data_readout"] = surface_data_readout
            parsed["surface_readout_stabilizers"] = {
                label: parity(surface_data_readout, support)
                for label, support in compatible_stabilizers.items()
            }
            logical_support = spec.logical_x if surface_readout_basis == "X" else spec.logical_z
            parsed["surface_logical_readout"] = {
                label: surface_data_readout[label] for label in logical_support
            }
        else:
            parsed["surface_logical_readout"] = bits[
                f"surface_logical_{surface_readout_basis.lower()}"
            ]

    # Record the known-+1 X preparation boundary, then change-detect the three
    # merge rounds from their measured pre-merge X/Z references.
    initial_compatible_x_delta = {
        label: bits["surface_init_x"][label] for label in spec.merged_x_labels
    }
    parsed["surface_init_x_delta"] = bits["surface_init_x"]
    x_frame = StabilizerFrame()
    x_frame.record_many(initial_compatible_x_delta)
    z_frame = StabilizerFrame()
    z_frame.record_many(bits["surface_init_z"])
    merged_delta_history = []
    for merged_z, merged_x in zip(
        merged_surface_z_rounds,
        merged_surface_x_rounds,
        strict=True,
    ):
        merged_delta_history.append(
            {
                "z": z_frame.advance(merged_z),
                "x": x_frame.advance(merged_x),
            }
        )
    parsed["merged_surface_syndrome_delta_history"] = merged_delta_history

    # At transfer completion, X_ab is restored by m_ab but X_cd remains
    # undefined on d=5 until the first complete post-transfer round.
    transfer_reference_values = {
        **merged_surface_z_rounds[-1],
        **merged_surface_x_rounds[-1],
        **bits["surface_x_recovery"],
    }
    expected_transfer_reference_labels = {
        *spec.z_labels,
        *spec.merged_x_labels,
        *SURFACE_RECOVERY_X_LABELS,
    }
    if set(transfer_reference_values) != expected_transfer_reference_labels:
        raise ValueError("transfer-boundary surface references are inconsistent.")
    transfer_reference_frame = StabilizerFrame()
    transfer_reference_frame.record_many(transfer_reference_values)
    parsed["post_transfer_surface_stabilizer_frame"] = transfer_reference_frame
    if post_transfer_stabilizer_rounds:
        post_transfer_frame = StabilizerFrame()
        post_transfer_frame.record_many(transfer_reference_values)
        post_transfer_delta_history = []
        post_transfer_z_delta_history = []
        post_transfer_x_delta_history = []
        for z_round, x_round in zip(
            parsed["post_transfer_surface_z_rounds"],
            parsed["post_transfer_surface_x_rounds"],
            strict=True,
        ):
            referenced_x_labels = post_transfer_frame.as_dict().keys()
            z_delta = post_transfer_frame.advance(z_round)
            x_delta = post_transfer_frame.advance(
                {label: bit for label, bit in x_round.items() if label in referenced_x_labels}
            )
            post_transfer_frame.record_many(x_round)
            complete_x_delta = {label: x_delta.get(label, 0) for label in spec.x_labels}
            post_transfer_z_delta_history.append(z_delta)
            post_transfer_x_delta_history.append(complete_x_delta)
            post_transfer_delta_history.append({"z": z_delta, "x": complete_x_delta})
        if full_data_readout:
            readout_delta = post_transfer_frame.advance(parsed["surface_readout_stabilizers"])
            post_transfer_delta_history.append({surface_readout_basis.lower(): readout_delta})
            if surface_readout_basis == "Z":
                post_transfer_z_delta_history.append(readout_delta)
            elif surface_readout_basis == "X":
                post_transfer_x_delta_history.append(readout_delta)
            else:  # pragma: no cover - validated by the transfer builder.
                raise ValueError(f"Unsupported surface readout basis: {surface_readout_basis!r}.")
        parsed["post_transfer_surface_syndrome_delta_history"] = post_transfer_delta_history

    # Reconstruct the missing color X checks from demolition and X_ab recovery.
    final_x_bits = {
        **bits["steane_x"],
        "X_ab": bits["surface_x_recovery"]["X_ab"],
    }
    parsed["final_x_verification_checks"] = {
        name: parity(final_x_bits, support) for name, support in FINAL_X_VERIFICATION_CHECKS.items()
    }

    # Decode surface detector layers jointly so timelike edges remain available.
    decoder = SpaceTimeRotatedDecoder(
        data_qubits=spec.surface_order,
        z_stabilizers=spec.z_stabilizers,
        x_stabilizers={label: spec.x_stabilizers[label] for label in spec.merged_x_labels},
        logical_x=spec.logical_x,
        logical_z=spec.logical_z,
    )
    recovery = decoder.decode(
        z_delta_history=[delta_layer["z"] for delta_layer in merged_delta_history],
        x_delta_history=[
            initial_compatible_x_delta,
            *(delta_layer["x"] for delta_layer in merged_delta_history),
        ],
    )
    parsed["merged_surface_qec_recovery"] = recovery
    parsed["merged_surface_logical_x_byproduct"] = (
        len(recovery.x_error_sites & frozenset(spec.logical_z)) % 2
    )
    parsed["merged_surface_logical_z_byproduct"] = (
        len(recovery.z_error_sites & frozenset(spec.logical_x)) % 2
    )
    if post_transfer_stabilizer_rounds:
        post_transfer_decoder = SpaceTimeRotatedDecoder(
            data_qubits=spec.surface_order,
            z_stabilizers=spec.z_stabilizers,
            x_stabilizers=spec.x_stabilizers,
            logical_x=spec.logical_x,
            logical_z=spec.logical_z,
        )
        post_transfer_recovery = post_transfer_decoder.decode(
            z_delta_history=post_transfer_z_delta_history,
            x_delta_history=post_transfer_x_delta_history,
        )
        parsed["post_transfer_surface_qec_recovery"] = post_transfer_recovery
        parsed["post_transfer_surface_logical_x_byproduct"] = (
            len(post_transfer_recovery.x_error_sites & frozenset(spec.logical_z)) % 2
        )
        parsed["post_transfer_surface_logical_z_byproduct"] = (
            len(post_transfer_recovery.z_error_sites & frozenset(spec.logical_x)) % 2
        )
    return parsed


def _round_bits(
    bits: Mapping[str, int],
    labels: Sequence[str],
    *,
    rounds: int,
) -> list[dict[str, int]]:
    return [
        {label: bits[f"round{round_index + 1}_{label}"] for label in labels}
        for round_index in range(rounds)
    ]


# =============================================================================
# Labeled detectors, logical observables, and detector postselection
# =============================================================================


# Every record in these Steane families is a known-zero postselection check.
STEANE_POSTSELECTION_FAMILIES: tuple[str, ...] = (
    "enc_flags",
    "steane_qed",
    "cult",
    "steane_verify",
    "steane_bell_partner_flags",
)
MeasurementRecord = tuple[str, str]
TransferDetectorRole = Literal["protocol", "persistent_surface_x", "surface"]
# One detector: the parity of shared-schema ``(family, label)`` records and its
# postselection role. Protocol detectors always reject, persistent-surface-X
# detectors reject when that policy is enabled, and surface detectors are
# decoded unless every detector is postselected.
TransferDetector = tuple[tuple[MeasurementRecord, ...], TransferDetectorRole]


def transfer_detectors(spec: SurfaceCodeTransferSpec) -> tuple[TransferDetector, ...]:
    """Return the transfer detectors that follow the Steane-local checks.

    Stim builders and the Qiskit circuit-level bridge resolve these labeled
    rules into their own record positions. Circuit order is:

    1. changes between consecutive interface rounds;
    2. final X checks reconstructed from Steane-X demolition and ``m_ab``;
    3. known-+1 initial surface X checks (random initial Z results are only
       references);
    4. merged Z rounds, referenced to the initial Z round;
    5. merged compatible X rounds, referenced to the initial X round.
    """
    detectors = [
        (
            (
                ("interface_z", f"round{round_index - 1}_{name}"),
                ("interface_z", f"round{round_index}_{name}"),
            ),
            "protocol",
        )
        for round_index in range(2, LATTICE_SURGERY_ROUNDS + 1)
        for name, _, _ in spec.merged_boundary_checks
    ]
    final_x_records = {
        **{label: ("steane_x", label) for label in HIRANO_STEANE_DATA_ORDER},
        **{label: ("surface_x_recovery", label) for label in SURFACE_RECOVERY_X_LABELS},
    }
    detectors.extend(
        (tuple(final_x_records[label] for label in support), "protocol")
        for support in FINAL_X_VERIFICATION_CHECKS.values()
    )
    detectors.extend(
        ((("surface_init_x", label),), _surface_detector_role(label))
        for label in spec.x_labels
    )
    detectors.extend(
        _stabilizer_change_detectors(
            "merged_surface_z",
            spec.z_labels,
            rounds=LATTICE_SURGERY_ROUNDS,
            first_round_references={label: ("surface_init_z", label) for label in spec.z_labels},
        )
    )
    detectors.extend(
        _stabilizer_change_detectors(
            "merged_surface_x",
            spec.merged_x_labels,
            rounds=LATTICE_SURGERY_ROUNDS,
            first_round_references={
                label: ("surface_init_x", label) for label in spec.merged_x_labels
            },
        )
    )
    return tuple(detectors)


def post_transfer_detectors(
    spec: SurfaceCodeTransferSpec,
    *,
    post_transfer_stabilizer_rounds: int,
) -> tuple[TransferDetector, ...]:
    """Return merge-to-full continuity detectors for the complete surface rounds.

    Round-1 Z checks compare with merge round 3. Compatible X checks compare
    with merge round 3 and ``X_ab`` with ``m_ab``. The withheld ``X_cd`` has no
    post-merge reference, so its first full-round result starts a new stream
    without a detector. Later rounds compare every check with the round before.
    All Z detectors precede all X detectors.
    """
    final_merge_round = f"round{LATTICE_SURGERY_ROUNDS}_"
    return (
        *_stabilizer_change_detectors(
            "post_transfer_surface_z",
            spec.z_labels,
            rounds=post_transfer_stabilizer_rounds,
            first_round_references={
                label: ("merged_surface_z", final_merge_round + label) for label in spec.z_labels
            },
        ),
        *_stabilizer_change_detectors(
            "post_transfer_surface_x",
            spec.x_labels,
            rounds=post_transfer_stabilizer_rounds,
            first_round_references={
                **{
                    label: ("merged_surface_x", final_merge_round + label)
                    for label in spec.merged_x_labels
                },
                **{label: ("surface_x_recovery", label) for label in SURFACE_RECOVERY_X_LABELS},
            },
        ),
    )


def readout_boundary_detectors(
    spec: SurfaceCodeTransferSpec,
    *,
    surface_readout_basis: SurfaceReadoutBasis,
    post_transfer_stabilizer_rounds: int,
) -> tuple[TransferDetector, ...]:
    """Compare stabilizers reconstructed from full-data readout with the last round."""
    surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    if surface_readout_basis == "X":
        stabilizers = spec.x_stabilizers
    elif surface_readout_basis == "Z":
        stabilizers = spec.z_stabilizers
    else:
        raise ValueError("full-data surface readout basis must be X or Z.")
    basis_name = surface_readout_basis.lower()
    return tuple(
        (
            (
                (
                    f"post_transfer_surface_{basis_name}",
                    f"round{post_transfer_stabilizer_rounds}_{label}",
                ),
                *((f"surface_data_{basis_name}", data_label) for data_label in support),
            ),
            "surface",
        )
        for label, support in stabilizers.items()
    )


def _stabilizer_change_detectors(
    family: str,
    labels: Sequence[str],
    *,
    rounds: int,
    first_round_references: Mapping[str, MeasurementRecord],
) -> list[TransferDetector]:
    """Compare each round with the previous one; round 1 uses referenced labels only."""
    detectors = []
    for round_index in range(1, rounds + 1):
        for label in labels:
            if round_index == 1 and label not in first_round_references:
                continue
            previous = (
                first_round_references[label]
                if round_index == 1
                else (family, f"round{round_index - 1}_{label}")
            )
            detectors.append(
                (
                    (previous, (family, f"round{round_index}_{label}")),
                    _surface_detector_role(label),
                )
            )
    return detectors


def _surface_detector_role(label: str) -> TransferDetectorRole:
    """Mark Hirano's persistently postselected X streams; other surface checks decode."""
    if label in PERSISTENT_POSTSELECTED_SURFACE_X_LABELS:
        return "persistent_surface_x"
    return "surface"


def transfer_logical_frame_records(
    spec: SurfaceCodeTransferSpec,
    basis: SurfaceReadoutBasis,
) -> tuple[MeasurementRecord, ...]:
    """Return the transfer records whose parity is one readout basis's byproduct.

    Steane-X demolition ``X146`` determines the logical Z byproduct, the final
    interface round determines the logical X byproduct, and Y needs both. A
    logical observable adds these records to its readout support exactly once.
    """
    basis = normalize_pauli_char(basis)
    interface = tuple(
        ("interface_z", f"round{LATTICE_SURGERY_ROUNDS}_{name}")
        for name, _, _ in spec.merged_boundary_checks
    )
    demolition = tuple(("steane_x", label) for label in HIRANO_STEANE_LOGICAL_X)
    if basis == "X":
        return demolition
    elif basis == "Y":
        return interface + demolition
    elif basis == "Z":
        return interface
    else:  # pragma: no cover - normalize_pauli_char admits only X, Y, and Z.
        raise ValueError(f"Unsupported logical readout basis: {basis!r}.")


def transfer_postselection_detector_indices(
    roles: Sequence[TransferDetectorRole],
    *,
    full_post_selection: bool,
    persistent_surface_x_post_selection: bool,
) -> tuple[int, ...]:
    """Select hard-postselection detectors from roles listed in circuit order.

    Full postselection rejects every listed detector. Otherwise protocol
    detectors always reject and the persistent ``X_bcgh``/``X_fgkl`` streams
    reject when enabled; the remaining surface detectors are decoded.
    """
    return tuple(
        index
        for index, role in enumerate(roles)
        if full_post_selection
        or role == "protocol"
        or (persistent_surface_x_post_selection and role == "persistent_surface_x")
    )


# =============================================================================
# Postselection and logical frame
# =============================================================================


def merged_boundary_checks_stable(measurements: dict[str, object]) -> bool:
    """Return whether each interface check is unchanged across all three rounds.

    Stability is checked per interface component before their product is
    interpreted as the logical ``Z_L Z_L`` teleportation outcome.
    """
    rounds = measurements["interface_z_rounds"]
    return bool(rounds) and all(round_bits == rounds[0] for round_bits in rounds[1:])


def merged_boundary_logical_zz_outcome(measurements: dict[str, object]) -> int:
    """Return the logical ``Z_L Z_L`` outcome of stable interface checks.

    A measurement bit ``b`` represents eigenvalue ``(-1)^b``, so multiplying
    the stable interface-check eigenvalues is their mod-2 parity. An unstable
    history is a postselection failure and has no logical interpretation.
    """
    if not merged_boundary_checks_stable(measurements):
        raise ValueError("Merged-boundary checks are not individually stable.")
    stable_boundary_bits = measurements["interface_z_rounds"][0]
    return parity(stable_boundary_bits, tuple(stable_boundary_bits))


def transfer_rejection_stages(measurements: dict[str, object]) -> tuple[tuple[str, bool], ...]:
    """Return transfer rejection conditions in measurement-record order."""
    merged_delta_history = measurements["merged_surface_syndrome_delta_history"]
    post_transfer_delta_history = measurements.get(
        "post_transfer_surface_syndrome_delta_history", ()
    )
    if measurements.get("full_post_selection", False):
        surface_init_rejects = any(measurements["surface_init_x"].values())
        merged_surface_rejects = any(
            bit
            for delta_layer in merged_delta_history
            for sector in delta_layer.values()
            for bit in sector.values()
        )
        post_transfer_rejects = any(
            bit
            for delta_layer in post_transfer_delta_history
            for sector in delta_layer.values()
            for bit in sector.values()
        )
    elif measurements.get("persistent_surface_x_post_selection", True):
        persistent_x_labels = tuple(
            label
            for label in PERSISTENT_POSTSELECTED_SURFACE_X_LABELS
            if label in measurements["surface_init_x"]
        )
        surface_init_rejects = any(
            measurements["surface_init_x"][label] for label in persistent_x_labels
        )
        merged_surface_rejects = any(
            delta_layer["x"][label]
            for delta_layer in merged_delta_history
            for label in persistent_x_labels
        )
        post_transfer_round_count = len(measurements.get("post_transfer_surface_x_rounds", ()))
        post_transfer_rejects = any(
            delta_layer["x"][label]
            for delta_layer in post_transfer_delta_history[:post_transfer_round_count]
            for label in persistent_x_labels
        )
    else:
        surface_init_rejects = merged_surface_rejects = post_transfer_rejects = False

    return (
        ("encoder_flags", any(measurements.get("enc_flags", {}).values())),
        ("post_injection_qed", any(measurements["post_injection_qed"].values())),
        ("cultivation", any(measurements.get("cultivation", {}).values())),
        ("surface_initialization", surface_init_rejects),
        ("merge_instability", not merged_boundary_checks_stable(measurements)),
        ("merged_surface_checks", merged_surface_rejects),
        ("steane_verification", any(measurements["steane_verification"].values())),
        ("final_x_verification", any(measurements["final_x_verification_checks"].values())),
        (
            "bell_partner_flags",
            any(measurements.get("steane_bell_partner_flags", {}).values()),
        ),
        ("post_transfer_checks", post_transfer_rejects),
    )


def transfer_acceptance_succeeds(measurements: dict[str, object]) -> bool:
    """Apply protocol postselection before surface-code correction.

    Encoder flags, post-injection QED, optional cultivation, color-code
    verification and its 14-site Bell-partner flags, final demolition checks,
    and interface changes always cause rejection. Full postselection also
    rejects every surface event. Otherwise the optional persistent
    ``X_bcgh``/``X_fgkl`` streams reject; all other surface events are decoded.

    Complementary-gap postselection belongs to the global circuit-level Stim
    analyzer, which owns the complete detector syndrome and matching graph.
    It cannot be derived from these parsed phase-local recovery objects.
    """
    return not any(rejects for _, rejects in transfer_rejection_stages(measurements))


# =============================================================================
# Pauli frame and logical readout
# =============================================================================


def transfer_pauli_frame(measurements: dict[str, object]) -> PauliFrame:
    """Build the target Pauli frame from teleportation and decoded-QEC outcomes.

    The logical ``Z_L Z_L`` interface outcome toggles an X byproduct, while the
    destructive Steane ``X146`` outcome toggles a Z byproduct. Merge and
    post-transfer spacetime decoding independently contribute logical X and Z
    corrections. Future logical readout is interpreted through this frame
    instead of applying physical correction gates.
    """
    pauli_frame = PauliFrame()
    pauli_frame.register("surface")
    pauli_frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    pauli_frame.update_x("surface", bool(measurements["merged_surface_logical_x_byproduct"]))
    pauli_frame.update_z("surface", bool(parity(measurements["steane_x"], HIRANO_STEANE_LOGICAL_X)))
    pauli_frame.update_z("surface", bool(measurements["merged_surface_logical_z_byproduct"]))
    pauli_frame.update_x(
        "surface",
        bool(measurements.get("post_transfer_surface_logical_x_byproduct", 0)),
    )
    pauli_frame.update_z(
        "surface",
        bool(measurements.get("post_transfer_surface_logical_z_byproduct", 0)),
    )
    return pauli_frame


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Interpret destructive logical-support parity through the transfer frame.

    The raw bit is the parity of the measured physical logical terms. This
    operation applies the same shot's teleportation and spacetime-decoder
    byproducts before returning the logical X, Y, or Z outcome.
    """
    if "surface_logical_readout" not in measurements:
        raise ValueError("measurements do not include a surface logical readout.")
    raw_bit = sum(measurements["surface_logical_readout"].values()) % 2
    pauli_frame = transfer_pauli_frame(measurements)
    match basis:
        case "X":
            return pauli_frame.interpret_x_readout("surface", raw_bit)
        case "Y":
            return pauli_frame.interpret_y_readout("surface", raw_bit)
        case "Z":
            return pauli_frame.interpret_z_readout("surface", raw_bit)
    assert_never(basis)
