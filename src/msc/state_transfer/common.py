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
from msc.rotated_surface_code.decoding.decoder import SpaceTimeRotatedDecoder
from msc.steane_code.state_injection_common import (
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_HELPER_SITES,
    post_injection_qed_stabilizer_labels,
)
from msc.utils.circuit_common import resolve_logical_measurement_terms
from msc.utils.normalize import normalize_pauli_char

SurfaceReadoutBasis = Literal["X", "Y", "Z"]
SteanePatchSites = Literal[11, 14]
SteaneVerificationBasis = Literal["X", "Z", "BOTH"]
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
class SteaneSuperdenseVerificationGroup:
    """One Bell-pair preparation shared by selected 14-site face checks."""

    basis: SteaneVerificationBasis
    stabilizer_indices: tuple[int, ...]
    check_names_in_readout_order: tuple[str, ...]


def group_14_site_superdense_verification_checks(
    check_names: Sequence[str],
) -> tuple[SteaneSuperdenseVerificationGroup, ...]:
    """Turn the requested checks into physical superdense extractions.

    The input names specify which stabilizers each merge round requires; this
    helper does not change that logical schedule. It groups same-face X and Z
    requests into one ``basis="BOTH"`` extraction so one prepared Bell pair
    measures both checks simultaneously. Without this grouping, the old code
    measured each requested check separately and recreated the Bell pair
    between X and Z on the same face.
    """
    check_by_basis_and_index: dict[tuple[str, int], str] = {}
    bases_by_stabilizer_index: dict[int, set[str]] = {}
    for check_name in check_names:
        try:
            stabilizer_index = STEANE_14_SITE_MERGE_VERIFICATION_STABILIZER_INDICES[check_name]
        except KeyError as error:
            raise ValueError(f"unknown 14-site verification check {check_name!r}.") from error
        key = (check_name[0], stabilizer_index)
        if key in check_by_basis_and_index:
            raise ValueError(f"duplicate 14-site verification check {check_name!r}.")
        check_by_basis_and_index[key] = check_name
        bases_by_stabilizer_index.setdefault(stabilizer_index, set()).add(check_name[0])

    stabilizer_indices_by_basis: dict[SteaneVerificationBasis, list[int]] = {}
    for stabilizer_index, bases in bases_by_stabilizer_index.items():
        basis: SteaneVerificationBasis
        if bases == {"X", "Z"}:
            basis = "BOTH"
        elif bases == {"X"}:
            basis = "X"
        elif bases == {"Z"}:
            basis = "Z"
        else:
            raise ValueError(f"unsupported verification bases {bases!r}.")
        stabilizer_indices_by_basis.setdefault(basis, []).append(stabilizer_index)

    groups = []
    for basis, stabilizer_indices in stabilizer_indices_by_basis.items():
        readout_bases = ("X", "Z") if basis == "BOTH" else (basis,)
        groups.append(
            SteaneSuperdenseVerificationGroup(
                basis=basis,
                stabilizer_indices=tuple(stabilizer_indices),
                check_names_in_readout_order=tuple(
                    check_by_basis_and_index[(readout_basis, stabilizer_index)]
                    for stabilizer_index in stabilizer_indices
                    for readout_basis in readout_bases
                ),
            )
        )
    return tuple(groups)


@dataclass(frozen=True)
class SteaneTransferPatchSpec:
    """Patch-dependent Steane geometry and measurement layout for transfer."""

    site_labels: tuple[str, ...]
    data_sites_in_hirano_order: tuple[int, ...]
    encoder_flag_labels: tuple[str, ...]
    qed_labels: tuple[str, ...]
    cultivation_labels: tuple[str, ...]
    merge_verification_stabilizer_indices: Mapping[str, int]
    merge_bell_partner_flag_labels: tuple[str, ...]


def resolve_steane_transfer_patch_spec(
    patch_sites: SteanePatchSites,
) -> SteaneTransferPatchSpec:
    """Resolve every transfer contract that differs between Steane patches."""
    if patch_sites == NUM_QUBITS_PATCH_11_SITES:
        return SteaneTransferPatchSpec(
            site_labels=tuple(
                HIRANO_PATCH_LABEL_BY_GOOGLE_SITE[site] for site in range(NUM_QUBITS_PATCH_11_SITES)
            ),
            data_sites_in_hirano_order=tuple(
                HIRANO_LABEL_TO_GOOGLE_SITE[label] for label in HIRANO_STEANE_DATA_ORDER
            ),
            encoder_flag_labels=("enc_f1", "enc_f2", "enc_f6", "enc_f7"),
            qed_labels=post_injection_qed_stabilizer_labels(NUM_QUBITS_PATCH_11_SITES),
            cultivation_labels=(
                "root_x",
                "cult_f1",
                "cult_f2",
                "cult_f6",
                "cult_f7",
            ),
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
                *post_injection_qed_stabilizer_labels(NUM_QUBITS_PATCH_14_SITES),
                STEANE_14_SITE_QED_D2_COPY_FLAG_LABEL,
            ),
            cultivation_labels=(
                "root_x",
                *(f"cult_f{site}" for site in PATCH_14_SITE_HELPER_SITES),
            ),
            merge_verification_stabilizer_indices=(
                STEANE_14_SITE_MERGE_VERIFICATION_STABILIZER_INDICES
            ),
            merge_bell_partner_flag_labels=(STEANE_14_SITE_MERGE_BELL_PARTNER_FLAG_LABELS),
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


def parity(bits: Mapping[str, int], labels: Sequence[str]) -> int:
    """Return the mod-2 parity over selected entries of a labeled bit mapping."""
    return sum(bits[label] for label in labels) % 2


def transfer_measurement_labels(
    spec: SurfaceCodeTransferSpec,
    *,
    patch_sites: SteanePatchSites = 14,
    include_cultivation: bool,
    surface_readout_basis: SurfaceReadoutBasis | None,
    post_transfer_stabilizer_rounds: int = 0,
    full_data_readout: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return ordered measurement-family labels for construction and parsing.

    The result is authoritative for Qiskit register allocation and parsing and
    for Stim family labels. Stim separately maps these labels to global record
    positions because physical append order interleaves families within every
    merged round.
    """
    patch_spec = resolve_steane_transfer_patch_spec(patch_sites)
    if post_transfer_stabilizer_rounds < 0:
        raise ValueError("post_transfer_stabilizer_rounds must be nonnegative.")
    labels = {
        "enc_flags": patch_spec.encoder_flag_labels,
        "steane_qed": patch_spec.qed_labels,
    }
    if include_cultivation:
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
        steane_verify=STEANE_MERGE_VERIFICATION_LABELS,
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
    post_transfer_stabilizer_rounds: int = 0,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    full_data_readout: bool = False,
    full_post_selection: bool = False,
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
        "enc_flags": bits["enc_flags"],
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
    }
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
# Postselection and logical frame
# =============================================================================


def merged_boundary_checks_stable(measurements: dict[str, object]) -> bool:
    """Return whether each interface check is unchanged across all three rounds.

    Stability is checked per interface component before their product is
    interpreted as the logical ``Z_L Z_L`` teleportation outcome.
    """
    rounds = measurements["interface_z_rounds"]
    if not rounds:
        return False
    check_names = tuple(rounds[0])
    if any(set(round_bits) != set(check_names) for round_bits in rounds[1:]):
        return False
    return all(
        round_bits[check_name] == rounds[0][check_name]
        for check_name in check_names
        for round_bits in rounds[1:]
    )


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


def transfer_acceptance_succeeds(measurements: dict[str, object]) -> bool:
    """Apply protocol postselection before surface-code correction.

    Encoder flags, post-injection QED, optional cultivation, color-code
    verification and its 14-site Bell-partner flags, final demolition checks,
    and interface changes always cause rejection. Full postselection also
    rejects every surface event. Otherwise only the persistent
    ``X_bcgh``/``X_fgkl`` streams reject; all other surface events are decoded.

    Complementary-gap postselection belongs to the global circuit-level Stim
    analyzer, which owns the complete detector syndrome and matching graph.
    It cannot be derived from these parsed phase-local recovery objects.
    """
    protocol_accepts = not any(
        (
            *measurements["enc_flags"].values(),
            *measurements["post_injection_qed"].values(),
            *measurements.get("cultivation", {}).values(),
            *measurements["steane_verification"].values(),
            *measurements.get("steane_bell_partner_flags", {}).values(),
            *measurements["final_x_verification_checks"].values(),
        )
    ) and merged_boundary_checks_stable(measurements)
    if not protocol_accepts:
        return False

    merged_delta_history = measurements["merged_surface_syndrome_delta_history"]
    post_transfer_delta_history = measurements.get(
        "post_transfer_surface_syndrome_delta_history", ()
    )
    if measurements.get("full_post_selection", False):
        return not any(
            (
                *measurements["surface_init_x"].values(),
                *(
                    bit
                    for delta_layer in merged_delta_history
                    for sector in delta_layer.values()
                    for bit in sector.values()
                ),
                *(
                    bit
                    for delta_layer in post_transfer_delta_history
                    for sector in delta_layer.values()
                    for bit in sector.values()
                ),
            )
        )

    post_transfer_round_count = len(measurements.get("post_transfer_surface_x_rounds", ()))
    persistent_x_labels = tuple(
        label
        for label in PERSISTENT_POSTSELECTED_SURFACE_X_LABELS
        if label in measurements["surface_init_x"]
    )
    return not any(
        (
            *(measurements["surface_init_x"][label] for label in persistent_x_labels),
            *(
                delta_layer["x"][label]
                for delta_layer in merged_delta_history
                for label in persistent_x_labels
            ),
            *(
                delta_layer["x"][label]
                for delta_layer in post_transfer_delta_history[:post_transfer_round_count]
                for label in persistent_x_labels
            ),
        )
    )


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
    frame = PauliFrame()
    frame.register("surface")
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    frame.update_x("surface", bool(measurements["merged_surface_logical_x_byproduct"]))
    frame.update_z("surface", bool(parity(measurements["steane_x"], HIRANO_STEANE_LOGICAL_X)))
    frame.update_z("surface", bool(measurements["merged_surface_logical_z_byproduct"]))
    frame.update_x(
        "surface",
        bool(measurements.get("post_transfer_surface_logical_x_byproduct", 0)),
    )
    frame.update_z(
        "surface",
        bool(measurements.get("post_transfer_surface_logical_z_byproduct", 0)),
    )
    return frame


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
