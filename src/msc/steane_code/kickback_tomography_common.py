"""Framework-neutral layouts and result types for kickback tomography."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_ROOT_CHECK_INDEX,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    STEANE_QED_MEASUREMENTS_PER_ROUND,
    CultivatedStateName,
)
from msc.steane_code.state_injection_common import (
    PATCH_11_SITE_DATA_SITES,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_11_SITE_X_STABILIZERS,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_DATA_SITES,
    PATCH_14_SITE_X_STABILIZERS,
)

# Basis of the final destructive data readout, whose face parities check the
# stabilizers of that type.
FinalReadoutBasis = Literal["X", "Z"]
KickbackTomographyPatchSites = Literal[11, 14]

KT_STABILIZER_BITS_PER_CYCLE = STEANE_QED_MEASUREMENTS_PER_ROUND
KT_DEFAULT_QED_CYCLE_INDICES: tuple[int, ...] = (1, 3, 4)
KT_FULL_QED_CYCLE_INDICES: tuple[int, ...] = (1, 2, 3, 4)


def normalize_kickback_tomography_state(state: str) -> CultivatedStateName:
    """Validate and return a supported KT magic-state axis."""
    if state == "T":
        return "T"
    if state == "Ty":
        return "Ty"
    raise ValueError("cultivated_state must be 'T' or 'Ty'.")


@dataclass(frozen=True)
class KickbackTomographyPatchSpec:
    """Framework-neutral record sizes and data geometry for one supported KT patch.

    ``stabilizer_faces`` are the face supports shared by the X and Z
    stabilizers, as patch sites.
    """

    patch_sites: KickbackTomographyPatchSites
    encoder_flag_count: int
    cultivation_measurement_count: int
    cultivation_root_check_index: int | None
    qed_measurement_count: int
    data_sites: tuple[int, ...]
    stabilizer_faces: tuple[tuple[int, ...], ...]


def resolve_kickback_tomography_patch_spec(
    patch_sites: KickbackTomographyPatchSites,
) -> KickbackTomographyPatchSpec:
    """Resolve patch geometry, encoder flags, and cultivation record layout."""
    if patch_sites == 11:
        return KickbackTomographyPatchSpec(
            patch_sites=11,
            encoder_flag_count=len(PATCH_11_SITE_HELPER_SITES),
            cultivation_measurement_count=PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            cultivation_root_check_index=PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX,
            qed_measurement_count=STEANE_QED_MEASUREMENTS_PER_ROUND,
            data_sites=PATCH_11_SITE_DATA_SITES,
            stabilizer_faces=PATCH_11_SITE_X_STABILIZERS,
        )
    elif patch_sites == 14:
        return KickbackTomographyPatchSpec(
            patch_sites=14,
            encoder_flag_count=len(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES),
            cultivation_measurement_count=PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            cultivation_root_check_index=PATCH_14_SITE_CULTIVATION_ROOT_CHECK_INDEX,
            qed_measurement_count=PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
            data_sites=PATCH_14_SITE_DATA_SITES,
            stabilizer_faces=PATCH_14_SITE_X_STABILIZERS,
        )
    else:
        raise ValueError("patch_sites must be 11 or 14.")


@dataclass(frozen=True)
class KickbackCultivationRoundBits:
    """Logical-root, zero-flag, and optional root-check positions for one KT round."""

    folded_root: int
    flag_positions: tuple[int, ...]
    folded_root_check: int | None

    @property
    def all_positions(self) -> tuple[int, ...]:
        """Return every round position in measurement-record order."""
        positions = (self.folded_root, *self.flag_positions)
        if self.folded_root_check is not None:
            positions += (self.folded_root_check,)
        return tuple(sorted(positions))


@dataclass(frozen=True)
class KickbackTomographyBitLayout:
    """Framework-neutral measurement positions used by a KT experiment.

    ``final_readout_faces`` lists, for each stabilizer face, the destructive
    data-readout positions whose parity KT postselection requires to be even.
    """

    preparation_round: KickbackCultivationRoundBits | None
    tomography_measure_round: KickbackCultivationRoundBits
    tomography_check_round: KickbackCultivationRoundBits
    qed_cycles: tuple[tuple[int, ...], ...]
    qed_cycle_indices: tuple[int, ...]
    final_readout_faces: tuple[tuple[int, ...], ...]
    encoder_flags: tuple[int, ...] = ()

    @property
    def detector_positions(self) -> tuple[int, ...]:
        """Return single positions required to be zero by KT postselection."""
        positions: list[int] = [*self.encoder_flags]
        if self.preparation_round is not None:
            positions.extend(self.preparation_round.all_positions)
        positions.extend(self.tomography_measure_round.flag_positions)
        positions.extend(self.tomography_check_round.flag_positions)
        for cycle in self.qed_cycles:
            positions.extend(cycle)
        return tuple(positions)


@dataclass(frozen=True)
class KickbackTomographySummary:
    """Shot counts after KT postselection and logical classification.

    ``measure_check_disagreement_shots`` counts the shots that pass every other
    check but are discarded because the measure and check roots disagree.
    """

    total_shots: int
    retained_shots: int
    rejected_shots: int
    logical_success_shots: int
    logical_error_shots: int
    measure_check_disagreement_shots: int

    @property
    def retainment_rate(self) -> float:
        """Return the fraction of all shots retained by postselection."""
        return self.retained_shots / self.total_shots if self.total_shots else float("nan")

    @property
    def logical_error_probability(self) -> float:
        """Return the conditional KT logical-error probability."""
        return (
            self.logical_error_shots / self.retained_shots if self.retained_shots else float("nan")
        )

    @property
    def fidelity_lower_bound(self) -> float:
        """Return the fidelity lower bound ``1 - p_KT``."""
        return 1.0 - self.logical_error_probability


def cultivation_round_bits_from_positions(
    positions: Sequence[int],
    *,
    root_check_index: int | None,
) -> KickbackCultivationRoundBits:
    """Partition one patch record into its logical root, flags, and optional root check.

    The 11-site schedule repeats site 7 at index four. The 14-site schedule
    records H8-H12 in site order after the root, so repeated H10 is at index
    three and the other four helper outcomes are fixed-zero checks.
    """
    positions = tuple(positions)
    if not positions:
        raise ValueError("cultivation round must contain a folded-root position.")
    if root_check_index is None:
        return KickbackCultivationRoundBits(
            folded_root=positions[0],
            flag_positions=positions[1:],
            folded_root_check=None,
        )
    if root_check_index <= 0 or root_check_index >= len(positions):
        raise ValueError("root_check_index must select a post-root round position.")
    return KickbackCultivationRoundBits(
        folded_root=positions[0],
        flag_positions=positions[1:root_check_index] + positions[root_check_index + 1 :],
        folded_root_check=positions[root_check_index],
    )


def normalize_qed_cycle_indices(qed_cycle_indices: Iterable[int]) -> tuple[int, ...]:
    """Validate QED cycle numbers and return them in protocol order."""
    indices = tuple(qed_cycle_indices)
    if any(index not in KT_FULL_QED_CYCLE_INDICES for index in indices):
        raise ValueError("qed_cycle_indices may only contain 1, 2, 3, and 4.")
    if len(set(indices)) != len(indices):
        raise ValueError("qed_cycle_indices must not contain duplicates.")
    return tuple(sorted(indices))


def final_readout_face_positions(
    spec: KickbackTomographyPatchSpec,
    data_readout_positions: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Map each stabilizer face to its positions in a data readout ordered like ``data_sites``."""
    position_by_site = dict(zip(spec.data_sites, data_readout_positions, strict=True))
    return tuple(tuple(position_by_site[site] for site in face) for face in spec.stabilizer_faces)
