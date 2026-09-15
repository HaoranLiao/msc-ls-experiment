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
    PATCH_11_SITE_HELPER_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
)

FinalStabilizerMode = Literal["X", "Z", "BOTH", "NONE"]
KickbackTomographyPatchSites = Literal[11, 14]

KT_STABILIZER_BITS_PER_CYCLE = STEANE_QED_MEASUREMENTS_PER_ROUND
KT_DEFAULT_QED_CYCLE_INDICES: tuple[int, ...] = (1, 3)
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
    """Framework-neutral record sizes for one supported KT patch."""

    patch_sites: KickbackTomographyPatchSites
    encoder_flag_count: int
    cultivation_measurement_count: int
    cultivation_root_check_index: int | None
    qed_measurement_count: int


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
        )
    elif patch_sites == 14:
        return KickbackTomographyPatchSpec(
            patch_sites=14,
            encoder_flag_count=len(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES),
            cultivation_measurement_count=PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            cultivation_root_check_index=PATCH_14_SITE_CULTIVATION_ROOT_CHECK_INDEX,
            qed_measurement_count=PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
        )
    else:
        raise ValueError("patch_sites must be 11 or 14.")


@dataclass(frozen=True)
class KickbackCultivationRoundBits:
    """Logical-root, zero-flag, and optional root-check positions for one KT round."""

    folded_root: int
    flag_positions: tuple[int, ...]
    folded_root_check: int | None
    reset_before_root_check: bool

    @property
    def all_positions(self) -> tuple[int, ...]:
        """Return every round position in measurement-record order."""
        positions = (self.folded_root, *self.flag_positions)
        if self.folded_root_check is not None:
            positions += (self.folded_root_check,)
        return tuple(sorted(positions))


@dataclass(frozen=True)
class KickbackTomographyBitLayout:
    """Framework-neutral measurement positions used by a KT experiment."""

    preparation_round: KickbackCultivationRoundBits | None
    tomography_measure_round: KickbackCultivationRoundBits
    tomography_check_round: KickbackCultivationRoundBits
    qed_cycles: tuple[tuple[int, ...], ...]
    qed_cycle_indices: tuple[int, ...]
    final_stabilizers: tuple[int, ...]
    final_bell_partner_flags: tuple[int, ...] = ()
    final_d2_copy_flag: int | None = None
    encoder_flags: tuple[int, ...] = ()

    @property
    def detector_positions(self) -> tuple[int, ...]:
        """Return positions required to be zero by KT postselection."""
        positions: list[int] = [*self.encoder_flags]
        if self.preparation_round is not None:
            positions.extend(self.preparation_round.all_positions)
        positions.extend(self.tomography_measure_round.flag_positions)
        positions.extend(self.tomography_check_round.flag_positions)
        for cycle in self.qed_cycles:
            positions.extend(cycle)
        positions.extend(self.final_stabilizers)
        positions.extend(self.final_bell_partner_flags)
        if self.final_d2_copy_flag is not None:
            positions.append(self.final_d2_copy_flag)
        return tuple(positions)


@dataclass(frozen=True)
class KickbackTomographySummary:
    """Shot counts after KT postselection and logical classification."""

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
    reset_before_root_check: bool,
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
            reset_before_root_check=False,
        )
    if root_check_index <= 0 or root_check_index >= len(positions):
        raise ValueError("root_check_index must select a post-root round position.")
    return KickbackCultivationRoundBits(
        folded_root=positions[0],
        flag_positions=positions[1:root_check_index] + positions[root_check_index + 1 :],
        folded_root_check=positions[root_check_index],
        reset_before_root_check=reset_before_root_check,
    )


def normalize_qed_cycle_indices(qed_cycle_indices: Iterable[int]) -> tuple[int, ...]:
    """Validate QED cycle numbers and return them in protocol order."""
    indices = tuple(qed_cycle_indices)
    if any(index not in KT_FULL_QED_CYCLE_INDICES for index in indices):
        raise ValueError("qed_cycle_indices may only contain 1, 2, 3, and 4.")
    if len(set(indices)) != len(indices):
        raise ValueError("qed_cycle_indices must not contain duplicates.")
    return tuple(sorted(indices))


def final_stabilizer_bit_count(mode: FinalStabilizerMode) -> int:
    """Return the number of outcomes produced by the final stabilizer mode."""
    if mode == "BOTH":
        return KT_STABILIZER_BITS_PER_CYCLE
    if mode in {"X", "Z"}:
        return KT_STABILIZER_BITS_PER_CYCLE // 2
    if mode == "NONE":
        return 0
    raise ValueError("final_stabilizer_basis must be 'X', 'Z', 'BOTH', or 'NONE'.")
