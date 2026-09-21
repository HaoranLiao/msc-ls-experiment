"""Framework-neutral geometry, schedules, and helpers for Steane cultivation.

Shared by :mod:`magic_state_cultivation_qiskit` and
:mod:`magic_state_cultivation_stim`: the 11-site patch layout, the fold/unfold
CNOT schedule, the per-round measurement counts, and the optional-CX filter.
Framework-specific circuit construction lives in the two framework modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from msc.steane_code.state_injection_common import (
    NUM_QUBITS_PATCH_11_SITES,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_HELPER_SITES,
    post_injection_qed_stabilizer_labels,
)

Patch14SiteEncoderName = Literal["abstract", "abstract_routed", "hirano"]
CultivatedStateName = Literal["T", "Ty"]

# =============================================================================
# 11-site cultivation patch layout
# =============================================================================

# 11-site labeling used in the cultivation patch
# ----------------------------------------------
#
# The physical sites are labeled as
#
#               0
#
#           1       2
#
#       3       4       5
#
#           6       7
#
#       8       9       10
#
# The seven Steane/color-code data sites are
#     PATCH_11_SITE_DATA_SITES = (0, 3, 4, 5, 8, 9, 10)
# and the four helper/measurement sites are
#     PATCH_11_SITE_HELPER_SITES = (1, 2, 6, 7).
#
# The step-3-to-step-6 fold schedule from the figure is a 4-3-2-1
# CNOT schedule:
#
# Step 3:
#     CX 2 -> 0
#     CX 1 -> 4
#     CX 6 -> 3
#     CX 7 -> 10
#
# Step 4:
#     CX 6 -> 8
#     CX 4 -> 2
#     CX 7 -> 5
#
# Step 5:
#     CX 4 -> 6
#     CX 7 -> 9
#
# Step 6:
#     CX 7 -> 4
#
# The folded observable is then measured on site 7 in the X basis. After the
# folded root, ``PATCH_11_SITE_HELPER_SITES`` is measured in its established
# order; site 7's second measurement is the root check.
#
# The unfold is the exact inverse of the above CNOT schedule.

PATCH_11_SITE_FOLDED_MEASUREMENT_SITE: int = 7
PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX: int = 1 + PATCH_11_SITE_HELPER_SITES.index(
    PATCH_11_SITE_FOLDED_MEASUREMENT_SITE
)
PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND: int = 1 + len(PATCH_11_SITE_HELPER_SITES)
STEANE_QED_MEASUREMENTS_PER_ROUND: int = len(
    post_injection_qed_stabilizer_labels(NUM_QUBITS_PATCH_11_SITES)
)


# =============================================================================
# 14-site cultivation patch layout
# =============================================================================

PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND: int = STEANE_QED_MEASUREMENTS_PER_ROUND + 1

# 14-site labeling used in the cultivation patch
# ----------------------------------------------
#
# The rotated (45°-tilted) Hirano patch is laid out on local coordinates.
# ``D4`` is the mid-encoding injection site and ``D2'`` (site 13) is the
# superdense second copy of data qubit 2, untouched by cultivation:
#
#           x=0    x=1    x=2    x=3    x=4    x=5    x=6
#    y=0                   D6
#    y=1                          H9            D2'
#    y=2                   D4            H10           D2
#    y=3           H7            D0            H12
#    y=4                  H8            H11
#    y=5           D1            D5            D3
#
# The fold is a 4-4-2-1 CNOT schedule:
#
# Layer 1:
#     CX 9 -> 6
#     CX 12 -> 2
#     CX 8 -> 1
#     CX 11 -> 3
#
# Layer 2:
#     CX 10 -> 12
#     CX 9 -> 4
#     CX 0 -> 8
#     CX 11 -> 5
#
# Layer 3:
#     CX 10 -> 9
#     CX 0 -> 11
#
# Layer 4:
#     CX 10 -> 0
#
# The unfold is the exact inverse of the above CNOT schedule.
#
# The site layout, data/helper sites, and encoder schedule live in
# state_injection_common.py (``PATCH_14_SITE_COORDS`` and neighbours).
# Cultivation folds the seven data sites onto helper root H10 (site 10), resets
# and remeasures H10 after unfolding, and reads H8, H9, H11, and H12 as four
# fixed-zero helper checks. H7 and site 13 (D2') are untouched by cultivation.
# After the folded root, the H8-H12 suffix of ``PATCH_14_SITE_HELPER_SITES`` is
# measured in its established order; H10's second measurement is the root check.
PATCH_14_SITE_FOLDED_MEASUREMENT_SITE: int = 10
PATCH_14_SITE_CULTIVATION_ROOT_CHECK_INDEX: int = 1 + PATCH_14_SITE_HELPER_SITES[1:].index(
    PATCH_14_SITE_FOLDED_MEASUREMENT_SITE
)
PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND: int = 1 + len(PATCH_14_SITE_HELPER_SITES[1:])


def measured_14_site_cultivation_helpers(*, measure_flags: bool) -> tuple[int, ...]:
    """Helper sites read in the X basis after unfolding one 14-site cultivation round.

    These follow the folded H10 root in record order. Measuring flags reads
    H8-H12 (the H10 root check plus the four fixed-zero ancilla flags); skipping
    flags reads only H10, keeping the folded root and its root check and dropping
    the four ancilla-flag measurements. Callers derive the per-round measurement
    count (``1 + len``) and the root-check index (``1 + index of H10``).
    """
    if measure_flags:
        return PATCH_14_SITE_HELPER_SITES[1:]
    return (PATCH_14_SITE_FOLDED_MEASUREMENT_SITE,)


# =============================================================================
# Fold / unfold network schedule
# =============================================================================

# 11-site patch fold network (Google Fig. 1 panel-C steps 3-6, a 4-3-2-1 CNOT
# schedule that folds onto helper site 7).
PATCH_11_SITE_FOLD_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    # Step 3: four CNOTs.
    (
        (2, 0),
        (1, 4),
        (6, 3),
        (7, 10),
    ),
    # Step 4: three CNOTs.
    (
        (6, 8),
        (4, 2),
        (7, 5),
    ),
    # Step 5: two CNOTs.
    (
        (4, 6),
        (7, 9),
    ),
    # Step 6: one CNOT.
    ((7, 4),),
)

# Four-layer 14-site fold network rooted at helper H10 (site 10). The network
# folds all seven data sites (0-6) without using H7; site 13 (D2') is untouched.
# Every ``(control_site, target_site)`` pair is a diagonally local CX.
PATCH_14_SITE_FOLD_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    # Layer 1: four CNOTs.
    (
        (9, 6),
        (12, 2),
        (8, 1),
        (11, 3),
    ),
    # Layer 2: four CNOTs.
    (
        (10, 12),
        (9, 4),
        (0, 8),
        (11, 5),
    ),
    # Layer 3: two CNOTs.
    (
        (10, 9),
        (0, 11),
    ),
    # Layer 4: one CNOT onto the root.
    ((10, 0),),
)


def filter_11_site_patch_fold_layer(
    layer: Sequence[tuple[int, int]],
    *,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Optionally remove a specific CX gate from a fold layer.

    Parameters
    ----------
    skip_cx_1_4
        shorthand for ``skip_cx=(1, 4)``.
    skip_cx
        ``(control, target)`` pair to remove. Takes precedence over
        ``skip_cx_1_4`` if both are given.
    """
    gate_to_skip = skip_cx if skip_cx is not None else ((1, 4) if skip_cx_1_4 else None)
    if gate_to_skip is None:
        return list(layer)
    return [(control, target) for control, target in layer if (control, target) != gate_to_skip]


def patch_14_site_encoder_flag_count(encoder: Patch14SiteEncoderName) -> int:
    """Return the number of optional postselection flags emitted by an encoder."""
    if encoder == "abstract":
        return 0
    if encoder == "abstract_routed":
        return len(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES)
    if encoder == "hirano":
        return len(PATCH_14_SITE_HELPER_SITES)
    raise ValueError("encoder must be 'abstract', 'abstract_routed', or 'hirano'.")
