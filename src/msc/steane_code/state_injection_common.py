"""Framework-neutral geometry and schedules for Steane state injection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

# =============================================================================
# Shared types and stabilizer selection
# =============================================================================


PatchEncoderGateName = Literal[
    "H",
    "CZ",
    "CX",
    "RESET_X",
    "RESET_Z",
    "INJECT",
]
PatchStabilizerBasis = Literal["X", "Z", "BOTH"]


def resolve_patch_stabilizer_indices(
    stabilizer_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate and return an ordered selection of patch stabilizer indices."""
    if stabilizer_indices is None:
        selected_indices = (0, 1, 2)
    else:
        selected_indices = tuple(stabilizer_indices)
    if not selected_indices:
        raise ValueError("stabilizer_indices must select at least one stabilizer.")
    if len(set(selected_indices)) != len(selected_indices) or any(
        not 0 <= index < 3 for index in selected_indices
    ):
        raise ValueError("stabilizer_indices must contain distinct indices in [0, 2].")
    return selected_indices


# =============================================================================
# Abstract Steane code
# =============================================================================


# -----------------------------------------------------------------------------
# Stabilizers and logical operators
# -----------------------------------------------------------------------------


# Steane [[7,1,3]] qubit layout (Fano plane in triangular form):
#
#        0
#     1     2
#        3
# 4      5      6
#
# Faces (stabilizers, weight-4):
#   Top: {0, 1, 2, 3}
#   Bottom-left: {1, 3, 4, 5}
#   Bottom-right: {2, 3, 5, 6}
#
# Sides (logicals, weight-3):
#   Left: {0, 1, 4}    Right: {0, 2, 6}    Bottom: {4, 5, 6}

# Steane [[7,1,3]] stabilizers and logical operators for qubits 0–6.
X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 3, 4, 5),
    (2, 3, 5, 6),
)
Z_STABILIZERS: tuple[tuple[int, ...], ...] = X_STABILIZERS

# May not be used but useful as a reference
DEFAULT_LOGICAL_X: tuple[int, ...] = (0, 1, 4)
DEFAULT_LOGICAL_Z: tuple[int, ...] = DEFAULT_LOGICAL_X

# All weight-3 logical representatives (Fano plane lines).
LOGICAL_X_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 1, 4),
    (0, 2, 6),
    (0, 3, 5),
    (1, 2, 5),
    (1, 3, 6),
    (2, 3, 4),
    (4, 5, 6),
)
LOGICAL_Z_REPRESENTATIVES: tuple[tuple[int, ...], ...] = LOGICAL_X_REPRESENTATIVES


# -----------------------------------------------------------------------------
# Encoder schedule
# -----------------------------------------------------------------------------


# Abstract Steane encoder schedule grouped into seven collision-free CX layers.
# Both frameworks consume this single representation; Stim additionally emits
# an explicit ``TICK`` after each layer.
ABSTRACT_ENCODER_CNOT_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((6, 2),),
    ((0, 6), (2, 5)),
    ((4, 0), (2, 3)),
    ((0, 1),),
    ((1, 3), (0, 2)),
    ((1, 4),),
    ((4, 5),),
)


# =============================================================================
# 11-site patch
# =============================================================================


# -----------------------------------------------------------------------------
# Geometry and site roles
# -----------------------------------------------------------------------------


# 11-site labeling used in the patch:
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
# Data sites:   (0, 3, 4, 5, 8, 9, 10)
# Helper sites: (1, 2, 6, 7)

NUM_QUBITS_PATCH_11_SITES: int = 11
PATCH_11_SITE_DATA_SITES: tuple[int, ...] = (0, 3, 4, 5, 8, 9, 10)
PATCH_11_SITE_HELPER_SITES: tuple[int, ...] = (1, 2, 6, 7)

# Site index -> local (x, y) coordinate on the same diagonal grid convention
# used by the 14-site patch.
PATCH_11_SITE_COORDS: tuple[tuple[int, int], ...] = (
    (2, 0),  # 0  data  (Steane 0; injection)
    (1, 1),  # 1  helper (unused routing site)
    (3, 1),  # 2  helper (face {0,3,4,5} ancilla)
    (0, 2),  # 3  data  (Steane 1)
    (2, 2),  # 4  data  (Steane 3)
    (4, 2),  # 5  data  (Steane 2)
    (1, 3),  # 6  helper (face {3,4,8,9} ancilla)
    (3, 3),  # 7  helper (face {4,5,9,10} ancilla)
    (0, 4),  # 8  data  (Steane 4)
    (2, 4),  # 9  data  (Steane 5)
    (4, 4),  # 10 data  (Steane 6)
)


# -----------------------------------------------------------------------------
# Abstract encoder mapping
# -----------------------------------------------------------------------------


# The 7 abstract Steane qubits (0–6) map to 11-site patch data sites as:
#   abstract 0 → patch site 0  (injection)
#   abstract 1 → patch site 3  (|+⟩)
#   abstract 2 → patch site 5  (|0⟩)
#   abstract 3 → patch site 4  (|0⟩)
#   abstract 4 → patch site 8  (|+⟩)
#   abstract 5 → patch site 9  (|0⟩)
#   abstract 6 → patch site 10 (|+⟩)
# Map from abstract Steane qubit index (0–6) to 11-site patch data site.
_STEANE_TO_11_SITE_PATCH: tuple[int, ...] = (0, 3, 5, 4, 8, 9, 10)


# -----------------------------------------------------------------------------
# Stabilizers and logical operators
# -----------------------------------------------------------------------------


PATCH_11_SITE_X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 3, 4, 5),
    (3, 4, 8, 9),
    (4, 5, 9, 10),
)
PATCH_11_SITE_Z_STABILIZERS: tuple[tuple[int, ...], ...] = PATCH_11_SITE_X_STABILIZERS

PATCH_11_SITE_DEFAULT_LOGICAL_X: tuple[int, ...] = (8, 9, 10)
PATCH_11_SITE_DEFAULT_LOGICAL_Z: tuple[int, ...] = PATCH_11_SITE_DEFAULT_LOGICAL_X

PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 3, 8),
    (0, 5, 10),
    (0, 4, 9),
    (3, 5, 9),
    (3, 4, 10),
    (4, 5, 8),
    (8, 9, 10),
)
PATCH_11_SITE_LOGICAL_Z_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    PATCH_11_SITE_LOGICAL_X_REPRESENTATIVES
)


# -----------------------------------------------------------------------------
# Rosenfeld encoder schedule
# -----------------------------------------------------------------------------


# Original site 5 in Rosenfeld et al.'s schedule maps to patch site 4, the
# center data site.  In the Hirano-labeled view used by lattice surgery, this is
# color-code label "0".
PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE: int = 4

# Rosenfeld et al.'s gate schedule mapped into repo patch labels.  Their initial
# H[injection] and I[injection] on original site 5 (patch site 4) are replaced
# by whatever physical state we want to inject into the logical state; this
# preparation is not part of the encoder layers below.
PATCH_11_SITE_ROSENFELD_ENCODER_LAYERS: tuple[tuple[PatchEncoderGateName, tuple], ...] = (
    ("H", (8, 3, 6, 1, 0, 2, 5)),
    ("CZ", ((3, 1), (6, 4), (0, 2))),
    ("H", (6, 9, 0)),
    ("CZ", ((6, 9),)),
    ("H", (9, 4, 7)),
    ("CZ", ((3, 6), (9, 7), (1, 4))),
    ("H", (6, 7)),
    ("CZ", ((6, 9), (4, 7))),
    ("H", (7,)),
    ("CZ", ((8, 6), (7, 5))),
    ("H", (6, 9, 1, 7, 10)),
    ("CZ", ((3, 6), (9, 7))),
    ("CZ", ((6, 4), (1, 0), (7, 10))),
    ("H", (1, 10, 5)),
    ("CZ", ((3, 1), (6, 9), (4, 7), (2, 5))),
    ("H", (3, 6, 9, 7, 2, 5)),
    ("CZ", ((8, 6), (1, 4), (7, 10), (0, 2))),
    ("H", (6, 1, 4, 7, 2)),
)


# -----------------------------------------------------------------------------
# Stabilizer-measurement ancillas
# -----------------------------------------------------------------------------


# Map from stabilizer index to the helper site used as its ancilla.
# Each helper sits at the geometric center of the corresponding face:
#   Stabilizer 0 (0, 3, 4, 5)  → helper site 2
#   Stabilizer 1 (3, 4, 8, 9)  → helper site 6
#   Stabilizer 2 (4, 5, 9, 10) → helper site 7
PATCH_11_SITE_STABILIZER_ANCILLA_SITES: tuple[int, ...] = (2, 6, 7)


# =============================================================================
# 14-site patch
# =============================================================================


# -----------------------------------------------------------------------------
# Geometry and site roles
# -----------------------------------------------------------------------------


# 14-site patch layout on the rotated (45°-tilted) grid used by Hirano's
# lattice-surgery cultivation.  Coordinates are *local to the Steane patch*:
# ``x`` runs 0–6 left-to-right and ``y`` runs 0–5 top-to-bottom (Stim's diagram
# convention, ``y`` increasing downward). Data sites carry the seven Steane
# qubits. The direct abstract encoder does not touch the helpers; the routed
# abstract encoder uses H8–H12 as |0> routing workspace; Hirano's encoder uses
# all six helpers and returns them to |+>. The helpers then serve as superdense
# Bell-pair ancillas. The fourteenth site is D2' (site 13), a second copy of D2
# used to reach the distant face-{0,2,4,6} ancillas:
#
#          x=0    x=1    x=2    x=3    x=4    x=5    x=6
#   y=0                   D6
#   y=1                          H9            D2'
#   y=2                   D4            H10           D2
#   y=3           H7            D0            H12
#   y=4                  H8            H11
#   y=5           D1            D5            D3
#
#   D<i> = data site (site index == Steane qubit index i); D4 is Hirano's phase
#          site, D6 is the direct abstract input, and D3 is the routed input.
#   H<j> = helper/ancilla site (site index j).
#   D2'  = site 13, the superdense second copy of data qubit 2.
#
# Diagonal ``(±1, ±1)`` neighbours define the fixed patch connectivity used by
# the fold and superdense schedules. The direct abstract encoder is
# intentionally unrouted. ``PATCH_14_SITE_HIRANO_ENCODER_COORDS`` separately
# places D1 beside H7 for Hirano's specialized encoder.

NUM_QUBITS_PATCH_14_SITES: int = 14
PATCH_14_SITE_DATA_SITES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
PATCH_14_SITE_HELPER_SITES: tuple[int, ...] = (7, 8, 9, 10, 11, 12)

# Site index → local (x, y) coordinate.  Indices 0–6 are the data sites (equal
# to the Steane qubit index); indices 7–12 are helper sites; index 13 is D2'.
PATCH_14_SITE_COORDS: tuple[tuple[int, int], ...] = (
    (3, 3),  # 0  data  (Steane 0)
    (1, 5),  # 1  data  (Steane 1)
    (6, 2),  # 2  data  (Steane 2)
    (5, 5),  # 3  data  (Steane 3)
    (2, 2),  # 4  data  (Steane 4; Hirano phase-injection site)
    (3, 5),  # 5  data  (Steane 5)
    (2, 0),  # 6  data  (Steane 6)
    (1, 3),  # 7  helper (face {0,1,4,5} X-ancilla)
    (2, 4),  # 8  helper (face {0,1,4,5} Z-ancilla)
    (3, 1),  # 9  helper (face {0,2,4,6} X-ancilla)
    (4, 2),  # 10 helper (face {0,2,4,6} Z-ancilla)
    (4, 4),  # 11 helper (face {0,2,3,5} X-ancilla)
    (5, 3),  # 12 helper (face {0,2,3,5} Z-ancilla)
    (5, 1),  # 13 D2' (superdense second copy of data qubit 2)
)

# Hirano's specialized injection-stage geometry differs only at D1. Its final
# two CXs transfer the D1 state from H7 into (0, 4), keeping every encoder CX
# diagonally local. The Hirano encoder retains that geometry. The mapped
# abstract encoder and fixed post-injection schedules instead use
# ``PATCH_14_SITE_COORDS``, with D1 at (1, 5).
PATCH_14_SITE_HIRANO_ENCODER_COORDS: tuple[tuple[int, int], ...] = (
    PATCH_14_SITE_COORDS[:1] + ((0, 4),) + PATCH_14_SITE_COORDS[2:]
)

# -----------------------------------------------------------------------------
# Abstract encoder mapping
# -----------------------------------------------------------------------------


# The 7 abstract Steane qubits (0–6) map to 14-site patch data sites as:
#   abstract 0 → patch site 6 (D6, injection)
#   abstract 1 → patch site 4 (D4, |+⟩)
#   abstract 2 → patch site 2 (D2, |0⟩)
#   abstract 3 → patch site 0 (D0, |0⟩)
#   abstract 4 → patch site 1 (D1, |+⟩)
#   abstract 5 → patch site 5 (D5, |0⟩)
#   abstract 6 → patch site 3 (D3, |+⟩)
# Map from abstract Steane qubit index (0–6) to 14-site patch data site.
_STEANE_TO_14_SITE_PATCH: tuple[int, ...] = (6, 4, 2, 0, 1, 5, 3)


# -----------------------------------------------------------------------------
# Stabilizers and logical operators
# -----------------------------------------------------------------------------


PATCH_14_SITE_X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 4, 5),
    (0, 2, 3, 5),
    (0, 2, 4, 6),
)
PATCH_14_SITE_Z_STABILIZERS: tuple[tuple[int, ...], ...] = PATCH_14_SITE_X_STABILIZERS

# Hirano cultivates the logical-Y eigenstate on data sites {1, 3, 5}; the
# logical X and Z operators share that weight-3 support.
PATCH_14_SITE_DEFAULT_LOGICAL_X: tuple[int, ...] = (1, 3, 5)
PATCH_14_SITE_DEFAULT_LOGICAL_Z: tuple[int, ...] = PATCH_14_SITE_DEFAULT_LOGICAL_X

# The seven weight-3 logical representatives (Fano lines of this encoder).
PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (1, 3, 5),
    (0, 3, 4),
    (0, 1, 2),
    (2, 4, 5),
    (2, 3, 6),
    (1, 4, 6),
    (0, 5, 6),
)
PATCH_14_SITE_LOGICAL_Z_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    PATCH_14_SITE_LOGICAL_X_REPRESENTATIVES
)

# -----------------------------------------------------------------------------
# Abstract-routed encoder
# -----------------------------------------------------------------------------


# The routed abstract encoder uses the fixed patch geometry, starts its
# arbitrary input on D3, and returns all seven encoded qubits to D0–D6. H8–H12
# are routing workspace; H7 and D2' are idle. Every non-data site finishes in
# |0>, so the superdense and cultivation schedules can follow immediately.
PATCH_14_SITE_ABSTRACT_ROUTED_IBM_MIAMI_QUBITS: tuple[int, ...] = (
    23,
    21,
    15,
    3,
    33,
    12,
    44,
    32,
    22,
    34,
    24,
    13,
    14,
    25,
)
PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_INJECTION_SITE: int = 3
# H8–H12 are the routing workspace and provide five known-zero Z-basis flags.
PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES: tuple[int, ...] = (8, 9, 10, 11, 12)

# The arbitrary-state abstract isometry routed by reversing an IBM Miami route
# of the inverse encoder. The search covers all 168 valid output mappings and
# SABRE seeds 0–63, ranking native depth before involved-site count. The winner
# uses 33 CXs in 14 two-qubit layers on 12 patch sites after reducing its final
# four-CX word to an equivalent two-CX word. IBM basis translation preserves
# the 14-layer depth with 33 CZs. The input state is prepared separately on D3.
PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_LAYERS: tuple[
    tuple[tuple[PatchEncoderGateName, tuple], ...], ...
] = (
    (("H", (12, 5, 11)),),
    (("CX", ((11, 0),)),),
    (("CX", ((0, 4), (11, 3))),),
    (("CX", ((0, 10), (3, 11))),),
    (("CX", ((5, 11), (0, 8))),),
    (("CX", ((11, 12), (8, 0))),),
    (("CX", ((12, 10), (0, 8), (11, 5))),),
    (("CX", ((10, 9), (5, 11))),),
    (("CX", ((9, 10), (11, 5))),),
    (("CX", ((10, 9), (12, 11), (5, 8))),),
    (("CX", ((9, 6), (11, 0), (8, 1), (12, 2))),),
    (("CX", ((6, 9), (0, 11), (1, 8), (2, 12))),),
    (("CX", ((9, 6), (11, 0), (8, 1), (12, 2))),),
    (("CX", ((4, 0),)),),
    (("CX", ((0, 4),)),),
)

# -----------------------------------------------------------------------------
# Hirano encoder schedule
# -----------------------------------------------------------------------------


# Tick-by-tick port of ``injection_generator`` from Hirano's ``steane_code.py``,
# rewritten into 14-site labels.  The encoder only touches the 13 core sites
# (0–12); D2' (site 13) is idle here and is prepared fresh by the superdense
# round. Hirano's final two CXs transfer D1 from H7 into the injection-stage
# cell (0, 4) in ``PATCH_14_SITE_HIRANO_ENCODER_COORDS``. The mapped abstract
# encoder instead prepares D1 directly at the fixed (1, 5) cell. Each tick is a
# tuple of operations applied in parallel:
#   ("RESET_X", (sites...))    reset the listed sites to |+>
#   ("RESET_Z", (sites...))    reset the listed sites to |0>
#   ("CX", ((ctrl, tgt)...))   controlled-X in the reference schedule
#   ("INJECT", (site,))        apply the magic-state phase gate on the injection
#                              site (S / S_DAG proxy for T / T_DAG)
# Hirano's original applies ``S_DAG`` at the injection tick; the framework
# module chooses S or S_DAG from ``injected_state``.
PATCH_14_SITE_HIRANO_ENCODER_LAYERS: tuple[tuple[tuple[PatchEncoderGateName, tuple], ...], ...] = (
    (("RESET_X", (12, 0, 5)), ("RESET_Z", (10, 11, 8))),
    (("CX", ((12, 10), (0, 11), (5, 8))),),
    (("CX", ((12, 11), (0, 8))), ("RESET_X", (7, 2))),
    (("CX", ((0, 10), (5, 11), (7, 8), (2, 12))), ("RESET_X", (9, 3, 4))),
    (("CX", ((9, 10), (3, 11), (4, 7), (12, 2))),),
    (("CX", ((4, 9), (11, 3), (8, 7))),),
    (("CX", ((10, 9), (7, 4))),),
    (("CX", ((9, 4),)),),
    (("INJECT", (4,)),),
    (("CX", ((9, 4),)),),
    (("CX", ((7, 4),)), ("RESET_X", (6, 1))),
    (("CX", ((6, 9), (1, 7))),),
    (("CX", ((9, 6), (7, 1))),),
)

# -----------------------------------------------------------------------------
# Superdense stabilizer-measurement schedule
# -----------------------------------------------------------------------------


# Each of the three weight-4 faces owns a neighboring pair of helper sites,
# used as a Bell pair: an X-ancilla (A) and a Z-ancilla (B).  Face → (A, B):
#
#   Face {0,1,4,5}: A = 7 (1,3),  B = 8  (2,4).
#   Face {0,2,3,5}: A = 11 (4,4), B = 12 (5,3).
#   Face {0,2,4,6}: A = 9 (3,1),  B = 10 (4,2).
#
# The Bell pair (|00>+|11>) is a +1 eigenstate of both X_A X_B and Z_A Z_B, so
# each face's X- and Z-stabilizer can be extracted *simultaneously*: measuring A
# in X returns the X-stabilizer and B in Z returns the Z-stabilizer. Every data
# qubit couples only to whichever ancilla it neighbours — data → ancilla for the
# Z parity, ancilla → data for the X parity — and the Bell correlation combines
# the spatially split contributions to each requested parity.

# The subtlety is data qubit 2: it is shared by faces {0,2,3,5} and {0,2,4,6}
# but neighbours only the former's Z-ancilla (12).  D2' (site 13) neighbours the
# latter's Z-ancilla (10).  We first build a Z-basis repetition code between D2
# and D2' (RZ D2'; CX D2→D2'), so their Z parity is shared and can couple to two
# Z-plaquettes at once; then we unentangle them and build an X-basis repetition
# code (RX D2'; CX D2'→D2), so their X parity is shared for the two X-plaquettes.
#
# The round is one clean tick-scheduled schedule (no consecutive-round detector
# history, so no ``xz``/``zx`` variants).  Each tick is a tuple of operations
# applied in parallel:
#   ("RESET_X"/"RESET_Z", (sites...))   reset the listed sites to |+> / |0>
#   ("CX", ((ctrl, tgt)...))            controlled-X between diagonally adjacent
#                                       sites
#   ("MEASURE_X"/"MEASURE_Z", (sites))  destructive X / Z ancilla readout (one
#                                       detector per measurement, +1 reference)
# The six stabilizer readouts are ordered
# [X0145, Z0145, X0235, Z0235, X0246, Z0246]. A terminal D2' flag follows when
# face 0246 is selected and D2' is not immediately reset for another parity.
PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS: tuple[tuple[int, int], ...] = (
    (7, 8),
    (11, 12),
    (9, 10),
)
_SUPERDENSE_BELL_PREPARATION_LAYERS = (
    (("RESET_X", (7, 11, 9)), ("RESET_Z", (8, 12, 10))),
    (("CX", ((7, 8), (11, 12), (9, 10))),),
)
_SUPERDENSE_D2_Z_COPY_LAYERS = (
    (("RESET_Z", (13,)),),
    (("CX", ((2, 13),)),),
)
_SUPERDENSE_Z_COUPLING_LAYERS = (
    (("CX", ((0, 8), (3, 11), (2, 12), (4, 9), (13, 10))),),
    (("CX", ((4, 7), (5, 8), (0, 11), (6, 9))),),
    (("CX", ((1, 8), (5, 11), (0, 10))),),
)
_SUPERDENSE_D2_Z_UNCOMPUTE_LAYER = (("CX", ((2, 13),)),)
_SUPERDENSE_D2_X_COPY_LAYERS = (
    (("RESET_X", (13,)),),
    (("CX", ((13, 2),)),),
)
_SUPERDENSE_X_COUPLING_LAYERS = (
    (("CX", ((8, 0), (11, 3), (12, 2), (9, 4), (10, 13))),),
    (("CX", ((7, 4), (8, 5), (11, 0), (9, 6))),),
    (("CX", ((8, 1), (11, 5), (10, 0))),),
)
_SUPERDENSE_D2_X_UNCOMPUTE_LAYER = (("CX", ((13, 2),)),)
_SUPERDENSE_BELL_UNCOMPUTE_LAYER = (("CX", ((7, 8), (11, 12), (9, 10))),)
_SUPERDENSE_BELL_READOUT_LAYER = (
    ("MEASURE_X", (7,)),
    ("MEASURE_Z", (8,)),
    ("MEASURE_X", (11,)),
    ("MEASURE_Z", (12,)),
    ("MEASURE_X", (9,)),
    ("MEASURE_Z", (10,)),
)

PATCH_14_SITE_SUPERDENSE_STABILIZER_MEASUREMENT_LAYERS = (
    *_SUPERDENSE_BELL_PREPARATION_LAYERS,
    *_SUPERDENSE_D2_Z_COPY_LAYERS,
    *_SUPERDENSE_Z_COUPLING_LAYERS,
    _SUPERDENSE_D2_Z_UNCOMPUTE_LAYER,
    *_SUPERDENSE_D2_X_COPY_LAYERS,
    *_SUPERDENSE_X_COUPLING_LAYERS,
    _SUPERDENSE_D2_X_UNCOMPUTE_LAYER,
    _SUPERDENSE_BELL_UNCOMPUTE_LAYER,
    _SUPERDENSE_BELL_READOUT_LAYER,
)


def select_14_site_patch_superdense_measurement_layers(
    basis: PatchStabilizerBasis,
    stabilizer_indices: Sequence[int] | None,
):
    """Select superdense layers and label every ancilla readout by its role.

    Unselected faces and opposite-basis coupling phases are removed. For a
    single-basis extraction, the unused Bell-ancilla result is labeled
    ``BELL_PARTNER_FLAG`` instead of ``STABILIZER``.

    Face 0246 uses a temporary repetition encoding of D2 into D2' (site 13).
    Its final known-state measurement is labeled ``D2_COPY_FLAG``: Z after a
    Z-only detour and X otherwise. ``"BOTH"`` omits the intermediate Z flag
    because D2' is immediately reset for the X detour.
    """
    if basis not in ("X", "Z", "BOTH"):
        raise ValueError("basis must be 'X', 'Z', or 'BOTH'.")
    selected_indices = resolve_patch_stabilizer_indices(stabilizer_indices)
    measure_d2_copy_flag = 2 in selected_indices
    measure_x_stabilizers = basis in ("X", "BOTH")
    measure_z_stabilizers = basis in ("Z", "BOTH")
    selected_ancilla_sites = {
        site for index in selected_indices for site in PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS[index]
    }
    if measure_d2_copy_flag:
        selected_ancilla_sites.add(13)
    active_layers = list(_SUPERDENSE_BELL_PREPARATION_LAYERS)

    if measure_z_stabilizers:
        if measure_d2_copy_flag:
            active_layers.extend(_SUPERDENSE_D2_Z_COPY_LAYERS)
        active_layers.extend(_SUPERDENSE_Z_COUPLING_LAYERS)
        if measure_d2_copy_flag:
            active_layers.append(_SUPERDENSE_D2_Z_UNCOMPUTE_LAYER)

    if measure_x_stabilizers:
        if measure_d2_copy_flag:
            active_layers.extend(_SUPERDENSE_D2_X_COPY_LAYERS)
        active_layers.extend(_SUPERDENSE_X_COUPLING_LAYERS)
        if measure_d2_copy_flag:
            active_layers.append(_SUPERDENSE_D2_X_UNCOMPUTE_LAYER)

    active_layers.append(_SUPERDENSE_BELL_UNCOMPUTE_LAYER)

    selected_layers = []
    for tick_layer in active_layers:
        selected_operations = []
        for gate_name, operands in tick_layer:
            if gate_name in ("RESET_X", "RESET_Z"):
                selected_operands = tuple(
                    site for site in operands if site in selected_ancilla_sites
                )
            elif gate_name == "CX":
                selected_operands = tuple(
                    pair
                    for pair in operands
                    if pair[0] in selected_ancilla_sites or pair[1] in selected_ancilla_sites
                )
            else:
                raise ValueError(f"Unexpected operation before readout: {gate_name}.")
            if selected_operands:
                selected_operations.append((gate_name, selected_operands))
        if selected_operations:
            selected_layers.append(tuple(selected_operations))

    if basis == "X":
        x_readout_role = "STABILIZER"
        z_readout_role = "BELL_PARTNER_FLAG"
    elif basis == "Z":
        x_readout_role = "BELL_PARTNER_FLAG"
        z_readout_role = "STABILIZER"
    elif basis == "BOTH":
        x_readout_role = "STABILIZER"
        z_readout_role = "STABILIZER"
    else:  # pragma: no cover - basis is validated above.
        raise ValueError(f"Unsupported stabilizer basis: {basis!r}.")

    readouts = []
    for index in selected_indices:
        x_ancilla, z_ancilla = PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS[index]
        readouts.append(("X", x_ancilla, x_readout_role))
        readouts.append(("Z", z_ancilla, z_readout_role))
    if measure_d2_copy_flag:
        d2_copy_basis = "Z" if basis == "Z" else "X"
        readouts.append((d2_copy_basis, 13, "D2_COPY_FLAG"))
    return tuple(selected_layers), tuple(readouts)


# =============================================================================
# Shared post-injection QED labels
# =============================================================================


def post_injection_qed_stabilizer_labels(num_patch_qubits: int) -> tuple[str, ...]:
    """Return the six post-injection QED stabilizer labels for a Steane patch.

    Cultivation and state transfer measure these checks between injection and
    any later stage. Labels follow each patch's physical measurement order:
    three X checks then three Z checks for 11 sites, and interleaved X/Z checks
    by face for the 14-site superdense round.
    """
    if num_patch_qubits == NUM_QUBITS_PATCH_11_SITES:
        labeled_stabilizers = (
            *(("X", support) for support in PATCH_11_SITE_X_STABILIZERS),
            *(("Z", support) for support in PATCH_11_SITE_Z_STABILIZERS),
        )
    elif num_patch_qubits == NUM_QUBITS_PATCH_14_SITES:
        labeled_stabilizers = tuple(
            (basis, stabilizers[index])
            for index in range(3)
            for basis, stabilizers in (
                ("X", PATCH_14_SITE_X_STABILIZERS),
                ("Z", PATCH_14_SITE_Z_STABILIZERS),
            )
        )
    else:
        raise ValueError(
            f"No post-injection QED labels for a {num_patch_qubits}-qubit patch; "
            f"expected {NUM_QUBITS_PATCH_11_SITES} or {NUM_QUBITS_PATCH_14_SITES}."
        )
    return tuple(
        f"{basis}_{''.join(str(site) for site in support)}"
        for basis, support in labeled_stabilizers
    )
