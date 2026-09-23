"""Framework-neutral geometry and schedules for Steane state injection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, assert_never

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


def _resolve_patch_stabilizer_indices(
    stabilizer_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate and return an ordered selection of patch stabilizer indices."""
    if stabilizer_indices is None:
        selected_indices: tuple[int, ...] = (0, 1, 2)
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


# Abstract Steane encoder: the arbitrary input is prepared on data qubit 0, the
# |+> seeds on data qubits 1, 2, 3, and data qubits 4, 5, 6 start in |0>. The
# twelve-CX seed-addition network, grouped into nine collision-free layers, is
# the routing-friendly factorization for the 14-site patch, relabeled here to
# the abstract [[7,1,3]] stabilizer convention above.
ABSTRACT_ENCODER_CNOT_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((3, 5),),
    ((5, 4), (3, 0)),
    ((5, 6), (0, 3)),
    ((1, 3),),
    ((3, 2),),
    ((2, 6), (3, 5)),
    ((2, 1),),
    ((4, 1),),
    ((1, 4),),
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
    (1, 1),  # 1  helper (face {0,3,4,5} routing ancilla)
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
#   abstract 2 → patch site 5  (|+⟩)
#   abstract 3 → patch site 4  (|+⟩)
#   abstract 4 → patch site 8  (|0⟩)
#   abstract 5 → patch site 9  (|0⟩)
#   abstract 6 → patch site 10 (|0⟩)
# Map from abstract Steane qubit index (0–6) to 11-site patch data site. The
# |+> seeds are abstract qubits 1, 2, 3 (see ``ABSTRACT_ENCODER_CNOT_LAYERS``).
STEANE_TO_11_SITE_PATCH: tuple[int, ...] = (0, 3, 5, 4, 8, 9, 10)


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
# Stabilizer-measurement ancillas and routed schedules
# -----------------------------------------------------------------------------


# Map from stabilizer index to the helper site used as its ancilla.
# The first ancilla is adjacent to 0, 4, 5 but NOT 3; site 1 routes that check.
#   Stabilizer 0 (0, 3, 4, 5)  → helper site 2
#   Stabilizer 1 (3, 4, 8, 9)  → helper site 6
#   Stabilizer 2 (4, 5, 9, 10) → helper site 7
PATCH_11_SITE_STABILIZER_ANCILLA_SITES: tuple[int, ...] = (2, 6, 7)


# Per-face Z-extraction schedules, indexed by stabilizer then two-qubit layer.
# Geometry: Rosenfeld et al., "Magic state cultivation on a superconducting
# quantum processor", Figs. 1 and S10. The extraction uses only our 11 sites,
# with every CX between geometric neighbors (hardware-routable, no SWAPs).
#
# Face 0345 is not simply connected: ancilla 2 neighbors data 0, 4, 5 but not
# 3. Site 1 bridges data 3 into the syndrome. Prepare ancilla 2 in |0> and
# routing ancilla 1 in |+>, thread data 3 through 1 and data 0, then restore
# data 0; measuring Z2 gives the syndrome. Site 1 returns to |+> disentangled
# from the data, so it is not measured. Reversing every CX and exchanging X/Z
# preparation and readout gives X0345. Faces 3489 and 45910 are simply
# connected and use plain four-layer data->ancilla extraction.
PATCH_11_SITE_Z_STABILIZER_CNOT_LAYERS = (
    (
        ((1, 0),),
        ((0, 2),),
        ((4, 2), (1, 0)),
        ((0, 2), (3, 1)),
        ((5, 2), (1, 0)),
        ((0, 2),),
        ((1, 0),),
    ),
    (((3, 6),), ((4, 6),), ((8, 6),), ((9, 6),)),
    (((4, 7),), ((5, 7),), ((9, 7),), ((10, 7),)),
)


def select_11_site_patch_stabilizer_measurement_layers(
    basis: PatchStabilizerBasis,
    stabilizer_indices: Sequence[int] | None,
):
    """Schedule selected local checks in parallel, preserving requested readout order.

    Return the ordered indices and one ``(basis, CX layers)`` phase per measured
    basis. Each phase prepares and reads its selected ancilla sites in the native
    basis; face 0345 additionally uses routing ancilla 1 (prepared in the
    opposite basis and left unmeasured).
    """
    selected_indices = _resolve_patch_stabilizer_indices(stabilizer_indices)
    selected_schedules = tuple(
        PATCH_11_SITE_Z_STABILIZER_CNOT_LAYERS[index] for index in selected_indices
    )
    z_layers = tuple(
        tuple(
            pair
            for schedule in selected_schedules
            if layer_index < len(schedule)
            for pair in schedule[layer_index]
        )
        for layer_index in range(max(map(len, selected_schedules)))
    )
    # X extraction reverses every CX and swaps preparation/readout bases.
    x_layers = tuple(tuple((target, control) for control, target in layer) for layer in z_layers)
    if basis == "X":
        return selected_indices, (("X", x_layers),)
    elif basis == "Z":
        return selected_indices, (("Z", z_layers),)
    elif basis == "BOTH":
        return selected_indices, (("X", x_layers), ("Z", z_layers))
    else:
        raise ValueError("basis must be 'X', 'Z', or 'BOTH'.")


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
#          site, and D3 is the input site of both the abstract and routed
#          encoders.
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


# The abstract Steane encoder (``ABSTRACT_ENCODER_CNOT_LAYERS``) mapped, unrouted,
# onto the 14-site data sites. Abstract encoder qubit (0–6) → 14-site data site:
#   abstract 0 → site 3 (D3, injection)
#   abstract 1 → site 0 (D0, |+⟩)
#   abstract 2 → site 2 (D2, |+⟩)
#   abstract 3 → site 5 (D5, |+⟩)
#   abstract 4 → site 4 (D4, |0⟩)
#   abstract 5 → site 1 (D1, |0⟩)
#   abstract 6 → site 6 (D6, |0⟩)
# This reproduces the routing-friendly twelve-CX network on the fixed patch
# labels (the seed-addition network the routed encoder is derived from), encoding
# into ``PATCH_14_SITE_X_STABILIZERS`` with logical support {1, 3, 5}. The CXs are
# *not* routed through the diagonal connectivity; the routed encoder imposes that.
STEANE_TO_14_SITE_PATCH: tuple[int, ...] = (3, 0, 2, 5, 4, 1, 6)


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
# |0>, so the superdense and cultivation schedules can follow immediately. This
# constant is the patch-site -> IBM Miami physical-qubit hardware layout.
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

# The fixed-patch realization of the seed-addition network above, obtained by
# routing the twelve-CX abstract network with seven SWAPs, deleting eight
# state-trivial CXs (a control known |0> or a target known |+>), cancelling the
# one resulting commuting CX pair, and rescheduling. The result uses 23 CXs in
# nine diagonal-local two-qubit layers on 12 patch sites.
PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_LAYERS: tuple[
    tuple[tuple[PatchEncoderGateName, tuple], ...], ...
] = (
    (("H", (12, 5, 11)),),
    (("CX", ((11, 0),)),),
    (("CX", ((0, 4), (11, 3))),),
    (("CX", ((0, 10), (3, 11))),),
    (("CX", ((5, 11), (0, 8), (12, 10))),),
    (("CX", ((8, 0), (10, 9), (12, 11))),),
    (("CX", ((9, 10), (5, 8), (11, 0), (12, 2))),),
    (("CX", ((9, 6), (8, 1), (0, 11), (2, 12))),),
    (("CX", ((6, 9), (1, 8), (4, 0))),),
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
    basis: PatchStabilizerBasis | Sequence[PatchStabilizerBasis],
    stabilizer_indices: Sequence[int] | None,
):
    """Select superdense layers and label every ancilla readout by its role.

    ``basis`` is one basis for every selected face, or one basis per selected
    index. Mixed X/Z/BOTH faces share one extraction schedule, not sequential
    rounds. Unselected faces and opposite-basis coupling phases are removed. For
    a single-basis extraction, the unused Bell-ancilla result is labeled
    ``BELL_PARTNER_FLAG`` instead of ``STABILIZER``.

    Face 0246 uses a temporary repetition encoding of D2 into D2' (site 13) to
    reach its distant ancilla. The Z detour runs before the X detour, and each
    copies D2 into D2', couples through it, then uncomputes it. D2' is never
    measured between detours: a ``"BOTH"`` extraction resets and reuses D2' for
    the X detour instead of reading an intermediate flag, so even when both
    parities are extracted D2' is measured only once, at the very end, as a
    single ``D2_COPY_FLAG`` on its final known state: Z after a Z-only detour and
    X otherwise (X-only or ``"BOTH"``, whose last detour is X).
    """
    selected_indices = _resolve_patch_stabilizer_indices(stabilizer_indices)
    selected_bases = (basis,) * len(selected_indices) if isinstance(basis, str) else tuple(basis)
    if len(selected_bases) != len(selected_indices) or any(
        face_basis not in ("X", "Z", "BOTH") for face_basis in selected_bases
    ):
        raise ValueError("basis must select X, Z, or BOTH for each selected face.")
    basis_by_face = dict(zip(selected_indices, selected_bases, strict=True))
    selected_ancilla_sites = {
        site for index in selected_indices for site in PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS[index]
    }
    # Filter each parity phase separately, while preparing each face's Bell pair once.
    active_layers = [
        (layer, selected_ancilla_sites) for layer in _SUPERDENSE_BELL_PREPARATION_LAYERS
    ]
    for parity, copy_layers, coupling_layers, uncompute_layer in (
        (
            "Z",
            _SUPERDENSE_D2_Z_COPY_LAYERS,
            _SUPERDENSE_Z_COUPLING_LAYERS,
            _SUPERDENSE_D2_Z_UNCOMPUTE_LAYER,
        ),
        (
            "X",
            _SUPERDENSE_D2_X_COPY_LAYERS,
            _SUPERDENSE_X_COUPLING_LAYERS,
            _SUPERDENSE_D2_X_UNCOMPUTE_LAYER,
        ),
    ):
        phase_ancilla_sites = {
            site
            for index, face_basis in basis_by_face.items()
            if face_basis in (parity, "BOTH")
            for site in PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS[index]
        }
        copy_d2 = basis_by_face.get(2) in (parity, "BOTH")
        if copy_d2:
            phase_ancilla_sites.add(13)
            active_layers.extend((layer, phase_ancilla_sites) for layer in copy_layers)
        active_layers.extend((layer, phase_ancilla_sites) for layer in coupling_layers)
        if copy_d2:
            active_layers.append((uncompute_layer, phase_ancilla_sites))
    active_layers.append((_SUPERDENSE_BELL_UNCOMPUTE_LAYER, selected_ancilla_sites))

    selected_layers = []
    for tick_layer, active_ancilla_sites in active_layers:
        selected_operations = []
        for gate_name, operands in tick_layer:
            if gate_name in ("RESET_X", "RESET_Z"):
                selected_operands = tuple(site for site in operands if site in active_ancilla_sites)
            elif gate_name == "CX":
                selected_operands = tuple(
                    pair
                    for pair in operands
                    if pair[0] in active_ancilla_sites or pair[1] in active_ancilla_sites
                )
            else:
                raise ValueError(f"Unexpected operation before readout: {gate_name}.")
            if selected_operands:
                selected_operations.append((gate_name, selected_operands))
        if selected_operations:
            selected_layers.append(tuple(selected_operations))

    readouts = []
    for index, face_basis in basis_by_face.items():
        for readout_basis, ancilla in zip(
            ("X", "Z"), PATCH_14_SITE_SUPERDENSE_ANCILLA_PAIRS[index], strict=True
        ):
            readout_role = (
                "STABILIZER" if face_basis in (readout_basis, "BOTH") else "BELL_PARTNER_FLAG"
            )
            readouts.append((readout_basis, ancilla, readout_role))
    if 2 in basis_by_face:
        d2_copy_basis = "Z" if basis_by_face[2] == "Z" else "X"
        readouts.append((d2_copy_basis, 13, "D2_COPY_FLAG"))
    return tuple(selected_layers), tuple(readouts)


# =============================================================================
# Shared post-injection QED labels / More generally, shared stabilizer labels
# =============================================================================


def patch_stabilizer_labels(num_patch_qubits: int) -> tuple[str, ...]:
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


PATCH_14_SITE_STABILIZER_CHECK_NAMES = patch_stabilizer_labels(NUM_QUBITS_PATCH_14_SITES)


def resolve_14_site_patch_stabilizer_checks(
    check_names: Sequence[str] | None,
) -> tuple[tuple[PatchStabilizerBasis, ...], tuple[int, ...]]:
    """Parse named post-injection QED checks into per-face bases and indices.

    The returned ``(bases, indices)`` feed ``select_14_site_patch_superdense_
    measurement_layers``; unlike ``_resolve_patch_stabilizer_indices``, which only
    normalizes an integer face selection, this maps ``PATCH_14_SITE_QED_CHECK_NAMES``
    into that face selection plus its per-face basis. Bases correspond to selected
    face indices. Same-face X/Z requests use BOTH; a single-basis face shares the
    schedule and reads its Bell partner as a flag.
    """
    if check_names is None:
        check_names = PATCH_14_SITE_STABILIZER_CHECK_NAMES
    if len(set(check_names)) != len(check_names) or any(
        name not in PATCH_14_SITE_STABILIZER_CHECK_NAMES for name in check_names
    ):
        raise ValueError(
            f"QED checks must be distinct names from {PATCH_14_SITE_STABILIZER_CHECK_NAMES}."
        )
    bases_by_face: dict[int, set[str]] = {}
    for name in check_names:
        bases_by_face.setdefault(PATCH_14_SITE_STABILIZER_CHECK_NAMES.index(name) // 2, set()).add(
            name[0]
        )
    selected_indices = tuple(sorted(bases_by_face))
    selected_bases = tuple(
        "BOTH" if bases_by_face[index] == {"X", "Z"} else next(iter(bases_by_face[index]))
        for index in selected_indices
    )
    return selected_bases, selected_indices


PostInjectionQedChecks = Literal["minimal", "full"]


def resolve_post_injection_qed_checks(
    post_injection_qed_checks: PostInjectionQedChecks,
) -> tuple[str, ...] | None:
    """Map a named post-injection QED preset to concrete 14-site check names.

    ``"full"`` keeps all six Steane stabilizers (``None`` downstream, which the
    11-site patch also accepts). ``"minimal"`` keeps only the distance-three
    subset Z_0145, X_0235, Z_0235: checking just these on the fully noisy MSC-LS
    path preserves the certified fault distance of every X/Z/Y probe (verified by
    the 14-site end-to-end Stim fault-distance scan), and it requires the 14-site
    patch.
    """
    if post_injection_qed_checks == "minimal":
        return ("Z_0145", "X_0235", "Z_0235")
    elif post_injection_qed_checks == "full":
        return None
    else:
        assert_never(post_injection_qed_checks)
