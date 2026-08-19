"""Framework-neutral geometry and schedules for Steane state injection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

PatchEncoderGateName = Literal["H", "CZ", "CX", "RESET_X", "RESET_Z", "INJECT"]
PatchSuperdenseGateName = Literal["CX", "RESET_X", "RESET_Z", "MEASURE_X", "MEASURE_Z"]
PatchStabilizerBasis = Literal["X", "Z", "BOTH"]


# =============================================================================
# Abstract Steane code
# =============================================================================


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

STEANE_ENCODER_CNOTS: tuple[tuple[int, int], ...] = (
    (6, 2),
    (0, 6),
    (4, 0),
    (2, 5),
    (2, 3),
    (0, 1),
    (1, 3),
    (1, 4),
    (4, 5),
    (0, 2),
)


# =============================================================================
# Map to 11-site patch
# =============================================================================


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

NUM_QUBITS_PATCH_11_SITES: int = 11
PATCH_11_SITE_DATA_SITES: tuple[int, ...] = (0, 3, 4, 5, 8, 9, 10)
PATCH_11_SITE_HELPER_SITES: tuple[int, ...] = (1, 2, 6, 7)

# ---------------------------------------------------------------------------
# Mapped stabilizers (expressed in 11-site patch labels)
# ---------------------------------------------------------------------------

PATCH_11_SITE_X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 3, 4, 5),
    (3, 4, 8, 9),
    (4, 5, 9, 10),
)
PATCH_11_SITE_Z_STABILIZERS: tuple[tuple[int, ...], ...] = PATCH_11_SITE_X_STABILIZERS

# ---------------------------------------------------------------------------
# Mapped logical operators (expressed in 11-site patch labels)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Rosenfeld et al. encoder schedule for this 11-site patch
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Stabilizer measurement on the 11-site patch
# ---------------------------------------------------------------------------

# Map from stabilizer index to the helper site used as its ancilla.
# Each helper sits at the geometric center of the corresponding face:
#   Stabilizer 0 (0, 3, 4, 5)  → helper site 2
#   Stabilizer 1 (3, 4, 8, 9)  → helper site 6
#   Stabilizer 2 (4, 5, 9, 10) → helper site 7
PATCH_11_SITE_STABILIZER_ANCILLA_SITES: tuple[int, ...] = (2, 6, 7)


def resolve_11_site_patch_stabilizer_indices(
    stabilizer_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate and return an ordered selection of patch stabilizer indices."""
    if stabilizer_indices is None:
        selected_indices = tuple(range(len(PATCH_11_SITE_X_STABILIZERS)))
    else:
        selected_indices = tuple(stabilizer_indices)
    if not selected_indices:
        raise ValueError("stabilizer_indices must select at least one stabilizer.")
    if len(set(selected_indices)) != len(selected_indices) or any(
        not 0 <= index < len(PATCH_11_SITE_X_STABILIZERS) for index in selected_indices
    ):
        raise ValueError("stabilizer_indices must contain distinct indices in [0, 2].")
    return selected_indices


# =============================================================================
# Map to 14-site patch (Hirano et al., arXiv:2510.24615)
# =============================================================================


# 14-site patch layout on the rotated (45°-tilted) grid used by Hirano's
# lattice-surgery cultivation.  Coordinates are *local to the Steane patch*:
# ``x`` runs 0–6 left-to-right and ``y`` runs 0–5 top-to-bottom (Stim's diagram
# convention, ``y`` increasing downward).  Data sites carry the seven Steane
# qubits; the six helper sites are used transiently by the injection encoder and
# return to |+> once encoding completes, then serve as the superdense Bell-pair
# ancillas.  The fourteenth site is D2' — a second copy of data qubit 2 that the
# superdense round couples to the distant face-{0,2,4,6} ancillas (see below):
#
#          x=0    x=1    x=2    x=3    x=4    x=5    x=6
#   y=0                   D6
#   y=1                          H9            D2'
#   y=2                   D4            H10           D2
#   y=3           H7            D0            H12
#   y=4    D1            H8            H11
#   y=5                          D5            D3
#
#   D<i> = data site (site index == Steane qubit index i); D4 is the injection
#          site where the magic-state phase gate is applied mid-encoding.
#   H<j> = helper/ancilla site (site index j).
#   D2'  = site 13, the superdense second copy of data qubit 2.
#
# Diagonal ``(±1, ±1)`` neighbours are the only pairs entangled by any schedule
# here, so every CX (injection and superdense) is geometrically local.

# Hirano's encoder uses its own Fano-plane coordinatization, so the stabilizers
# and logicals below are expressed directly in these data-site labels (0–6) —
# this is NOT the abstract Steane labeling above.

NUM_QUBITS_PATCH_14_SITES: int = 14

# Site index → local (x, y) coordinate.  Indices 0–6 are the data sites (equal
# to the Steane qubit index); indices 7–12 are helper sites; index 13 is D2'.
PATCH_14_SITE_COORDS: tuple[tuple[int, int], ...] = (
    (3, 3),  # 0  data  (Steane 0)
    (0, 4),  # 1  data  (Steane 1)
    (6, 2),  # 2  data  (Steane 2)
    (5, 5),  # 3  data  (Steane 3)
    (2, 2),  # 4  data  (Steane 4, injection site)
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

PATCH_14_SITE_DATA_SITES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
PATCH_14_SITE_HELPER_SITES: tuple[int, ...] = (7, 8, 9, 10, 11, 12)

# Data site 4 receives the injected magic state mid-encoding.
PATCH_14_SITE_ENCODER_INJECTION_SITE: int = 4

# ---------------------------------------------------------------------------
# Mapped stabilizers and logical operators (14-site patch / Steane labels)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Hirano et al. encoder schedule for this 14-site patch
# ---------------------------------------------------------------------------

# Tick-by-tick port of ``injection_generator`` from Hirano's ``steane_code.py``,
# rewritten into 14-site labels.  The encoder only touches the 13 core sites
# (0–12); D2' (site 13) is idle here and is prepared fresh by the superdense
# round.  Each tick is a tuple of operations applied in parallel:
#   ("RESET_X", (sites...))    reset the listed sites to |+>
#   ("RESET_Z", (sites...))    reset the listed sites to |0>
#   ("CX", ((ctrl, tgt)...))   controlled-X between diagonally adjacent sites
#   ("INJECT", (site,))        apply the magic-state phase gate on the injection
#                              site (S / S_DAG proxy for T / T_DAG)
# Hirano's original applies ``S_DAG`` at the injection tick; the framework
# module chooses S or S_DAG from ``injected_state``.
PATCH_14_SITE_ENCODER_LAYERS: tuple[tuple[tuple[PatchEncoderGateName, tuple], ...], ...] = (
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

# ---------------------------------------------------------------------------
# Conceptually-clear superdense stabilizer measurement on the 14-site patch
# ---------------------------------------------------------------------------

# Site 13 is D2', the superdense second copy of data qubit 2.
PATCH_14_SITE_D2_PRIME_SITE: int = 13

# Each of the three weight-4 faces owns a neighbouring pair of helper sites,
# used as a Bell pair: an X-ancilla (A) and a Z-ancilla (B).  Face → (A, B):
#
#   Face {0,1,4,5}: A = 7 (1,3),  B = 8  (2,4).
#   Face {0,2,3,5}: A = 11 (4,4), B = 12 (5,3).
#   Face {0,2,4,6}: A = 9 (3,1),  B = 10 (4,2).
#
# The Bell pair (|00>+|11>) is a +1 eigenstate of both X_A X_B and Z_A Z_B, so
# each face's X- and Z-stabilizer can be extracted *simultaneously*: measuring A
# in X returns the X-stabilizer and B in Z returns the Z-stabilizer.  Every data
# qubit couples only to whichever ancilla it neighbours — data → ancilla for the
# Z parity, ancilla → data for the X parity — and the Bell correlation carries
# each contribution into both readouts.  Ordered face → (X-ancilla, Z-ancilla):
PATCH_14_SITE_SUPERDENSE_BELL_PAIRS: tuple[tuple[int, int], ...] = (
    (7, 8),
    (11, 12),
    (9, 10),
)

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
# The six readouts are ordered [X0145, Z0145, X0235, Z0235, X0246, Z0246].
PATCH_14_SITE_SUPERDENSE_STABILIZER_MEASUREMENT_LAYERS: tuple[
    tuple[tuple[PatchSuperdenseGateName, tuple], ...], ...
] = (
    # Phase 1 — prepare the three helper Bell pairs (A in |+>, B in |0>, CX A→B).
    (("RESET_X", (7, 11, 9)), ("RESET_Z", (8, 12, 10))),
    (("CX", ((7, 8), (11, 12), (9, 10))),),
    # Phase 2 — build the D2/D2' Z-basis repetition code (shared Z parity).
    (("RESET_Z", (13,)),),
    (("CX", ((2, 13),)),),
    # Phase 3 — Z-stabilizer couplings CX(data → its neighbouring ancilla).
    (("CX", ((1, 7), (0, 8), (3, 11), (2, 12), (4, 9), (13, 10))),),
    (("CX", ((4, 7), (5, 8), (0, 11), (6, 9))),),
    (("CX", ((5, 11), (0, 10))),),
    # Phase 4 — swap the D2/D2' Z-rep code for an X-basis repetition code.
    (("CX", ((2, 13),)),),
    (("RESET_X", (13,)),),
    (("CX", ((13, 2),)),),
    # Phase 5 — X-stabilizer couplings CX(ancilla → its neighbouring data).
    (("CX", ((7, 1), (8, 0), (11, 3), (12, 2), (9, 4), (10, 13))),),
    (("CX", ((7, 4), (8, 5), (11, 0), (9, 6))),),
    (("CX", ((11, 5), (10, 0))),),
    # Phase 6 — disentangle the D2/D2' X-rep code, restoring data qubit 2.
    (("CX", ((13, 2),)),),
    # Phase 7 — unentangle each Bell pair with a final CX A→B.
    (("CX", ((7, 8), (11, 12), (9, 10))),),
    # Phase 8 — Bell-basis readout (MEASURE_X on each A, MEASURE_Z on each B).
    (
        ("MEASURE_X", (7,)),
        ("MEASURE_Z", (8,)),
        ("MEASURE_X", (11,)),
        ("MEASURE_Z", (12,)),
        ("MEASURE_X", (9,)),
        ("MEASURE_Z", (10,)),
    ),
)
