"""Lookup-table decoder for the Steane [[7,1,3]] code on the 11-site patch.

The Steane code is a CSS self-dual [[7,1,3]] code. Because its stabilizers
have identical X and Z supports, syndrome decoding separates cleanly into two
independent 3-bit lookups:

- Z-stabilizer syndrome (3 bits) → identifies the qubit with an X error.
- X-stabilizer syndrome (3 bits) → identifies the qubit with a Z error.

This module provides the lookup tables and a decoder class that returns a
:class:`SteaneRecovery` indicating the correction and its logical effect
(i.e., whether the physical correction anticommutes with the tracked logical
operators).
"""

from __future__ import annotations

from dataclasses import dataclass

from research_qec_haoran.steane_code.state_injection_qiskit import (
    PATCH_DATA_SITES,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    PATCH_X_STABILIZERS,
    PATCH_Z_STABILIZERS,
)


# ---------------------------------------------------------------------------
# Stabilizer labels (canonical ordering)
# ---------------------------------------------------------------------------

STEANE_PATCH_X_STABILIZER_LABELS: tuple[str, ...] = tuple(
    f"X_{''.join(str(s) for s in support)}" for support in PATCH_X_STABILIZERS
)

STEANE_PATCH_Z_STABILIZER_LABELS: tuple[str, ...] = tuple(
    f"Z_{''.join(str(s) for s in support)}" for support in PATCH_Z_STABILIZERS
)

STEANE_PATCH_ALL_STABILIZER_LABELS: tuple[str, ...] = (
    STEANE_PATCH_X_STABILIZER_LABELS + STEANE_PATCH_Z_STABILIZER_LABELS
)


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

def _build_syndrome_to_qubit_table(
    stabilizers: tuple[tuple[int, ...], ...],
    data_sites: tuple[int, ...],
) -> dict[tuple[int, ...], int | None]:
    """Build a map from 3-bit syndrome → data-site index (or None for no error).

    For each data site, compute which stabilizers it appears in.  The resulting
    binary syndrome vector is the lookup key.
    """
    table: dict[tuple[int, ...], int | None] = {}
    # No-error syndrome
    table[(0, 0, 0)] = None

    for site in data_sites:
        syndrome = tuple(1 if site in stab else 0 for stab in stabilizers)
        table[syndrome] = site

    return table


# Z-stabilizer syndrome → qubit with X error
_Z_SYNDROME_TABLE: dict[tuple[int, ...], int | None] = _build_syndrome_to_qubit_table(
    PATCH_Z_STABILIZERS, PATCH_DATA_SITES
)

# X-stabilizer syndrome → qubit with Z error
_X_SYNDROME_TABLE: dict[tuple[int, ...], int | None] = _build_syndrome_to_qubit_table(
    PATCH_X_STABILIZERS, PATCH_DATA_SITES
)


# ---------------------------------------------------------------------------
# Recovery dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SteaneRecovery:
    """Decoded recovery for the Steane code on the 11-site patch.

    Attributes
    ----------
    x_correction_site : int | None
        Patch site to apply X correction (detected by Z stabilizers), or None.
    z_correction_site : int | None
        Patch site to apply Z correction (detected by X stabilizers), or None.
    flips_logical_z : bool
        Whether the X correction anticommutes with the tracked logical Z
        (i.e., X correction site is in the default logical Z support).
    flips_logical_x : bool
        Whether the Z correction anticommutes with the tracked logical X
        (i.e., Z correction site is in the default logical X support).
    """

    x_correction_site: int | None
    z_correction_site: int | None
    flips_logical_z: bool
    flips_logical_x: bool

    @property
    def is_trivial(self) -> bool:
        """True if no correction is needed."""
        return self.x_correction_site is None and self.z_correction_site is None


# ---------------------------------------------------------------------------
# Decoder class
# ---------------------------------------------------------------------------

class SteaneDecoder:
    """Lookup-table decoder for the Steane [[7,1,3]] code on the 11-site patch.

    Uses the default logical X and Z representatives to determine whether
    a physical correction induces a logical flip.

    Parameters
    ----------
    logical_x_support : tuple[int, ...]
        Patch sites forming the tracked logical X representative.
    logical_z_support : tuple[int, ...]
        Patch sites forming the tracked logical Z representative.
    """

    def __init__(
        self,
        *,
        logical_x_support: tuple[int, ...] = PATCH_DEFAULT_LOGICAL_X,
        logical_z_support: tuple[int, ...] = PATCH_DEFAULT_LOGICAL_Z,
    ) -> None:
        self._logical_x_set = set(logical_x_support)
        self._logical_z_set = set(logical_z_support)

    def decode(
        self,
        *,
        z_syndrome: tuple[int, int, int],
        x_syndrome: tuple[int, int, int],
    ) -> SteaneRecovery:
        """Decode a 6-bit syndrome into a physical recovery.

        Parameters
        ----------
        z_syndrome
            3-bit tuple from Z-stabilizer change-detected outcomes (flags X errors).
            Order matches PATCH_Z_STABILIZERS: (0345), (3489), (4_5_9_10).
        x_syndrome
            3-bit tuple from X-stabilizer change-detected outcomes (flags Z errors).
            Order matches PATCH_X_STABILIZERS: (0345), (3489), (4_5_9_10).

        Returns
        -------
        SteaneRecovery
            The decoded correction and its logical effect.
        """
        x_site = _Z_SYNDROME_TABLE.get(z_syndrome)
        z_site = _X_SYNDROME_TABLE.get(x_syndrome)

        flips_z = x_site is not None and x_site in self._logical_z_set
        flips_x = z_site is not None and z_site in self._logical_x_set

        return SteaneRecovery(
            x_correction_site=x_site,
            z_correction_site=z_site,
            flips_logical_z=flips_z,
            flips_logical_x=flips_x,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_DECODER: SteaneDecoder | None = None


def get_steane_decoder() -> SteaneDecoder:
    """Return the module-level SteaneDecoder singleton (default logical supports)."""
    global _DEFAULT_DECODER
    if _DEFAULT_DECODER is None:
        _DEFAULT_DECODER = SteaneDecoder()
    assert _DEFAULT_DECODER is not None
    return _DEFAULT_DECODER


def decode_steane_syndrome(
    *,
    z_syndrome: tuple[int, int, int],
    x_syndrome: tuple[int, int, int],
) -> SteaneRecovery:
    """Convenience function: decode a Steane-code syndrome using default logicals."""
    return get_steane_decoder().decode(z_syndrome=z_syndrome, x_syndrome=x_syndrome)


