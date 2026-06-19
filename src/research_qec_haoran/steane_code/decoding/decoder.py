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
    """

    x_correction_site: int | None
    z_correction_site: int | None

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
            The decoded correction.
        """
        x_site = _Z_SYNDROME_TABLE.get(z_syndrome)
        z_site = _X_SYNDROME_TABLE.get(x_syndrome)

        return SteaneRecovery(
            x_correction_site=x_site,
            z_correction_site=z_site,
        )

    def update_pauli_frame(
        self,
        frame,
        label: str,
        *,
        z_syndrome: tuple[int, int, int],
        x_syndrome: tuple[int, int, int],
    ) -> SteaneRecovery:
        """Decode the syndrome, update the Pauli frame, and return the recovery.

        An X correction on a site in the logical-Z support anticommutes with
        Z_L, producing a logical X flip. A Z correction on a site in the
        logical-X support anticommutes with X_L, producing a logical Z flip.

        Parameters
        ----------
        frame : PauliFrame
            The Pauli frame to update.
        label
            Logical-qubit label in the frame (e.g. ``"L0"``).
        z_syndrome
            3-bit Z-stabilizer change-detected outcomes.
        x_syndrome
            3-bit X-stabilizer change-detected outcomes.

        Returns
        -------
        SteaneRecovery
            The decoded correction.
        """
        recovery = self.decode(z_syndrome=z_syndrome, x_syndrome=x_syndrome)

        # X correction anticommutes with Z_L → logical X flip
        if recovery.x_correction_site is not None:
            frame.update_x(label, recovery.x_correction_site in self._logical_z_set)
        # Z correction anticommutes with X_L → logical Z flip
        if recovery.z_correction_site is not None:
            frame.update_z(label, recovery.z_correction_site in self._logical_x_set)

        return recovery


