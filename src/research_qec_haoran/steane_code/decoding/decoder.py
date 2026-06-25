"""Lookup-table decoder for the Steane [[7,1,3]] code on the 11-site patch.

The Steane code is a CSS self-dual [[7,1,3]] code. Because its stabilizers
have identical X and Z supports, syndrome decoding separates cleanly into two
independent 3-bit lookups:

- Z-stabilizer syndrome (3 bits) → identifies the qubit with an X error.
- X-stabilizer syndrome (3 bits) → identifies the qubit with a Z error.

This module provides the lookup tables and a decoder class that returns a
:class:`SteaneDetectedErrors` indicating the correction and its logical effect
(i.e., whether the physical correction anticommutes with the tracked logical
operators).
"""

from __future__ import annotations

from typing import NamedTuple

from research_qec_haoran.steane_code.state_injection_qiskit import (
    PATCH_DATA_SITES,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    PATCH_X_STABILIZERS,
    PATCH_Z_STABILIZERS,
)

# ---------------------------------------------------------------------------
# Canonical stabilizer labels
# ---------------------------------------------------------------------------

X_STABILIZER_LABELS: tuple[str, ...] = tuple(
    f"X_{''.join(str(s) for s in support)}" for support in PATCH_X_STABILIZERS
)
Z_STABILIZER_LABELS: tuple[str, ...] = tuple(
    f"Z_{''.join(str(s) for s in support)}" for support in PATCH_Z_STABILIZERS
)
SYNDROME_BIT_LABELS: tuple[str, ...] = ("x0", "x1", "x2", "z0", "z1", "z2")

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
    table: dict[tuple[int, ...], int | None] = {(0, 0, 0): None}
    for site in data_sites:
        syndrome = tuple(1 if site in stab else 0 for stab in stabilizers)
        table[syndrome] = site
    return table


# Z-stabilizer syndrome → qubit with X error
_Z_SYNDROME_TABLE = _build_syndrome_to_qubit_table(PATCH_Z_STABILIZERS, PATCH_DATA_SITES)

# X-stabilizer syndrome → qubit with Z error
_X_SYNDROME_TABLE = _build_syndrome_to_qubit_table(PATCH_X_STABILIZERS, PATCH_DATA_SITES)


# ---------------------------------------------------------------------------
# Recovery result
# ---------------------------------------------------------------------------


class SteaneDetectedError(NamedTuple):
    """Decoded correction for the Steane code.

    Each field is the patch data site where the decoder identified an error,
    or ``None`` if that error type was not detected.

    - ``x_error_site``: site with an X error (detected by Z stabilizers)
    - ``z_error_site``: site with a Z error (detected by X stabilizers)
    """

    x_error_site: int | None
    z_error_site: int | None


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

    # -- Core decode --

    def decode(
        self,
        *,
        z_syndrome: tuple[int, ...],
        x_syndrome: tuple[int, ...],
    ) -> SteaneDetectedError:
        """Decode a 6-bit change-detected syndrome into a physical recovery.

        Parameters
        ----------
        z_syndrome
            3-bit Z-stabilizer change-detected outcomes (flags X errors).
        x_syndrome
            3-bit X-stabilizer change-detected outcomes (flags Z errors).
        """
        return SteaneDetectedError(
            x_error_site=_Z_SYNDROME_TABLE.get(z_syndrome),
            z_error_site=_X_SYNDROME_TABLE.get(x_syndrome),
        )

    # -- Frame update --

    def apply_recovery(self, frame, label: str, recovery: SteaneDetectedError) -> None:
        """Apply a decoded recovery to a PauliFrame.

        An X correction on a site in the logical-Z support anticommutes with
        Z_L, producing a logical X flip. A Z correction on a site in the
        logical-X support anticommutes with X_L, producing a logical Z flip.

        Parameters
        ----------
        frame : PauliFrame
            The Pauli frame to update.
        label
            Logical-qubit label in the frame (e.g. ``"L0"``).
        recovery
            The decoded correction to apply.
        """
        if recovery.x_error_site is not None:
            frame.update_x(label, recovery.x_error_site in self._logical_z_set)
        if recovery.z_error_site is not None:
            frame.update_z(label, recovery.z_error_site in self._logical_x_set)

    # -- Convenience pipeline --

    def decode_from_raw_syndromes(
        self,
        frame,
        label: str,
        *,
        x_raw: tuple[int, ...],
        z_raw: tuple[int, ...],
        stab_frame,
    ) -> SteaneDetectedError:
        """Change-detect raw syndrome measurements, decode, and update the Pauli frame.

        Full pipeline for one syndrome extraction round:

        1. Advance the StabilizerFrame to get change-detected deltas.
        2. Decode the delta syndrome.
        3. Apply the recovery to the PauliFrame.

        Parameters
        ----------
        frame : PauliFrame
            The Pauli frame to update.
        label
            Logical-qubit label in the frame (e.g. ``"L0"``).
        x_raw
            3-bit raw X-stabilizer measurement outcomes.
        z_raw
            3-bit raw Z-stabilizer measurement outcomes.
        stab_frame : StabilizerFrame
            Tracks the previous syndrome reference; advanced in place.
        """
        x_meas = {lab: x_raw[i] for i, lab in enumerate(X_STABILIZER_LABELS)}
        z_meas = {lab: z_raw[i] for i, lab in enumerate(Z_STABILIZER_LABELS)}

        x_deltas = stab_frame.advance(x_meas)
        z_deltas = stab_frame.advance(z_meas)

        x_syn = tuple(x_deltas[lab] for lab in X_STABILIZER_LABELS)
        z_syn = tuple(z_deltas[lab] for lab in Z_STABILIZER_LABELS)

        recovery = self.decode(z_syndrome=z_syn, x_syndrome=x_syn)
        self.apply_recovery(frame, label, recovery)
        return recovery
