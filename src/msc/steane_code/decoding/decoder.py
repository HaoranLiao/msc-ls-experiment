"""Lookup decoder for the Steane [[7,1,3]] code on the 11-site patch.

The self-dual CSS code decodes Z-check and X-check syndromes independently to
physical X and Z recoveries, then folds their logical effect into a Pauli frame.
"""

from __future__ import annotations

from typing import NamedTuple

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame
from msc.steane_code.state_injection_qiskit import (
    PATCH_DATA_SITES,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    PATCH_X_STABILIZERS,
    PATCH_Z_STABILIZERS,
)

# =============================================================================
# Canonical stabilizer labels
# =============================================================================

X_STABILIZER_LABELS: tuple[str, ...] = tuple(
    f"X_{''.join(str(s) for s in support)}" for support in PATCH_X_STABILIZERS
)
Z_STABILIZER_LABELS: tuple[str, ...] = tuple(
    f"Z_{''.join(str(s) for s in support)}" for support in PATCH_Z_STABILIZERS
)
SYNDROME_BIT_LABELS: tuple[str, ...] = ("x0", "x1", "x2", "z0", "z1", "z2")

# =============================================================================
# Lookup tables
# =============================================================================

_Z_SYNDROME_TABLE: dict[tuple[int, ...], int | None] = {(0, 0, 0): None}
_X_SYNDROME_TABLE: dict[tuple[int, ...], int | None] = {(0, 0, 0): None}
for data_site in PATCH_DATA_SITES:
    _Z_SYNDROME_TABLE[tuple(int(data_site in support) for support in PATCH_Z_STABILIZERS)] = (
        data_site
    )
    _X_SYNDROME_TABLE[tuple(int(data_site in support) for support in PATCH_X_STABILIZERS)] = (
        data_site
    )


# =============================================================================
# Recovery result
# =============================================================================


class SteaneDetectedError(NamedTuple):
    """Physical X/Z recovery sites decoded from one syndrome round."""

    x_error_site: int | None
    z_error_site: int | None


# =============================================================================
# Decoder class
# =============================================================================


class SteaneDecoder:
    """Decode one Steane syndrome round and track its logical Pauli recovery."""

    def __init__(
        self,
        *,
        logical_x_support: tuple[int, ...] = PATCH_DEFAULT_LOGICAL_X,
        logical_z_support: tuple[int, ...] = PATCH_DEFAULT_LOGICAL_Z,
    ) -> None:
        """Store the logical representatives used to interpret recoveries."""
        self._logical_x_support = frozenset(logical_x_support)
        self._logical_z_support = frozenset(logical_z_support)

    def decode(
        self,
        *,
        z_syndrome: tuple[int, int, int],
        x_syndrome: tuple[int, int, int],
    ) -> SteaneDetectedError:
        """Decode Z/X three-bit delta syndromes to physical X/Z recoveries."""
        return SteaneDetectedError(
            x_error_site=_Z_SYNDROME_TABLE[z_syndrome],
            z_error_site=_X_SYNDROME_TABLE[x_syndrome],
        )

    def apply_recovery(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        detected_errors: SteaneDetectedError,
    ) -> None:
        """Fold physical recoveries crossing opposite logicals into the frame."""
        x_error_site = detected_errors.x_error_site
        z_error_site = detected_errors.z_error_site
        if x_error_site is not None:
            pauli_frame.update_x(
                logical_qubit_label,
                x_error_site in self._logical_z_support,
            )
        if z_error_site is not None:
            pauli_frame.update_z(
                logical_qubit_label,
                z_error_site in self._logical_x_support,
            )

    def decode_raw_syndromes(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        *,
        x_raw: tuple[int, int, int],
        z_raw: tuple[int, int, int],
        stab_frame: StabilizerFrame,
    ) -> SteaneDetectedError:
        """Advance both frames from one raw syndrome round and return its recovery."""
        x_deltas = stab_frame.advance(dict(zip(X_STABILIZER_LABELS, x_raw, strict=True)))
        z_deltas = stab_frame.advance(dict(zip(Z_STABILIZER_LABELS, z_raw, strict=True)))
        x_delta_syndrome = tuple(x_deltas[label] for label in X_STABILIZER_LABELS)
        z_delta_syndrome = tuple(z_deltas[label] for label in Z_STABILIZER_LABELS)
        detected_errors = self.decode(
            z_syndrome=z_delta_syndrome,
            x_syndrome=x_delta_syndrome,
        )
        self.apply_recovery(
            pauli_frame,
            logical_qubit_label,
            detected_errors,
        )
        return detected_errors
