"""General matching-based decoder for rotated surface-code patches.

This module provides a ``RotatedDecoder`` that builds a PyMatching decoder
from stabilizer definitions. It works for any distance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from research_qec_haoran.utils.normalize import normalize_bit

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatedDetectedErrors:
    """Decoded recovery for a rotated surface-code patch.

    Attributes
    ----------
    x_error_sites : frozenset[str]
        Data-qubit labels where X errors were identified (by Z stabilizers).
    z_error_sites : frozenset[str]
        Data-qubit labels where Z errors were identified (by X stabilizers).
    z_deltas : dict[str, int]
        Change-detected Z-stabilizer syndrome used for decoding.
    x_deltas : dict[str, int]
        Change-detected X-stabilizer syndrome used for decoding.
    """

    x_error_sites: frozenset[str]
    z_error_sites: frozenset[str]
    z_deltas: dict[str, int]
    x_deltas: dict[str, int]

    @property
    def is_trivial(self) -> bool:
        """True if no correction is needed."""
        return not self.x_error_sites and not self.z_error_sites

    def label(self) -> str:
        """Human-readable label for the correction (convenience for d=3)."""
        if self.is_trivial:
            return "I"
        all_sites = self.x_error_sites | self.z_error_sites
        if len(all_sites) == 1:
            site = next(iter(all_sites))
            has_x = site in self.x_error_sites
            has_z = site in self.z_error_sites
            if has_x and has_z:
                return f"Y_{site}"
            elif has_x:
                return f"X_{site}"
            else:
                return f"Z_{site}"
        parts = []
        for site in sorted(self.x_error_sites):
            parts.append(f"X_{site}")
        for site in sorted(self.z_error_sites):
            parts.append(f"Z_{site}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# General matching decoder for a rotated surface-code patch
# ---------------------------------------------------------------------------


class RotatedDecoder:
    """Matching-based decoder for a rotated surface-code patch.

    Parameters
    ----------
    data_qubits : sequence of qubit labels
    z_stabilizers : mapping stabilizer_name -> tuple of qubit labels in its support
    x_stabilizers : mapping stabilizer_name -> tuple of qubit labels in its support
    logical_x : qubit labels forming a logical-X representative
    logical_z : qubit labels forming a logical-Z representative
    """

    def __init__(
        self,
        *,
        data_qubits: Sequence[str],
        z_stabilizers: Mapping[str, tuple[str, ...]],
        x_stabilizers: Mapping[str, tuple[str, ...]],
        logical_x: Sequence[str] = (),
        logical_z: Sequence[str] = (),
    ) -> None:
        import pymatching

        self.data_qubits = list(data_qubits)
        self.z_stabilizer_names = list(z_stabilizers.keys())
        self.x_stabilizer_names = list(x_stabilizers.keys())
        self._qubit_index = {q: i for i, q in enumerate(self.data_qubits)}
        self._logical_x_set = frozenset(logical_x)
        self._logical_z_set = frozenset(logical_z)
        n = len(self.data_qubits)

        # Build X-error decoder (detects X errors via Z stabilizers)
        z_check = np.zeros((len(self.z_stabilizer_names), n), dtype=np.uint8)
        for row, name in enumerate(self.z_stabilizer_names):
            for q in z_stabilizers[name]:
                z_check[row, self._qubit_index[q]] = 1
        self._x_decoder = pymatching.Matching(z_check)

        # Build Z-error decoder (detects Z errors via X stabilizers)
        x_check = np.zeros((len(self.x_stabilizer_names), n), dtype=np.uint8)
        for row, name in enumerate(self.x_stabilizer_names):
            for q in x_stabilizers[name]:
                x_check[row, self._qubit_index[q]] = 1
        self._z_decoder = pymatching.Matching(x_check)

    def decode(
        self,
        *,
        z_deltas: Mapping[str, int | bool],
        x_deltas: Mapping[str, int | bool],
    ) -> RotatedDetectedErrors:
        """Decode a syndrome and return a physical recovery.

        z_deltas: change-detected Z-stabilizer outcomes (flags X/Y errors)
        x_deltas: change-detected X-stabilizer outcomes (flags Z/Y errors)
        """
        z_syn = np.array(
            [normalize_bit(z_deltas[name]) for name in self.z_stabilizer_names],
            dtype=np.uint8,
        )
        x_syn = np.array(
            [normalize_bit(x_deltas[name]) for name in self.x_stabilizer_names],
            dtype=np.uint8,
        )

        x_correction = self._x_decoder.decode(z_syn)
        z_correction = self._z_decoder.decode(x_syn)

        x_sites = frozenset(self.data_qubits[i] for i in np.flatnonzero(x_correction))
        z_sites = frozenset(self.data_qubits[i] for i in np.flatnonzero(z_correction))

        z_dict = {name: int(z_syn[i]) for i, name in enumerate(self.z_stabilizer_names)}
        x_dict = {name: int(x_syn[i]) for i, name in enumerate(self.x_stabilizer_names)}

        return RotatedDetectedErrors(
            x_error_sites=x_sites,
            z_error_sites=z_sites,
            z_deltas=z_dict,
            x_deltas=x_dict,
        )

    # -- Frame update --

    def apply_recovery(self, frame, label: str, recovery: RotatedDetectedErrors) -> None:
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
        frame.update_x(label, len(recovery.x_error_sites & self._logical_z_set) % 2 == 1)
        frame.update_z(label, len(recovery.z_error_sites & self._logical_x_set) % 2 == 1)

    def update_pauli_frame(
        self,
        frame,
        label: str,
        *,
        z_deltas: Mapping[str, int | bool],
        x_deltas: Mapping[str, int | bool],
    ) -> RotatedDetectedErrors:
        """Decode the syndrome, update the Pauli frame, and return the recovery.

        Convenience method that calls :meth:`decode` then :meth:`apply_recovery`.
        """
        recovery = self.decode(z_deltas=z_deltas, x_deltas=x_deltas)
        self.apply_recovery(frame, label, recovery)
        return recovery
