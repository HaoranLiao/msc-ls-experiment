"""General matching-based decoder for rotated surface-code patches.

This module provides a ``RotatedDecoder`` that builds a PyMatching decoder
from stabilizer definitions. It works for any distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from research_qec_haoran.utils.normalize import normalize_bit


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatedRecovery:
    """A decoded single-qubit recovery for a rotated surface-code patch."""

    qubit_label: str | None
    pauli: str | None
    z_deltas: dict[str, int]
    x_deltas: dict[str, int]

    @property
    def is_trivial(self) -> bool:
        return self.pauli is None or self.qubit_label is None

    def label(self) -> str:
        if self.is_trivial:
            return "I"
        return f"{self.pauli}_{self.qubit_label}"


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
        n = len(self.data_qubits)

        # Build X-error decoder (detects X errors via Z stabilizers)
        z_check = np.zeros((len(self.z_stabilizer_names), n), dtype=np.uint8)
        for row, name in enumerate(self.z_stabilizer_names):
            for q in z_stabilizers[name]:
                z_check[row, self._qubit_index[q]] = 1

        logical_x_vec = np.zeros((1, n), dtype=np.uint8)
        for q in logical_x:
            logical_x_vec[0, self._qubit_index[q]] = 1

        # Uniform weights (code-capacity model)
        self._x_decoder = pymatching.Matching(z_check)
        self._x_logical = logical_x_vec[0]

        # Build Z-error decoder (detects Z errors via X stabilizers)
        x_check = np.zeros((len(self.x_stabilizer_names), n), dtype=np.uint8)
        for row, name in enumerate(self.x_stabilizer_names):
            for q in x_stabilizers[name]:
                x_check[row, self._qubit_index[q]] = 1

        logical_z_vec = np.zeros((1, n), dtype=np.uint8)
        for q in logical_z:
            logical_z_vec[0, self._qubit_index[q]] = 1

        self._z_decoder = pymatching.Matching(x_check)
        self._z_logical = logical_z_vec[0]

    def decode(
        self,
        *,
        z_deltas: Mapping[str, int | bool],
        x_deltas: Mapping[str, int | bool],
    ) -> RotatedRecovery:
        """Decode a syndrome and return a physical recovery operator.

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

        # Combine into a single Pauli label
        combined = x_correction | z_correction  # not bitwise XOR; both are 0/1 arrays
        # Identify qubit with non-trivial correction
        x_locs = set(np.flatnonzero(x_correction))
        z_locs = set(np.flatnonzero(z_correction))
        all_locs = x_locs | z_locs

        z_dict = {name: int(z_syn[i]) for i, name in enumerate(self.z_stabilizer_names)}
        x_dict = {name: int(x_syn[i]) for i, name in enumerate(self.x_stabilizer_names)}

        if not all_locs:
            return RotatedRecovery(
                qubit_label=None, pauli=None, z_deltas=z_dict, x_deltas=x_dict
            )

        # Pick one representative qubit for the label (first in patch order)
        rep_idx = min(all_locs)
        qubit_label = self.data_qubits[rep_idx]
        if rep_idx in x_locs and rep_idx in z_locs:
            pauli = "Y"
        elif rep_idx in x_locs:
            pauli = "X"
        else:
            pauli = "Z"

        return RotatedRecovery(
            qubit_label=qubit_label, pauli=pauli, z_deltas=z_dict, x_deltas=x_dict
        )



