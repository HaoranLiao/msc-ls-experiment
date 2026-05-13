"""General matching-based decoder for rotated surface-code patches (Qiskit workflow).

This module provides a ``RotatedPatchDecoder`` that builds a PyMatching decoder
from stabilizer definitions. It works for any distance, though currently the
repo only defines the d=3 layout.

A convenience ``decode_rotated_patch_syndrome`` function preserves the
original API used by notebooks and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    ROTATED_D3_LOGICAL_X,
    ROTATED_D3_LOGICAL_Z,
    ROTATED_D3_SURFACE_ORDER,
    ROTATED_D3_X_STABILIZERS,
    ROTATED_D3_Z_STABILIZERS,
)
from research_qec_haoran.utils.normalize import normalize_bit, normalize_pauli_char

ROTATED_D3_Z_SYNDROME_ORDER: tuple[str, ...] = tuple(ROTATED_D3_Z_STABILIZERS)
ROTATED_D3_X_SYNDROME_ORDER: tuple[str, ...] = tuple(ROTATED_D3_X_STABILIZERS)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatedPatchRecovery:
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
# General matching decoder for a rotated patch
# ---------------------------------------------------------------------------


class RotatedPatchDecoder:
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
    ) -> RotatedPatchRecovery:
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
            return RotatedPatchRecovery(
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

        return RotatedPatchRecovery(
            qubit_label=qubit_label, pauli=pauli, z_deltas=z_dict, x_deltas=x_dict
        )


# ---------------------------------------------------------------------------
# Module-level d=3 decoder instance (lazy singleton)
# ---------------------------------------------------------------------------

_ROTATED_D3_DECODER: RotatedPatchDecoder | None = None


def _get_rotated_d3_decoder() -> RotatedPatchDecoder:
    global _ROTATED_D3_DECODER
    if _ROTATED_D3_DECODER is None:
        _ROTATED_D3_DECODER = RotatedPatchDecoder(
            data_qubits=ROTATED_D3_SURFACE_ORDER,
            z_stabilizers=ROTATED_D3_Z_STABILIZERS,
            x_stabilizers=ROTATED_D3_X_STABILIZERS,
            logical_x=ROTATED_D3_LOGICAL_X,
            logical_z=ROTATED_D3_LOGICAL_Z,
        )
    return _ROTATED_D3_DECODER


# ---------------------------------------------------------------------------
# Convenience helpers (preserve original API)
# ---------------------------------------------------------------------------


def rotated_d3_single_qubit_relative_syndrome(
    *,
    pauli: str,
    qubit_label: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return the relative stabilizer flips caused by one data-qubit Pauli."""
    normalized_pauli = normalize_pauli_char(pauli)
    if normalized_pauli in {None, "I"}:
        raise ValueError("single-qubit error syndrome expects one of 'X', 'Y', 'Z'")
    if qubit_label not in ROTATED_D3_SURFACE_ORDER:
        raise ValueError(f"unknown rotated-d3 data qubit label {qubit_label!r}")

    z_deltas = {name: 0 for name in ROTATED_D3_Z_SYNDROME_ORDER}
    x_deltas = {name: 0 for name in ROTATED_D3_X_SYNDROME_ORDER}

    if normalized_pauli in {"X", "Y"}:
        for name, support in ROTATED_D3_Z_STABILIZERS.items():
            z_deltas[name] = int(qubit_label in support)

    if normalized_pauli in {"Z", "Y"}:
        for name, support in ROTATED_D3_X_STABILIZERS.items():
            x_deltas[name] = int(qubit_label in support)

    return z_deltas, x_deltas


def decode_rotated_patch_syndrome(
    *,
    z_deltas: Mapping[str, int | bool],
    x_deltas: Mapping[str, int | bool],
) -> RotatedPatchRecovery:
    """Decode a no-error or single-qubit Pauli syndrome on rotated d=3.

    Uses minimum-weight perfect matching (PyMatching) under a uniform
    code-capacity noise model. The returned recovery is a canonical
    representative for that syndrome class.
    """
    return _get_rotated_d3_decoder().decode(z_deltas=z_deltas, x_deltas=x_deltas)
