"""Lookup decoder for the Steane [[7,1,3]] code.

The self-dual CSS code decodes Z-check and X-check syndromes independently to
physical X and Z recoveries, then folds their logical effect into a Pauli frame.
The caller supplies the patch geometry (data sites, stabilizer supports, and
logical representatives) at construction, so the decoder is patch-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame

# =============================================================================
# Decoding results data structure
# =============================================================================


@dataclass(frozen=True)
class SteaneDetectedErrors:
    """Physical X/Z recovery sites decoded from one syndrome round.

    Like the rotated-code result classes, ``x_error_sites`` and
    ``z_error_sites`` hold the at-most-one data site each CSS sector's lookup
    localizes, and stay empty when that sector is trivial.
    """

    x_error_sites: frozenset[int]
    z_error_sites: frozenset[int]


# =============================================================================
# Decoder class
# =============================================================================


class SteaneDecoder:
    """Decode one Steane syndrome round and track its logical Pauli recovery."""

    def __init__(
        self,
        *,
        data_sites: Sequence[int],
        x_stabilizers: Mapping[str, tuple[int, ...]],
        z_stabilizers: Mapping[str, tuple[int, ...]],
        logical_x: tuple[int, ...],
        logical_z: tuple[int, ...],
    ) -> None:
        """Build the single-error lookup tables and store the decode geometry.

        The stabilizer labels (mapping keys) key the supplied
        :class:`StabilizerFrame`, so they must match those the caller recorded.
        Each mapping value is the check's data-site support, from which the
        weight-one syndrome tables are precomputed.
        """
        self.x_stabilizer_labels = tuple(x_stabilizers)
        self.z_stabilizer_labels = tuple(z_stabilizers)
        self.logical_x = frozenset(logical_x)
        self.logical_z = frozenset(logical_z)
        # A Z-check syndrome localizes an X error and vice versa.
        self._z_syndrome_table = self._build_syndrome_table(data_sites, z_stabilizers.values())
        self._x_syndrome_table = self._build_syndrome_table(data_sites, x_stabilizers.values())

    @staticmethod
    def _build_syndrome_table(
        data_sites: Sequence[int],
        stabilizer_supports: Iterable[tuple[int, ...]],
    ) -> dict[tuple[int, ...], int | None]:
        """Map each weight-one syndrome to its data site (all-zero syndrome to None)."""
        supports = tuple(stabilizer_supports)
        table: dict[tuple[int, ...], int | None] = {(0,) * len(supports): None}
        for data_site in data_sites:
            table[tuple(int(data_site in support) for support in supports)] = data_site
        return table

    def decode(
        self,
        *,
        z_syndrome: tuple[int, int, int],
        x_syndrome: tuple[int, int, int],
    ) -> SteaneDetectedErrors:
        """Decode Z/X three-bit delta syndromes to physical X/Z recoveries."""
        x_error_site = self._z_syndrome_table[z_syndrome]
        z_error_site = self._x_syndrome_table[x_syndrome]
        return SteaneDetectedErrors(
            x_error_sites=frozenset() if x_error_site is None else frozenset({x_error_site}),
            z_error_sites=frozenset() if z_error_site is None else frozenset({z_error_site}),
        )

    def apply_recovery(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        detected_errors: SteaneDetectedErrors,
    ) -> None:
        """Fold physical recoveries crossing opposite logicals into the frame."""
        pauli_frame.update_x(
            logical_qubit_label,
            bool(len(detected_errors.x_error_sites & self.logical_z) % 2),
        )
        pauli_frame.update_z(
            logical_qubit_label,
            bool(len(detected_errors.z_error_sites & self.logical_x) % 2),
        )

    def decode_raw_syndromes(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        *,
        x_raw: tuple[int, int, int],
        z_raw: tuple[int, int, int],
        stab_frame: StabilizerFrame,
    ) -> SteaneDetectedErrors:
        """Advance both frames from one raw syndrome round and return its recovery."""
        x_deltas = stab_frame.advance(dict(zip(self.x_stabilizer_labels, x_raw, strict=True)))
        z_deltas = stab_frame.advance(dict(zip(self.z_stabilizer_labels, z_raw, strict=True)))
        x_delta_syndrome = tuple(x_deltas[label] for label in self.x_stabilizer_labels)
        z_delta_syndrome = tuple(z_deltas[label] for label in self.z_stabilizer_labels)
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
