"""General matching-based decoder for rotated surface-code patches.

This module provides a ``RotatedDecoder`` that builds a PyMatching decoder
from stabilizer definitions. It works for any distance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from msc.utils.normalize import normalize_bit

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotatedDetectedErrors:
    """Detected errors for a rotated surface-code patch.

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
        """True if no errors were detected."""
        return not self.x_error_sites and not self.z_error_sites

    def label(self) -> str:
        """Human-readable label for the detected errors."""
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


@dataclass(frozen=True)
class TemporalRotatedDetectedErrors:
    """Net data correction decoded from a repeated delta history.

    Timelike matching edges represent syndrome-measurement faults and therefore
    do not appear in ``x_error_sites`` or ``z_error_sites``. The two history
    fields preserve the change-detection layers supplied to the joint decoder.
    """

    x_error_sites: frozenset[str]
    z_error_sites: frozenset[str]
    z_delta_history: tuple[dict[str, int], ...]
    x_delta_history: tuple[dict[str, int], ...]

    @property
    def is_trivial(self) -> bool:
        """True if the joint decoder predicts no net data correction."""
        return not self.x_error_sites and not self.z_error_sites


@dataclass(frozen=True, eq=False)
class _RotatedMatchingModel:
    """Shared spatial foundation for rotated-code matching decoders.

    Both the single-layer and temporal decoders use these exact qubit/check
    orderings and parity-check matrices. They differ only in whether PyMatching
    builds one spatial graph or repeats it and adds timelike edges.
    """

    data_qubits: tuple[str, ...]
    z_stabilizer_names: tuple[str, ...]
    x_stabilizer_names: tuple[str, ...]
    z_check: np.ndarray
    x_check: np.ndarray
    logical_x: frozenset[str]
    logical_z: frozenset[str]

    @classmethod
    def from_stabilizers(
        cls,
        *,
        data_qubits: Sequence[str],
        z_stabilizers: Mapping[str, tuple[str, ...]],
        x_stabilizers: Mapping[str, tuple[str, ...]],
        logical_x: Sequence[str] = (),
        logical_z: Sequence[str] = (),
    ) -> _RotatedMatchingModel:
        ordered_data = tuple(data_qubits)
        qubit_index = {label: index for index, label in enumerate(ordered_data)}
        z_names = tuple(z_stabilizers)
        x_names = tuple(x_stabilizers)

        def check_matrix(
            stabilizers: Mapping[str, tuple[str, ...]],
            names: Sequence[str],
        ) -> np.ndarray:
            matrix = np.zeros((len(names), len(ordered_data)), dtype=np.uint8)
            for row, name in enumerate(names):
                for qubit in stabilizers[name]:
                    matrix[row, qubit_index[qubit]] = 1
            return matrix

        return cls(
            data_qubits=ordered_data,
            z_stabilizer_names=z_names,
            x_stabilizer_names=x_names,
            z_check=check_matrix(z_stabilizers, z_names),
            x_check=check_matrix(x_stabilizers, x_names),
            logical_x=frozenset(logical_x),
            logical_z=frozenset(logical_z),
        )

    def apply_recovery(
        self,
        pauli_frame,
        logical_qubit_label: str,
        detected_errors: RotatedDetectedErrors | TemporalRotatedDetectedErrors,
    ) -> None:
        """Fold decoded physical corrections into a logical Pauli frame."""
        pauli_frame.update_x(
            logical_qubit_label,
            len(detected_errors.x_error_sites & self.logical_z) % 2 == 1,
        )
        pauli_frame.update_z(
            logical_qubit_label,
            len(detected_errors.z_error_sites & self.logical_x) % 2 == 1,
        )


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

        self._model = _RotatedMatchingModel.from_stabilizers(
            data_qubits=data_qubits,
            z_stabilizers=z_stabilizers,
            x_stabilizers=x_stabilizers,
            logical_x=logical_x,
            logical_z=logical_z,
        )
        self.data_qubits = self._model.data_qubits
        self.z_stabilizer_names = self._model.z_stabilizer_names
        self.x_stabilizer_names = self._model.x_stabilizer_names

        # Build X-error decoder (detects X errors via Z stabilizers)
        self._x_decoder = pymatching.Matching(self._model.z_check)

        # Build Z-error decoder (detects Z errors via X stabilizers)
        self._z_decoder = pymatching.Matching(self._model.x_check)

    def decode(
        self,
        *,
        z_deltas: Mapping[str, int | bool],
        x_deltas: Mapping[str, int | bool],
    ) -> RotatedDetectedErrors:
        """Decode a syndrome and return detected error sites.

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

    def update_pauli_frame(
        self,
        pauli_frame,
        logical_qubit_label: str,
        *,
        z_deltas: Mapping[str, int | bool],
        x_deltas: Mapping[str, int | bool],
    ) -> RotatedDetectedErrors:
        """Decode the syndrome, update the Pauli frame, and return detected errors.

        An X correction crossing logical Z updates the logical X frame, while a
        Z correction crossing logical X updates the logical Z frame.
        """
        detected_errors = self.decode(z_deltas=z_deltas, x_deltas=x_deltas)
        self._model.apply_recovery(
            pauli_frame=pauli_frame,
            logical_qubit_label=logical_qubit_label,
            detected_errors=detected_errors,
        )
        return detected_errors

    def decode_from_raw_syndromes(
        self,
        pauli_frame,
        logical_qubit_label: str,
        *,
        x_raw: Mapping[str, int | bool],
        z_raw: Mapping[str, int | bool],
        stab_frame,
    ) -> RotatedDetectedErrors:
        """Change-detect raw syndromes, decode, and update the Pauli frame.

        ``stab_frame`` must already contain the reference value for every
        stabilizer used by this decoder. It is advanced in place to the raw
        measurements supplied for this round.

        Parameters
        ----------
        pauli_frame
            Pauli frame to update.
        logical_qubit_label
            Logical-qubit label registered in ``pauli_frame``.
        x_raw
            Raw X-stabilizer outcomes keyed by ``x_stabilizer_names``.
        z_raw
            Raw Z-stabilizer outcomes keyed by ``z_stabilizer_names``.
        stab_frame
            Stabilizer frame containing the previous raw outcomes.
        """
        x_deltas = stab_frame.advance({name: x_raw[name] for name in self.x_stabilizer_names})
        z_deltas = stab_frame.advance({name: z_raw[name] for name in self.z_stabilizer_names})
        return self.update_pauli_frame(
            pauli_frame=pauli_frame,
            logical_qubit_label=logical_qubit_label,
            z_deltas=z_deltas,
            x_deltas=x_deltas,
        )


class TemporalRotatedDecoder:
    """Joint MWPM decoder for repeated rotated-code syndrome measurements.

    A separate spacetime graph is built for each CSS sector. Spatial edges are
    data errors; timelike edges are measurement errors. Histories may have
    different lengths, which is needed when one sector has a known preparation
    boundary while the other sector uses its first measurement as a reference.
    """

    def __init__(
        self,
        *,
        data_qubits: Sequence[str],
        z_stabilizers: Mapping[str, tuple[str, ...]],
        x_stabilizers: Mapping[str, tuple[str, ...]],
        logical_x: Sequence[str] = (),
        logical_z: Sequence[str] = (),
        data_weight: float = 1.0,
        measurement_weight: float = 1.0,
    ) -> None:
        self._model = _RotatedMatchingModel.from_stabilizers(
            data_qubits=data_qubits,
            z_stabilizers=z_stabilizers,
            x_stabilizers=x_stabilizers,
            logical_x=logical_x,
            logical_z=logical_z,
        )
        self.data_qubits = self._model.data_qubits
        self.z_stabilizer_names = self._model.z_stabilizer_names
        self.x_stabilizer_names = self._model.x_stabilizer_names
        self._data_weight = data_weight
        self._measurement_weight = measurement_weight

    @staticmethod
    def _delta_history_array(
        history: Sequence[Mapping[str, int | bool]],
        names: Sequence[str],
    ) -> np.ndarray:
        if not history:
            raise ValueError("delta history must contain at least one time layer.")
        expected = set(names)
        for layer in history:
            if set(layer) != expected:
                raise ValueError(
                    "every delta-history layer must contain exactly the decoder checks."
                )
        return np.array(
            [[normalize_bit(layer[name]) for layer in history] for name in names],
            dtype=np.uint8,
        )

    def _decode_sector(
        self,
        *,
        check_matrix: np.ndarray,
        delta_history: Sequence[Mapping[str, int | bool]],
        names: Sequence[str],
    ) -> np.ndarray:
        import pymatching

        syndrome = self._delta_history_array(delta_history, names)
        matching = pymatching.Matching.from_check_matrix(
            check_matrix,
            weights=self._data_weight,
            repetitions=syndrome.shape[1],
            timelike_weights=self._measurement_weight,
            use_virtual_boundary_node=True,
        )
        correction = matching.decode(syndrome)
        return correction[0] if isinstance(correction, tuple) else correction

    def decode(
        self,
        *,
        z_delta_history: Sequence[Mapping[str, int | bool]],
        x_delta_history: Sequence[Mapping[str, int | bool]],
    ) -> TemporalRotatedDetectedErrors:
        """Jointly decode all supplied delta layers in both CSS sectors."""
        x_correction = self._decode_sector(
            check_matrix=self._model.z_check,
            delta_history=z_delta_history,
            names=self.z_stabilizer_names,
        )
        z_correction = self._decode_sector(
            check_matrix=self._model.x_check,
            delta_history=x_delta_history,
            names=self.x_stabilizer_names,
        )
        return TemporalRotatedDetectedErrors(
            x_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(x_correction)
            ),
            z_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(z_correction)
            ),
            z_delta_history=tuple(
                {name: normalize_bit(layer[name]) for name in self.z_stabilizer_names}
                for layer in z_delta_history
            ),
            x_delta_history=tuple(
                {name: normalize_bit(layer[name]) for name in self.x_stabilizer_names}
                for layer in x_delta_history
            ),
        )

    def update_pauli_frame(
        self,
        pauli_frame,
        logical_qubit_label: str,
        *,
        z_delta_history: Sequence[Mapping[str, int | bool]],
        x_delta_history: Sequence[Mapping[str, int | bool]],
    ) -> TemporalRotatedDetectedErrors:
        """Decode all delta layers, update the Pauli frame, and return the recovery."""
        detected_errors = self.decode(
            z_delta_history=z_delta_history,
            x_delta_history=x_delta_history,
        )
        self._model.apply_recovery(
            pauli_frame=pauli_frame,
            logical_qubit_label=logical_qubit_label,
            detected_errors=detected_errors,
        )
        return detected_errors
