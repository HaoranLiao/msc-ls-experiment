"""Distance-generic matching decoders for rotated surface-code patches.

Use ``decode_raw_syndromes`` for normal scalar workflows: it advances the
supplied :class:`StabilizerFrame`, decodes the resulting changes, and updates
the supplied :class:`PauliFrame`. ``decode`` is the lower-level pure API
for protocols that already own their change-detection boundaries. The
spacetime decoder also provides a logical-parity-only batch path for LER
sampling.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pymatching

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame
from msc.utils.normalize import normalize_bit

# =============================================================================
# Decoding results data structure
# =============================================================================


class _RotatedDetectedErrorsBase:
    """Provide the common interface for spatial and spacetime recoveries."""

    x_error_sites: frozenset[str]
    z_error_sites: frozenset[str]

    @property
    def is_trivial(self) -> bool:
        """Return whether the decoder predicts no net data correction."""
        return not self.x_error_sites and not self.z_error_sites

    def label(self) -> str:
        """Return a compact label for the predicted net data correction."""
        if self.is_trivial:
            return "I"
        all_sites = self.x_error_sites | self.z_error_sites
        if len(all_sites) == 1:
            site = next(iter(all_sites))
            if site in self.x_error_sites and site in self.z_error_sites:
                return f"Y_{site}"
            return f"X_{site}" if site in self.x_error_sites else f"Z_{site}"
        return " ".join(
            [
                *(f"X_{site}" for site in sorted(self.x_error_sites)),
                *(f"Z_{site}" for site in sorted(self.z_error_sites)),
            ]
        )


@dataclass(frozen=True)
class RotatedDetectedErrors(_RotatedDetectedErrorsBase):
    """Single-shot spatial decoding result.

    The error-site fields, :attr:`is_trivial`, and :meth:`label` have exactly
    the same meaning as on :class:`SpaceTimeRotatedDetectedErrors`. The only
    difference is dimensional: this result stores one ``z_deltas`` mapping
    and one ``x_deltas`` mapping rather than repeated histories.
    """

    x_error_sites: frozenset[str]
    z_error_sites: frozenset[str]
    z_deltas: dict[str, int]
    x_deltas: dict[str, int]


@dataclass(frozen=True)
class SpaceTimeRotatedDetectedErrors(_RotatedDetectedErrorsBase):
    """Single-shot spacetime decoding result.

    The common error-site fields describe the net physical data correction,
    just as on :class:`RotatedDetectedErrors`; corrections that use only
    timelike measurement-error edges therefore leave those fields empty. The
    delta fields contain one normalized mapping per decoded round.
    """

    x_error_sites: frozenset[str]
    z_error_sites: frozenset[str]
    z_delta_history: tuple[dict[str, int], ...]
    x_delta_history: tuple[dict[str, int], ...]


# =============================================================================
# Shared rotated-code geometry and frame update
# =============================================================================


class _RotatedDecoderBase:
    """Own check matrices and logical supports shared by both matching modes."""

    def __init__(
        self,
        *,
        data_qubits: Sequence[str],
        z_stabilizers: Mapping[str, tuple[str, ...]],
        x_stabilizers: Mapping[str, tuple[str, ...]],
        logical_x: Sequence[str],
        logical_z: Sequence[str],
        data_weight: float,
    ) -> None:
        """Store shared geometry, logical supports, and spatial-edge weight."""
        self.data_qubits = tuple(data_qubits)
        self.z_stabilizer_names = tuple(z_stabilizers)
        self.x_stabilizer_names = tuple(x_stabilizers)
        self._logical_x = frozenset(logical_x)
        self._logical_z = frozenset(logical_z)
        self._data_weight = data_weight
        data_index = {label: index for index, label in enumerate(self.data_qubits)}
        self._logical_x_indices = tuple(data_index[label] for label in self._logical_x)
        self._logical_z_indices = tuple(data_index[label] for label in self._logical_z)
        check_matrices = []
        for stabilizers in (z_stabilizers, x_stabilizers):
            matrix = np.zeros((len(stabilizers), len(self.data_qubits)), dtype=np.uint8)
            for row, support in enumerate(stabilizers.values()):
                for qubit in support:
                    matrix[row, data_index[qubit]] = 1
            check_matrices.append(matrix)
        self._z_check_matrix, self._x_check_matrix = check_matrices

    @staticmethod
    def _normalize_syndrome_round(
        syndrome_round: Mapping[str, int | bool],
        names: Sequence[str],
    ) -> dict[str, int]:
        """Normalize simultaneous check outcomes from one measurement round.

        A ``syndrome_round`` is a mapping of stabilizer names to outcomes at
        one time step; it is not a Tanner-graph or matching-graph layer.
        """
        if set(syndrome_round) != set(names):
            raise ValueError("syndrome round must contain exactly the decoder checks.")
        return {name: normalize_bit(syndrome_round[name]) for name in names}

    @classmethod
    def _normalize_syndrome_history(
        cls,
        syndrome_history: Sequence[Mapping[str, int | bool]],
        names: Sequence[str],
    ) -> tuple[dict[str, int], ...]:
        """Normalize a nonempty chronological sequence of syndrome rounds."""
        if not syndrome_history:
            raise ValueError("syndrome history must contain at least one round.")
        return tuple(
            cls._normalize_syndrome_round(syndrome_round, names)
            for syndrome_round in syndrome_history
        )

    def apply_recovery(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        detected_errors: RotatedDetectedErrors | SpaceTimeRotatedDetectedErrors,
    ) -> None:
        """Fold a physical recovery's logical crossings into a Pauli frame."""
        pauli_frame.update_x(
            logical_qubit_label,
            bool(len(detected_errors.x_error_sites & self._logical_z) % 2),
        )
        pauli_frame.update_z(
            logical_qubit_label,
            bool(len(detected_errors.z_error_sites & self._logical_x) % 2),
        )


# =============================================================================
# Spatial decoder: one syndrome round
# =============================================================================


class RotatedDecoder(_RotatedDecoderBase):
    """Single-round MWPM decoder with configurable spatial-edge weight."""

    def __init__(
        self,
        *,
        data_qubits: Sequence[str],
        z_stabilizers: Mapping[str, tuple[str, ...]],
        x_stabilizers: Mapping[str, tuple[str, ...]],
        logical_x: Sequence[str] = (),
        logical_z: Sequence[str] = (),
        data_weight: float = 1.0,
    ) -> None:
        """Build the two CSS matching graphs for one syndrome round.

        ``data_weight`` is the weight assigned to every spatial edge, exactly
        as in :class:`SpaceTimeRotatedDecoder`. Only the spacetime decoder has
        ``measurement_weight`` because single-round decoding has no timelike edges.
        """
        super().__init__(
            data_qubits=data_qubits,
            z_stabilizers=z_stabilizers,
            x_stabilizers=x_stabilizers,
            logical_x=logical_x,
            logical_z=logical_z,
            data_weight=data_weight,
        )
        self._x_matching = pymatching.Matching(
            self._z_check_matrix,
            weights=self._data_weight,
        )
        self._z_matching = pymatching.Matching(
            self._x_check_matrix,
            weights=self._data_weight,
        )

    def decode(
        self,
        *,
        z_deltas: Mapping[str, int | bool],
        x_deltas: Mapping[str, int | bool],
    ) -> RotatedDetectedErrors:
        """Decode one shot's named delta round to a physical data correction.

        This is the pure, single-round analogue of
        :meth:`SpaceTimeRotatedDecoder.decode`: neither method updates a
        stabilizer or Pauli frame.
        """
        z_deltas = self._normalize_syndrome_round(
            z_deltas,
            self.z_stabilizer_names,
        )
        x_deltas = self._normalize_syndrome_round(
            x_deltas,
            self.x_stabilizer_names,
        )
        x_correction = self._x_matching.decode(
            tuple(z_deltas.values()),
            return_weight=False,
        )
        z_correction = self._z_matching.decode(
            tuple(x_deltas.values()),
            return_weight=False,
        )
        return RotatedDetectedErrors(
            x_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(x_correction)
            ),
            z_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(z_correction)
            ),
            z_deltas=z_deltas,
            x_deltas=x_deltas,
        )

    def decode_raw_syndromes(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        *,
        x_raw: Mapping[str, int | bool],
        z_raw: Mapping[str, int | bool],
        stab_frame: StabilizerFrame,
    ) -> RotatedDetectedErrors:
        """Mutate both frames from one raw round and return its recovery.

        Call structure: advance ``stab_frame`` -> :meth:`decode` -> apply
        the decoded logical correction to ``pauli_frame``.
        """
        x_raw = self._normalize_syndrome_round(x_raw, self.x_stabilizer_names)
        z_raw = self._normalize_syndrome_round(z_raw, self.z_stabilizer_names)
        detected_errors = self.decode(
            z_deltas=stab_frame.advance(z_raw),
            x_deltas=stab_frame.advance(x_raw),
        )
        self.apply_recovery(pauli_frame, logical_qubit_label, detected_errors)
        return detected_errors


# =============================================================================
# Spacetime decoder: repeated syndrome rounds
# =============================================================================


class SpaceTimeRotatedDecoder(_RotatedDecoderBase):
    """Repeated-round MWPM decoder for full or partial delta histories.

    Spatial edges represent data errors; timelike edges represent readout
    errors. The X and Z delta histories may have different lengths, and a round
    may omit a check that has no valid comparison in that round; the omitted
    check then has no detector node in that round of the graph.
    ``decode_raw_syndromes`` takes raw rounds that contain every check; MSC
    surgery delta histories, computed from Qiskit hardware counts, go to
    ``decode``.
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
        z_stabilizers_measured_initial_reference: Sequence[str] = (),
        x_stabilizers_measured_initial_reference: Sequence[str] = (),
    ) -> None:
        """Store edge weights and the stabilizers whose initial reference is measured.

        ``data_weight`` weights spatial edges. ``measurement_weight`` weights
        readout-error edges: timelike edges and time-boundary edges.

        Each detector is the XOR of two consecutive outcomes of one stabilizer.
        A stabilizer's first detector is the XOR of the stabilizer's first
        decoded outcome and the stabilizer's initial reference. If the initial
        reference is known from preparation, a readout error on the first
        decoded outcome flips the first two detectors, and the timelike edge
        between those two detectors represents that readout error. If the
        initial reference is a measured outcome, a readout error on that
        measurement flips only the first detector. A time-boundary edge from the
        first detector represents that readout error; without the time-boundary
        edge, the decoder would explain the lone detector as a data error.

        In MSC lattice surgery, every Z stabilizer's initial reference is the Z
        stabilizer's initialization outcome, so the Z delta history starts at
        merge round 1. X_cd's initial reference is X_cd's first post-surgery
        outcome, because X_cd is not measured during the merge.
        """
        super().__init__(
            data_qubits=data_qubits,
            z_stabilizers=z_stabilizers,
            x_stabilizers=x_stabilizers,
            logical_x=logical_x,
            logical_z=logical_z,
            data_weight=data_weight,
        )
        self._measurement_weight = measurement_weight
        self._stabilizers_measured_initial_reference = {
            "z": frozenset(z_stabilizers_measured_initial_reference),
            "x": frozenset(x_stabilizers_measured_initial_reference),
        }
        if self._stabilizers_measured_initial_reference["z"] - set(self.z_stabilizer_names) or (
            self._stabilizers_measured_initial_reference["x"] - set(self.x_stabilizer_names)
        ):
            raise ValueError("*_stabilizers_measured_initial_reference must list decoder checks.")

    def _decode_check_basis(
        self,
        check_basis: str,
        delta_history: Sequence[Mapping[str, int | bool]],
    ) -> tuple[tuple[dict[str, int], ...], np.ndarray]:
        """Return one basis's normalized delta history and decoded data correction.

        Z-check detectors locate X data errors; X-check detectors locate Z data
        errors. Each round's detectors are the checks in that round's mapping.
        """
        if check_basis == "z":
            check_names = self.z_stabilizer_names
        elif check_basis == "x":
            check_names = self.x_stabilizer_names
        else:
            raise ValueError(f"unsupported check basis: {check_basis!r}.")
        if not delta_history:
            raise ValueError("delta history must contain at least one round.")
        if any(set(delta_round) - set(check_names) for delta_round in delta_history):
            raise ValueError("delta rounds must contain only decoder checks.")
        # Order each round's checks as in the decoder, matching the graph's node order.
        delta_history = tuple(
            self._normalize_syndrome_round(
                delta_round, [name for name in check_names if name in delta_round]
            )
            for delta_round in delta_history
        )
        detector_bits = [bit for delta_round in delta_history for bit in delta_round.values()]
        return delta_history, self._matching_graph(check_basis, delta_history).decode(detector_bits)

    def _matching_graph(
        self,
        check_basis: str,
        detector_checks_by_round: Sequence[Collection[str]],
    ) -> pymatching.Matching:
        """Build one basis's matching graph from each round's detector checks.

        Detector nodes are numbered round by round, in each round's check
        order, so the node numbers match the positions of the flattened
        detector bits.
        """
        if check_basis == "z":
            check_names, check_matrix = self.z_stabilizer_names, self._z_check_matrix
        elif check_basis == "x":
            check_names, check_matrix = self.x_stabilizer_names, self._x_check_matrix
        else:
            raise ValueError(f"unsupported check basis: {check_basis!r}.")
        stabilizers_measured_initial_reference = self._stabilizers_measured_initial_reference[
            check_basis
        ]
        check_index_by_name = {name: index for index, name in enumerate(check_names)}
        matching_graph = pymatching.Matching()
        detector_node_by_round_check = {}
        checks_with_earlier_detectors = set()
        for round_index, checks in enumerate(detector_checks_by_round):
            for check in checks:
                detector_node = len(detector_node_by_round_check)
                detector_node_by_round_check[round_index, check] = detector_node
                if (
                    check in stabilizers_measured_initial_reference
                    and check not in checks_with_earlier_detectors
                ):
                    # A readout error on this check's measured initial reference
                    # flips only this first detector. Adding this time-boundary
                    # edge, weighted as a readout error, puts that scenario in
                    # the decoding graph.
                    matching_graph.add_boundary_edge(detector_node, weight=self._measurement_weight)
                checks_with_earlier_detectors.add(check)
            # Spatial edges. The same edge can be added more than once; keep the
            # lowest-weight copy.
            for data_index in range(len(self.data_qubits)):
                detector_nodes = [
                    detector_node_by_round_check[round_index, check]
                    for check in checks
                    if check_matrix[check_index_by_name[check], data_index]
                ]
                if len(detector_nodes) == 2:
                    matching_graph.add_edge(
                        *detector_nodes,
                        fault_ids=data_index,
                        weight=self._data_weight,
                        merge_strategy="smallest-weight",
                    )
                elif len(detector_nodes) == 1:
                    # Only one check detects this data qubit in this round (patch
                    # boundary or omitted check), so the edge ends on the boundary.
                    matching_graph.add_boundary_edge(
                        detector_nodes[0],
                        fault_ids=data_index,
                        weight=self._data_weight,
                        merge_strategy="smallest-weight",
                    )
                elif len(detector_nodes) > 2:
                    raise ValueError("a data error must touch at most two checks.")
            # Timelike edges: a readout error in the previous round flips this
            # check's detectors in both rounds. No edge if the previous round
            # omitted this check.
            for check in checks:
                if (round_index - 1, check) in detector_node_by_round_check:
                    matching_graph.add_edge(
                        detector_node_by_round_check[round_index - 1, check],
                        detector_node_by_round_check[round_index, check],
                        weight=self._measurement_weight,
                    )
        matching_graph.ensure_num_fault_ids(len(self.data_qubits))
        return matching_graph

    def decode(
        self,
        *,
        z_delta_history: Sequence[Mapping[str, int | bool]],
        x_delta_history: Sequence[Mapping[str, int | bool]],
    ) -> SpaceTimeRotatedDetectedErrors:
        """Decode one shot's named delta histories to the shot's net data correction.

        Each basis has one chronological ``{check_name: delta_bit}`` mapping per
        round, holding exactly that round's detectors. Omit a check that has no
        valid comparison instead of passing 0, which counts as a detector that
        did not fire.
        """
        z_delta_history, x_data_correction = self._decode_check_basis("z", z_delta_history)
        x_delta_history, z_data_correction = self._decode_check_basis("x", x_delta_history)
        return SpaceTimeRotatedDetectedErrors(
            x_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(x_data_correction)
            ),
            z_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(z_data_correction)
            ),
            z_delta_history=z_delta_history,
            x_delta_history=x_delta_history,
        )

    def decode_raw_syndromes(
        self,
        pauli_frame: PauliFrame,
        logical_qubit_label: str,
        *,
        x_raw_history: Sequence[Mapping[str, int | bool]],
        z_raw_history: Sequence[Mapping[str, int | bool]],
        stab_frame: StabilizerFrame,
    ) -> SpaceTimeRotatedDetectedErrors:
        """Mutate both frames across raw histories and return the net recovery.

        Call structure: advance ``stab_frame`` through every raw round ->
        :meth:`decode` both delta histories -> apply the logical correction to
        ``pauli_frame``. Raw rounds must contain every check; delta histories
        that omit checks go to :meth:`decode`.
        """
        x_raw_history = self._normalize_syndrome_history(x_raw_history, self.x_stabilizer_names)
        z_raw_history = self._normalize_syndrome_history(z_raw_history, self.z_stabilizer_names)
        x_delta_history = tuple(stab_frame.advance(raw_round) for raw_round in x_raw_history)
        z_delta_history = tuple(stab_frame.advance(raw_round) for raw_round in z_raw_history)
        detected_errors = self.decode(
            z_delta_history=z_delta_history,
            x_delta_history=x_delta_history,
        )
        self.apply_recovery(pauli_frame, logical_qubit_label, detected_errors)
        return detected_errors

    def decode_logical_corrections_batch(
        self,
        *,
        z_delta_histories: np.ndarray,
        x_delta_histories: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Speed up logical-error-rate sampling by batch-decoding logical corrections.

        Unlike :meth:`decode`, this method takes ``(shots, rounds, checks)``
        arrays, so every round must contain every check, and returns only the
        logical X and Z correction bits of each shot.
        """
        normalized_delta_history_batches = []
        for check_basis, delta_history_batch, check_count in (
            ("z", z_delta_histories, len(self.z_stabilizer_names)),
            ("x", x_delta_histories, len(self.x_stabilizer_names)),
        ):
            delta_history_batch = np.asarray(delta_history_batch)
            if (
                delta_history_batch.ndim != 3
                or not delta_history_batch.shape[0]
                or not delta_history_batch.shape[1]
                or delta_history_batch.shape[2] != check_count
            ):
                raise ValueError(
                    f"{check_basis}_delta_histories must have nonempty "
                    f"(shots, rounds, {check_count}) shape."
                )
            if not np.all((delta_history_batch == 0) | (delta_history_batch == 1)):
                raise ValueError(f"{check_basis}_delta_histories must contain only 0 or 1.")
            normalized_delta_history_batches.append(
                delta_history_batch.astype(np.uint8, copy=False)
            )
        z_delta_histories, x_delta_histories = normalized_delta_history_batches
        if z_delta_histories.shape[0] != x_delta_histories.shape[0]:
            raise ValueError("Z and X delta-history batches must have the same shot count.")

        x_data_corrections, z_data_corrections = (
            self._matching_graph(
                check_basis, (check_names,) * delta_history_batch.shape[1]
            ).decode_batch(
                delta_history_batch.reshape(len(delta_history_batch), -1),
                return_weights=False,
            )
            for check_basis, delta_history_batch, check_names in (
                ("z", z_delta_histories, self.z_stabilizer_names),
                ("x", x_delta_histories, self.x_stabilizer_names),
            )
        )
        return (
            np.logical_xor.reduce(x_data_corrections[:, self._logical_z_indices], axis=1),
            np.logical_xor.reduce(z_data_corrections[:, self._logical_x_indices], axis=1),
        )
