"""Distance-generic matching decoders for rotated surface-code patches.

Use ``decode_raw_syndromes`` for normal scalar workflows: it advances the
supplied :class:`StabilizerFrame`, decodes the resulting changes, and updates
the supplied :class:`PauliFrame`. ``decode`` is the lower-level pure API
for protocols that already own their change-detection boundaries. The
spacetime decoder also reports complementary logical-sector gaps and provides
a logical-parity-only batch path for LER sampling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

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
    ``x_complementary_gap`` and ``z_complementary_gap`` compare the best MWPM
    recovery with the best recovery in the opposite logical sector.
    """

    x_error_sites: frozenset[str]
    z_error_sites: frozenset[str]
    z_delta_history: tuple[dict[str, int], ...]
    x_delta_history: tuple[dict[str, int], ...]
    x_complementary_gap: float
    z_complementary_gap: float

    @property
    def complementary_gap(self) -> float:
        """Return the least confidence gap across the two CSS sectors."""
        return min(self.x_complementary_gap, self.z_complementary_gap)


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
        self._logical_x_indices = frozenset(data_index[label] for label in self._logical_x)
        self._logical_z_indices = frozenset(data_index[label] for label in self._logical_z)
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
    """Repeated-round MWPM decoder; prefer ``decode_raw_syndromes``.

    Spatial edges represent data errors and timelike edges represent syndrome-
    measurement errors. X and Z histories may have different lengths when
    their preparation boundaries differ.
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
        """Store graph weights used to build each requested history length.

        ``data_weight`` has the same spatial-edge meaning as on
        :class:`RotatedDecoder`; ``measurement_weight`` applies only to the
        additional timelike edges in repeated-round matching graphs.
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

    def _build_spacetime_matching_graph(
        self,
        check_matrix: np.ndarray,
        repetitions: int,
    ) -> pymatching.Matching:
        """Build one CSS sector's matching graph for ``repetitions`` rounds."""
        return pymatching.Matching.from_check_matrix(
            check_matrix,
            weights=self._data_weight,
            repetitions=repetitions,
            timelike_weights=self._measurement_weight,
            use_virtual_boundary_node=True,
        )

    @staticmethod
    def _decode_complementary_sectors(
        matching: pymatching.Matching,
        syndrome: Sequence[int | bool] | np.ndarray,
        logical_support: frozenset[int],
    ) -> tuple[np.ndarray, float]:
        """Return the ordinary recovery and opposite-logical-sector weight gap.

        A syndrome fixes the endpoints of a recovery chain but not its logical
        homology class. Two chains with the same endpoints may differ by a
        logical operator. PyMatching's ordinary decode minimizes over both
        classes, so obtaining the other class requires an explicit constraint;
        this is not a request for the second-best unconstrained matching.

        For each edge ``e``, define ``p(e)`` as the parity of its ``fault_ids``
        on ``logical_support``. A recovery is in one logical sector or the other
        according to the XOR of ``p(e)`` over its selected edges. We represent
        this edge parity as a graph cut by coloring every detector node ``v``
        with a bit ``c(v)`` satisfying

        ``c(u) XOR c(v) = p(e)``

        for every internal edge ``e = (u, v)``. A boundary edge incident to
        ``v`` then belongs to boundary class ``p(e) XOR c(v)``. Keeping class 0
        as the ordinary matching boundary and replacing class 1 with one extra
        detector exposes the recovery's logical-sector parity as that added
        detector's syndrome bit.

        The physical syndrome determines one consistent value for the added
        bit. Decoding once with that value and once with its complement forces
        the minimum-weight recovery in each logical sector while keeping every
        physical detector unchanged.

        The ordinary MWPM correction is returned to preserve PyMatching's
        normal tie-breaking. The gap is the absolute difference between the
        minimum weights in the two constrained sectors; an absent logical
        support has no complementary sector and therefore has infinite gap.
        """
        # First retain PyMatching's ordinary recovery. The augmented graph below
        # is used only to compare sector weights, not to replace this correction.
        syndrome_array = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        selected_correction = cast(
            np.ndarray,
            matching.decode(syndrome_array, return_weight=False),
        )
        if not logical_support:
            return selected_correction, math.inf

        boundary_nodes = matching.boundary
        matching_edges = tuple(matching.edges())

        # Step 1: turn each internal edge's logical crossing parity p(e) into a
        # parity-labeled adjacency graph. Boundary edges are handled only after
        # detector colors have been determined.
        logical_parity_adjacency: dict[int, list[tuple[int, int]]] = {
            node: [] for node in range(matching.num_nodes) if node not in boundary_nodes
        }
        for node_a, node_b, edge_data in matching_edges:
            if node_b is None or node_a in boundary_nodes or node_b in boundary_nodes:
                continue
            logical_parity = len(edge_data["fault_ids"] & logical_support) % 2
            logical_parity_adjacency[node_a].append((node_b, logical_parity))
            logical_parity_adjacency[node_b].append((node_a, logical_parity))

        # Step 2: solve c(u) XOR c(v) = p(e) by graph traversal. Each disconnected
        # component may choose color 0 at its first node; this fixes only that
        # component's sector-labeling convention, not the pair of sectors compared.
        logical_side_by_detector: dict[int, int] = {}
        for component_start in logical_parity_adjacency:
            if component_start in logical_side_by_detector:
                continue
            logical_side_by_detector[component_start] = 0
            pending_detectors = [component_start]
            while pending_detectors:
                node_a = pending_detectors.pop()
                for node_b, logical_parity in logical_parity_adjacency[node_a]:
                    expected_neighbor_side = logical_side_by_detector[node_a] ^ logical_parity
                    if node_b not in logical_side_by_detector:
                        logical_side_by_detector[node_b] = expected_neighbor_side
                        pending_detectors.append(node_b)
                    elif logical_side_by_detector[node_b] != expected_neighbor_side:
                        # Conflicting colors mean these edge parities are not a
                        # valid cut and cannot define two consistent sectors.
                        raise ValueError(
                            "logical support cannot define complementary matching sectors."
                        )

        # Step 3: copy the graph while splitting its matching boundary into two
        # classes. Class-0 edges remain boundary edges. Class-1 edges terminate
        # at one synthetic detector, converting their use into a syndrome parity
        # that PyMatching can constrain.
        sector_detector = matching.num_detectors
        constrained_matching = pymatching.Matching()
        has_complementary_boundary = False
        for node_a, node_b, edge_data in matching_edges:
            fault_ids = edge_data["fault_ids"]
            weight = edge_data["weight"]
            if node_b is not None and node_a not in boundary_nodes and node_b not in boundary_nodes:
                constrained_matching.add_edge(
                    node_a,
                    node_b,
                    fault_ids=fault_ids,
                    weight=weight,
                )
                continue

            if node_b is None:
                detector_node = node_a
            elif node_a in boundary_nodes:
                detector_node = node_b
            else:
                detector_node = node_a
            if detector_node in boundary_nodes:
                raise ValueError("matching edge has no detector endpoint.")

            # c(v) accounts for logical parity accumulated through internal
            # edges before this path reaches its boundary edge e.
            logical_parity = len(fault_ids & logical_support) % 2
            boundary_side = logical_parity ^ logical_side_by_detector[detector_node]
            if boundary_side:
                constrained_matching.add_edge(
                    detector_node,
                    sector_detector,
                    fault_ids=fault_ids,
                    weight=weight,
                )
                has_complementary_boundary = True
            else:
                constrained_matching.add_boundary_edge(
                    detector_node,
                    fault_ids=fault_ids,
                    weight=weight,
                )
        if not has_complementary_boundary:
            raise ValueError("logical support does not connect opposite matching boundaries.")

        # Step 4: XOR the colors of the physical defects (the 1s in the syndrome).
        # This is the synthetic detector value consistent with one sector. Only
        # toggling this added bit leaves the physical syndrome fixed but forces
        # the opposite boundary parity, and therefore the opposite logical class.
        sector_bit = (
            sum(
                logical_side_by_detector[node] * int(bit) for node, bit in enumerate(syndrome_array)
            )
            % 2
        )
        sector_weights: list[float] = []
        for constrained_sector_bit in (sector_bit, sector_bit ^ 1):
            _, weight = constrained_matching.decode(
                np.append(syndrome_array, constrained_sector_bit),
                return_weight=True,
            )
            sector_weights.append(float(weight))

        # A small gap means two logical explanations have similar likelihood;
        # a large gap means the selected logical class is comparatively clear.
        complementary_gap = abs(sector_weights[0] - sector_weights[1])
        return selected_correction, complementary_gap

    def decode(
        self,
        *,
        z_delta_history: Sequence[Mapping[str, int | bool]],
        x_delta_history: Sequence[Mapping[str, int | bool]],
    ) -> SpaceTimeRotatedDetectedErrors:
        """Decode one shot's named delta histories to its net data correction.

        Each sector is a sequence of ``{stabilizer_name: bit}`` mappings and
        may contain a different number of measurement rounds.
        """
        z_delta_history = self._normalize_syndrome_history(z_delta_history, self.z_stabilizer_names)
        x_delta_history = self._normalize_syndrome_history(x_delta_history, self.x_stabilizer_names)
        z_delta_history_array = np.asarray(
            [
                [delta_round[name] for delta_round in z_delta_history]
                for name in self.z_stabilizer_names
            ],
            dtype=np.uint8,
        )
        x_delta_history_array = np.asarray(
            [
                [delta_round[name] for delta_round in x_delta_history]
                for name in self.x_stabilizer_names
            ],
            dtype=np.uint8,
        )
        x_correction, x_complementary_gap = self._decode_complementary_sectors(
            self._build_spacetime_matching_graph(
                self._z_check_matrix,
                len(z_delta_history),
            ),
            z_delta_history_array.T.reshape(-1),
            self._logical_z_indices,
        )
        z_correction, z_complementary_gap = self._decode_complementary_sectors(
            self._build_spacetime_matching_graph(
                self._x_check_matrix,
                len(x_delta_history),
            ),
            x_delta_history_array.T.reshape(-1),
            self._logical_x_indices,
        )
        return SpaceTimeRotatedDetectedErrors(
            x_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(x_correction)
            ),
            z_error_sites=frozenset(
                self.data_qubits[index] for index in np.flatnonzero(z_correction)
            ),
            z_delta_history=z_delta_history,
            x_delta_history=x_delta_history,
            x_complementary_gap=x_complementary_gap,
            z_complementary_gap=z_complementary_gap,
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
        :meth:`decode` jointly in spacetime -> apply the decoded logical
        correction to ``pauli_frame``.
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
        """Speed up logical-error-rate sampling by batch-decoding correction bits.

        This is not a vectorized :meth:`decode`: it accepts delta-history
        arrays with axes ``(shots, rounds, checks)`` but returns only the
        logical parity with which each physical recovery crosses the opposite
        logical representative.
        """
        normalized_delta_history_batches = []
        for sector, delta_history_batch, check_count in (
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
                    f"{sector}_delta_histories must have nonempty "
                    f"(shots, rounds, {check_count}) shape."
                )
            if not np.all((delta_history_batch == 0) | (delta_history_batch == 1)):
                raise ValueError(f"{sector}_delta_histories must contain only 0 or 1.")
            normalized_delta_history_batches.append(
                delta_history_batch.astype(np.uint8, copy=False)
            )
        z_delta_histories, x_delta_histories = normalized_delta_history_batches
        if z_delta_histories.shape[0] != x_delta_histories.shape[0]:
            raise ValueError("Z and X delta-history batches must have the same shot count.")

        x_corrections = self._build_spacetime_matching_graph(
            self._z_check_matrix,
            z_delta_histories.shape[1],
        ).decode_batch(
            z_delta_histories.reshape(len(z_delta_histories), -1),
            return_weights=False,
        )
        z_corrections = self._build_spacetime_matching_graph(
            self._x_check_matrix,
            x_delta_histories.shape[1],
        ).decode_batch(
            x_delta_histories.reshape(len(x_delta_histories), -1),
            return_weights=False,
        )
        data_index = {label: index for index, label in enumerate(self.data_qubits)}
        logical_x = tuple(data_index[label] for label in self._logical_x)
        logical_z = tuple(data_index[label] for label in self._logical_z)
        return (
            np.logical_xor.reduce(x_corrections[:, logical_z], axis=1),
            np.logical_xor.reduce(z_corrections[:, logical_x], axis=1),
        )
