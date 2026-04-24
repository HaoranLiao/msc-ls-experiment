"""Helpers for decoding distance-3 unrotated surface-code memory experiments.

Public API
----------
- ``generate_d3_unrotated_memory_decoder_circuit`` – Stim circuit with small noise for decoder construction
- ``build_d3_unrotated_memory_matching_decoder`` – PyMatching decoder from that circuit
- ``decode_memory_shots`` – sample, decode, and return structured ``DecodedMemoryShot`` results
- ``detector_coords_by_index`` – map detector indices to ``(x, y, t)`` coordinates

Data classes
------------
- ``DecodedMemoryShot`` – one decoded shot with detection events, corrections, and logical outcome
- ``DetectionEvent`` – a single fired detector with sector and time-layer metadata
- ``InferredCorrection`` – a matched pair from the decoder with inferred error description
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import stim

# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

_UNROTATED_D3_DISTANCE = 3

DetectorCoord = tuple[int, int, int]
QubitCoord = tuple[int, int]

DEFAULT_DECODER_ERROR_PROBABILITY = 1e-3

_MEMORY_TASK_BY_BASIS = {
    "Z": "surface_code:unrotated_memory_z",
    "X": "surface_code:unrotated_memory_x",
}

# All data-qubit coordinates in the d=3 unrotated layout (Stim's (x, y) convention).
_D3_DATA_QUBIT_COORDS: frozenset[QubitCoord] = frozenset(
    (x, y) for x in range(5) for y in range(5) if (x + y) % 2 == 0
)

# ---------------------------------------------------------------------------
# Data classes (public)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionEvent:
    """A single fired detector with its metadata."""

    detector_index: int
    coord: DetectorCoord
    sector: str  # "X" or "Z"
    time_label: str  # "preparation-boundary", "bulk", "final-boundary"

    def __str__(self) -> str:
        x, y, t = self.coord
        return f"D{self.detector_index} {self.sector}-stabilizer({x},{y}) t={t} [{self.time_label}]"


@dataclass(frozen=True)
class InferredCorrection:
    """One matched pair from the decoder: the decoder's hypothesis about what error occurred."""

    event_a: DetectionEvent | str  # DetectionEvent or "boundary"
    event_b: DetectionEvent | str
    inferred_error_type: str  # e.g. "X error on data qubit (2,2), t=1"
    flips_logical: bool  # whether this matched edge flips L0

    def __str__(self) -> str:
        a_str = str(self.event_a) if isinstance(self.event_a, DetectionEvent) else self.event_a
        b_str = str(self.event_b) if isinstance(self.event_b, DetectionEvent) else self.event_b
        flip_str = " [flips L0]" if self.flips_logical else ""
        return f"{a_str}  ↔  {b_str}  →  {self.inferred_error_type}{flip_str}"


@dataclass(frozen=True)
class DecodedMemoryShot:
    """One decoded shot from a logical-memory experiment."""

    logical_readout_basis: str
    logical_observable_qubits: tuple[QubitCoord, ...]
    detection_events: tuple[DetectionEvent, ...]
    decoder_corrections: tuple[InferredCorrection, ...]
    actual_logical_flip: bool
    decoder_inferred_logical_flip: bool
    residual_logical_error: bool

    def as_dict(self) -> dict[str, object]:
        """Return a notebook-friendly summary."""
        has_detection = len(self.detection_events) > 0

        if not self.residual_logical_error:
            if not has_detection and not self.actual_logical_flip:
                product_str = "I — no errors occurred"
            elif has_detection and not self.actual_logical_flip:
                product_str = "I — error detected and corrected (no logical impact, no frame update needed)"
            elif has_detection and self.actual_logical_flip:
                product_str = "I — error detected, logical flip tracked via Pauli frame update"
            else:
                product_str = "I — no residual error"
        else:
            product_str = f"logical {self.logical_readout_basis} FLIP — uncorrected logical error"

        return {
            "logical_observable": f"logical {self.logical_readout_basis} on qubits {list(self.logical_observable_qubits)}",
            "detection_events": (
                [str(e) for e in self.detection_events]
                if self.detection_events
                else "none — no detectors fired"
            ),
            "decoder_corrections": (
                [str(c) for c in self.decoder_corrections]
                if self.decoder_corrections
                else "none — decoder inferred no errors"
            ),
            "actual_logical_flip": self.actual_logical_flip,
            "decoder_inferred_logical_flip": self.decoder_inferred_logical_flip,
            "error x correction product": product_str,
            "decoding_succeeded": not self.residual_logical_error,
        }


# ---------------------------------------------------------------------------
# Public API: circuit generation, decoder construction, decoding
# ---------------------------------------------------------------------------


def generate_d3_unrotated_memory_decoder_circuit(
    *,
    rounds: int = 2,
    basis: str = "Z",
    error_probability: float = DEFAULT_DECODER_ERROR_PROBABILITY,
) -> stim.Circuit:
    """Return a small-noise circuit used to build a matching decoder."""
    if not 0 <= error_probability <= 1:
        raise ValueError("error_probability must be between 0 and 1")

    normalized_basis = _normalized_basis(basis)
    return stim.Circuit.generated(
        _MEMORY_TASK_BY_BASIS[normalized_basis],
        distance=_UNROTATED_D3_DISTANCE,
        rounds=rounds,
        after_clifford_depolarization=error_probability,
        before_round_data_depolarization=error_probability,
        before_measure_flip_probability=error_probability,
        after_reset_flip_probability=error_probability,
    )


def build_d3_unrotated_memory_matching_decoder(
    *,
    rounds: int = 2,
    basis: str = "Z",
    error_probability: float = DEFAULT_DECODER_ERROR_PROBABILITY,
) -> Any:
    """Build a PyMatching decoder for Stim's native d=3 unrotated memory circuit."""
    try:
        import pymatching
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pymatching is required for decoding helpers; use the repo's qec environment."
        ) from exc

    decoder_circuit = generate_d3_unrotated_memory_decoder_circuit(
        rounds=rounds,
        basis=basis,
        error_probability=error_probability,
    )
    detector_error_model = decoder_circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(detector_error_model)


def detector_coords_by_index(circuit: stim.Circuit) -> dict[int, DetectorCoord]:
    """Map each detector index in *circuit* to its ``(x, y, t)`` coordinate."""
    return {
        detector: _to_detector_coord(coords)
        for detector, coords in circuit.get_detector_coordinates().items()
    }


def decode_memory_shots(
    circuit: stim.Circuit,
    *,
    matching: Any,
    shots: int = 1,
    logical_readout_basis: str = "Z",
) -> list[DecodedMemoryShot]:
    """Sample and decode memory-experiment shots from *circuit*.

    Returns one :class:`DecodedMemoryShot` per shot, each containing:
    - which detectors fired and what stabilizer sector they belong to,
    - the decoder's matched pairs with inferred error descriptions,
    - the actual vs. decoder-inferred logical flip, and the residual outcome.
    """
    if shots < 1:
        raise ValueError("shots must be at least 1")

    normalized_basis = _normalized_basis(logical_readout_basis)

    # Sample detector outcomes and observable flips
    sampler = circuit.compile_detector_sampler()
    detector_samples, observable_flips = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    if observable_flips.shape[1] != 1:
        raise ValueError("this helper expects exactly one logical observable, namely Stim's L0")

    # Pre-compute metadata shared across all shots
    detector_coord_map = detector_coords_by_index(circuit)
    obs_qubit_coords = _observable_data_qubit_coords(circuit)
    max_t = max(c[2] for c in detector_coord_map.values()) if detector_coord_map else 0

    decoded_shots: list[DecodedMemoryShot] = []
    for detector_sample, measured_observable_flip in zip(
        detector_samples, observable_flips, strict=True,
    ):
        detection_events = _build_detection_events(detector_sample, detector_coord_map, max_t)
        decoder_inferred_flip, corrections = _decode_and_build_corrections(
            detector_sample, matching, detector_coord_map, max_t,
        )

        actual_flip = bool(measured_observable_flip[0])
        decoded_shots.append(
            DecodedMemoryShot(
                logical_readout_basis=normalized_basis,
                logical_observable_qubits=obs_qubit_coords,
                detection_events=tuple(detection_events),
                decoder_corrections=tuple(corrections),
                actual_logical_flip=actual_flip,
                decoder_inferred_logical_flip=decoder_inferred_flip,
                residual_logical_error=(actual_flip != decoder_inferred_flip),
            )
        )
    return decoded_shots


# ---------------------------------------------------------------------------
# Private helpers: decoding internals
# ---------------------------------------------------------------------------


def _build_detection_events(
    detector_sample: Any,
    detector_coord_map: dict[int, DetectorCoord],
    max_t: int,
) -> list[DetectionEvent]:
    """Convert a raw detector-sample array into a list of :class:`DetectionEvent`."""
    events: list[DetectionEvent] = []
    for det_idx, det_bit in enumerate(detector_sample):
        if bool(det_bit):
            x, y, t = detector_coord_map[det_idx]
            events.append(DetectionEvent(
                detector_index=det_idx,
                coord=(x, y, t),
                sector=_detector_sector(x, y),
                time_label=_detector_time_label(t, max_t),
            ))
    return events


def _decode_and_build_corrections(
    detector_sample: Any,
    matching: Any,
    detector_coord_map: dict[int, DetectorCoord],
    max_t: int,
) -> tuple[bool, list[InferredCorrection]]:
    """Run the decoder and return ``(inferred_logical_flip, corrections)``."""
    matched_dict = matching.decode_to_matched_dets_dict(detector_sample)
    decoder_inferred_flip = bool(matching.decode(detector_sample)[0])

    seen_pairs: set[tuple[int, int | None]] = set()
    corrections: list[InferredCorrection] = []

    for det_a, det_b in matched_dict.items():
        pair_key = (
            min(det_a, det_b) if det_b is not None else det_a,
            max(det_a, det_b) if det_b is not None else None,
        )
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        evt_a = _make_detection_event(det_a, detector_coord_map, max_t)
        evt_b: DetectionEvent | str = (
            "boundary" if det_b is None
            else _make_detection_event(det_b, detector_coord_map, max_t)
        )

        corrections.append(InferredCorrection(
            event_a=evt_a,
            event_b=evt_b,
            inferred_error_type=_infer_error_description(evt_a, evt_b),
            flips_logical=(det_b is None and decoder_inferred_flip),
        ))

    return decoder_inferred_flip, corrections


def _make_detection_event(
    det_idx: int,
    detector_coord_map: dict[int, DetectorCoord],
    max_t: int,
) -> DetectionEvent:
    """Create a :class:`DetectionEvent` for a given detector index."""
    coord = detector_coord_map[det_idx]
    return DetectionEvent(
        detector_index=det_idx,
        coord=coord,
        sector=_detector_sector(coord[0], coord[1]),
        time_label=_detector_time_label(coord[2], max_t),
    )


# ---------------------------------------------------------------------------
# Private helpers: error description inference
# ---------------------------------------------------------------------------


def _infer_error_description(
    event_a: DetectionEvent | str,
    event_b: DetectionEvent | str,
) -> str:
    """Infer a human-readable error description from a matched pair."""

    # Boundary match: one detector matched to the virtual boundary node
    if isinstance(event_a, str) or isinstance(event_b, str):
        det = event_a if isinstance(event_a, DetectionEvent) else event_b
        if isinstance(det, DetectionEvent):
            x, y, t = det.coord
            error_pauli = "X" if det.sector == "Z" else "Z" if det.sector == "X" else "?"
            data_q = _boundary_data_qubit(x, y)
            loc = f"on data qubit ({data_q[0]},{data_q[1]})" if data_q else f"near stabilizer ({x},{y})"
            return f"{error_pauli} error {loc}, t={t}"
        return "boundary match"

    # Two detectors matched together
    assert isinstance(event_a, DetectionEvent) and isinstance(event_b, DetectionEvent)
    ax, ay, at = event_a.coord
    bx, by, bt = event_b.coord

    if at == bt:
        # Same time layer → spatial (data-qubit) error
        if event_a.sector == "Z" and event_b.sector == "Z":
            error_pauli = "X"
        elif event_a.sector == "X" and event_b.sector == "X":
            error_pauli = "Z"
        else:
            error_pauli = "Y"  # cross-sector
        data_q = _shared_data_qubit(ax, ay, bx, by)
        if data_q:
            return f"{error_pauli} error on data qubit ({data_q[0]},{data_q[1]}), t={at}"
        return f"{error_pauli} error on data qubit near ({(ax+bx)/2:.0f},{(ay+by)/2:.0f}), t={at}"

    if ax == bx and ay == by:
        # Same spatial location, different time → measurement error
        return f"measurement error on stabilizer ({ax},{ay}) between t={min(at,bt)} and t={max(at,bt)}"

    # Space-time diagonal
    return f"space-time error between ({ax},{ay},t={at}) and ({bx},{by},t={bt})"


# ---------------------------------------------------------------------------
# Private helpers: layout geometry
# ---------------------------------------------------------------------------


def _adjacent_data_qubits(sx: int, sy: int) -> list[QubitCoord]:
    """Return data-qubit coordinates adjacent to stabilizer ancilla at ``(sx, sy)``."""
    return [
        (sx + dx, sy + dy)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if (sx + dx, sy + dy) in _D3_DATA_QUBIT_COORDS
    ]


def _shared_data_qubit(ax: int, ay: int, bx: int, by: int) -> QubitCoord | None:
    """Return the data qubit shared by two adjacent stabilizers, or None."""
    mid = ((ax + bx) / 2, (ay + by) / 2)
    if mid[0] == int(mid[0]) and mid[1] == int(mid[1]):
        candidate = (int(mid[0]), int(mid[1]))
        if candidate in _D3_DATA_QUBIT_COORDS:
            return candidate
    return None


def _boundary_data_qubit(sx: int, sy: int) -> QubitCoord | None:
    """Return the most likely boundary data qubit for a single-detector boundary match.

    When a stabilizer has fewer than 4 data-qubit neighbors (boundary stabilizer),
    the boundary error is on one of its neighbors that sits on the patch edge.
    We pick the neighbor with the smallest number of adjacent stabilizers (most boundary-like).
    """
    neighbors = _adjacent_data_qubits(sx, sy)
    if not neighbors:
        return None

    def _num_adjacent_stabilizers(qx: int, qy: int) -> int:
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = qx + dx, qy + dy
            if 0 <= nx <= 4 and 0 <= ny <= 4 and (nx, ny) not in _D3_DATA_QUBIT_COORDS:
                count += 1
        return count

    # Prefer neighbors at the actual patch boundary (x=0 or x=4 or y=0 or y=4)
    boundary_neighbors = [(qx, qy) for qx, qy in neighbors if qx in (0, 4) or qy in (0, 4)]
    if boundary_neighbors:
        return min(boundary_neighbors, key=lambda q: _num_adjacent_stabilizers(*q))
    return min(neighbors, key=lambda q: _num_adjacent_stabilizers(*q))


# ---------------------------------------------------------------------------
# Private helpers: detector metadata
# ---------------------------------------------------------------------------


def _detector_sector(x: int, y: int) -> str:
    """Return ``'X'`` or ``'Z'`` for the stabilizer sector at Stim coordinate ``(x, y)``.

    In Stim's unrotated surface-code layout:
    - X-stabilizer ancillas (with H gates) sit at (odd x, even y)
    - Z-stabilizer ancillas (no H gates) sit at (even x, odd y)

    An X-stabilizer detects Z errors; a Z-stabilizer detects X errors.
    """
    if x % 2 == 1 and y % 2 == 0:
        return "X"
    if x % 2 == 0 and y % 2 == 1:
        return "Z"
    return "?"


def _detector_time_label(t: int, max_t: int) -> str:
    """Return a human-readable label for the detector time layer."""
    if t == 0:
        return "preparation-boundary"
    if t == max_t:
        return "final-boundary"
    return "bulk"


# ---------------------------------------------------------------------------
# Private helpers: observable extraction, basis normalization, coord conversion
# ---------------------------------------------------------------------------


def _observable_data_qubit_coords(circuit: stim.Circuit) -> tuple[QubitCoord, ...]:
    """Return the ``(x, y)`` coordinates of data qubits that make up the logical observable."""
    qcoords = {
        q: (int(cs[0]), int(cs[1]))
        for q, cs in circuit.get_final_qubit_coordinates().items()
    }
    flat = circuit.flattened()
    m_targets: list[int] = []
    for inst in flat:
        if inst.name in ("M", "MX"):
            m_targets.extend(t.value for t in inst.targets_copy())

    total_m = len(m_targets)
    obs_qubit_coords: list[QubitCoord] = []
    for line in str(circuit).splitlines():
        if "OBSERVABLE_INCLUDE" not in line:
            continue
        for token in line.split():
            if token.startswith("rec["):
                offset = int(token[4:-1])
                qubit_index = m_targets[total_m + offset]
                obs_qubit_coords.append(qcoords[qubit_index])
    return tuple(obs_qubit_coords)


def _normalized_basis(basis: str) -> str:
    normalized = basis.upper()
    if normalized not in _MEMORY_TASK_BY_BASIS:
        raise ValueError("basis must be 'X' or 'Z'")
    return normalized


def _to_detector_coord(raw_coords: list[float]) -> DetectorCoord:
    padded = list(raw_coords) + [0.0] * (3 - len(raw_coords))
    return tuple(int(v) for v in padded[:3])
