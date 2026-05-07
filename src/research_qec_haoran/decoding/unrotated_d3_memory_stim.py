"""Helpers for decoding distance-3 unrotated surface-code memory experiments.

This module wraps PyMatching to produce human-readable ``DecodedMemoryShot``
objects for pedagogical notebooks.

Architecture
------------
The actual decoding is two PyMatching calls per shot:

    inferred_flip = matching_decoder.decode(detector_sample)[0]
    matched_pairs = matching_decoder.decode_to_matched_dets_dict(detector_sample)

Everything else in this module is **annotation and presentation**, bundled
in the ``D3ErrorInterpreter`` class.  That class translates PyMatching's raw
output (detector indices and matched pairs) into human-readable descriptions
of *what type of error* happened and *where*.

PyMatching itself never returns error types or locations — it only returns
which detectors to match and whether to flip the logical observable.  The
interpreter uses the d=3 unrotated layout geometry to work backwards from
matched detector pairs to inferred Pauli errors on specific data qubits.

For high-throughput logical error rate estimation (no per-shot annotation),
use :func:`research_qec_haoran.decoding.memory_circuit.estimate_logical_error_rate`
instead — it calls ``decoder.decode_batch(...)`` directly.

Public API
----------
- ``decode_memory_shots`` – sample, decode, and return structured results
- ``detector_coords_by_index`` – map detector indices to ``(x, y, t)`` coordinates
- ``D3ErrorInterpreter`` – translates decoder output into human-readable annotations

Output containers (presentation)
---------------------------------
- ``DecodedMemoryShot`` – one decoded shot with annotations and logical outcome
- ``AnnotatedDetector`` – a fired detector labeled with sector and time layer
- ``AnnotatedCorrection`` – a matched pair labeled with inferred error description
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import stim

from research_qec_haoran.decoding.memory_circuit_stim import normalized_basis

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DetectorCoord = tuple[int, int, int]  # (x, y, t) in Stim's coordinate system
QubitCoord = tuple[int, int]          # (x, y) in Stim's coordinate system

# All data-qubit coordinates in the d=3 unrotated layout.
# In Stim's convention, data qubits sit at (x, y) where x + y is even,
# with 0 <= x, y <= 4.  There are 13 such positions.
_D3_DATA_QUBIT_COORDS: frozenset[QubitCoord] = frozenset(
    (x, y) for x in range(5) for y in range(5) if (x + y) % 2 == 0
)


# ===========================================================================
# Public API
# ===========================================================================


def detector_coords_by_index(circuit: stim.Circuit) -> dict[int, DetectorCoord]:
    """Map each detector index in *circuit* to its ``(x, y, t)`` coordinate.

    Stim assigns up to 3 coordinates per detector via ``DETECTOR(...) rec[...]``
    annotations.  For surface-code memory circuits, these are ``(x, y, t)`` where
    ``x, y`` identify the stabilizer ancilla and ``t`` is the time layer.
    """
    return {
        det: _to_detector_coord(coords)
        for det, coords in circuit.get_detector_coordinates().items()
    }


def decode_memory_shots(
    circuit: stim.Circuit,
    *,
    matching_decoder: Any,
    shots: int = 1,
    logical_readout_basis: str = "Z",
) -> list[DecodedMemoryShot]:
    """Sample and decode memory-experiment shots from *circuit*.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit to sample from.  Can be noiseless (for deterministic
        fault-injection) or noisy.
    matching_decoder : pymatching.Matching
        A PyMatching decoder built from a noisy version of the same circuit
        layout.  See :func:`build_matching_decoder`.
    shots : int
        Number of shots to sample and decode.
    logical_readout_basis : str
        ``"X"`` or ``"Z"`` — used only for labeling in the output.

    Returns
    -------
    list[DecodedMemoryShot]
        One structured result per shot.
    """
    if shots < 1:
        raise ValueError("shots must be at least 1")

    basis = normalized_basis(logical_readout_basis)

    # Sample detector outcomes and the logical observable flip for each shot.
    sampler = circuit.compile_detector_sampler()
    detector_samples, observable_flips = sampler.sample(
        shots=shots, separate_observables=True,
    )
    if observable_flips.shape[1] != 1:
        raise ValueError("this helper expects exactly one logical observable, namely Stim's L0")

    # Build the interpreter once — it pre-computes layout metadata from the circuit.
    interpreter = D3ErrorInterpreter(circuit)

    decoded_shots: list[DecodedMemoryShot] = []
    for det_sample, obs_flip in zip(detector_samples, observable_flips, strict=True):
        # ── Decoding (PyMatching) ─────────────────────────────────────
        # These two calls are the actual decoder.  Everything below is
        # annotation for human-readable output.
        inferred_flip = bool(matching_decoder.decode(det_sample)[0])
        matched_pairs = matching_decoder.decode_to_matched_dets_dict(det_sample)
        # matched_pairs is a dict of detector-index pairs from PyMatching:
        #   no error:          {}
        #   bulk error:        {9: 14, 14: 9}        (two detectors matched to each other, bidirectional)
        #   boundary error:    {18: None}             (one detector matched to the virtual boundary node)
        #   mixed:             {3: 8, 8: 3, 12: None} (one bulk pair + one boundary match)

        # ── Annotation (presentation layer) ───────────────────────────
        # The interpreter translates the decoder's raw output into
        # human-readable error descriptions.  PyMatching itself never
        # returns error types or locations — only detector-index pairs.
        detection_events = interpreter.annotate_fired_detectors(det_sample)
        corrections = interpreter.annotate_matched_pairs(matched_pairs, inferred_flip)

        # ── Assemble result ───────────────────────────────────────────
        actual_flip = bool(obs_flip[0])  # ground truth from Stim
        decoded_shots.append(DecodedMemoryShot(
            logical_readout_basis=basis,
            logical_observable_qubits=interpreter.observable_qubit_coords,
            detection_events=tuple(detection_events),
            decoder_corrections=tuple(corrections),
            actual_logical_flip=actual_flip,
            decoder_inferred_logical_flip=inferred_flip,
            residual_logical_error=(actual_flip != inferred_flip),
        ))
    return decoded_shots


# ===========================================================================
# Output containers (presentation only)
# ===========================================================================
#
# These dataclasses hold annotated decoding results for notebook display.
# They are not part of the decoding algorithm.
# ===========================================================================


@dataclass(frozen=True)
class AnnotatedDetector:
    """A fired detector labeled with its stabilizer sector and time layer.

    This is presentation metadata — the decoder itself only sees detector
    indices and a matching graph.

    Attributes
    ----------
    detector_index : int
        The Stim detector index (0-based).
    coord : DetectorCoord
        The ``(x, y, t)`` coordinate assigned by Stim, where ``t`` is the
        time layer (0 = preparation-boundary, max = final-boundary).
    sector : str
        ``"X"`` or ``"Z"`` — the stabilizer type this detector belongs to.
        An X-stabilizer detects Z errors; a Z-stabilizer detects X errors.
    time_label : str
        Human-readable label: ``"preparation-boundary"``, ``"bulk"``, or
        ``"final-boundary"``.
    """

    detector_index: int
    coord: DetectorCoord
    sector: str
    time_label: str

    def __str__(self) -> str:
        x, y, t = self.coord
        return f"D{self.detector_index} {self.sector}-stabilizer({x},{y}) t={t} [{self.time_label}]"


@dataclass(frozen=True)
class AnnotatedCorrection:
    """A matched pair labeled with an inferred error description.

    This is presentation metadata — the decoder's actual output is just
    a pair of detector indices (or detector-to-boundary).

    Attributes
    ----------
    event_a : AnnotatedDetector | str
        One endpoint of the matched pair, or ``"boundary"`` if matched to a
        virtual boundary node.
    event_b : AnnotatedDetector | str
        The other endpoint.
    inferred_error_type : str
        Human-readable description such as ``"X error on data qubit (2,2), t=1"``
        or ``"measurement error on stabilizer (0,1) between t=0 and t=1"``.
    flips_logical : bool
        Whether this matched edge flips the logical observable ``L0``.
    """

    event_a: AnnotatedDetector | str
    event_b: AnnotatedDetector | str
    inferred_error_type: str
    flips_logical: bool

    def __str__(self) -> str:
        flip_str = " [flips L0]" if self.flips_logical else ""
        return f"{self.event_a}  ↔  {self.event_b}  →  {self.inferred_error_type}{flip_str}"


@dataclass(frozen=True)
class DecodedMemoryShot:
    """One decoded shot from a logical-memory experiment.

    Combines the decoder's actual output (``decoder_inferred_logical_flip``)
    with presentation annotations (``detection_events``, ``decoder_corrections``)
    and Stim's ground truth (``actual_logical_flip``).

    Attributes
    ----------
    logical_readout_basis : str
        ``"X"`` or ``"Z"`` — the basis of the logical observable.
    logical_observable_qubits : tuple[QubitCoord, ...]
        The ``(x, y)`` coordinates of data qubits in Stim's ``L0``.
    detection_events : tuple[AnnotatedDetector, ...]
        All fired detectors in this shot (presentation only).
    decoder_corrections : tuple[AnnotatedCorrection, ...]
        The decoder's matched pairs with inferred error descriptions
        (presentation only).
    actual_logical_flip : bool
        Whether the logical observable ``L0`` was actually flipped (from Stim's
        sampler).  This is the uncorrected logical readout bit.
    decoder_inferred_logical_flip : bool
        Whether the decoder inferred that ``L0`` was flipped.
    residual_logical_error : bool
        ``True`` if ``actual_logical_flip != decoder_inferred_logical_flip`` —
        i.e. the decoder's correction was wrong, and a logical error remains.
    """

    logical_readout_basis: str
    logical_observable_qubits: tuple[QubitCoord, ...]
    detection_events: tuple[AnnotatedDetector, ...]
    decoder_corrections: tuple[AnnotatedCorrection, ...]
    actual_logical_flip: bool
    decoder_inferred_logical_flip: bool
    residual_logical_error: bool

    def as_dict(self) -> dict[str, object]:
        """Return a notebook-friendly summary dict.

        The ``"error x correction product"`` key describes the net outcome:
        whether the error and the decoder's correction compose to the identity
        (success) or to a logical operator (failure).
        """
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


# ===========================================================================
# D3ErrorInterpreter — translates decoder output into human-readable labels
# ===========================================================================
#
# PyMatching returns detector-index pairs and a logical flip bit.  It never
# says "X error on qubit (2,2)."  This class uses the d=3 unrotated layout
# geometry to work backwards from matched pairs to inferred error descriptions.
# ===========================================================================


class D3ErrorInterpreter:
    """Translates PyMatching decoder output into human-readable error annotations.

    This is the presentation layer.  It takes the raw decoder output — detector
    index pairs and a logical flip bit — and produces ``AnnotatedDetector`` and
    ``AnnotatedCorrection`` objects with inferred error types and locations.

    The interpreter is initialized from a Stim circuit (to extract detector
    coordinates and observable qubit coordinates) and then reused across shots.

    Parameters
    ----------
    circuit : stim.Circuit
        The circuit whose detector coordinates define the layout.
    """

    def __init__(self, circuit: stim.Circuit) -> None:
        self.detector_coord_map = detector_coords_by_index(circuit)
        self.observable_qubit_coords = _interpret_observable_qubit_coords(circuit)
        self._max_t = (
            max(c[2] for c in self.detector_coord_map.values())
            if self.detector_coord_map else 0
        )

    # -- Annotate detectors ------------------------------------------------

    def annotate_detector(self, det_idx: int) -> AnnotatedDetector:
        """Label a single detector with its stabilizer sector and time layer."""
        x, y, t = self.detector_coord_map[det_idx]
        return AnnotatedDetector(
            detector_index=det_idx,
            coord=(x, y, t),
            sector=_interpret_stabilizer_sector(x, y),
            time_label=_interpret_time_label(t, self._max_t),
        )

    def annotate_fired_detectors(self, detector_sample: Any) -> list[AnnotatedDetector]:
        """Label all fired detectors (bit = 1) in a detector-sample bit-array."""
        return [
            self.annotate_detector(idx)
            for idx, fired in enumerate(detector_sample)
            if fired
        ]

    # -- Annotate matched pairs --------------------------------------------

    def annotate_matched_pairs(
        self,
        matched_dict: dict[int, int | None],
        inferred_flip: bool,
    ) -> list[AnnotatedCorrection]:
        """Translate PyMatching's matched-pairs dict into annotated corrections.

        Parameters
        ----------
        matched_dict : dict[int, int | None]
            From ``matching_decoder.decode_to_matched_dets_dict(...)``.
            Bidirectional: ``{a: b, b: a}`` for detector–detector pairs,
            ``{a: None}`` for boundary matches.
        inferred_flip : bool
            From ``matching_decoder.decode(...)[0]``.
        """
        seen: set[tuple[int, int | None]] = set()
        corrections: list[AnnotatedCorrection] = []
        for det_a, det_b in matched_dict.items():
            key = (min(det_a, det_b), max(det_a, det_b)) if det_b is not None else (det_a, None)
            if key in seen:
                continue
            seen.add(key)

            evt_a = self.annotate_detector(det_a)
            evt_b: AnnotatedDetector | str = (
                "boundary" if det_b is None
                else self.annotate_detector(det_b)
            )
            corrections.append(AnnotatedCorrection(
                event_a=evt_a,
                event_b=evt_b,
                inferred_error_type=self.interpret_error_description(evt_a, evt_b),
                flips_logical=(det_b is None and inferred_flip),
            ))
        return corrections

    # -- Interpret error type and location from a matched pair -------------

    @staticmethod
    def interpret_error_description(
        event_a: AnnotatedDetector | str,
        event_b: AnnotatedDetector | str,
    ) -> str:
        """Infer a human-readable error description from a matched pair.

        PyMatching only returns detector-index pairs.  This method uses the
        d=3 layout geometry to infer *what kind of error* produced that pair:

        1. **Boundary match** (one side is ``"boundary"``): error on a boundary
           data qubit adjacent to the fired stabilizer.
        2. **Same time layer**: spatial data-qubit error.  Pauli type from sector:
           both Z-stabilizers → X error, both X → Z error, mixed → Y.
        3. **Same location, different ``t``**: measurement error on the ancilla.
        4. **Everything else**: space-time diagonal error.
        """
        # Case 1: boundary match.
        if isinstance(event_a, str) or isinstance(event_b, str):
            det = event_a if isinstance(event_b, str) else event_b
            assert isinstance(det, AnnotatedDetector)
            x, y, t = det.coord
            pauli = "X" if det.sector == "Z" else "Z" if det.sector == "X" else "?"
            data_q = _boundary_data_qubit(x, y)
            loc = f"on data qubit ({data_q[0]},{data_q[1]})" if data_q else f"near stabilizer ({x},{y})"
            return f"{pauli} error {loc}, t={t}"

        # Cases 2–4: both sides are AnnotatedDetector.
        ax, ay, at = event_a.coord
        bx, by, bt = event_b.coord

        # Case 2: same time layer → spatial data-qubit error.
        if at == bt:
            sector_pair = frozenset({event_a.sector, event_b.sector})
            if sector_pair == {"Z"}:
                pauli = "X"
            elif sector_pair == {"X"}:
                pauli = "Z"
            else:
                pauli = "Y"
            data_q = _shared_data_qubit(ax, ay, bx, by)
            if data_q:
                return f"{pauli} error on data qubit ({data_q[0]},{data_q[1]}), t={at}"
            return f"{pauli} error on data qubit near ({(ax + bx) / 2:.0f},{(ay + by) / 2:.0f}), t={at}"

        # Case 3: same stabilizer, different time → measurement error.
        if (ax, ay) == (bx, by):
            return f"measurement error on stabilizer ({ax},{ay}) between t={min(at, bt)} and t={max(at, bt)}"

        # Case 4: space-time diagonal.
        return f"space-time error between ({ax},{ay},t={at}) and ({bx},{by},t={bt})"


# ===========================================================================
# Layout geometry and detector metadata (used by D3ErrorInterpreter)
# ===========================================================================
#
# These are module-level helpers because they depend only on the d=3
# unrotated layout, not on any per-circuit state.
# ===========================================================================


def _interpret_stabilizer_sector(x: int, y: int) -> str:
    """Return ``'X'`` or ``'Z'`` for the stabilizer sector at Stim coordinate ``(x, y)``.

    In Stim's unrotated surface-code layout:

    - **X-stabilizer** ancillas sit at ``(odd x, even y)``  — they have
      Hadamard gates and detect Z errors on neighboring data qubits.
    - **Z-stabilizer** ancillas sit at ``(even x, odd y)`` — they have no
      Hadamard gates and detect X errors on neighboring data qubits.
    - **Data qubits** sit at ``(x, y)`` where ``x + y`` is even.

    Returns ``'?'`` if the coordinate is neither (should not happen for
    valid detector coordinates).
    """
    if x % 2 == 1 and y % 2 == 0:
        return "X"
    if x % 2 == 0 and y % 2 == 1:
        return "Z"
    return "?"


def _interpret_time_label(t: int, max_t: int) -> str:
    """Return a human-readable label for the detector time layer.

    - ``t == 0``: preparation-boundary detectors compare the initial state
      against the first round of stabilizer measurements.
    - ``t == max_t``: final-boundary detectors compare the last stabilizer
      round against the destructive data-qubit readout.
    - Otherwise: bulk (round-to-round) detectors.
    """
    if t == 0:
        return "preparation-boundary"
    if t == max_t:
        return "final-boundary"
    return "bulk"


def _adjacent_data_qubits(sx: int, sy: int) -> list[QubitCoord]:
    """Return data-qubit ``(x, y)`` coordinates adjacent to the stabilizer at ``(sx, sy)``."""
    return [
        (sx + dx, sy + dy)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if (sx + dx, sy + dy) in _D3_DATA_QUBIT_COORDS
    ]


def _shared_data_qubit(ax: int, ay: int, bx: int, by: int) -> QubitCoord | None:
    """Return the data qubit shared by two adjacent stabilizers, or ``None``."""
    mid_x, mid_y = (ax + bx) / 2, (ay + by) / 2
    if mid_x == int(mid_x) and mid_y == int(mid_y):
        candidate = (int(mid_x), int(mid_y))
        if candidate in _D3_DATA_QUBIT_COORDS:
            return candidate
    return None


def _boundary_data_qubit(sx: int, sy: int) -> QubitCoord | None:
    """Return the most likely boundary data qubit for a boundary match.

    Picks the neighbor with the fewest adjacent stabilizers (most boundary-like),
    preferring neighbors on the patch perimeter (``x ∈ {0,4}`` or ``y ∈ {0,4}``).
    """
    neighbors = _adjacent_data_qubits(sx, sy)
    if not neighbors:
        return None

    def _stabilizer_count(qx: int, qy: int) -> int:
        return sum(
            1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= qx + dx <= 4 and 0 <= qy + dy <= 4
            and (qx + dx, qy + dy) not in _D3_DATA_QUBIT_COORDS
        )

    edge = [(qx, qy) for qx, qy in neighbors if qx in (0, 4) or qy in (0, 4)]
    return min(edge or neighbors, key=lambda q: _stabilizer_count(*q))


# ---------------------------------------------------------------------------
# Observable extraction and coordinate conversion
# ---------------------------------------------------------------------------


def _interpret_observable_qubit_coords(circuit: stim.Circuit) -> tuple[QubitCoord, ...]:
    """Return the ``(x, y)`` coordinates of data qubits in the logical observable.

    For ``surface_code:unrotated_memory_z`` at d=3, returns the three
    qubits along the top row: ``(0,0), (2,0), (4,0)``.
    """
    qubit_coords = {
        q: (int(cs[0]), int(cs[1]))
        for q, cs in circuit.get_final_qubit_coordinates().items()
    }
    flat = circuit.flattened()

    measurement_qubit_indices: list[int] = []
    for inst in flat:
        if inst.name in ("M", "MX"):
            measurement_qubit_indices.extend(t.value for t in inst.targets_copy())

    total_measurements = len(measurement_qubit_indices)
    obs_coords: list[QubitCoord] = []
    for inst in flat:
        if inst.name != "OBSERVABLE_INCLUDE":
            continue
        for target in inst.targets_copy():
            if target.is_measurement_record_target:
                qubit_idx = measurement_qubit_indices[total_measurements + target.value]
                obs_coords.append(qubit_coords[qubit_idx])
    return tuple(obs_coords)


def _to_detector_coord(stim_coords: list[float]) -> DetectorCoord:
    """Convert Stim detector coordinates to an ``(x, y, t)`` integer tuple."""
    padded = list(stim_coords) + [0.0] * (3 - len(stim_coords))
    return (int(padded[0]), int(padded[1]), int(padded[2]))



