"""Helpers for decoding distance-3 unrotated surface-code memory experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import stim

_UNROTATED_D3_DISTANCE = 3

DetectorCoord = tuple[int, int, int]

DEFAULT_DECODER_ERROR_PROBABILITY = 1e-3

_MEMORY_TASK_BY_BASIS = {
    "Z": "surface_code:unrotated_memory_z",
    "X": "surface_code:unrotated_memory_x",
}


@dataclass(frozen=True)
class DecodedMemoryShot:
    """One decoded shot from a logical-memory experiment.

    Stim labels logical observables by index, e.g. ``L0``. The native unrotated-memory circuits
    used here have exactly one logical observable, so ``stim_observable_index=0`` means the circuit
    output bit labeled ``L0`` by Stim.

    For ``surface_code:unrotated_memory_z``, that single observable is the final logical-``Z``
    readout bit. For ``surface_code:unrotated_memory_x``, it is the final logical-``X`` readout
    bit.
    """

    stim_observable_index: int
    logical_readout_basis: str
    triggered_detectors: tuple[int, ...]
    triggered_detector_coords: tuple[DetectorCoord, ...]
    measured_logical_readout_flip: int
    decoder_inferred_logical_readout_flip: int
    post_decoding_logical_readout_mismatch: int

    def as_dict(self) -> dict[str, object]:
        """Return a notebook-friendly summary."""

        return {
            "stim_observable_index": self.stim_observable_index,
            "logical_readout_basis": self.logical_readout_basis,
            "triggered_detectors": self.triggered_detectors,
            "triggered_detector_coords": self.triggered_detector_coords,
            "measured_logical_readout_flip": self.measured_logical_readout_flip,
            "decoder_inferred_logical_readout_flip": (
                self.decoder_inferred_logical_readout_flip
            ),
            "post_decoding_logical_readout_mismatch": (
                self.post_decoding_logical_readout_mismatch
            ),
        }


def detector_coords_by_index(circuit: stim.Circuit) -> dict[int, DetectorCoord]:
    """Map each detector index in ``circuit`` to its visualized ``(x, y, t)`` coordinate."""

    return {
        detector: _to_detector_coord(coords)
        for detector, coords in circuit.get_detector_coordinates().items()
    }


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
    except ImportError as exc:  # pragma: no cover - exercised in the qec env
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


def decode_memory_shots(
    circuit: stim.Circuit,
    *,
    matching: Any,
    shots: int = 1,
    logical_readout_basis: str = "Z",
) -> list[DecodedMemoryShot]:
    """Sample and decode memory-experiment shots from ``circuit``.

    The input circuit must expose exactly one Stim logical observable, ``L0``. The
    ``logical_readout_basis`` argument states what that single logical readout means:

    - ``"Z"`` for ``surface_code:unrotated_memory_z``
    - ``"X"`` for ``surface_code:unrotated_memory_x``
    """

    if shots < 1:
        raise ValueError("shots must be at least 1")

    normalized_basis = _normalized_basis(logical_readout_basis)

    sampler = circuit.compile_detector_sampler()
    detector_samples, observable_flips = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    decoder_inferred_observable_flips = matching.decode_batch(detector_samples)

    if observable_flips.shape[1] != 1 or decoder_inferred_observable_flips.shape[1] != 1:
        raise ValueError("this helper expects exactly one logical observable, namely Stim's L0")

    detector_coord_map = detector_coords_by_index(circuit)
    decoded_shots: list[DecodedMemoryShot] = []

    for detector_sample, measured_observable_flip, decoder_inferred_observable_flip in zip(
        detector_samples,
        observable_flips,
        decoder_inferred_observable_flips,
        strict=True,
    ):
        triggered_detectors = tuple(
            detector_index
            for detector_index, detector_bit in enumerate(detector_sample)
            if bool(detector_bit)
        )
        triggered_detector_coords = tuple(
            detector_coord_map[detector_index] for detector_index in triggered_detectors
        )
        measured_logical_readout_flip = int(measured_observable_flip[0])
        decoder_inferred_logical_readout_flip = int(decoder_inferred_observable_flip[0])
        decoded_shots.append(
            DecodedMemoryShot(
                stim_observable_index=0,
                logical_readout_basis=normalized_basis,
                triggered_detectors=triggered_detectors,
                triggered_detector_coords=triggered_detector_coords,
                measured_logical_readout_flip=measured_logical_readout_flip,
                decoder_inferred_logical_readout_flip=decoder_inferred_logical_readout_flip,
                post_decoding_logical_readout_mismatch=(
                    measured_logical_readout_flip ^ decoder_inferred_logical_readout_flip
                ),
            )
        )

    return decoded_shots


def _normalized_basis(basis: str) -> str:
    normalized_basis = basis.upper()
    if normalized_basis not in _MEMORY_TASK_BY_BASIS:
        raise ValueError("basis must be 'X' or 'Z'")
    return normalized_basis


def _to_detector_coord(raw_coords: list[float]) -> DetectorCoord:
    padded_coords = list(raw_coords) + [0.0] * (3 - len(raw_coords))
    return tuple(int(value) for value in padded_coords[:3])
