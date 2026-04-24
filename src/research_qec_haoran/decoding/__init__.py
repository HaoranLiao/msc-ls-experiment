"""Decoding and syndrome-processing utilities."""

from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.unrotated_d3_memory import (
    DEFAULT_DECODER_ERROR_PROBABILITY,
    DecodedMemoryShot,
    DetectionEvent,
    InferredCorrection,
    build_d3_unrotated_memory_matching_decoder,
    decode_memory_shots,
    detector_coords_by_index,
    generate_d3_unrotated_memory_decoder_circuit,
)

__all__ = [
    "DEFAULT_DECODER_ERROR_PROBABILITY",
    "DecodedMemoryShot",
    "DetectionEvent",
    "InferredCorrection",
    "PauliFrame",
    "build_d3_unrotated_memory_matching_decoder",
    "decode_memory_shots",
    "detector_coords_by_index",
    "generate_d3_unrotated_memory_decoder_circuit",
]
