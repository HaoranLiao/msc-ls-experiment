"""Decoding and syndrome-processing utilities."""

from research_qec_haoran.decoding.memory_circuit import (
    build_matching_decoder,
    estimate_logical_error_rate,
    generate_noisy_memory_circuit,
)
from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.unrotated_d3_memory import (
    AnnotatedCorrection,
    AnnotatedDetector,
    D3ErrorInterpreter,
    DecodedMemoryShot,
    decode_memory_shots,
    detector_coords_by_index,
)

__all__ = [
    "AnnotatedCorrection",
    "AnnotatedDetector",
    "D3ErrorInterpreter",
    "DecodedMemoryShot",
    "PauliFrame",
    "build_matching_decoder",
    "decode_memory_shots",
    "detector_coords_by_index",
    "estimate_logical_error_rate",
    "generate_noisy_memory_circuit",
]
