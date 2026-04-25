"""Decoding and syndrome-processing utilities."""

from research_qec_haoran.decoding.memory_circuit import (
    build_matching_decoder,
    generate_noisy_memory_circuit,
)
from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.threshold_sim import (
    SweepResult,
    estimate_logical_error_rate,
    sweep_physical_error_rates,
)
from research_qec_haoran.decoding.unrotated_d3_memory import (
    DecodedMemoryShot,
    DetectionEvent,
    InferredCorrection,
    decode_memory_shots,
    detector_coords_by_index,
)

__all__ = [
    "DecodedMemoryShot",
    "DetectionEvent",
    "InferredCorrection",
    "PauliFrame",
    "SweepResult",
    "build_matching_decoder",
    "decode_memory_shots",
    "detector_coords_by_index",
    "estimate_logical_error_rate",
    "generate_noisy_memory_circuit",
    "sweep_physical_error_rates",
]
