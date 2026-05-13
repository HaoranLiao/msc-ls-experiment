"""Decoding and syndrome-processing utilities."""

from research_qec_haoran.surface_code.decoding.memory_circuit_stim import (
    _MEMORY_TASK_BY_BASIS,
    build_matching_decoder,
    estimate_logical_error_rate,
    generate_noisy_memory_circuit,
    normalized_basis,
)
from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.stabilizer_frame import StabilizerFrame
from research_qec_haoran.surface_code.decoding.rotated_d3_decoder_qiskit import (
    RotatedD3SingleQubitRecovery,
    decode_rotated_d3_single_qubit_syndrome,
    rotated_d3_single_qubit_relative_syndrome,
)
from research_qec_haoran.surface_code.decoding.unrotated_d3_memory_stim import (
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
    "RotatedD3SingleQubitRecovery",
    "StabilizerFrame",
    "build_matching_decoder",
    "decode_rotated_d3_single_qubit_syndrome",
    "decode_memory_shots",
    "detector_coords_by_index",
    "estimate_logical_error_rate",
    "generate_noisy_memory_circuit",
    "rotated_d3_single_qubit_relative_syndrome",
]
