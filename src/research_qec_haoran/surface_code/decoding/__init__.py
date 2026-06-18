"""Decoding and syndrome-processing utilities."""

from archive.surface_code.decoding.memory_circuit_stim import (
    build_matching_decoder,
    estimate_logical_error_rate,
    generate_noisy_memory_circuit,
)
from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.stabilizer_frame import StabilizerFrame
from research_qec_haoran.surface_code.decoding.rotated_patch_decoder_qiskit import (
    RotatedPatchRecovery,
    RotatedPatchDecoder,
    decode_rotated_patch_syndrome,
    rotated_d3_single_qubit_relative_syndrome,
)
from archive.surface_code.decoding.unrotated_d3_memory_stim import (
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
    "RotatedPatchRecovery",
    "RotatedPatchDecoder",
    "StabilizerFrame",
    "build_matching_decoder",
    "decode_rotated_patch_syndrome",
    "decode_memory_shots",
    "detector_coords_by_index",
    "estimate_logical_error_rate",
    "generate_noisy_memory_circuit",
    "rotated_d3_single_qubit_relative_syndrome",
]
