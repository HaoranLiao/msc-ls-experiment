"""Stim-native surface-code helpers and geometry utilities."""

from research_qec_haoran.surface_code.decoding import (
    PauliFrame,
    RotatedD3SingleQubitRecovery,
    StabilizerFrame,
    apply_rotated_d3_data_pauli,
    decode_rotated_d3_single_qubit_syndrome,
    rotated_d3_single_qubit_relative_syndrome,
)
from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    ROTATED_D3_X_STABILIZER_CENTERS,
    ROTATED_D3_Z_STABILIZER_CENTERS,
    ROTATED_D3_LOGICAL_X,
    ROTATED_D3_LOGICAL_Z,
    ROTATED_D3_SURFACE_COORDS,
    ROTATED_D3_SURFACE_ORDER,
    ROTATED_D3_X_STABILIZERS,
    ROTATED_D3_Z_STABILIZERS,
    draw_rotated_d3_patch,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    ROTATED_D3_INIT_Z_LABELS,
    append_prepare_plus_state,
    append_prepare_zero_state,
    append_rotated_d3_x_stabilizer_measurements,
    append_rotated_d3_z_stabilizer_measurements,
    append_surface_d3_stabilizer_round,
    build_rotated_d3_logical_plus_statevector,
    build_rotated_d3_pauli_operator,
    rotated_d3_logical_plus_expectations,
    rotated_d3_z_syndrome_bits_from_measurements,
)
from research_qec_haoran.surface_code.unrotated_d3_stim import generate_d3_unrotated_memory_circuit
from research_qec_haoran.surface_code.unrotated_d3_layout_stim import (
    D3_UNROTATED_DATA_COORDS,
    D3_UNROTATED_X_STABILIZERS,
    D3_UNROTATED_Z_STABILIZERS,
)

__all__ = [
    "D3_UNROTATED_DATA_COORDS",
    "D3_UNROTATED_X_STABILIZERS",
    "D3_UNROTATED_Z_STABILIZERS",
    "PauliFrame",
    "ROTATED_D3_INIT_Z_LABELS",
    "ROTATED_D3_LOGICAL_X",
    "ROTATED_D3_LOGICAL_Z",
    "RotatedD3SingleQubitRecovery",
    "ROTATED_D3_SURFACE_COORDS",
    "ROTATED_D3_SURFACE_ORDER",
    "ROTATED_D3_X_STABILIZER_CENTERS",
    "ROTATED_D3_X_STABILIZERS",
    "ROTATED_D3_Z_STABILIZER_CENTERS",
    "ROTATED_D3_Z_STABILIZERS",
    "StabilizerFrame",
    "apply_rotated_d3_data_pauli",
    "append_prepare_plus_state",
    "append_prepare_zero_state",
    "append_rotated_d3_x_stabilizer_measurements",
    "append_rotated_d3_z_stabilizer_measurements",
    "append_surface_d3_stabilizer_round",
    "build_rotated_d3_logical_plus_statevector",
    "build_rotated_d3_pauli_operator",
    "decode_rotated_d3_single_qubit_syndrome",
    "draw_rotated_d3_patch",
    "generate_d3_unrotated_memory_circuit",
    "rotated_d3_logical_plus_expectations",
    "rotated_d3_single_qubit_relative_syndrome",
    "rotated_d3_z_syndrome_bits_from_measurements",
]
