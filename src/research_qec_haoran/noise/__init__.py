"""Noise-model definitions and architecture assumptions."""

from research_qec_haoran.noise.fault_injection_stim import (
    inject_pauli_errors_after_instruction_occurrence,
    inject_pauli_errors_after_measurement_round,
    inject_pauli_errors_before_final_measurement,
)
from research_qec_haoran.noise.steane_patch_fault_injection_qiskit import (
    append_patch_pauli_faults,
)

__all__ = [
    "inject_pauli_errors_after_instruction_occurrence",
    "inject_pauli_errors_after_measurement_round",
    "inject_pauli_errors_before_final_measurement",
    "append_patch_pauli_faults",
]
