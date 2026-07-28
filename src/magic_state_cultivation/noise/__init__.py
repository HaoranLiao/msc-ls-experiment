"""Noise-model definitions and architecture assumptions."""

from magic_state_cultivation.noise.fault_injection_qiskit import (
    append_pauli_faults,
    build_depolarizing_noise_model,
    build_noisy_aer_simulator,
)

__all__ = [
    "append_pauli_faults",
    "build_depolarizing_noise_model",
    "build_noisy_aer_simulator",
]
