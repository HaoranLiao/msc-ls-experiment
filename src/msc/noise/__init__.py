"""Noise-model definitions and architecture assumptions."""

from msc.noise.fault_injection_qiskit import (
    append_pauli_faults,
    build_depolarizing_noise_model,
    build_noisy_aer_simulator,
)

__all__ = [
    "append_pauli_faults",
    "build_depolarizing_noise_model",
    "build_noisy_aer_simulator",
]
