"""General helpers for Stim surface-code memory circuits and PyMatching decoders.

These are distance-general building blocks shared by the d=3 decoding helpers
(``unrotated_d3_memory``) and the threshold estimation notebook.

Public API
----------
- ``generate_noisy_memory_circuit`` – Stim circuit with uniform circuit-level noise
- ``build_matching_decoder`` – PyMatching decoder from a noisy Stim circuit
- ``estimate_logical_error_rate`` – sample many shots, decode, return logical error rate
"""

from __future__ import annotations

import numpy as np
import stim

_MEMORY_TASK_BY_BASIS = {
    "Z": "surface_code:unrotated_memory_z",
    "X": "surface_code:unrotated_memory_x",
}


def normalized_basis(basis: str) -> str:
    """Return the upper-cased basis letter after validation."""
    result = basis.upper()
    if result not in _MEMORY_TASK_BY_BASIS:
        raise ValueError("basis must be 'X' or 'Z'")
    return result


def generate_noisy_memory_circuit(
    *,
    distance: int,
    rounds: int,
    basis: str = "Z",
    error_probability: float,
) -> stim.Circuit:
    """Return a Stim memory circuit with uniform circuit-level depolarizing noise.

    Works with any distance supported by Stim's ``surface_code:unrotated_memory_z``
    (and ``_x``) generators.
    """
    basis = normalized_basis(basis)
    if not 0 < error_probability < 1:
        raise ValueError("error_probability must be in (0, 1)")

    return stim.Circuit.generated(
        _MEMORY_TASK_BY_BASIS[basis],
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=error_probability,
        before_round_data_depolarization=error_probability,
        before_measure_flip_probability=error_probability,
        after_reset_flip_probability=error_probability,
    )


def build_matching_decoder(circuit: stim.Circuit):
    """Build a PyMatching decoder from a noisy Stim circuit.

    The input *circuit* **must** contain noise (e.g. depolarizing channels,
    measurement flip probabilities).  Without noise the detector error model
    has no error mechanisms, which means PyMatching's matching graph has no
    edges and the decoder cannot decode anything.

    The noise probabilities do not need to match the physical error rate of
    the actual samples — they only set the relative edge weights that guide
    minimum-weight perfect matching.  A small nominal value such as ``1e-3``
    is typical when the samples come from a separate ideal or fault-injected
    circuit.
    """
    import pymatching

    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def estimate_logical_error_rate(
    circuit: stim.Circuit,
    *,
    decoder=None,
    shots: int = 10_000,
) -> tuple[float, float]:
    """Estimate the logical error rate by sampling and decoding.

    Returns ``(logical_error_rate, standard_error)``.
    The standard error assumes binomial statistics.
    """
    if decoder is None:
        decoder = build_matching_decoder(circuit)

    sampler = circuit.compile_detector_sampler()
    detector_samples, observable_flips = sampler.sample(
        shots=shots,
        separate_observables=True,
    )

    predictions = decoder.decode_batch(detector_samples)
    num_errors = int(np.sum(predictions != observable_flips))

    p_logical = num_errors / shots
    std_err = np.sqrt(p_logical * (1 - p_logical) / shots) if shots > 0 else 0.0
    return p_logical, std_err
