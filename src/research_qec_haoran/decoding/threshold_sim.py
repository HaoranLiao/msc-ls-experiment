"""Logical error rate simulation for unrotated surface-code memory experiments.

This module provides helpers for sweeping physical error rates and computing
logical error rates via Stim + PyMatching.

Circuit generation and decoder construction are delegated to
:mod:`research_qec_haoran.decoding.memory_circuit`.

Public API
----------
- ``estimate_logical_error_rate`` – sample many shots, decode, return logical error rate
- ``sweep_physical_error_rates`` – run a full (distance, p_physical) sweep
- ``SweepResult`` – structured result from a sweep
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from research_qec_haoran.decoding.memory_circuit import (
    build_matching_decoder,
    generate_noisy_memory_circuit,
)


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


@dataclass
class SweepResult:
    """Results from a physical-error-rate sweep for one code distance."""

    distance: int
    rounds: int
    basis: str
    physical_error_rates: list[float]
    logical_error_rates: list[float]
    standard_errors: list[float]
    shots_per_point: int


def sweep_physical_error_rates(
    *,
    distance: int,
    rounds: int | None = None,
    basis: str = "Z",
    physical_error_rates: list[float],
    shots: int = 10_000,
) -> SweepResult:
    """Run a logical error rate sweep over physical error rates for one distance.

    If ``rounds`` is not specified, it defaults to ``distance`` (a common choice
    that gives the code enough time layers to see the full space-time structure).
    """
    if rounds is None:
        rounds = distance

    logical_rates: list[float] = []
    std_errs: list[float] = []

    for p in physical_error_rates:
        circuit = generate_noisy_memory_circuit(
            distance=distance,
            rounds=rounds,
            basis=basis,
            error_probability=p,
        )
        decoder = build_matching_decoder(circuit)
        p_l, se = estimate_logical_error_rate(circuit, decoder=decoder, shots=shots)
        logical_rates.append(p_l)
        std_errs.append(se)

    return SweepResult(
        distance=distance,
        rounds=rounds,
        basis=basis,
        physical_error_rates=physical_error_rates,
        logical_error_rates=logical_rates,
        standard_errors=std_errs,
        shots_per_point=shots,
    )

