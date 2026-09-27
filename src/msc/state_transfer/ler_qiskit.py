"""Per-round logical error rate from Qiskit Ty-state memory tomography."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import exp, hypot, isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from msc.state_transfer import common
from msc.state_transfer.tomography_qiskit import (
    CircuitLevelTomographyVariant,
    TY_TOMOGRAPHY_CONFIG_PATH,
    TyTomographyConfig,
    TyTomographySummary,
    analyze_ty_tomography_counts,
    build_ty_tomography_circuits,
)

if TYPE_CHECKING:
    from msc.experiment.results_io import DeviceCalibration

TY_LOGICAL_ERROR_RATE_CONFIG_PATH = Path(__file__).with_name("ler_config.yaml")


@dataclass(frozen=True)
class TyLogicalErrorRateConfig:
    """Validated settings for the grafted-code Ty memory experiment."""

    tomography: TyTomographyConfig
    post_transfer_stabilizer_rounds: tuple[int, ...]

    def __post_init__(self) -> None:
        rounds = self.post_transfer_stabilizer_rounds
        if len(rounds) < 3:
            raise ValueError("post_transfer_stabilizer_rounds must contain at least three values.")
        if any(type(round_count) is not int or round_count < 1 for round_count in rounds):
            raise ValueError("post_transfer_stabilizer_rounds must contain positive integers.")
        if any(left >= right for left, right in pairwise(rounds)):
            raise ValueError("post_transfer_stabilizer_rounds must be strictly increasing.")


@dataclass(frozen=True)
class TyLogicalErrorRateSummary:
    """Per-round tomography summaries and their fitted per-round logical error rate.

    ``tomography[i]`` is the postselected logical tomography at memory duration
    ``post_transfer_stabilizer_rounds[i]``. ``logical_error_rate_per_round`` is
    fit in the binary logical-flip convention ``R(N) = R(0) (1 - 2 p_L)**N`` over
    the per-point Bloch radii ``sqrt(<X_L>**2 + <Z_L>**2)``. The 95% interval
    propagates accepted-shot noise through the radii and nonlinear fit.
    """

    post_transfer_stabilizer_rounds: tuple[int, ...]
    tomography: tuple[TyTomographySummary, ...]
    logical_error_rate_per_round: float
    logical_error_rate_confidence_interval_95: tuple[float, float]
    bootstrap_samples: int
    bootstrap_seed: int


def load_ty_logical_error_rate_config(
    config_path: str | Path = TY_LOGICAL_ERROR_RATE_CONFIG_PATH,
) -> TyLogicalErrorRateConfig:
    """Load and validate one grafted-code memory-experiment YAML file."""
    config_path = Path(config_path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid logical-error-rate YAML in {config_path}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"logical-error-rate config {config_path} must contain a mapping.")

    expected_fields = set(TyTomographyConfig.__annotations__)
    if set(raw_config) != expected_fields:
        raise ValueError(
            f"logical-error-rate config {config_path} fields must be {sorted(expected_fields)}."
        )
    return parse_ty_logical_error_rate_config(raw_config)


def parse_ty_logical_error_rate_config(
    raw_config: Mapping[str, object],
) -> TyLogicalErrorRateConfig:
    """Split a flat LER mapping into tomography settings and memory durations.

    The YAML loader and the saved-hardware-job loader share this layout: the
    tomography fields with ``post_transfer_stabilizer_rounds`` as a list.
    """
    rounds = raw_config["post_transfer_stabilizer_rounds"]
    if not isinstance(rounds, list):
        raise ValueError("post_transfer_stabilizer_rounds must be a YAML list.")
    return TyLogicalErrorRateConfig(
        tomography=TyTomographyConfig(
            **{**raw_config, "post_transfer_stabilizer_rounds": None},
        ),
        post_transfer_stabilizer_rounds=tuple(rounds),
    )


def build_ty_logical_error_rate_circuits(
    *,
    config_path: str | Path = TY_LOGICAL_ERROR_RATE_CONFIG_PATH,
) -> dict[int, dict[common.SurfaceReadoutBasis, QuantumCircuit]]:
    """Build paired logical-X/Z circuits for every configured memory duration."""
    config = load_ty_logical_error_rate_config(config_path)
    tomography = config.tomography
    return {
        round_count: build_ty_tomography_circuits(
            circuit_config_path=TY_TOMOGRAPHY_CONFIG_PATH,
            target_distance=tomography.target_distance,
            patch_sites=tomography.patch_sites,
            merge_schedule=tomography.merge_schedule,
            full_post_selection=tomography.full_post_selection,
            persistent_surface_x_post_selection=tomography.persistent_surface_x_post_selection,
            measure_encoder_flags=tomography.measure_encoder_flags,
            measure_cultivation_flags=tomography.measure_cultivation_flags,
            post_injection_qed_checks=tomography.post_injection_qed_checks,
            post_transfer_stabilizer_rounds=round_count,
            add_barriers=tomography.add_barriers,
        )
        for round_count in config.post_transfer_stabilizer_rounds
    }


def analyze_ty_logical_error_rate_counts(
    counts_by_round: Mapping[
        int,
        Mapping[common.SurfaceReadoutBasis, Mapping[str, int]],
    ],
    *,
    config_path: str | Path = TY_LOGICAL_ERROR_RATE_CONFIG_PATH,
    logical_error_rate_config: TyLogicalErrorRateConfig | None = None,
    measurement_schemas_by_round: (
        Mapping[int, Mapping[common.SurfaceReadoutBasis, Mapping[str, Sequence[str]]]] | None
    ) = None,
    submitted_circuits_by_round: (
        Mapping[int, Mapping[common.SurfaceReadoutBasis, QuantumCircuit]] | None
    ) = None,
    calibrations_by_round: (
        Mapping[int, Mapping[common.SurfaceReadoutBasis, DeviceCalibration]] | None
    ) = None,
    circuit_level_variants_by_round: (
        Mapping[
            int,
            Mapping[common.SurfaceReadoutBasis, Sequence[CircuitLevelTomographyVariant]],
        ]
        | None
    ) = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> TyLogicalErrorRateSummary:
    """Analyze an X/Z tomography sweep and fit its per-round logical error rate.

    Every memory duration is analyzed as one tomography point with the
    tomography settings of this config at that point's round count. Hardware
    sweeps are decoded circuit-level when submitted circuits, calibrations, and
    measurement schemas are passed by round count; each point then builds its
    own calibrated model, and the configured complementary-gap threshold
    applies.
    """
    config = logical_error_rate_config or load_ty_logical_error_rate_config(config_path)
    expected_rounds = set(config.post_transfer_stabilizer_rounds)
    if set(counts_by_round) != expected_rounds:
        raise ValueError(
            "counts_by_round must contain exactly the configured "
            f"post-transfer rounds {sorted(expected_rounds)}."
        )

    tomography_summaries = tuple(
        analyze_ty_tomography_counts(
            counts_by_round[round_count],
            tomography_config=replace(
                config.tomography,
                post_transfer_stabilizer_rounds=round_count,
            ),
            measurement_schemas=(
                None
                if measurement_schemas_by_round is None
                else measurement_schemas_by_round[round_count]
            ),
            submitted_circuits_by_basis=(
                None
                if submitted_circuits_by_round is None
                else submitted_circuits_by_round[round_count]
            ),
            calibrations_by_basis=(
                None if calibrations_by_round is None else calibrations_by_round[round_count]
            ),
            circuit_level_variants_by_basis=(
                None
                if circuit_level_variants_by_round is None
                else circuit_level_variants_by_round[round_count]
            ),
        )
        for round_count in config.post_transfer_stabilizer_rounds
    )
    logical_error_rate_per_round = fit_ty_logical_error_rate(
        config.post_transfer_stabilizer_rounds,
        [hypot(summary.expectation_x, summary.expectation_z) for summary in tomography_summaries],
    )
    logical_error_rate_confidence_interval_95 = (
        bootstrap_ty_logical_error_rate_confidence_interval(
            config.post_transfer_stabilizer_rounds,
            tomography_summaries,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    )
    return TyLogicalErrorRateSummary(
        post_transfer_stabilizer_rounds=config.post_transfer_stabilizer_rounds,
        tomography=tomography_summaries,
        logical_error_rate_per_round=logical_error_rate_per_round,
        logical_error_rate_confidence_interval_95=(
            logical_error_rate_confidence_interval_95
        ),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def fit_ty_logical_error_rate(
    post_transfer_stabilizer_rounds: Sequence[int],
    bloch_radii_xz: Sequence[float],
) -> float:
    """Fit the Section XIV memory decay and return the per-round LER ``p_L``.

    Supplementary Section XIV and Fig. S22 specify an exponential fit to
    ``R(N) = sqrt(<X_L>**2 + <Z_L>**2)`` but do not print its equation. This
    implementation directly fits the standard binary logical-flip convention
    ``R(N) = A (1 - 2 p_L)**N`` and returns ``p_L``; the amplitude ``A`` is a
    nuisance parameter. Inputs must be consistently postselected and decoded
    logical tomography results across all memory durations.
    """
    rounds = tuple(post_transfer_stabilizer_rounds)
    radii = tuple(bloch_radii_xz)
    if len(rounds) != len(radii):
        raise ValueError("post-transfer round and Bloch-radius sequences must have equal length.")
    if len(rounds) < 3:
        raise ValueError("at least three memory durations are required for the decay fit.")
    if any(type(round_count) is not int or round_count < 1 for round_count in rounds):
        raise ValueError("post-transfer rounds must be positive integers.")
    if len(set(rounds)) != len(rounds):
        raise ValueError("post-transfer rounds must be distinct.")
    if any(not isfinite(radius) or radius < 0 for radius in radii):
        raise ValueError("Bloch radii must be finite and nonnegative.")
    if sum(radius > 0 for radius in radii) < 2:
        raise ValueError("at least two Bloch radii must be positive.")

    order = np.argsort(rounds)
    round_array = np.asarray(rounds, dtype=float)[order]
    radius_array = np.asarray(radii, dtype=float)[order]
    positive = radius_array > 0
    slope, intercept = np.polyfit(
        round_array[positive],
        np.log(radius_array[positive]),
        deg=1,
    )
    initial_logical_error_rate = (1.0 - exp(min(0.0, float(slope)))) / 2.0
    initial_radius = exp(float(intercept))

    def logical_decay(
        round_count: np.ndarray,
        radius: float,
        logical_error_rate: float,
    ) -> np.ndarray:
        return radius * (1.0 - 2.0 * logical_error_rate) ** round_count

    parameters = np.asarray(
        curve_fit(
            logical_decay,
            round_array,
            radius_array,
            p0=(initial_radius, initial_logical_error_rate),
            bounds=((0.0, 0.0), (np.inf, 0.5)),
            maxfev=100_000,
            full_output=False,
        )[0],
        dtype=float,
    )
    return float(parameters[1])


def bootstrap_ty_logical_error_rate_confidence_interval(
    post_transfer_stabilizer_rounds: Sequence[int],
    tomography_summaries: Sequence[TyTomographySummary],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[float, float]:
    """Estimate a 95% LER interval from accepted logical X/Z shot noise.

    Each bootstrap replicate draws accepted logical outcomes from the measured
    binomial distributions, recomputes the XZ Bloch radii, and repeats the full
    nonlinear decay fit. The interval is conditional on the accepted ensemble
    and does not include decoder-model or calibration uncertainty.
    """
    rounds = tuple(post_transfer_stabilizer_rounds)
    summaries = tuple(tomography_summaries)
    if len(rounds) != len(summaries):
        raise ValueError("post-transfer round and tomography-summary sequences must match.")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")

    random_number_generator = np.random.default_rng(bootstrap_seed)
    expectation_x_samples = np.column_stack(
        [
            2.0
            * random_number_generator.binomial(
                summary.accepted_x_shots,
                (1.0 + summary.expectation_x) / 2.0,
                size=bootstrap_samples,
            )
            / summary.accepted_x_shots
            - 1.0
            for summary in summaries
        ]
    )
    expectation_z_samples = np.column_stack(
        [
            2.0
            * random_number_generator.binomial(
                summary.accepted_z_shots,
                (1.0 + summary.expectation_z) / 2.0,
                size=bootstrap_samples,
            )
            / summary.accepted_z_shots
            - 1.0
            for summary in summaries
        ]
    )
    logical_error_rate_samples = np.asarray(
        [
            fit_ty_logical_error_rate(rounds, np.hypot(expectation_x, expectation_z))
            for expectation_x, expectation_z in zip(
                expectation_x_samples,
                expectation_z_samples,
                strict=True,
            )
        ]
    )
    lower, upper = np.quantile(logical_error_rate_samples, (0.025, 0.975))
    return float(lower), float(upper)
