"""Per-round logical error rate from Qiskit Ty-state memory tomography."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import exp, hypot, isfinite
from pathlib import Path

import numpy as np
import yaml
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from msc.state_transfer import common
from msc.state_transfer.tomography_qiskit import (
    TY_TOMOGRAPHY_CONFIG_PATH,
    TyTomographyConfig,
    TyTomographySummary,
    analyze_ty_tomography_counts,
    build_ty_tomography_circuits,
)

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
class TyMemoryExperimentPoint:
    """Logical tomography result after one number of memory rounds."""

    post_transfer_stabilizer_rounds: int
    tomography: TyTomographySummary

    @property
    def bloch_radius_xz(self) -> float:
        """Return ``sqrt(<X_L>**2 + <Z_L>**2)`` for this sweep point."""
        return hypot(self.tomography.expectation_x, self.tomography.expectation_z)


@dataclass(frozen=True)
class TyLogicalErrorRateFit:
    """Bloch-decay fit in the per-round binary logical-error convention."""

    zero_round_bloch_radius: float
    logical_error_rate_per_round: float

    def fitted_bloch_radius(self, post_transfer_stabilizer_rounds: int | float) -> float:
        """Evaluate the fitted ``R(N) = R(0) (1 - 2 p_L)**N`` curve."""
        return (
            self.zero_round_bloch_radius
            * (1.0 - 2.0 * self.logical_error_rate_per_round) ** post_transfer_stabilizer_rounds
        )


@dataclass(frozen=True)
class TyLogicalErrorRateSummary:
    """Tomography sweep points and their fitted per-round logical error rate."""

    points: tuple[TyMemoryExperimentPoint, ...]
    fit: TyLogicalErrorRateFit


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
    rounds = raw_config["post_transfer_stabilizer_rounds"]
    if not isinstance(rounds, list):
        raise ValueError("post_transfer_stabilizer_rounds must be a YAML list.")
    tomography_values = dict(raw_config)
    tomography_values["post_transfer_stabilizer_rounds"] = None
    return TyLogicalErrorRateConfig(
        tomography=TyTomographyConfig(**tomography_values),
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
            full_post_selection=tomography.full_post_selection,
            persistent_surface_x_post_selection=tomography.persistent_surface_x_post_selection,
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
) -> TyLogicalErrorRateSummary:
    """Analyze an X/Z tomography sweep and fit its per-round logical error rate."""
    config = load_ty_logical_error_rate_config(config_path)
    tomography = config.tomography
    expected_rounds = set(config.post_transfer_stabilizer_rounds)
    if set(counts_by_round) != expected_rounds:
        raise ValueError(
            "counts_by_round must contain exactly the configured "
            f"post-transfer rounds {sorted(expected_rounds)}."
        )

    points = tuple(
        TyMemoryExperimentPoint(
            post_transfer_stabilizer_rounds=round_count,
            tomography=analyze_ty_tomography_counts(
                counts_by_round[round_count],
                circuit_config_path=TY_TOMOGRAPHY_CONFIG_PATH,
                target_distance=tomography.target_distance,
                patch_sites=tomography.patch_sites,
                full_post_selection=tomography.full_post_selection,
                persistent_surface_x_post_selection=(
                    tomography.persistent_surface_x_post_selection
                ),
                post_transfer_stabilizer_rounds=round_count,
            ),
        )
        for round_count in config.post_transfer_stabilizer_rounds
    )
    fit = fit_ty_logical_error_rate(
        [point.post_transfer_stabilizer_rounds for point in points],
        [point.bloch_radius_xz for point in points],
    )
    return TyLogicalErrorRateSummary(points=points, fit=fit)


def fit_ty_logical_error_rate(
    post_transfer_stabilizer_rounds: Sequence[int],
    bloch_radii_xz: Sequence[float],
) -> TyLogicalErrorRateFit:
    """Fit the Section XIV memory decay and convert it to LER per round.

    Supplementary Section XIV and Fig. S22 specify an exponential fit to
    ``R(N) = sqrt(<X_L>**2 + <Z_L>**2)`` but do not print its equation. This
    implementation directly fits the standard binary logical-flip convention
    ``R(N) = A (1 - 2 p_L)**N``. Inputs must be consistently postselected and
    decoded logical tomography results across all memory durations.
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
    zero_round_radius, logical_error_rate = float(parameters[0]), float(parameters[1])
    return TyLogicalErrorRateFit(
        zero_round_bloch_radius=zero_round_radius,
        logical_error_rate_per_round=logical_error_rate,
    )
