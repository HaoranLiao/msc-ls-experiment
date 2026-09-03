from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from msc.noise.fault_tolerance_stim import LogicalErrorEstimate, fit_power_law


def test_fit_power_law_matches_hirano_curve_fit_reference() -> None:
    """Match Hirano's independent ``curve_fit`` convention on the same data.

    Hirano's linked Figure 9 workbook identifies its original fit as SciPy
    ``curve_fit`` of ``a * p**b`` in linear probability space. The reference
    call below spells out that fit independently of our production helper and
    uses a fixed ``(1000, 3)`` starting point instead of our helper's log-linear
    initializer. Comparing the two on the fixed eight-point no-idle Z-probe
    counts guards against changing our helper to log-space or weighted
    regression, using sampled rather than accepted shots, or selecting a
    different fitting window.

    Hirano's plotting script also calls ``sinter.plot_error_rate``, but leaves
    its ``line_fits`` argument at the default ``None``. Sinter therefore fits a
    binomial rate and uncertainty interval independently at each point; it does
    not fit or return a power-law exponent. Using ``sinter.fit_line_slope`` on
    log-transformed points here would instead define a different regression
    from Hirano's workbook fit.
    """
    probabilities = (5e-4, 6e-4, 8e-4, 1e-3, 1.2e-3, 1.4e-3, 1.7e-3, 2e-3)
    accepted_shots = (
        25_674_705,
        24_888_720,
        23_387_259,
        21_975_607,
        20_654_637,
        19_409_954,
        17_688_923,
        16_116_543,
    )
    logical_errors = (447, 633, 1_281, 2_062, 3_086, 4_289, 6_821, 9_455)
    estimates = [
        LogicalErrorEstimate(
            experiment_name="hirano_no_idle_z_error",
            physical_error_probability=probability,
            sampled_shots=30_000_000,
            accepted_shots=accepted,
            logical_errors=errors,
        )
        for probability, accepted, errors in zip(
            probabilities,
            accepted_shots,
            logical_errors,
            strict=True,
        )
    ]

    reference_fit, _ = curve_fit(
        lambda probability, coefficient, exponent: (coefficient * probability**exponent),
        np.asarray(probabilities),
        np.asarray(logical_errors) / np.asarray(accepted_shots),
        p0=(1_000, 3),
        maxfev=100_000,
    )
    coefficient, exponent = fit_power_law(estimates)

    assert coefficient == pytest.approx(reference_fit[0], rel=1e-7)
    assert exponent == pytest.approx(reference_fit[1], rel=1e-7)
