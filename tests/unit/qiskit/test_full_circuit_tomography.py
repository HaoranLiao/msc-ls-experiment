"""Tests for logical Ty tomography after Qiskit state transfer."""

from __future__ import annotations

import math

import pytest
from qiskit_aer import AerSimulator

from msc.state_transfer.tomography_qiskit import (
    TOMOGRAPHY_BASES,
    build_ty_tomography_circuits,
    summarize_ty_tomography_counts,
)


def test_ty_tomography_builds_full_x_and_z_readout_circuits() -> None:
    """Both experiments include continuation rounds and all-data readout."""
    circuits = build_ty_tomography_circuits(
        target_distance=3,
        patch_sites=14,
        add_barriers=False,
    )

    assert tuple(circuits) == TOMOGRAPHY_BASES
    for basis, circuit in circuits.items():
        registers = {register.name: register for register in circuit.cregs}
        assert "post_transfer_surface_z" in registers
        assert "post_transfer_surface_x" in registers
        assert len(registers[f"surface_data_{basis.lower()}"]) == 9


@pytest.mark.slow
def test_noiseless_transferred_ty_tomography_has_unit_fidelity() -> None:
    """The paired full circuits recover the cultivated Ty Bloch vector."""
    circuits = build_ty_tomography_circuits(
        target_distance=3,
        add_barriers=False,
    )
    result = AerSimulator(method="matrix_product_state").run(
        [circuits[basis] for basis in TOMOGRAPHY_BASES],
        shots=128,
        seed_simulator=42,
    ).result()
    summary = summarize_ty_tomography_counts(
        {
            basis: result.get_counts(index)
            for index, basis in enumerate(TOMOGRAPHY_BASES)
        },
        target_distance=3,
    )

    assert summary.accepted_x_shots == 128
    assert summary.accepted_z_shots == 128
    assert summary.expectation_x == pytest.approx(1 / math.sqrt(2), abs=0.25)
    assert summary.expectation_z == pytest.approx(1 / math.sqrt(2), abs=0.25)
    assert summary.fidelity == pytest.approx(1.0, abs=0.18)
