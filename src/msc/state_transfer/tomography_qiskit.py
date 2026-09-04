"""Logical X/Z tomography after Qiskit Steane-to-surface state transfer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from qiskit import QuantumCircuit

from msc.state_transfer import common
from msc.state_transfer import to_d3_rotated_qiskit as d3
from msc.state_transfer import to_d5_rotated_qiskit as d5

TomographyTargetDistance = Literal[3, 5]
TOMOGRAPHY_BASES: tuple[common.SurfaceReadoutBasis, ...] = ("X", "Z")


@dataclass(frozen=True)
class TyTomographySummary:
    """Postselected logical Bloch components and fidelity with ``|Ty>``."""

    expectation_x: float
    expectation_z: float
    fidelity: float
    accepted_x_shots: int
    accepted_z_shots: int
    total_x_shots: int
    total_z_shots: int


def build_ty_tomography_circuits(
    *,
    target_distance: TomographyTargetDistance = 5,
    patch_sites: common.SteanePatchSites = 14,
    full_post_selection: bool = False,
    add_barriers: bool = True,
) -> dict[common.SurfaceReadoutBasis, QuantumCircuit]:
    """Build full logical-X and logical-Z readout circuits for transferred ``|Ty>``.

    Each circuit includes physical ``|Ty>`` injection, post-injection Steane
    QED, optional logical-H_XZ cultivation, all three lattice-surgery merge
    rounds, the target-distance post-transfer rounds, and destructive readout
    of every surface-code data qubit in the selected basis.
    """
    if target_distance == 3:
        build_transfer_circuit = d3.build_steane_to_rotated_d3_lattice_surgery_circuit
    elif target_distance == 5:
        build_transfer_circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit
    else:
        raise ValueError("target_distance must be 3 or 5.")

    return {
        basis: build_transfer_circuit(
            patch_sites=patch_sites,
            injected_state="Ty",
            cultivated_state="Ty",
            include_cultivation=True,
            surface_readout_basis=basis,
            full_post_selection=full_post_selection,
            add_barriers=add_barriers,
        )
        for basis in TOMOGRAPHY_BASES
    }


def summarize_ty_tomography_counts(
    counts_by_basis: Mapping[common.SurfaceReadoutBasis, Mapping[str, int]],
    *,
    target_distance: TomographyTargetDistance = 5,
    patch_sites: common.SteanePatchSites = 14,
    full_post_selection: bool = False,
) -> TyTomographySummary:
    """Postselect and decode paired Qiskit counts into ``|Ty>`` fidelity."""
    if target_distance == 3:
        parse_transfer_measurements = d3.parse_transfer_measurements
    elif target_distance == 5:
        parse_transfer_measurements = d5.parse_transfer_measurements
    else:
        raise ValueError("target_distance must be 3 or 5.")

    expectations: dict[common.SurfaceReadoutBasis, float] = {}
    accepted_shots: dict[common.SurfaceReadoutBasis, int] = {}
    total_shots: dict[common.SurfaceReadoutBasis, int] = {}
    for basis in TOMOGRAPHY_BASES:
        if basis not in counts_by_basis:
            raise ValueError(f"counts_by_basis is missing the {basis} experiment.")
        counts = counts_by_basis[basis]
        total_shots[basis] = sum(counts.values())
        signed_accepted_shots = 0
        accepted_shots[basis] = 0
        for bitstring, count in counts.items():
            measurements = parse_transfer_measurements(
                bitstring,
                patch_sites=patch_sites,
                include_cultivation=True,
                surface_readout_basis=basis,
                full_post_selection=full_post_selection,
            )
            if not common.transfer_acceptance_succeeds(measurements):
                continue
            logical_bit = common.corrected_surface_logical_readout(
                measurements,
                basis=basis,
            )
            accepted_shots[basis] += count
            signed_accepted_shots += (1 if logical_bit == 0 else -1) * count
        if accepted_shots[basis] == 0:
            raise ValueError(f"no accepted {basis}-basis tomography shots.")
        expectations[basis] = signed_accepted_shots / accepted_shots[basis]

    expectation_x = expectations["X"]
    expectation_z = expectations["Z"]
    linear_fidelity = (1.0 + (expectation_x + expectation_z) / 2**0.5) / 2.0
    return TyTomographySummary(
        expectation_x=expectation_x,
        expectation_z=expectation_z,
        # Independent finite-shot estimates can fall outside the Bloch ball.
        fidelity=min(1.0, max(0.0, linear_fidelity)),
        accepted_x_shots=accepted_shots["X"],
        accepted_z_shots=accepted_shots["Z"],
        total_x_shots=total_shots["X"],
        total_z_shots=total_shots["Z"],
    )
