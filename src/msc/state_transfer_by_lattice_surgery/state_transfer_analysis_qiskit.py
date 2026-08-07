"""Qiskit parsing for Steane-to-surface lattice-surgery results.

Circuit and register construction live in ``state_transfer_helper_qiskit``.
This module converts Qiskit count keys into framework-neutral measurement
families, then delegates detector reconstruction, acceptance, surface decoding,
and Pauli-frame interpretation to ``state_transfer_common``.
"""

from __future__ import annotations

from msc.state_transfer_by_lattice_surgery.state_transfer_common import (
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    interpret_transfer_measurement_families,
    transfer_measurement_specs,
)
from msc.utils.normalize import parse_qiskit_registers


def parse_transfer_measurements(
    bitstring: str,
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    """Parse and interpret one fixed-protocol Qiskit transfer shot.

    The returned ``registers`` entry preserves each raw count-key chunk, its
    semantic labels, and parsed bits. Other entries expose the paper's three
    merged rounds, Steane checks, demolition outcomes, recovered top-edge X
    checks, surface detector histories, decoded merge recovery, and complete
    post-transfer surface stabilizer reference.
    """
    measurement_specs = transfer_measurement_specs(
        spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
    labels_by_register = [
        (measurement_spec.name, measurement_spec.labels) for measurement_spec in measurement_specs
    ]
    bits_by_register = parse_qiskit_registers(bitstring, labels_by_register)
    measurement_families: dict[str, dict[str, object]] = {}
    for measurement_spec in measurement_specs:
        bits = bits_by_register[measurement_spec.name]
        measurement_families[measurement_spec.name] = {
            "raw": "".join(str(bits[label]) for label in reversed(measurement_spec.labels)),
            "labels": measurement_spec.labels,
            "bits": bits,
        }
    return interpret_transfer_measurement_families(
        measurement_families,
        spec=spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
