from __future__ import annotations

from msc.state_transfer.common import (
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    interpret_transfer_measurements,
    transfer_measurement_labels,
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
    """Label one count key, then derive checks, recovery, and Pauli byproducts."""
    labels = transfer_measurement_labels(
        spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
    bits = parse_qiskit_registers(bitstring, tuple(labels.items()))
    parsed = interpret_transfer_measurements(
        bits,
        spec=spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
    parsed["registers"] = {
        name: {
            "raw": "".join(str(family[label]) for label in reversed(labels[name])),
            "bits": family,
        }
        for name, family in bits.items()
    }
    return parsed
