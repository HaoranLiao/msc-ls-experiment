"""Stim parsing for Steane-to-surface lattice-surgery results.

Circuit construction and global record positions live in
``state_transfer_helper_stim``. This module converts one Stim record row into
framework-neutral measurement families, then delegates detector
reconstruction, acceptance, surface decoding, and Pauli-frame interpretation
to ``state_transfer_common``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from msc.state_transfer_by_lattice_surgery.state_transfer_common import (
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    interpret_transfer_measurement_families,
)
from msc.state_transfer_by_lattice_surgery.state_transfer_helper_stim import (
    TransferMeasurementLayout,
    transfer_measurement_layout,
)


def parse_transfer_measurements(
    shot: Sequence[bool | int] | np.ndarray,
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    layout: TransferMeasurementLayout | None = None,
) -> dict[str, object]:
    """Parse and interpret one fixed-protocol Stim measurement-record row.

    The returned ``registers`` entry preserves semantic labels, global record
    positions, and parsed bits for each family. Other entries expose the three
    merged rounds, Steane checks, demolition outcomes, recovered top-edge X
    checks, surface detector histories, decoded merge recovery, and complete
    post-transfer surface stabilizer reference.

    ``layout`` may be reused across many shots to avoid rebuilding identical
    record-position metadata. Its width must exactly match ``shot``.
    """
    shot = np.asarray(shot, dtype=bool)
    if shot.ndim != 1:
        raise ValueError("shot must be a one-dimensional array.")
    if layout is None:
        layout = transfer_measurement_layout(
            spec,
            include_cultivation=include_cultivation,
            include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
            surface_readout_basis=surface_readout_basis,
        )
    if shot.shape[0] != layout.num_measurements:
        raise ValueError(
            "shot width does not match the transfer measurement layout: "
            f"expected {layout.num_measurements}, got {shot.shape[0]}."
        )

    measurement_families: dict[str, dict[str, object]] = {}
    for record in layout.records:
        bits = record.bits(shot)
        measurement_families[record.name] = {
            "raw": "".join(str(bits[label]) for label in record.labels),
            "labels": record.labels,
            "positions": record.positions,
            "bits": bits,
        }
    return interpret_transfer_measurement_families(
        measurement_families,
        spec=spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
