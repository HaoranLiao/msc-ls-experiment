from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from msc.state_transfer.common import (
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    interpret_transfer_measurements,
)
from msc.state_transfer.helper_stim import (
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
    """Label one record row, then derive checks, recovery, and Pauli byproducts."""
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

    bits = layout.extract(shot)
    parsed = interpret_transfer_measurements(
        bits,
        spec=spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
    parsed["registers"] = {
        name: {
            "raw": "".join(map(str, family.values())),
            "bits": family,
            "positions": layout.positions[name],
        }
        for name, family in bits.items()
    }
    return parsed
