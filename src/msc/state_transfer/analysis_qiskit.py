from __future__ import annotations

from collections.abc import Sequence

from msc.state_transfer.common import (
    MergeSchedule,
    SteanePatchSites,
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
    patch_sites: SteanePatchSites = 14,
    merge_schedule: MergeSchedule = "hirano",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = False,
    post_injection_qed_checks: Sequence[str] | None = None,
    post_transfer_stabilizer_rounds: int = 0,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    full_data_readout: bool = False,
    full_post_selection: bool = False,
    persistent_surface_x_post_selection: bool = True,
) -> dict[str, object]:
    """Parse one count key and decode its merge and continuation histories.

    Full-data X/Z readout reconstructs the compatible final stabilizers and
    logical support from the same canonical surface-data register.

    ``measure_encoder_flags`` and ``measure_cultivation_flags`` must match
    construction; each ``False`` shrinks the record layout (dropping the
    ``enc_flags`` family, or the four ancilla flags per ``cult`` round).
    """
    labels = transfer_measurement_labels(
        spec,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        merge_schedule=merge_schedule,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        full_data_readout=full_data_readout,
    )
    bits = parse_qiskit_registers(bitstring, tuple(labels.items()))
    parsed = interpret_transfer_measurements(
        bits,
        spec=spec,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        full_data_readout=full_data_readout,
        full_post_selection=full_post_selection,
        persistent_surface_x_post_selection=persistent_surface_x_post_selection,
    )
    parsed["registers"] = {
        name: {
            "raw": "".join(str(family[label]) for label in reversed(labels[name])),
            "bits": family,
        }
        for name, family in bits.items()
    }
    return parsed
