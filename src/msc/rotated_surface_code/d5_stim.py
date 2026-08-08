"""Rotated distance-5 surface-code Stim API over shared patch geometry.

The generated 25-qubit geometry, stabilizers, logical representatives,
drawing metadata, and canonical schedules are owned by ``common``.
This module parallels the public Qiskit d=5 API while delegating Stim circuit
primitives to ``helper_stim``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import stim
from matplotlib.axes import Axes

from msc.rotated_surface_code import common
from msc.rotated_surface_code import helper_stim as helper

PATCH_SPEC = common.D5_PATCH_SPEC
DISTANCE = PATCH_SPEC.distance
NUM_DATA_QUBITS = len(PATCH_SPEC.surface_order)
SURFACE_ORDER = PATCH_SPEC.surface_order
SURFACE_INDEX = PATCH_SPEC.surface_index
SURFACE_COORDS = PATCH_SPEC.surface_coords
X_STABILIZERS = PATCH_SPEC.x_stabilizers
Z_STABILIZERS = PATCH_SPEC.z_stabilizers
X_STABILIZER_CENTERS = PATCH_SPEC.x_stabilizer_centers
Z_STABILIZER_CENTERS = PATCH_SPEC.z_stabilizer_centers
DEFAULT_LOGICAL_X = PATCH_SPEC.default_logical_x
DEFAULT_LOGICAL_Z = PATCH_SPEC.default_logical_z
LOGICAL_X_REPRESENTATIVES = PATCH_SPEC.logical_x_representatives
LOGICAL_Z_REPRESENTATIVES = PATCH_SPEC.logical_z_representatives
INIT_Z_LABELS = PATCH_SPEC.init_z_labels
X_STABILIZER_SCHEDULE = PATCH_SPEC.x_stabilizer_schedule
Z_STABILIZER_SCHEDULE = PATCH_SPEC.z_stabilizer_schedule
FULL_STABILIZER_SCHEDULE = PATCH_SPEC.full_stabilizer_schedule


def label2qubit(surface_qubits: Sequence[int]) -> dict[str, int]:
    """Map surface labels ``a..y`` to supplied Stim qubit indices."""
    return common.label2qubit(surface_qubits, surface_index=SURFACE_INDEX)


def draw_patch(ax: Axes | None = None) -> tuple[plt.Figure, Axes]:
    """Draw the shared rotated distance-5 patch geometry."""
    return common.draw_patch(PATCH_SPEC, ax)


# These two preparation helpers need no patch geometry, so expose them directly.
reset_to_zero = helper.reset_to_zero
prepare_product_plus = helper.prepare_product_plus


def append_scheduled_stabilizer_round(
    circuit: stim.Circuit,
    *,
    surface_qubits: Sequence[int],
    selected_stabilizer_names: Sequence[str],
    ancillas: Sequence[int],
    round_label: str,
    reset_ancillas_initially: bool = False,
    add_ticks: bool = True,
) -> tuple[int, ...]:
    """Measure selected rotated-d=5 stabilizers in the canonical schedule.

    For every position ``i``, ``selected_stabilizer_names[i]`` is measured
    with ``ancillas[i]`` and appears in the same Stim record order. X-only,
    Z-only, and mixed-basis subsets all retain their canonical rows and require
    at most four two-qubit layers.

    ``surface_qubits`` must contain at least 25 indices in ``SURFACE_ORDER``;
    names and used ancillas must be unique. An empty selection is valid. When
    ``add_ticks`` is true, layer separators and a trailing round separator are
    appended. ``round_label`` is conceptual because Stim ticks have no labels.
    Returns the round's Stim record positions for
    :func:`helper.append_stabilizer_detectors`.
    """
    selected_stabilizer_names = tuple(selected_stabilizer_names)
    if len(selected_stabilizer_names) != len(set(selected_stabilizer_names)):
        raise ValueError("selected_stabilizer_names must be unique.")
    selected_stabilizer_schedules = {
        name: FULL_STABILIZER_SCHEDULE[name] for name in selected_stabilizer_names
    }
    return helper.append_scheduled_stabilizer_round(
        circuit,
        label_to_qubit=label2qubit(surface_qubits),
        ancillas=ancillas,
        selected_stabilizer_schedules=selected_stabilizer_schedules,
        reset_ancillas_initially=reset_ancillas_initially,
        add_ticks=add_ticks,
        tick_label=f"{round_label} scheduled stabilizer round",
    )


def append_logical_measurement(
    circuit: stim.Circuit,
    basis: str = "Z",
    *,
    surface_qubits: Sequence[int],
    support: tuple[str, ...] | None = None,
    ancilla_qubit: int | None = None,
    destructive: bool = True,
    observable_index: int | None = None,
) -> None:
    """Append an X, Y, or Z logical measurement on the d=5 patch.

    ``surface_qubits`` contains 25 indices in ``SURFACE_ORDER``. ``support``
    overrides the default X or Z representative; for Y, omit it to measure
    ``i X_L Z_L`` from the distinct default supports. A destructive
    measurement records one result per term. A non-destructive measurement
    requires ``ancilla_qubit`` and records one parity result. Set
    ``observable_index`` to include the result parity as a Stim observable.
    """
    helper.append_logical_measurement(
        circuit,
        basis,
        surface_qubits=surface_qubits,
        surface_index=SURFACE_INDEX,
        default_logical_x=DEFAULT_LOGICAL_X,
        default_logical_z=DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        destructive=destructive,
        observable_index=observable_index,
    )


def append_all_data_readout_detectors_and_observable(
    circuit: stim.Circuit,
    basis: str,
    *,
    final_stabilizer_measurement_positions: Mapping[str, int],
    data_measurement_positions: Mapping[str, int],
    logical_frame_measurement_positions: Sequence[int] = (),
    observable_index: int = 0,
) -> tuple[int, ...]:
    """Annotate a destructive d=5 X- or Z-basis all-data readout."""
    return helper.append_all_data_readout_detectors_and_observable(
        circuit,
        basis,
        x_stabilizers=X_STABILIZERS,
        z_stabilizers=Z_STABILIZERS,
        logical_x=DEFAULT_LOGICAL_X,
        logical_z=DEFAULT_LOGICAL_Z,
        final_stabilizer_measurement_positions=final_stabilizer_measurement_positions,
        data_measurement_positions=data_measurement_positions,
        logical_frame_measurement_positions=logical_frame_measurement_positions,
        observable_index=observable_index,
    )
