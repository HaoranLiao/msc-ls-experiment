"""Rotated distance-3 surface-code Qiskit API over shared patch geometry.

Coordinates, stabilizers, logical operators, drawing metadata, and canonical
schedules are owned by ``common``. This module preserves the public
d=3 preparation, scheduled-round, logical-measurement, drawing, and label
mapping API while delegating Qiskit circuit primitives to
``helper_qiskit``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from qiskit import QuantumCircuit

from msc.rotated_surface_code import common
from msc.rotated_surface_code import helper_qiskit as helper

PATCH_SPEC = common.D3_PATCH_SPEC
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


def label2qubit(surface_qubits: Sequence[object]) -> dict[str, object]:
    """Map surface labels ``a..i`` to supplied Qiskit qubits."""
    return common.label2qubit(surface_qubits, surface_index=SURFACE_INDEX)


def draw_patch(ax: Axes | None = None) -> tuple[plt.Figure, Axes]:
    """Draw the shared rotated distance-3 patch geometry."""
    return common.draw_patch(PATCH_SPEC, ax)


# These two preparation helpers need no patch geometry, so expose them directly.
reset_to_zero = helper.reset_to_zero
prepare_product_plus = helper.prepare_product_plus


def append_scheduled_stabilizer_round(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    selected_stabilizer_names: Sequence[str],
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    round_label: str,
    reset_ancillas_initially: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append selected rotated-d=3 stabilizers in the canonical schedule.

    For every position ``i``, ``selected_stabilizer_names[i]`` is measured
    with ``ancillas[i]`` and stored in ``classical_bits[i]``. The selection may
    contain only X checks, only Z checks, or any mixed-basis subset. Every
    selected check retains its row from ``FULL_STABILIZER_SCHEDULE``, so every
    valid selection has at most four two-qubit layers.

    Parameters
    ----------
    circuit
        Circuit to modify.
    surface_qubits
        At least nine data qubits ordered according to ``SURFACE_ORDER``.
    selected_stabilizer_names
        Unique names from ``FULL_STABILIZER_SCHEDULE``. Their order aligns
        positionally with ``ancillas`` and ``classical_bits``.
    ancillas, classical_bits
        One distinct ancilla and one result bit per selected stabilizer;
        additional entries are untouched.
    round_label
        Human-readable label for the optional trailing barrier.
    reset_ancillas_initially
        Whether to reset every used ancilla to ``|0>`` before the round.
    add_barriers
        Whether to append one labeled barrier after all measurements.

    Raises
    ------
    KeyError
        If a selected name is absent from ``FULL_STABILIZER_SCHEDULE``.
    ValueError
        If names or used ancillas repeat, inputs are too short, or schedules
        collide.

    Notes
    -----
    An empty selection appends no gates or measurements. With
    ``add_barriers=True``, it still appends the requested trailing barrier.
    """
    selected_stabilizer_names = tuple(selected_stabilizer_names)
    if len(selected_stabilizer_names) != len(set(selected_stabilizer_names)):
        raise ValueError("selected_stabilizer_names must be unique.")
    helper.append_scheduled_stabilizer_round(
        circuit,
        label_to_qubit=label2qubit(surface_qubits),
        ancillas=ancillas,
        classical_bits=classical_bits,
        selected_stabilizer_schedules={
            name: FULL_STABILIZER_SCHEDULE[name] for name in selected_stabilizer_names
        },
        reset_ancillas_initially=reset_ancillas_initially,
        add_barriers=add_barriers,
        barrier_label=f"{round_label} scheduled stabilizer round",
    )


def append_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Z",
    *,
    surface_qubits: Sequence[object],
    support: tuple[str, ...] | None = None,
    ancilla_qubit=None,
    classical_bits: Sequence[object],
    destructive: bool = True,
) -> None:
    """Append an X, Y, or Z logical measurement on the d=3 patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Logical Pauli ``"X"``, ``"Y"``, or ``"Z"``.
    surface_qubits
        Nine data qubits ordered according to ``SURFACE_ORDER``.
    support
        Optional logical representative labels. This overrides the default for
        X or Z. For Y, omit it to measure ``i X_L Z_L`` from the distinct
        default X and Z supports.
    ancilla_qubit
        Ancilla required for non-destructive measurement.
    classical_bits
        One result bit per measured physical term when destructive, or at
        least one bit when non-destructive.
    destructive
        Whether to measure data qubits directly instead of an ancilla parity.
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
        classical_bits=classical_bits,
        destructive=destructive,
    )
