"""Rotated distance-3 surface-code patch: layout, geometry, and Qiskit circuit helpers.

Layout data (coordinates, stabilizers, logical operators, drawing) and the
circuit API (preparation, stabilizer rounds, fault injection) live together
because they describe the same d=3 patch.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from qiskit import QuantumCircuit

from magic_state_cultivation.surface_code import rotated_common_qiskit as rotated_common

# ---------------------------------------------------------------------------
# Patch geometry
# ---------------------------------------------------------------------------

SURFACE_ORDER: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h", "i")
SURFACE_INDEX: dict[str, int] = {label: index for index, label in enumerate(SURFACE_ORDER)}

SURFACE_COORDS: dict[str, tuple[int, int]] = {
    "a": (0, 0),
    "b": (1, 0),
    "c": (2, 0),
    "d": (0, 1),
    "e": (1, 1),
    "f": (2, 1),
    "g": (0, 2),
    "h": (1, 2),
    "i": (2, 2),
}

X_STABILIZERS: dict[str, tuple[str, ...]] = {
    "X_ab": ("a", "b"),
    "X_bcef": ("b", "c", "e", "f"),
    "X_degh": ("d", "e", "g", "h"),
    "X_hi": ("h", "i"),
}

Z_STABILIZERS: dict[str, tuple[str, ...]] = {
    "Z_abde": ("a", "b", "d", "e"),
    "Z_cf": ("c", "f"),
    "Z_dg": ("d", "g"),
    "Z_efhi": ("e", "f", "h", "i"),
}

X_STABILIZER_CENTERS: dict[str, tuple[float, float]] = {
    "X_ab": (0.5, 0.0),
    "X_bcef": (1.5, 0.5),
    "X_degh": (0.5, 1.5),
    "X_hi": (1.5, 2.0),
}

Z_STABILIZER_CENTERS: dict[str, tuple[float, float]] = {
    "Z_abde": (0.5, 0.5),
    "Z_cf": (2.0, 0.5),
    "Z_dg": (0.0, 1.5),
    "Z_efhi": (1.5, 1.5),
}

DEFAULT_LOGICAL_Z: tuple[str, ...] = ("a", "b", "c")
DEFAULT_LOGICAL_X: tuple[str, ...] = ("a", "d", "g")

# All logical-Z representatives: DEFAULT_LOGICAL_Z × products of Z-stabilizers.
LOGICAL_Z_REPRESENTATIVES: tuple[tuple[str, ...], ...] = (
    ("a", "b", "c"),
    ("a", "b", "f"),
    ("a", "b", "d", "e", "g", "h", "i"),
    ("a", "b", "d", "f", "g"),
    ("a", "b", "e", "h", "i"),
    ("c", "d", "e"),
    ("c", "e", "g"),
    ("c", "d", "f", "h", "i"),
    ("c", "f", "g", "h", "i"),
    ("d", "e", "f"),
    ("d", "h", "i"),
    ("e", "f", "g"),
    ("g", "h", "i"),
    ("a", "b", "c", "d", "g"),
    ("a", "b", "c", "e", "f", "h", "i"),
    ("a", "b", "c", "d", "e", "f", "g", "h", "i"),
)

# All logical-X representatives: DEFAULT_LOGICAL_X × products of X-stabilizers.
LOGICAL_X_REPRESENTATIVES: tuple[tuple[str, ...], ...] = (
    ("a", "d", "g"),
    ("a", "e", "h"),
    ("a", "e", "i"),
    ("a", "d", "g", "h", "i"),
    ("b", "d", "g"),
    ("b", "e", "h"),
    ("b", "e", "i"),
    ("b", "d", "g", "h", "i"),
    ("c", "d", "e", "f", "g"),
    ("c", "f", "h"),
    ("c", "f", "i"),
    ("c", "d", "e", "f", "g", "h", "i"),
    ("a", "b", "c", "d", "e", "f", "g"),
    ("a", "b", "c", "f", "h"),
    ("a", "b", "c", "f", "i"),
    ("a", "b", "c", "d", "e", "f", "g", "h", "i"),
)


# ---------------------------------------------------------------------------
# Stabilizer schedules and circuit constants
# ---------------------------------------------------------------------------

INIT_Z_LABELS: tuple[str, ...] = tuple(f"init_{name.lower()}" for name in Z_STABILIZERS)

Z_STABILIZER_SCHEDULE = rotated_common.build_full_canonical_stabilizer_schedule(
    stabilizers=Z_STABILIZERS,
    surface_coords=SURFACE_COORDS,
    basis="Z",
)

X_STABILIZER_SCHEDULE = rotated_common.build_full_canonical_stabilizer_schedule(
    stabilizers=X_STABILIZERS,
    surface_coords=SURFACE_COORDS,
    basis="X",
)
FULL_STABILIZER_SCHEDULE = {
    **Z_STABILIZER_SCHEDULE,
    **X_STABILIZER_SCHEDULE,
}


# ---------------------------------------------------------------------------
# Label / qubit mapping
# ---------------------------------------------------------------------------


def label2qubit(surface_qubits) -> dict[str, object]:
    """Map surface labels ``a..i`` to the supplied Qiskit qubits."""
    return rotated_common.label2qubit(
        surface_qubits,
        surface_index=SURFACE_INDEX,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_rotated_stabilizer(
    ax: Axes,
    name: str,
    support: tuple[str, ...],
    center: tuple[float, float],
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    x, y = center
    ax.add_patch(
        Rectangle(
            (x - 0.32, y - 0.32),
            0.64,
            0.64,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
            alpha=0.45,
            zorder=1,
        )
    )
    for label in support:
        dx, dy = SURFACE_COORDS[label]
        ax.plot([x, dx], [y, dy], color=edgecolor, linewidth=1.6, zorder=1)
    ax.text(x, y, name, ha="center", va="center", fontsize=10, weight="bold", zorder=2)


def draw_patch(ax: Axes | None = None) -> tuple[plt.Figure, Axes]:
    """Draw the rotated distance-3 surface-code patch with colored stabilizers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    for name, support in Z_STABILIZERS.items():
        _draw_rotated_stabilizer(
            ax,
            name,
            support,
            Z_STABILIZER_CENTERS[name],
            facecolor="#F6D55C",
            edgecolor="#A57C00",
        )
    for name, support in X_STABILIZERS.items():
        _draw_rotated_stabilizer(
            ax,
            name,
            support,
            X_STABILIZER_CENTERS[name],
            facecolor="#7FDBFF",
            edgecolor="#0B5D7A",
        )

    for index, label in enumerate(SURFACE_ORDER):
        x, y = SURFACE_COORDS[label]
        ax.scatter(x, y, s=260, color="#222222", edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(x, y + 0.18, f"{label} (d{index})", ha="center", va="bottom", fontsize=9)

    ax.set_title("Distance-3 Rotated Surface-Code Patch")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(2.6, -0.6)
    return fig, ax


# ---------------------------------------------------------------------------
# Circuit-building primitives
# ---------------------------------------------------------------------------


def reset_to_zero(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Reset a collection of qubits to ``|0>``."""
    rotated_common.reset_to_zero(
        circuit,
        qubits=qubits,
        add_barriers=add_barriers,
        barrier_label=barrier_label,
    )


def prepare_product_plus(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Prepare a collection of qubits in ``|+>``."""
    rotated_common.prepare_product_plus(
        circuit,
        qubits=qubits,
        add_barriers=add_barriers,
        barrier_label=barrier_label,
    )


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
    """Append selected rotated-``d=3`` stabilizers in the canonical schedule.

    For every position ``i``, ``selected_stabilizer_names[i]`` is measured
    with ``ancillas[i]`` and stored in ``classical_bits[i]``. The selection may
    contain only X checks, only Z checks, or any mixed-basis subset. Every
    selected check retains its row from ``FULL_STABILIZER_SCHEDULE``, so any
    valid selection has at most four two-qubit layers.

    Call structure::

        FULL_STABILIZER_SCHEDULE
        └── selected_stabilizer_names
            └── selected_stabilizer_schedules
                └── rotated_common.append_scheduled_stabilizer_round

    Parameters
    ----------
    circuit
        Circuit to modify.
    surface_qubits
        At least nine data qubits ordered according to ``SURFACE_ORDER``.
    selected_stabilizer_names
        Unique names selected from ``FULL_STABILIZER_SCHEDULE``. Order defines
        the positional alignment with ``ancillas`` and ``classical_bits``.
    ancillas
        Distinct measurement ancillas, one per selected stabilizer. Additional
        ancillas are untouched.
    classical_bits
        Result bits, one per selected stabilizer. Additional bits are untouched.
    round_label
        Human-readable label used when appending the optional trailing barrier.
    reset_ancillas_initially
        If true, reset every used ancilla to ``|0>`` before the round. If false,
        every used ancilla must already be in ``|0>``.
    add_barriers
        Whether to append one labeled barrier after all measurements.

    Raises
    ------
    KeyError
        If a selected stabilizer name is not in ``FULL_STABILIZER_SCHEDULE``.
    ValueError
        If selected names or used ancillas are repeated, too few ancillas or
        classical bits are supplied, or the selected schedules collide.

    Notes
    -----
    An empty selection is valid and appends no gates or measurements. With
    ``add_barriers=True``, it still appends the requested trailing barrier.
    """
    selected_stabilizer_names = tuple(selected_stabilizer_names)
    if len(selected_stabilizer_names) != len(set(selected_stabilizer_names)):
        raise ValueError("selected_stabilizer_names must be unique.")

    selected_stabilizer_schedules = {
        name: FULL_STABILIZER_SCHEDULE[name] for name in selected_stabilizer_names
    }
    rotated_common.append_scheduled_stabilizer_round(
        circuit,
        label_to_qubit=label2qubit(surface_qubits),
        ancillas=ancillas,
        classical_bits=classical_bits,
        selected_stabilizer_schedules=selected_stabilizer_schedules,
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
    """Append a logical measurement on the rotated surface-code patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Which logical Pauli to measure: ``"X"``, ``"Y"``, or ``"Z"``.
    surface_qubits
        Nine data qubits ordered according to ``SURFACE_ORDER``.
    support
        Data-qubit labels for the logical representative to measure. Defaults
        to ``DEFAULT_LOGICAL_Z`` or ``DEFAULT_LOGICAL_X`` for Z or X. For Y,
        omit this argument to measure ``i X_L Z_L`` from the distinct default
        X and Z supports.
    ancilla_qubit
        For non-destructive measurement, the ancilla qubit to use. Required
        when ``destructive=False``.
    classical_bits
        Classical bits for measurement results. For destructive: one per
        measured physical qubit (the union of the X and Z supports for Y).
        For non-destructive: at least one bit.
    destructive
        If True (default), directly measure the data qubits in the
        appropriate basis. If False, use an ancilla for indirect measurement.
    """
    rotated_common.append_logical_measurement(
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
