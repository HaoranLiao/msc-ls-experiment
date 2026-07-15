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

from research_qec_haoran.surface_code import rotated_common_qiskit as rotated_common

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

Z_STABILIZER_SCHEDULE: dict[str, tuple[str, ...]] = {
    "Z_abde": ("a", "b", "d", "e"),
    "Z_cf": ("", "c", "", "f"),
    "Z_dg": ("", "d", "", "g"),
    "Z_efhi": ("e", "f", "h", "i"),
}

X_STABILIZER_SCHEDULE: dict[str, tuple[str, ...]] = {
    "X_ab": ("", "a", "b", ""),
    "X_bcef": ("b", "c", "e", "f"),
    "X_degh": ("d", "e", "g", "h"),
    "X_hi": ("h", "i", "", ""),
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


def _append_parallel_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    label_to_qubit,
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    schedules,
    basis: str,
    prepare_ancillas: bool,
    add_barriers: bool,
    barrier_label: str | None,
) -> None:
    rotated_common.append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=label_to_qubit,
        ancillas=ancillas,
        classical_bits=classical_bits,
        schedules=schedules,
        basis=basis,
        prepare_ancillas=prepare_ancillas,
        add_barriers=add_barriers,
        barrier_label=barrier_label,
    )


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


def prepare_plus(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Prepare a collection of qubits in ``|+>``."""
    rotated_common.prepare_plus(
        circuit,
        qubits=qubits,
        add_barriers=add_barriers,
        barrier_label=barrier_label,
    )


def append_z_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    classical_bits: Sequence[object],
    round_label: str,
    stabilizer_names: Sequence[str] | None = None,
    prepare_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one parallel rotated-``d=3`` Z-check round."""
    schedules = (
        Z_STABILIZER_SCHEDULE
        if stabilizer_names is None
        else {name: Z_STABILIZER_SCHEDULE[name] for name in stabilizer_names}
    )
    _append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=label2qubit(surface_qubits),
        ancillas=z_ancillas,
        classical_bits=classical_bits,
        schedules=schedules,
        basis="Z",
        prepare_ancillas=prepare_ancillas,
        add_barriers=add_barriers,
        barrier_label=f"{round_label} parallel Z checks",
    )


def append_x_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    x_ancillas: Sequence[object],
    classical_bits: Sequence[object],
    round_label: str,
    stabilizer_names: Sequence[str] | None = None,
    prepare_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one parallel rotated-``d=3`` X-check round."""
    schedules = (
        X_STABILIZER_SCHEDULE
        if stabilizer_names is None
        else {name: X_STABILIZER_SCHEDULE[name] for name in stabilizer_names}
    )
    _append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=label2qubit(surface_qubits),
        ancillas=x_ancillas,
        classical_bits=classical_bits,
        schedules=schedules,
        basis="X",
        prepare_ancillas=prepare_ancillas,
        add_barriers=add_barriers,
        barrier_label=f"{round_label} parallel X checks",
    )


def append_stabilizer_round(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    x_ancillas: Sequence[object],
    z_bits: Sequence[object],
    x_bits: Sequence[object],
    round_label: str,
    prepare_z_ancillas: bool = False,
    prepare_x_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one full rotated-``d=3`` stabilizer round."""
    append_z_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=z_ancillas,
        classical_bits=z_bits,
        round_label=f"{round_label} Z",
        prepare_ancillas=prepare_z_ancillas,
        add_barriers=add_barriers,
    )
    append_x_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        x_ancillas=x_ancillas,
        classical_bits=x_bits,
        round_label=f"{round_label} X",
        prepare_ancillas=prepare_x_ancillas,
        add_barriers=add_barriers,
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
        Which logical to measure: ``"Z"`` for Z_L or ``"X"`` for X_L.
    surface_qubits
        Nine data qubits ordered according to ``SURFACE_ORDER``.
    support
        Data-qubit labels for the logical representative to measure. Defaults
        to ``DEFAULT_LOGICAL_Z`` or ``DEFAULT_LOGICAL_X``.
    ancilla_qubit
        For non-destructive measurement, the ancilla qubit to use. Required
        when ``destructive=False``.
    classical_bits
        Classical bits for measurement results. For destructive: one per
        support qubit. For non-destructive: at least one bit.
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
