"""Rotated distance-5 surface-code patch: layout and Qiskit circuit helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from string import ascii_lowercase

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from qiskit import QuantumCircuit

from magic_state_cultivation.surface_code import rotated_common_qiskit as rotated_common

DISTANCE = 5
NUM_DATA_QUBITS = DISTANCE * DISTANCE
SURFACE_ORDER: tuple[str, ...] = tuple(ascii_lowercase[:NUM_DATA_QUBITS])
SURFACE_INDEX: dict[str, int] = {label: index for index, label in enumerate(SURFACE_ORDER)}
SURFACE_COORDS: dict[str, tuple[int, int]] = {
    label: (index % DISTANCE, index // DISTANCE) for label, index in SURFACE_INDEX.items()
}


def _label(row: int, col: int) -> str:
    return SURFACE_ORDER[row * DISTANCE + col]


def _support_name(basis: str, support: tuple[str, ...]) -> str:
    return f"{basis}_{''.join(support)}"


def _cell_support(row: int, col: int) -> tuple[str, ...]:
    return (
        _label(row, col),
        _label(row, col + 1),
        _label(row + 1, col),
        _label(row + 1, col + 1),
    )


def _build_x_stabilizers() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[float, float]]]:
    stabilizers: dict[str, tuple[str, ...]] = {}
    centers: dict[str, tuple[float, float]] = {}

    for col in range(0, DISTANCE - 1, 2):
        support = (_label(0, col), _label(0, col + 1))
        name = _support_name("X", support)
        stabilizers[name] = support
        centers[name] = (col + 0.5, 0.0)

    for row in range(DISTANCE - 1):
        for col in range(DISTANCE - 1):
            if (row + col) % 2 != 1:
                continue
            support = _cell_support(row, col)
            name = _support_name("X", support)
            stabilizers[name] = support
            centers[name] = (col + 0.5, row + 0.5)

    for col in range(1, DISTANCE - 1, 2):
        support = (_label(DISTANCE - 1, col), _label(DISTANCE - 1, col + 1))
        name = _support_name("X", support)
        stabilizers[name] = support
        centers[name] = (col + 0.5, float(DISTANCE - 1))

    return stabilizers, centers


def _build_z_stabilizers() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[float, float]]]:
    stabilizers: dict[str, tuple[str, ...]] = {}
    centers: dict[str, tuple[float, float]] = {}

    for row in range(DISTANCE - 1):
        for col in range(DISTANCE - 1):
            if (row + col) % 2 != 0:
                continue
            support = _cell_support(row, col)
            name = _support_name("Z", support)
            stabilizers[name] = support
            centers[name] = (col + 0.5, row + 0.5)

    for row in range(0, DISTANCE - 1, 2):
        support = (_label(row, DISTANCE - 1), _label(row + 1, DISTANCE - 1))
        name = _support_name("Z", support)
        stabilizers[name] = support
        centers[name] = (float(DISTANCE - 1), row + 0.5)

    for row in range(1, DISTANCE - 1, 2):
        support = (_label(row, 0), _label(row + 1, 0))
        name = _support_name("Z", support)
        stabilizers[name] = support
        centers[name] = (0.0, row + 0.5)

    return stabilizers, centers


X_STABILIZERS, X_STABILIZER_CENTERS = _build_x_stabilizers()
Z_STABILIZERS, Z_STABILIZER_CENTERS = _build_z_stabilizers()

DEFAULT_LOGICAL_Z: tuple[str, ...] = tuple(_label(0, col) for col in range(DISTANCE))
DEFAULT_LOGICAL_X: tuple[str, ...] = tuple(_label(row, 0) for row in range(DISTANCE))


def _representatives(
    default: tuple[str, ...],
    stabilizers: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    supports = [frozenset(support) for support in stabilizers.values()]
    generated: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for count in range(len(supports) + 1):
        for combo in combinations(supports, count):
            rep = frozenset(default)
            for support in combo:
                rep = rep.symmetric_difference(support)
            if rep in seen:
                continue
            seen.add(rep)
            generated.append(tuple(label for label in SURFACE_ORDER if label in rep))
    return tuple(generated)


LOGICAL_Z_REPRESENTATIVES = _representatives(DEFAULT_LOGICAL_Z, Z_STABILIZERS)
LOGICAL_X_REPRESENTATIVES = _representatives(DEFAULT_LOGICAL_X, X_STABILIZERS)

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


def label2qubit(surface_qubits) -> dict[str, object]:
    """Map surface labels ``a..y`` to the supplied Qiskit qubits."""
    return rotated_common.label2qubit(surface_qubits, surface_index=SURFACE_INDEX)


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
    ax.text(x, y, name, ha="center", va="center", fontsize=8, weight="bold", zorder=2)


def draw_patch(ax: Axes | None = None) -> tuple[plt.Figure, Axes]:
    """Draw the rotated distance-5 surface-code patch with colored stabilizers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
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
        ax.scatter(x, y, s=220, color="#222222", edgecolors="white", linewidths=1.1, zorder=3)
        ax.text(x, y + 0.18, f"{label} (d{index})", ha="center", va="bottom", fontsize=8)

    ax.set_title("Distance-5 Rotated Surface-Code Patch")
    ax.set_xticks(range(DISTANCE))
    ax.set_yticks(range(DISTANCE))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, DISTANCE - 0.4)
    ax.set_ylim(DISTANCE - 0.4, -0.6)
    return fig, ax


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
    """Append selected rotated-``d=5`` stabilizers in the canonical schedule.

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
        At least 25 data qubits ordered according to ``SURFACE_ORDER``.
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
    """Append a logical measurement on the rotated distance-5 patch."""
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
