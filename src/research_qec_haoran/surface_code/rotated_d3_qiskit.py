"""Rotated distance-3 surface-code patch: layout, geometry, and Qiskit circuit helpers.

Layout data (coordinates, stabilizers, logical operators, drawing) and the
circuit API (preparation, stabilizer rounds, fault injection) live together
because they describe the same d=3 patch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Patch geometry
# ---------------------------------------------------------------------------

SURFACE_ORDER: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h", "i")
SURFACE_INDEX: dict[str, int] = {
    label: index
    for index, label in enumerate(SURFACE_ORDER)
}

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

LOGICAL_Z: tuple[str, ...] = ("a", "b", "c")
LOGICAL_X: tuple[str, ...] = ("a", "d", "g")


# ---------------------------------------------------------------------------
# Label / qubit mapping
# ---------------------------------------------------------------------------


def _label2qubit(surface_qubits) -> dict[str, object]:
    """Map surface labels ``a..i`` to the supplied Qiskit qubits."""
    if len(surface_qubits) != 9:
        raise ValueError("surface_qubits must contain exactly 9 qubits.")
    return {
        label: surface_qubits[index]
        for label, index in SURFACE_INDEX.items()
    }


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
# Stabilizer schedules and circuit constants
# ---------------------------------------------------------------------------

INIT_Z_LABELS: tuple[str, ...] = tuple(
    f"init_{name.lower()}" for name in Z_STABILIZERS
)

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
# Circuit-building primitives
# ---------------------------------------------------------------------------


def append_data_pauli(
    circuit: QuantumCircuit,
    surface_qubits: Sequence[object],
    pauli: str,
    qubit_label: str,
) -> None:
    """Append a physical Pauli gate on a named rotated-d3 data qubit."""
    qubit = surface_qubits[SURFACE_INDEX[qubit_label]]
    gate = pauli.upper()
    if gate == "X":
        circuit.x(qubit)
    elif gate == "Y":
        circuit.y(qubit)
    elif gate == "Z":
        circuit.z(qubit)
    else:
        raise ValueError(f"unsupported Pauli {pauli!r}")


def _validate_round_width(
    *,
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    expected: int,
    label: str,
) -> None:
    if len(ancillas) != expected:
        raise ValueError(f"{label} ancillas must contain {expected} qubits.")
    if len(classical_bits) != expected:
        raise ValueError(f"{label} classical bits must contain {expected} bits.")


def _append_parallel_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    label_to_qubit: Mapping[str, object],
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    schedules: Mapping[str, tuple[str, ...]],
    basis: str,
    prepare_ancillas: bool,
    add_barriers: bool,
    barrier_label: str | None,
) -> None:
    ancilla_list = list(ancillas)
    classical_list = list(classical_bits)
    stabilizer_names = tuple(schedules)
    _validate_round_width(
        ancillas=ancilla_list,
        classical_bits=classical_list,
        expected=len(stabilizer_names),
        label=f"{basis}-check",
    )

    if prepare_ancillas:
        append_prepare_zero_state(circuit, qubits=ancilla_list, add_barriers=False)

    if basis == "X":
        for ancilla in ancilla_list:
            circuit.h(ancilla)

    num_steps = len(next(iter(schedules.values())))
    for step in range(num_steps):
        for ancilla, stabilizer_name in zip(ancilla_list, stabilizer_names, strict=False):
            data_label = schedules[stabilizer_name][step]
            if not data_label:
                continue
            if basis == "Z":
                circuit.cx(label_to_qubit[data_label], ancilla)
            elif basis == "X":
                circuit.cx(ancilla, label_to_qubit[data_label])
            else:
                raise ValueError(f"unsupported stabilizer basis {basis!r}")

    if basis == "X":
        for ancilla in ancilla_list:
            circuit.h(ancilla)

    for ancilla, classical_bit in zip(ancilla_list, classical_list, strict=False):
        circuit.measure(ancilla, classical_bit)

    if add_barriers:
        circuit.barrier(label=barrier_label or f"parallel {basis} stabilizer round")


def append_prepare_zero_state(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Reset a collection of qubits to ``|0>``."""
    qubit_list = list(qubits)
    for qubit in qubit_list:
        circuit.reset(qubit)
    if add_barriers:
        circuit.barrier(label=barrier_label or "prepare |0>")


def append_prepare_plus_state(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Prepare a collection of qubits in ``|+>``."""
    qubit_list = list(qubits)
    for qubit in qubit_list:
        circuit.h(qubit)
    if add_barriers:
        circuit.barrier(label=barrier_label or "prepare |+>")


def append_z_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    classical_bits: Sequence[object],
    round_label: str,
    prepare_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one parallel rotated-``d=3`` Z-check round."""
    _append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=_label2qubit(surface_qubits),
        ancillas=z_ancillas,
        classical_bits=classical_bits,
        schedules=Z_STABILIZER_SCHEDULE,
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
    prepare_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one parallel rotated-``d=3`` X-check round."""
    _append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=_label2qubit(surface_qubits),
        ancillas=x_ancillas,
        classical_bits=classical_bits,
        schedules=X_STABILIZER_SCHEDULE,
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

