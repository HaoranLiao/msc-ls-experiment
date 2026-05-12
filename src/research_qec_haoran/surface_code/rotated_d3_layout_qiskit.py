from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

ROTATED_D3_SURFACE_ORDER: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h", "i")
ROTATED_D3_SURFACE_INDEX: dict[str, int] = {
    label: index
    for index, label in enumerate(ROTATED_D3_SURFACE_ORDER)
}

ROTATED_D3_SURFACE_COORDS: dict[str, tuple[int, int]] = {
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

ROTATED_D3_X_STABILIZERS: dict[str, tuple[str, ...]] = {
    "X_ab": ("a", "b"),
    "X_bcef": ("b", "c", "e", "f"),
    "X_degh": ("d", "e", "g", "h"),
    "X_hi": ("h", "i"),
}

ROTATED_D3_Z_STABILIZERS: dict[str, tuple[str, ...]] = {
    "Z_abde": ("a", "b", "d", "e"),
    "Z_cf": ("c", "f"),
    "Z_dg": ("d", "g"),
    "Z_efhi": ("e", "f", "h", "i"),
}

ROTATED_D3_X_STABILIZER_CENTERS: dict[str, tuple[float, float]] = {
    "X_ab": (0.5, 0.0),
    "X_bcef": (1.5, 0.5),
    "X_degh": (0.5, 1.5),
    "X_hi": (1.5, 2.0),
}

ROTATED_D3_Z_STABILIZER_CENTERS: dict[str, tuple[float, float]] = {
    "Z_abde": (0.5, 0.5),
    "Z_cf": (2.0, 0.5),
    "Z_dg": (0.0, 1.5),
    "Z_efhi": (1.5, 1.5),
}

ROTATED_D3_LOGICAL_Z: tuple[str, ...] = ("a", "b", "c")
ROTATED_D3_LOGICAL_X: tuple[str, ...] = ("a", "d", "g")


def rotated_d3_surface_label_map(surface_qubits) -> dict[str, object]:
    """Map surface labels ``a..i`` to the supplied Qiskit qubits."""
    if len(surface_qubits) != 9:
        raise ValueError("surface_qubits must contain exactly 9 qubits.")
    return {
        label: surface_qubits[index]
        for label, index in ROTATED_D3_SURFACE_INDEX.items()
    }


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
        dx, dy = ROTATED_D3_SURFACE_COORDS[label]
        ax.plot([x, dx], [y, dy], color=edgecolor, linewidth=1.6, zorder=1)
    ax.text(x, y, name, ha="center", va="center", fontsize=10, weight="bold", zorder=2)


def draw_rotated_d3_patch(ax: Axes | None = None) -> tuple[plt.Figure, Axes]:
    """Draw the rotated distance-3 surface-code patch with colored stabilizers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    for name, support in ROTATED_D3_Z_STABILIZERS.items():
        _draw_rotated_stabilizer(
            ax,
            name,
            support,
            ROTATED_D3_Z_STABILIZER_CENTERS[name],
            facecolor="#F6D55C",
            edgecolor="#A57C00",
        )
    for name, support in ROTATED_D3_X_STABILIZERS.items():
        _draw_rotated_stabilizer(
            ax,
            name,
            support,
            ROTATED_D3_X_STABILIZER_CENTERS[name],
            facecolor="#7FDBFF",
            edgecolor="#0B5D7A",
        )

    for index, label in enumerate(ROTATED_D3_SURFACE_ORDER):
        x, y = ROTATED_D3_SURFACE_COORDS[label]
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
