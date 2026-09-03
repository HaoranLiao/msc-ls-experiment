"""Framework-neutral geometry and schedules for rotated surface-code patches.

This module is the single source of truth for d=3 and d=5 data-qubit labels,
coordinates, stabilizers, logical representatives, drawing metadata, and the
canonical four-step measurement schedules. Framework-specific gate appenders
live in ``helper_qiskit`` and ``helper_stim``; the public
distance modules bind those helpers to the corresponding shared patch spec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from string import ascii_lowercase
from typing import TypeVar

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

_QubitT = TypeVar("_QubitT")

# ---------------------------------------------------------------------------
# Label mapping and canonical schedules
# ---------------------------------------------------------------------------


def label2qubit(
    surface_qubits: Sequence[_QubitT],
    *,
    surface_index: Mapping[str, int],
) -> dict[str, _QubitT]:
    """Map surface labels to framework-specific qubit objects or indices."""
    if len(surface_qubits) < len(surface_index):
        raise ValueError(f"surface_qubits must contain at least {len(surface_index)} qubits.")
    return {label: surface_qubits[index] for label, index in surface_index.items()}


def resolve_selected_stabilizer_schedules(
    *,
    label_to_qubit: Mapping[str, _QubitT],
    selected_stabilizer_schedules: Mapping[str, tuple[str, ...]],
    ancillas: Sequence[_QubitT],
) -> tuple[tuple[tuple[_QubitT | None, ...], ...], tuple[_QubitT, ...]]:
    """Resolve selected label schedules and align their measurement ancillas.

    Mapping order defines row order. Every nonempty data label is replaced by
    its framework-specific qubit object or index; empty schedule positions
    become ``None``. Exactly the first required ancillas are returned, and too
    few ancillas are rejected.

    Data-flow structure::

        selected label schedules + label_to_qubit
        └── data-qubit schedules + aligned ancillas
            └── framework-specific scheduled parity measurements
    """
    num_checks = len(selected_stabilizer_schedules)
    if len(ancillas) < num_checks:
        raise ValueError(
            f"stabilizer round requires at least {num_checks} ancillas, got {len(ancillas)}."
        )
    data_qubit_schedules = tuple(
        tuple(label_to_qubit[label] if label else None for label in schedule)
        for schedule in selected_stabilizer_schedules.values()
    )
    return data_qubit_schedules, tuple(ancillas[:num_checks])


def build_full_canonical_stabilizer_schedule(
    *,
    stabilizers: Mapping[str, tuple[str, ...]],
    surface_coords: Mapping[str, tuple[int, int]],
    basis: str,
) -> dict[str, tuple[str, ...]]:
    """Build the full canonical four-step schedule for one stabilizer basis.

    X checks use ``NW, NE, SW, SE`` in the bulk, with top-boundary checks
    assigned to steps 3-4 and bottom-boundary checks to steps 1-2. Z checks use
    ``NW, SW, NE, SE`` in the bulk, with right-boundary checks assigned to
    steps 1-2 and left-boundary checks to steps 3-4. The completed schedule is
    checked for data-qubit collisions at every step.

    Data-flow structure::

        build_full_canonical_stabilizer_schedule
        └── full X or Z stabilizer schedule
            └── select any same-basis or mixed-basis subset
                └── framework-specific scheduled round

    Reference: ``CanonicalMeasurementScheduleSurfaceCode.png``.
    """
    if basis not in {"X", "Z"}:
        raise ValueError(f"unsupported stabilizer basis {basis!r}")
    if not surface_coords:
        raise ValueError("surface_coords must not be empty.")

    min_x = min(x for x, _ in surface_coords.values())
    max_x = max(x for x, _ in surface_coords.values())
    min_y = min(y for _, y in surface_coords.values())
    max_y = max(y for _, y in surface_coords.values())
    full_stabilizer_schedule: dict[str, tuple[str, ...]] = {}

    for stabilizer_name, support in stabilizers.items():
        try:
            label_by_coord = {surface_coords[label]: label for label in support}
        except KeyError as error:
            raise ValueError(
                f"{stabilizer_name} contains unknown data-qubit label {error.args[0]!r}."
            ) from error

        if len(support) == 4:
            support_xs = {x for x, _ in label_by_coord}
            support_ys = {y for _, y in label_by_coord}
            if len(support_xs) != 2 or len(support_ys) != 2:
                raise ValueError(f"{stabilizer_name} is not a four-corner bulk check.")
            west, east = min(support_xs), max(support_xs)
            north, south = min(support_ys), max(support_ys)
            try:
                northwest = label_by_coord[(west, north)]
                northeast = label_by_coord[(east, north)]
                southwest = label_by_coord[(west, south)]
                southeast = label_by_coord[(east, south)]
            except KeyError as error:
                raise ValueError(f"{stabilizer_name} is not a four-corner bulk check.") from error
            full_stabilizer_schedule[stabilizer_name] = (
                (northwest, northeast, southwest, southeast)
                if basis == "X"
                else (northwest, southwest, northeast, southeast)
            )
            continue

        if len(support) != 2:
            raise ValueError(
                f"{stabilizer_name} must be a weight-2 boundary or weight-4 bulk check."
            )
        first, second = sorted(
            support,
            key=lambda label: (
                surface_coords[label][0] if basis == "X" else surface_coords[label][1]
            ),
        )
        first_x, first_y = surface_coords[first]
        second_x, second_y = surface_coords[second]
        if basis == "X":
            if first_y != second_y:
                raise ValueError(f"{stabilizer_name} is not a horizontal X-boundary check.")
            if first_y == min_y:
                full_stabilizer_schedule[stabilizer_name] = ("", "", first, second)
            elif first_y == max_y:
                full_stabilizer_schedule[stabilizer_name] = (first, second, "", "")
            else:
                raise ValueError(f"{stabilizer_name} is not on an X boundary.")
        elif basis == "Z":
            if first_x != second_x:
                raise ValueError(f"{stabilizer_name} is not a vertical Z-boundary check.")
            if first_x == max_x:
                full_stabilizer_schedule[stabilizer_name] = (first, second, "", "")
            elif first_x == min_x:
                full_stabilizer_schedule[stabilizer_name] = ("", "", first, second)
            else:
                raise ValueError(f"{stabilizer_name} is not on a Z boundary.")
        else:  # pragma: no cover - stabilizer labels provide only X/Z bases.
            raise ValueError(f"Unsupported stabilizer basis: {basis!r}.")

    for step in range(4):
        active_labels = [
            schedule[step] for schedule in full_stabilizer_schedule.values() if schedule[step]
        ]
        if len(active_labels) != len(set(active_labels)):
            raise ValueError(f"canonical schedule has a data-qubit collision at step {step + 1}.")
    return full_stabilizer_schedule


# ---------------------------------------------------------------------------
# Logical representatives and patch specifications
# ---------------------------------------------------------------------------


def logical_representatives(
    *,
    surface_order: tuple[str, ...],
    default: tuple[str, ...],
    stabilizers: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    """Generate one logical equivalence class from all stabilizer products.

    Representatives are returned in ``surface_order`` and deduplicated after
    symmetric-difference multiplication by every stabilizer subset.
    """
    supports = tuple(frozenset(support) for support in stabilizers.values())
    generated: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for count in range(len(supports) + 1):
        for selected_supports in combinations(supports, count):
            representative = frozenset(default)
            for support in selected_supports:
                representative = representative.symmetric_difference(support)
            if representative in seen:
                continue
            seen.add(representative)
            generated.append(tuple(label for label in surface_order if label in representative))
    return tuple(generated)


def repeated_round_labels(labels: Sequence[str], *, rounds: int) -> tuple[str, ...]:
    """Prefix an arbitrary number of repeated-round semantic labels."""
    if rounds < 0:
        raise ValueError("rounds must be nonnegative.")
    return tuple(
        f"round{round_index + 1}_{label}" for round_index in range(rounds) for label in labels
    )


@dataclass(frozen=True)
class RotatedPatchSpec:
    """Framework-neutral geometry, logical data, and schedules for one patch.

    One immutable specification owns the label order, drawing coordinates,
    stabilizer supports and centers, default logical representatives, their
    stabilizer-equivalent classes, and the canonical X/Z schedules. Both
    Qiskit and Stim public modules expose these same objects.
    """

    distance: int
    surface_order: tuple[str, ...]
    surface_index: dict[str, int]
    surface_coords: dict[str, tuple[int, int]]
    x_stabilizers: dict[str, tuple[str, ...]]
    z_stabilizers: dict[str, tuple[str, ...]]
    x_stabilizer_centers: dict[str, tuple[float, float]]
    z_stabilizer_centers: dict[str, tuple[float, float]]
    default_logical_x: tuple[str, ...]
    default_logical_z: tuple[str, ...]
    logical_x_representatives: tuple[tuple[str, ...], ...]
    logical_z_representatives: tuple[tuple[str, ...], ...]
    x_stabilizer_schedule: dict[str, tuple[str, ...]]
    z_stabilizer_schedule: dict[str, tuple[str, ...]]

    @property
    def init_z_labels(self) -> tuple[str, ...]:
        """Preparation-round labels in canonical surface-Z order."""
        return tuple(f"init_{name.lower()}" for name in self.z_stabilizers)

    @property
    def full_stabilizer_schedule(self) -> dict[str, tuple[str, ...]]:
        """Combined Z-then-X canonical stabilizer schedule."""
        return {**self.z_stabilizer_schedule, **self.x_stabilizer_schedule}


def _make_patch_spec(
    *,
    distance: int,
    surface_order: tuple[str, ...],
    surface_coords: dict[str, tuple[int, int]],
    x_stabilizers: dict[str, tuple[str, ...]],
    z_stabilizers: dict[str, tuple[str, ...]],
    x_stabilizer_centers: dict[str, tuple[float, float]],
    z_stabilizer_centers: dict[str, tuple[float, float]],
    default_logical_x: tuple[str, ...],
    default_logical_z: tuple[str, ...],
    logical_x_representatives: tuple[tuple[str, ...], ...] | None = None,
    logical_z_representatives: tuple[tuple[str, ...], ...] | None = None,
) -> RotatedPatchSpec:
    surface_index = {label: index for index, label in enumerate(surface_order)}
    return RotatedPatchSpec(
        distance=distance,
        surface_order=surface_order,
        surface_index=surface_index,
        surface_coords=surface_coords,
        x_stabilizers=x_stabilizers,
        z_stabilizers=z_stabilizers,
        x_stabilizer_centers=x_stabilizer_centers,
        z_stabilizer_centers=z_stabilizer_centers,
        default_logical_x=default_logical_x,
        default_logical_z=default_logical_z,
        logical_x_representatives=(
            logical_x_representatives
            if logical_x_representatives is not None
            else logical_representatives(
                surface_order=surface_order,
                default=default_logical_x,
                stabilizers=x_stabilizers,
            )
        ),
        logical_z_representatives=(
            logical_z_representatives
            if logical_z_representatives is not None
            else logical_representatives(
                surface_order=surface_order,
                default=default_logical_z,
                stabilizers=z_stabilizers,
            )
        ),
        x_stabilizer_schedule=build_full_canonical_stabilizer_schedule(
            stabilizers=x_stabilizers,
            surface_coords=surface_coords,
            basis="X",
        ),
        z_stabilizer_schedule=build_full_canonical_stabilizer_schedule(
            stabilizers=z_stabilizers,
            surface_coords=surface_coords,
            basis="Z",
        ),
    )


def _build_d3_patch_spec() -> RotatedPatchSpec:
    surface_order = tuple("abcdefghi")
    surface_coords = {label: (index % 3, index // 3) for index, label in enumerate(surface_order)}
    x_stabilizers = {
        "X_ab": ("a", "b"),
        "X_bcef": ("b", "c", "e", "f"),
        "X_degh": ("d", "e", "g", "h"),
        "X_hi": ("h", "i"),
    }
    z_stabilizers = {
        "Z_abde": ("a", "b", "d", "e"),
        "Z_cf": ("c", "f"),
        "Z_dg": ("d", "g"),
        "Z_efhi": ("e", "f", "h", "i"),
    }
    # All logical-Z representatives: default logical Z times Z-stabilizer products.
    logical_z_representatives = (
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
    # All logical-X representatives: default logical X times X-stabilizer products.
    logical_x_representatives = (
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
    return _make_patch_spec(
        distance=3,
        surface_order=surface_order,
        surface_coords=surface_coords,
        x_stabilizers=x_stabilizers,
        z_stabilizers=z_stabilizers,
        x_stabilizer_centers={
            "X_ab": (0.5, 0.0),
            "X_bcef": (1.5, 0.5),
            "X_degh": (0.5, 1.5),
            "X_hi": (1.5, 2.0),
        },
        z_stabilizer_centers={
            "Z_abde": (0.5, 0.5),
            "Z_cf": (2.0, 0.5),
            "Z_dg": (0.0, 1.5),
            "Z_efhi": (1.5, 1.5),
        },
        default_logical_x=("a", "d", "g"),
        default_logical_z=("a", "b", "c"),
        logical_x_representatives=logical_x_representatives,
        logical_z_representatives=logical_z_representatives,
    )


def _build_d5_patch_spec() -> RotatedPatchSpec:
    distance = 5
    surface_order = tuple(ascii_lowercase[: distance * distance])

    def label(row: int, col: int) -> str:
        return surface_order[row * distance + col]

    def support_name(basis: str, support: tuple[str, ...]) -> str:
        return f"{basis}_{''.join(support)}"

    def cell_support(row: int, col: int) -> tuple[str, ...]:
        return (
            label(row, col),
            label(row, col + 1),
            label(row + 1, col),
            label(row + 1, col + 1),
        )

    x_stabilizers: dict[str, tuple[str, ...]] = {}
    x_centers: dict[str, tuple[float, float]] = {}
    for col in range(0, distance - 1, 2):
        support = (label(0, col), label(0, col + 1))
        name = support_name("X", support)
        x_stabilizers[name] = support
        x_centers[name] = (col + 0.5, 0.0)
    for row in range(distance - 1):
        for col in range(distance - 1):
            if (row + col) % 2 != 1:
                continue
            support = cell_support(row, col)
            name = support_name("X", support)
            x_stabilizers[name] = support
            x_centers[name] = (col + 0.5, row + 0.5)
    for col in range(1, distance - 1, 2):
        support = (label(distance - 1, col), label(distance - 1, col + 1))
        name = support_name("X", support)
        x_stabilizers[name] = support
        x_centers[name] = (col + 0.5, float(distance - 1))

    z_stabilizers: dict[str, tuple[str, ...]] = {}
    z_centers: dict[str, tuple[float, float]] = {}
    for row in range(distance - 1):
        for col in range(distance - 1):
            if (row + col) % 2 != 0:
                continue
            support = cell_support(row, col)
            name = support_name("Z", support)
            z_stabilizers[name] = support
            z_centers[name] = (col + 0.5, row + 0.5)
    for row in range(0, distance - 1, 2):
        support = (label(row, distance - 1), label(row + 1, distance - 1))
        name = support_name("Z", support)
        z_stabilizers[name] = support
        z_centers[name] = (float(distance - 1), row + 0.5)
    for row in range(1, distance - 1, 2):
        support = (label(row, 0), label(row + 1, 0))
        name = support_name("Z", support)
        z_stabilizers[name] = support
        z_centers[name] = (0.0, row + 0.5)

    return _make_patch_spec(
        distance=distance,
        surface_order=surface_order,
        surface_coords={
            surface_label: (index % distance, index // distance)
            for index, surface_label in enumerate(surface_order)
        },
        x_stabilizers=x_stabilizers,
        z_stabilizers=z_stabilizers,
        x_stabilizer_centers=x_centers,
        z_stabilizer_centers=z_centers,
        default_logical_x=tuple(label(row, 0) for row in range(distance)),
        default_logical_z=tuple(label(0, col) for col in range(distance)),
    )


# ---------------------------------------------------------------------------
# Shared drawing
# ---------------------------------------------------------------------------


def _draw_rotated_stabilizer(
    ax: Axes,
    *,
    spec: RotatedPatchSpec,
    name: str,
    support: tuple[str, ...],
    center: tuple[float, float],
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
        data_x, data_y = spec.surface_coords[label]
        ax.plot([x, data_x], [y, data_y], color=edgecolor, linewidth=1.6, zorder=1)
    stabilizer_font_size = 10 if spec.distance == 3 else 8
    ax.text(
        x,
        y,
        name,
        ha="center",
        va="center",
        fontsize=stabilizer_font_size,
        weight="bold",
        zorder=2,
    )


def draw_patch(
    spec: RotatedPatchSpec,
    ax: Axes | None = None,
) -> tuple[plt.Figure, Axes]:
    """Draw a rotated d=3 or d=5 patch with colored stabilizers and labels."""
    if ax is None:
        figure_size = 7 if spec.distance == 3 else 8
        figure, ax = plt.subplots(figsize=(figure_size, figure_size))
    else:
        figure = ax.figure
    for name, support in spec.z_stabilizers.items():
        _draw_rotated_stabilizer(
            ax,
            spec=spec,
            name=name,
            support=support,
            center=spec.z_stabilizer_centers[name],
            facecolor="#F6D55C",
            edgecolor="#A57C00",
        )
    for name, support in spec.x_stabilizers.items():
        _draw_rotated_stabilizer(
            ax,
            spec=spec,
            name=name,
            support=support,
            center=spec.x_stabilizer_centers[name],
            facecolor="#7FDBFF",
            edgecolor="#0B5D7A",
        )
    marker_size = 260 if spec.distance == 3 else 220
    marker_line_width = 1.2 if spec.distance == 3 else 1.1
    data_label_font_size = 9 if spec.distance == 3 else 8
    for index, label in enumerate(spec.surface_order):
        x, y = spec.surface_coords[label]
        ax.scatter(
            x,
            y,
            s=marker_size,
            color="#222222",
            edgecolors="white",
            linewidths=marker_line_width,
            zorder=3,
        )
        ax.text(
            x,
            y + 0.18,
            f"{label} (d{index})",
            ha="center",
            va="bottom",
            fontsize=data_label_font_size,
        )
    ax.set_title(f"Distance-{spec.distance} Rotated Surface-Code Patch")
    ax.set_xticks(range(spec.distance))
    ax.set_yticks(range(spec.distance))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, spec.distance - 0.4)
    ax.set_ylim(spec.distance - 0.4, -0.6)
    return figure, ax


D3_PATCH_SPEC = _build_d3_patch_spec()
D5_PATCH_SPEC = _build_d5_patch_spec()
