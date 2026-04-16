"""Distance-3 unrotated surface-code geometry and stabilizer layout."""

from __future__ import annotations

from dataclasses import dataclass

Coord = tuple[int, int]


@dataclass(frozen=True)
class StabilizerSpec:
    """A stabilizer and the data-qubit support it acts on."""

    name: str
    ancilla_coord: Coord
    data_coords: tuple[Coord, ...]


D3_UNROTATED_DATA_COORDS: tuple[Coord, ...] = (
    (0, 0),
    (0, 2),
    (0, 4),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 2),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 0),
    (4, 2),
    (4, 4),
)

_DATA_INDEX_BY_COORD = {coord: index for index, coord in enumerate(D3_UNROTATED_DATA_COORDS)}


def _neighbor_data_coords(ancilla_coord: Coord) -> tuple[Coord, ...]:
    """Return orthogonally adjacent data coordinates for an ancilla location."""

    row, col = ancilla_coord
    neighbors: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = (row + dr, col + dc)
        if candidate in _DATA_INDEX_BY_COORD:
            neighbors.append(candidate)
    return tuple(neighbors)


_D3_UNROTATED_X_ANCILLA_COORDS: tuple[Coord, ...] = (
    (0, 1),
    (0, 3),
    (2, 1),
    (2, 3),
    (4, 1),
    (4, 3),
)

_D3_UNROTATED_Z_ANCILLA_COORDS: tuple[Coord, ...] = (
    (1, 0),
    (1, 2),
    (1, 4),
    (3, 0),
    (3, 2),
    (3, 4),
)


D3_UNROTATED_X_STABILIZERS: tuple[StabilizerSpec, ...] = tuple(
    StabilizerSpec(
        name=f"X{i}",
        ancilla_coord=coord,
        data_coords=_neighbor_data_coords(coord),
    )
    for i, coord in enumerate(_D3_UNROTATED_X_ANCILLA_COORDS)
)

D3_UNROTATED_Z_STABILIZERS: tuple[StabilizerSpec, ...] = tuple(
    StabilizerSpec(
        name=f"Z{i}",
        ancilla_coord=coord,
        data_coords=_neighbor_data_coords(coord),
    )
    for i, coord in enumerate(_D3_UNROTATED_Z_ANCILLA_COORDS)
)
