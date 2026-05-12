from __future__ import annotations


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
