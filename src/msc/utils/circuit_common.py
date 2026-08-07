"""Framework-neutral circuit semantics shared by Qiskit and Stim helpers."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

_SupportT = TypeVar("_SupportT", bound=Hashable)


def logical_y_parity_is_inverted(x_z_overlap: Sequence[object]) -> bool:
    """Return whether measuring the physical overlap-Y string reverses logical Y.

    Let ``S_X`` and ``S_Z`` be X-only and Z-only logical representatives for
    the same encoded qubit, and let ``m = len(S_X & S_Z)``. Each shared site
    contributes ``XZ = -ZX``, while operators on different sites commute, so
    ``X_L Z_L = (-1)**m Z_L X_L``. The representatives must therefore have
    odd overlap. The physical Pauli string representing ``i X_L Z_L`` acquires
    a minus sign exactly when ``m mod 4 == 3``.

    The default rotated-surface-code representatives have overlap one; common
    weight-three Steane representatives have overlap three.
    """
    if len(x_z_overlap) % 2 == 0:
        raise ValueError(
            "X-only logical-X and Z-only logical-Z representatives for the same "
            "encoded qubit must have odd support overlap so they anticommute."
        )
    return len(x_z_overlap) % 4 == 3


def resolve_logical_measurement_terms(
    basis: str,
    *,
    logical_x_support: tuple[_SupportT, ...],
    logical_z_support: tuple[_SupportT, ...],
) -> tuple[tuple[tuple[_SupportT, str], ...], bool]:
    """Resolve a normalized logical Pauli into ordered physical terms and sign.

    X and Z retain their supplied representative order. Y represents
    ``i X_L Z_L`` with Y on the X/Z overlap, X on X-only sites, and Z on
    Z-only sites. The returned boolean indicates whether physical measurement
    parity must be inverted to preserve the logical ``+1`` convention.

    Callers own Pauli-label normalization so a label is validated exactly once
    at the public circuit or protocol entry point.
    """
    if basis == "X":
        return tuple((support_item, "X") for support_item in logical_x_support), False
    if basis == "Z":
        return tuple((support_item, "Z") for support_item in logical_z_support), False

    logical_x_set = set(logical_x_support)
    logical_z_set = set(logical_z_support)
    if len(logical_x_set) != len(logical_x_support):
        raise ValueError("logical_x_support must not contain duplicate indices.")
    if len(logical_z_set) != len(logical_z_support):
        raise ValueError("logical_z_support must not contain duplicate indices.")

    x_z_overlap = tuple(
        support_item for support_item in logical_x_support if support_item in logical_z_set
    )
    invert_parity = logical_y_parity_is_inverted(x_z_overlap)
    terms = tuple(
        (support_item, "Y" if support_item in logical_z_set else "X")
        for support_item in logical_x_support
    ) + tuple(
        (support_item, "Z")
        for support_item in logical_z_support
        if support_item not in logical_x_set
    )
    return terms, invert_parity
