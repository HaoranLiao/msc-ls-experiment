"""General-purpose normalization helpers used across the repo."""

from __future__ import annotations


def normalize_bit(bit: int | bool) -> int:
    """Return *bit* as an integer ``0`` or ``1``, raising on invalid input."""
    if bit not in (0, 1, False, True):
        raise ValueError(f"bit must be 0 or 1, got {bit!r}")
    return int(bit)


def normalize_pauli_char(pauli: str | None) -> str | None:
    """Normalize a single-character Pauli label to upper case.

    Returns ``None`` unchanged.  Raises on anything outside
    ``{None, "I", "X", "Y", "Z"}``.
    """
    if pauli is None:
        return None
    normalized = pauli.upper()
    if normalized not in {"I", "X", "Y", "Z"}:
        raise ValueError(f"unsupported Pauli {pauli!r}")
    return normalized

