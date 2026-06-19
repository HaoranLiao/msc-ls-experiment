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


def _flatten_qiskit_bitstring(bitstring: str) -> str:
    return bitstring.replace(" ", "")


def qiskit_bit_value(bitstring: str, position: int) -> int:
    """Return a bit from a Qiskit counts key using Qiskit bit positions.

    Qiskit count keys are printed most-significant bit first, so classical bit
    position 0 is the rightmost bit after removing register-separating spaces.
    """
    bits = _flatten_qiskit_bitstring(bitstring)
    if position < 0 or position >= len(bits):
        raise ValueError(f"bit position {position} is outside bitstring {bitstring!r}.")
    return int(bits[len(bits) - 1 - position])


def parse_qiskit_bitstring(
    bitstring: str,
    labels: tuple[str, ...] | list[str],
) -> dict[str, int]:
    """Map a Qiskit bitstring to a ``{label: 0|1}`` dict.

    Qiskit returns measurement bitstrings in big-endian order (most
    significant bit first).  This reverses the string so that index 0
    corresponds to classical bit 0, then zips with *labels*.
    """
    flattened = _flatten_qiskit_bitstring(bitstring)
    if len(flattened) != len(labels):
        raise ValueError(
            f"bitstring {bitstring!r} has {len(flattened)} bits, "
            f"but {len(labels)} labels were provided."
        )
    return {
        label: qiskit_bit_value(flattened, position)
        for position, label in enumerate(labels)
    }
