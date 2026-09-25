"""General-purpose normalization helpers used across the repo."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar

import numpy as np

_QubitT = TypeVar("_QubitT")


def resolve_patch_qubits(
    patch_qubits: Sequence[_QubitT],
    *,
    expected_count: int,
) -> list[_QubitT]:
    """Require enough patch qubits and return the required prefix as a list."""
    if len(patch_qubits) < expected_count:
        raise ValueError(f"patch_qubits must contain at least {expected_count} qubits.")
    return list(patch_qubits[:expected_count])


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


def merge_counts(counts: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Sum any number of Qiskit-format counts dicts into one joint counts dict.

    Twirling splits a submitted circuit into several randomized variants, each
    run for a fraction of the requested shots; the platform returns one counts
    dict per variant instead of merging them, so downstream analysis only sees
    a fraction of the shot budget unless the variants are combined first. This
    sums matching bitstrings key-wise across all dicts; a single-entry input
    is returned unchanged.
    """
    merged: dict[str, int] = {}
    for counts_dict in counts:
        for bitstring, count in counts_dict.items():
            merged[bitstring] = merged.get(bitstring, 0) + count
    return merged


def qiskit_bit_value(bitstring: str, position: int) -> int:
    """Return a bit from a Qiskit counts key using Qiskit bit positions.

    Qiskit count keys are printed most-significant bit first, so classical bit
    position 0 is the rightmost bit after removing register-separating spaces.
    """
    bits = bitstring.replace(" ", "")
    if position < 0 or position >= len(bits):
        raise ValueError(f"bit position {position} is outside bitstring {bitstring!r}.")
    return int(bits[len(bits) - 1 - position])


def qiskit_bit_values(bitstrings: Sequence[str]) -> np.ndarray:
    """Vectorize :func:`qiskit_bit_value` over many count keys and every bit position.

    Decoding whole count dictionaries needs every bit of every key at once:
    row ``k`` holds ``bitstrings[k]`` and column ``i`` holds classical bit
    ``i``. Register-separating spaces are ignored, and every key must have the
    same width.
    """
    flattened = [bitstring.replace(" ", "") for bitstring in bitstrings]
    width = len(flattened[0]) if flattened else 0
    if any(len(bitstring) != width for bitstring in flattened):
        raise ValueError("every Qiskit counts key must have the same number of bits.")
    characters = np.frombuffer("".join(flattened).encode("ascii"), dtype=np.uint8)
    return characters.reshape(len(flattened), width)[:, ::-1] == ord("1")


def parse_qiskit_bitstring(
    bitstring: str,
    labels: Sequence[str],
) -> dict[str, int]:
    """Map a Qiskit bitstring to a ``{label: 0|1}`` dict.

    Qiskit returns measurement bitstrings in big-endian order (most
    significant bit first).  This reverses the string so that index 0
    corresponds to classical bit 0, then zips with *labels*.
    """
    flattened = bitstring.replace(" ", "")
    if len(flattened) != len(labels):
        raise ValueError(
            f"bitstring {bitstring!r} has {len(flattened)} bits, "
            f"but {len(labels)} labels were provided."
        )
    return dict(zip(labels, map(int, reversed(flattened)), strict=True))


def parse_qiskit_registers(
    bitstring: str,
    register_labels: Sequence[tuple[str, Sequence[str]]],
) -> dict[str, dict[str, int]]:
    """Parse a Qiskit counts key into named classical-register bit maps.

    ``register_labels`` must be in circuit construction order.  Qiskit prints
    multi-register count keys in reverse classical-register order; this helper
    handles both the usual space-separated key and an already-flattened key.
    """
    parts = bitstring.split()
    if len(parts) == len(register_labels):
        return {
            name: parse_qiskit_bitstring(chunk, labels)
            for (name, labels), chunk in zip(reversed(register_labels), parts, strict=True)
        }

    flattened = bitstring.replace(" ", "")
    expected_width = sum(len(labels) for _, labels in register_labels)
    if len(flattened) != expected_width:
        raise ValueError(
            f"bitstring {bitstring!r} has {len(flattened)} bits, "
            f"but register labels describe {expected_width} bits."
        )

    parsed: dict[str, dict[str, int]] = {}
    offset = len(flattened)
    for name, labels in register_labels:
        offset -= len(labels)
        parsed[name] = parse_qiskit_bitstring(flattened[offset : offset + len(labels)], labels)
    return parsed
