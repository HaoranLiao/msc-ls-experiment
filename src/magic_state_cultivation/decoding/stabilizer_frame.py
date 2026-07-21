"""Classical tracker for stabilizer measurement references.

In repeated syndrome extraction, the meaningful quantity is often not the raw
stabilizer measurement but the *change* relative to a previously recorded
reference. This is the preparation-boundary logic used in surface-code memory
circuits, generalized into a reusable object.

Unlike :class:`PauliFrame`, this class does not track logical byproducts. It
tracks the most recently recorded measurement bit for each stabilizer label and
computes XOR-relative changes against future measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from magic_state_cultivation.utils.normalize import normalize_bit


@dataclass
class StabilizerFrame:
    """Track previously recorded stabilizer measurement bits.

    Each stabilizer is identified by a string label such as ``"Z_abde"`` or
    ``"init_z_abde"``. The frame stores the last recorded bit per label.

    The main operation is :meth:`advance`, which compares a new measurement
    round against the stored reference, returns the XOR-relative changes, and
    updates the stored reference to the new round.
    """

    _bits: dict[str, int] = field(default_factory=dict)

    def register(self, label: str, bit: int | bool = 0) -> None:
        """Register or overwrite ``label`` with a reference bit."""
        self._bits[label] = normalize_bit(bit)

    def record(self, label: str, bit: int | bool) -> None:
        """Record the current reference bit for ``label``."""
        self._bits[label] = normalize_bit(bit)

    def record_many(self, measurements: Mapping[str, int | bool]) -> None:
        """Record a batch of stabilizer reference bits."""
        for label, bit in measurements.items():
            self.record(label, bit)

    def current(self, label: str) -> int:
        """Return the currently stored reference bit for ``label``."""
        return self._bits[label]

    def compare(self, label: str, measured_bit: int | bool, *, update: bool = False) -> int:
        """Compare a new bit against the stored reference for ``label``.

        Returns the XOR-relative change bit. If ``update`` is True, the stored
        reference is advanced to ``measured_bit``.
        """
        bit = normalize_bit(measured_bit)
        delta = self._bits[label] ^ bit
        if update:
            self._bits[label] = bit
        return delta

    def compare_many(
        self,
        measurements: Mapping[str, int | bool],
        *,
        update: bool = False,
    ) -> dict[str, int]:
        """Return XOR-relative changes for a batch of measurements."""
        return {
            label: self.compare(label, bit, update=update) for label, bit in measurements.items()
        }

    def advance(self, measurements: Mapping[str, int | bool]) -> dict[str, int]:
        """Compare a new round against the stored reference and update it."""
        return self.compare_many(measurements, update=True)

    def reset(self, label: str, bit: int | bool = 0) -> None:
        """Reset ``label`` to a chosen reference bit, defaulting to 0."""
        self._bits[label] = normalize_bit(bit)

    def as_dict(self) -> dict[str, int]:
        """Return the stored reference bits."""
        return dict(self._bits)

    def __repr__(self) -> str:
        entries = ", ".join(f"{label}: {bit}" for label, bit in sorted(self._bits.items()))
        return f"StabilizerFrame({{{entries}}})"
