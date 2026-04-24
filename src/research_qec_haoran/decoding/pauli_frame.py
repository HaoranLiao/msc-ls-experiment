"""Pauli frame tracker for logical qubits.

In a fault-tolerant quantum computer, the Pauli frame is a classical register
that tracks accumulated Pauli byproducts on each logical qubit. Instead of
physically applying correction gates, the decoder updates the frame, and the
frame is consulted when interpreting future logical measurements.

In a simple memory experiment this reduces to a single XOR at the end, but
for lattice surgery, cultivation, or any multi-step protocol the frame must
persist across operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PauliFrame:
    """Classical tracker for logical Pauli byproducts.

    Each logical qubit is identified by a string label (e.g. ``"L0"``).
    The frame stores independent X and Z flip bits for each qubit, which
    together represent the four Pauli byproducts I, X, Z, Y = XZ.

    Usage example (memory experiment)::

        frame = PauliFrame()
        frame.register("L0")

        # Decoder says L0 was X-flipped
        frame.update_x("L0", flip=True)

        # Interpret the final Z-basis readout
        raw_readout = 1  # physical measurement said |1>
        corrected = frame.interpret_z_readout("L0", raw_readout)
        # corrected = 0 if frame had an X flip, because X anticommutes with Z

    """

    _x_flips: dict[str, bool] = field(default_factory=dict)
    _z_flips: dict[str, bool] = field(default_factory=dict)

    def register(self, label: str) -> None:
        """Register a new logical qubit with identity (no byproduct)."""
        self._x_flips[label] = False
        self._z_flips[label] = False

    def update_x(self, label: str, flip: bool) -> None:
        """Toggle the X byproduct on ``label`` if ``flip`` is True."""
        if flip:
            self._x_flips[label] = not self._x_flips[label]

    def update_z(self, label: str, flip: bool) -> None:
        """Toggle the Z byproduct on ``label`` if ``flip`` is True."""
        if flip:
            self._z_flips[label] = not self._z_flips[label]

    def has_x_flip(self, label: str) -> bool:
        """Return True if the frame currently records an X byproduct on ``label``."""
        return self._x_flips[label]

    def has_z_flip(self, label: str) -> bool:
        """Return True if the frame currently records a Z byproduct on ``label``."""
        return self._z_flips[label]

    def current_byproduct(self, label: str) -> str:
        """Return the current Pauli byproduct as a string: 'I', 'X', 'Z', or 'Y'."""
        x = self._x_flips[label]
        z = self._z_flips[label]
        if x and z:
            return "Y"
        elif x:
            return "X"
        elif z:
            return "Z"
        return "I"

    def interpret_z_readout(self, label: str, raw_bit: int) -> int:
        """Apply the frame to a raw Z-basis logical readout.

        An X byproduct anticommutes with Z measurement, so it flips the result.
        A Z byproduct commutes with Z measurement, so it has no effect.
        """
        return raw_bit ^ int(self._x_flips[label])

    def interpret_x_readout(self, label: str, raw_bit: int) -> int:
        """Apply the frame to a raw X-basis logical readout.

        A Z byproduct anticommutes with X measurement, so it flips the result.
        An X byproduct commutes with X measurement, so it has no effect.
        """
        return raw_bit ^ int(self._z_flips[label])

    def reset(self, label: str) -> None:
        """Reset the frame for ``label`` back to identity."""
        self._x_flips[label] = False
        self._z_flips[label] = False

    def as_dict(self) -> dict[str, str]:
        """Return a summary of all tracked qubits."""
        return {label: self.current_byproduct(label) for label in self._x_flips}

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{label}: {self.current_byproduct(label)}" for label in sorted(self._x_flips)
        )
        return f"PauliFrame({{{entries}}})"

