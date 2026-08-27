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
        uncorrected_readout = 1  # physical measurement said |1>
        corrected = frame.interpret_z_readout("L0", uncorrected_readout)
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
        if label not in self._x_flips:
            raise KeyError(
                f"Logical qubit {label!r} not registered. Call frame.register({label!r}) first."
            )
        if flip:
            self._x_flips[label] = not self._x_flips[label]

    def update_z(self, label: str, flip: bool) -> None:
        """Toggle the Z byproduct on ``label`` if ``flip`` is True."""
        if label not in self._x_flips:
            raise KeyError(
                f"Logical qubit {label!r} not registered. Call frame.register({label!r}) first."
            )
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

    def interpret_x_readout(self, label: str, uncorrected_bit: int) -> int:
        """Apply the frame to an uncorrected X-basis logical readout.

        A Z byproduct anticommutes with X measurement, so it flips the result.
        An X byproduct commutes with X measurement, so it has no effect.
        """
        return uncorrected_bit ^ int(self._z_flips[label])

    def interpret_y_readout(self, label: str, uncorrected_bit: int) -> int:
        """Apply the frame to an uncorrected Y-basis logical readout.

        X and Z byproducts each anticommute with Y and flip its readout. A Y
        byproduct contains both flips, so their effects cancel.
        """
        return uncorrected_bit ^ int(self._x_flips[label]) ^ int(self._z_flips[label])

    def interpret_z_readout(self, label: str, uncorrected_bit: int) -> int:
        """Apply the frame to an uncorrected Z-basis logical readout.

        An X byproduct anticommutes with Z measurement, so it flips the result.
        A Z byproduct commutes with Z measurement, so it has no effect.
        """
        return uncorrected_bit ^ int(self._x_flips[label])

    def correct_x_expectation(self, label: str, raw_value: float) -> float:
        """Apply the frame sign correction to a raw ⟨X_L⟩ expectation value.

        A Z byproduct anticommutes with X, flipping the sign.
        """
        return -raw_value if self._z_flips[label] else raw_value

    def correct_y_expectation(self, label: str, raw_value: float) -> float:
        """Apply the frame sign correction to a raw ⟨Y_L⟩ expectation value.

        Exactly one of an X or Z byproduct flips the sign; their product Y
        commutes with the measured logical Y.
        """
        return -raw_value if self._x_flips[label] != self._z_flips[label] else raw_value

    def correct_z_expectation(self, label: str, raw_value: float) -> float:
        """Apply the frame sign correction to a raw ⟨Z_L⟩ expectation value.

        An X byproduct anticommutes with Z, flipping the sign.
        """
        return -raw_value if self._x_flips[label] else raw_value

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
