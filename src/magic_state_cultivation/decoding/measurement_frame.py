"""Classical state tracker for measurement qubits reused without reset.

A computational-basis measurement with raw outcome ``b`` leaves its qubit in
``|b>``. If a later parity-measurement sequence reuses that qubit without its
usual reset, ``b`` is XORed into the later raw outcome. ``MeasurementFrame``
stores that residual bit per measurement-qubit label so the intended outcome
can be recovered.

Unlike ``StabilizerFrame``, this object does not compare semantic stabilizer
outcomes between rounds. Unlike ``PauliFrame``, it does not track logical
quantum byproducts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from magic_state_cultivation.utils.normalize import normalize_bit


def _resolve_measurement_label(label: str) -> str:
    """Validate and return one measurement-qubit label."""

    if not isinstance(label, str) or not label:
        raise ValueError("measurement label must be a non-empty string.")
    return label


@dataclass
class MeasurementFrame:
    """Track residual states of measurement qubits when resets are omitted."""

    _residual_bits: dict[str, int] = field(default_factory=dict)

    def current(self, label: str) -> int:
        """Return the residual bit for ``label``; an unseen label starts at zero."""

        return self._residual_bits.get(_resolve_measurement_label(label), 0)

    def advance(self, label: str, raw_bit: int | bool) -> int:
        """Interpret one readout and record the post-measurement residual state.

        The returned semantic bit is the raw readout XOR the qubit's residual
        bit before measurement. The new residual is the raw readout itself,
        because measurement leaves the qubit in that computational-basis
        state.
        """

        resolved_label = _resolve_measurement_label(label)
        resolved_raw_bit = normalize_bit(raw_bit)
        interpreted_bit = self.current(resolved_label) ^ resolved_raw_bit
        if resolved_raw_bit:
            self._residual_bits[resolved_label] = resolved_raw_bit
        else:
            self._residual_bits.pop(resolved_label, None)
        return interpreted_bit

    def reset(self, label: str) -> None:
        """Record that ``label`` was reset to ``|0>``."""

        self._residual_bits.pop(_resolve_measurement_label(label), None)

    def as_dict(self) -> dict[str, int]:
        """Return the nonzero residual measurement-qubit states."""

        return dict(self._residual_bits)

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{label}: {self._residual_bits[label]}" for label in sorted(self._residual_bits)
        )
        return f"MeasurementFrame({{{entries}}})"
