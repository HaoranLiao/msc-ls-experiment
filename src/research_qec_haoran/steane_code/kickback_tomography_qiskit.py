"""Kickback tomography helpers for the Steane-code cultivation patch.

This module builds the Fig. 3-style kickback tomography (KT) circuit from
Rosenfeld et al. using the repo's existing 11-site Steane patch primitives.
The KT measurement reuses cultivation rounds as repeated logical Hadamard-axis
measurements: one optional preparation round, one tomography measurement round,
and one consistency-check round.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    append_hadamard_axis_projection_round,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from research_qec_haoran.utils.normalize import qiskit_bit_value

FinalStabilizerMode = Literal["both", "none"]

KT_CULTIVATION_BITS_PER_ROUND = 5
KT_STABILIZER_BITS_PER_CYCLE = 6
KT_DEFAULT_QEC_CYCLE_INDICES: tuple[int, ...] = (0, 1, 2)
_VALID_QEC_CYCLE_INDICES = frozenset(KT_DEFAULT_QEC_CYCLE_INDICES)


# =============================================================================
# Classical-bit layout and result summaries
# =============================================================================


@dataclass(frozen=True)
class KickbackCultivationBits:
    """Classical bit positions for one cultivation round.

    Positions follow Qiskit's counts convention: bit position 0 is the
    rightmost bit in a flattened counts key.
    """

    folded_observable: int
    helper_checks: tuple[int, int, int, int]

    @property
    def all_positions(self) -> tuple[int, ...]:
        """Return folded-observable and helper-check positions."""

        return (self.folded_observable, *self.helper_checks)

    @property
    def flag_helper_checks(self) -> tuple[int, int, int]:
        """Helper bits that must be zero independent of the logical outcome."""

        return self.helper_checks[:3]

    @property
    def folded_site_helper_check(self) -> int:
        """Helper-site-7 bit that tracks the folded observable in KT rounds."""

        return self.helper_checks[3]


@dataclass(frozen=True)
class KickbackTomographyBitLayout:
    """Classical-bit positions used by a KT experiment."""

    preparation: KickbackCultivationBits | None
    tomography_measurement: KickbackCultivationBits
    tomography_check: KickbackCultivationBits
    qec_cycles: tuple[tuple[int, ...], ...]
    final_stabilizers: tuple[int, ...]

    @property
    def detector_positions(self) -> tuple[int, ...]:
        """Bits that must be zero for KT postselection.

        The tomography measurement root is the logical outcome, so it is not a
        detector. In KT rounds, the helper-site-7 bit tracks the folded root
        and is handled separately.
        """

        positions: list[int] = []
        if self.preparation is not None:
            positions.extend(self.preparation.all_positions)
        positions.extend(self.tomography_measurement.flag_helper_checks)
        positions.extend(self.tomography_check.flag_helper_checks)
        for cycle in self.qec_cycles:
            positions.extend(cycle)
        positions.extend(self.final_stabilizers)
        return tuple(positions)


@dataclass(frozen=True)
class KickbackTomographyExperiment:
    """A KT circuit together with its classical-bit layout."""

    circuit: QuantumCircuit
    layout: KickbackTomographyBitLayout


@dataclass(frozen=True)
class KickbackTomographySummary:
    """Shot counts after KT postselection and logical readout."""

    total_shots: int
    accepted_shots: int
    rejected_shots: int
    logical_success_shots: int
    logical_error_shots: int

    @property
    def retainment_rate(self) -> float:
        """Fraction of all shots retained by KT postselection."""

        if self.total_shots == 0:
            return float("nan")
        return self.accepted_shots / self.total_shots

    @property
    def logical_error_probability(self) -> float:
        """Conditional KT logical error probability among retained shots."""

        if self.accepted_shots == 0:
            return float("nan")
        return self.logical_error_shots / self.accepted_shots


# =============================================================================
# KT circuit builder
# =============================================================================


class _ClassicalAllocator:
    """Track Qiskit bit positions while registers are added."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._next_position = 0

    def add_register(self, size: int, name: str) -> tuple[ClassicalRegister, tuple[int, ...]]:
        creg = ClassicalRegister(size, name)
        self._circuit.add_register(creg)
        positions = tuple(range(self._next_position, self._next_position + size))
        self._next_position += size
        return creg, positions


def _cultivation_bits_from_positions(positions: Sequence[int]) -> KickbackCultivationBits:
    if len(positions) < KT_CULTIVATION_BITS_PER_ROUND:
        raise ValueError("cultivation register must contain at least 5 bit positions.")
    return KickbackCultivationBits(
        folded_observable=positions[0],
        helper_checks=(positions[1], positions[2], positions[3], positions[4]),
    )


def _normalise_qec_cycle_indices(qec_cycle_indices: Iterable[int]) -> tuple[int, ...]:
    indices = tuple(qec_cycle_indices)
    if any(index not in _VALID_QEC_CYCLE_INDICES for index in indices):
        raise ValueError("qec_cycle_indices may only contain 0, 1, and 2.")
    if len(set(indices)) != len(indices):
        raise ValueError("qec_cycle_indices must not contain duplicates.")
    return tuple(sorted(indices))


def _final_stabilizer_bit_count(mode: FinalStabilizerMode) -> int:
    if mode == "both":
        return KT_STABILIZER_BITS_PER_CYCLE
    if mode == "none":
        return 0
    raise ValueError("final_stabilizer_basis must be 'both' or 'none'.")


def build_kickback_tomography_experiment(
    injected_state: InjectedStateName = "H",
    *,
    include_preparation_cultivation: bool = True,
    qec_cycle_indices: Iterable[int] = KT_DEFAULT_QEC_CYCLE_INDICES,
    final_stabilizer_basis: FinalStabilizerMode = "both",
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> KickbackTomographyExperiment:
    """Build a Steane-patch kickback tomography experiment.

    Stage order is

    ``encode -> [prepare cultivation] -> QEC0 -> KT measure -> QEC1
    -> KT check -> QEC2 -> final stabilizer check``.

    ``qec_cycle_indices`` selects which of QEC0, QEC1, and QEC2 are included.
    Keeping all three matches the main KT circuit; using an empty tuple is the
    "without QEC cycles" variant.

    ``final_stabilizer_basis="both"`` appends the repo's existing full
    six-stabilizer patch measurement. ``"none"`` omits the final check.
    """

    qec_indices = _normalise_qec_cycle_indices(qec_cycle_indices)

    patch = QuantumRegister(11, "patch")
    circuit = QuantumCircuit(patch)
    alloc = _ClassicalAllocator(circuit)

    qec_registers: dict[int, tuple[ClassicalRegister, tuple[int, ...]]] = {}

    prep_creg: ClassicalRegister | None = None
    prep_positions: tuple[int, ...] | None = None
    if include_preparation_cultivation:
        prep_creg, prep_positions = alloc.add_register(KT_CULTIVATION_BITS_PER_ROUND, "prep_cult")

    for index in qec_indices:
        qec_registers[index] = alloc.add_register(KT_STABILIZER_BITS_PER_CYCLE, f"qec_{index}")

    kt_measure_creg, kt_measure_positions = alloc.add_register(
        KT_CULTIVATION_BITS_PER_ROUND,
        "kt_measure",
    )
    kt_check_creg, kt_check_positions = alloc.add_register(
        KT_CULTIVATION_BITS_PER_ROUND,
        "kt_check",
    )

    final_bit_count = _final_stabilizer_bit_count(final_stabilizer_basis)
    final_creg: ClassicalRegister | None = None
    final_positions: tuple[int, ...] = ()
    if final_bit_count:
        final_creg, final_positions = alloc.add_register(final_bit_count, "final_stab")

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(patch),
        injected_state=injected_state,
        add_barriers=add_barriers,
    )

    if include_preparation_cultivation:
        assert prep_creg is not None
        append_hadamard_axis_projection_round(
            circuit,
            patch_qubits=list(patch),
            classical_bits=list(prep_creg),
            add_barriers=add_barriers,
            skip_cx_1_4=skip_cx_1_4,
            skip_cx=skip_cx,
        )

    if 0 in qec_registers:
        qec_creg, _ = qec_registers[0]
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(patch),
            classical_bits=list(qec_creg),
            add_barriers=add_barriers,
        )

    append_hadamard_axis_projection_round(
        circuit,
        patch_qubits=list(patch),
        classical_bits=list(kt_measure_creg),
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    if 1 in qec_registers:
        qec_creg, _ = qec_registers[1]
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(patch),
            classical_bits=list(qec_creg),
            add_barriers=add_barriers,
        )

    append_hadamard_axis_projection_round(
        circuit,
        patch_qubits=list(patch),
        classical_bits=list(kt_check_creg),
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    if 2 in qec_registers:
        qec_creg, _ = qec_registers[2]
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(patch),
            classical_bits=list(qec_creg),
            add_barriers=add_barriers,
        )

    if final_creg is not None:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(patch),
            classical_bits=list(final_creg),
            add_barriers=add_barriers,
        )

    qec_positions = tuple(qec_registers[index][1] for index in sorted(qec_registers))
    preparation_bits = (
        _cultivation_bits_from_positions(prep_positions) if prep_positions is not None else None
    )
    layout = KickbackTomographyBitLayout(
        preparation=preparation_bits,
        tomography_measurement=_cultivation_bits_from_positions(kt_measure_positions),
        tomography_check=_cultivation_bits_from_positions(kt_check_positions),
        qec_cycles=qec_positions,
        final_stabilizers=final_positions,
    )
    circuit.metadata = {
        **(circuit.metadata or {}),
        "kickback_tomography_bit_layout": {
            "preparation": layout.preparation.all_positions if layout.preparation else None,
            "tomography_measurement": layout.tomography_measurement.all_positions,
            "tomography_check": layout.tomography_check.all_positions,
            "qec_cycles": layout.qec_cycles,
            "final_stabilizers": layout.final_stabilizers,
        },
    }
    return KickbackTomographyExperiment(circuit=circuit, layout=layout)


def build_kickback_tomography_circuit(
    injected_state: InjectedStateName = "H",
    **kwargs,
) -> QuantumCircuit:
    """Return only the Qiskit circuit for a KT experiment.

    Use :func:`build_kickback_tomography_experiment` when counts parsing is
    needed, because it also returns the bit layout.
    """

    return build_kickback_tomography_experiment(injected_state, **kwargs).circuit


# =============================================================================
# Counts parsing and KT estimators
# =============================================================================


def bit_value(bitstring: str, position: int) -> int:
    """Return a bit from a Qiskit counts key using Qiskit bit positions.

    This compatibility wrapper preserves the KT module's public helper while
    sharing the repo-wide implementation.
    """

    return qiskit_bit_value(bitstring, position)


def detector_bits_are_zero(bitstring: str, positions: Sequence[int]) -> bool:
    """Return whether all listed detector bit positions are zero."""

    return all(bit_value(bitstring, position) == 0 for position in positions)


def kickback_tomography_shot_is_accepted(
    bitstring: str,
    layout: KickbackTomographyBitLayout,
    *,
    require_check_agreement: bool = False,
) -> bool:
    """Return whether one KT shot passes detector postselection.

    The two KT roots are not required to agree by default.  A disagreement is
    kept and counted as a logical-error event by
    :func:`summarize_kickback_tomography_counts`.
    """

    if not detector_bits_are_zero(bitstring, layout.detector_positions):
        return False

    measurement = bit_value(bitstring, layout.tomography_measurement.folded_observable)
    measurement_site_check = bit_value(
        bitstring,
        layout.tomography_measurement.folded_site_helper_check,
    )
    if measurement_site_check != measurement:
        return False

    check = bit_value(bitstring, layout.tomography_check.folded_observable)
    check_site_check = bit_value(
        bitstring,
        layout.tomography_check.folded_site_helper_check,
    )
    if check_site_check != check:
        return False

    if require_check_agreement:
        return measurement == check

    return True


def postselect_kickback_tomography_counts(
    counts: dict[str, int],
    layout: KickbackTomographyBitLayout,
    *,
    require_check_agreement: bool = False,
) -> dict[str, int]:
    """Filter counts to shots accepted by KT detector postselection."""

    return {
        bitstring: count
        for bitstring, count in counts.items()
        if kickback_tomography_shot_is_accepted(
            bitstring,
            layout,
            require_check_agreement=require_check_agreement,
        )
    }


def summarize_kickback_tomography_counts(
    counts: dict[str, int],
    layout: KickbackTomographyBitLayout,
    *,
    require_check_agreement: bool = False,
) -> KickbackTomographySummary:
    """Summarize KT retainment and logical error from Qiskit counts.

    A no-error KT shot has both tomography roots equal to 0 after all detectors
    pass. If the measurement and check roots disagree, the shot is not discarded;
    it is counted as a logical-error shot, matching KT's role as a noisy
    metrology circuit whose reported error includes undetected logical faults
    and measurement faults.
    """

    total_shots = sum(counts.values())
    accepted_shots = 0
    logical_error_shots = 0

    for bitstring, count in counts.items():
        if not kickback_tomography_shot_is_accepted(
            bitstring,
            layout,
            require_check_agreement=require_check_agreement,
        ):
            continue

        accepted_shots += count
        measurement = bit_value(bitstring, layout.tomography_measurement.folded_observable)
        check = bit_value(bitstring, layout.tomography_check.folded_observable)
        if measurement or check:
            logical_error_shots += count

    return KickbackTomographySummary(
        total_shots=total_shots,
        accepted_shots=accepted_shots,
        rejected_shots=total_shots - accepted_shots,
        logical_success_shots=accepted_shots - logical_error_shots,
        logical_error_shots=logical_error_shots,
    )
