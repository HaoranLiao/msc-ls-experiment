"""Kickback tomography helpers for the Steane-code cultivation patch.

This module builds the Fig. 3-style kickback tomography (KT) circuit from
Rosenfeld et al. using the repo's existing 11-site Steane patch primitives.
The KT measurement reuses cultivation rounds as repeated logical H_XY-axis
measurements: one optional preparation round, one tomography measurement round,
and one consistency-check round.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from magic_state_cultivation.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
    append_h_xy_projection_round,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from magic_state_cultivation.utils.normalize import qiskit_bit_value

FinalStabilizerMode = Literal["x", "z", "both", "none"]

KT_STABILIZER_BITS_PER_CYCLE = 6
KT_DEFAULT_QEC_CYCLE_INDICES: tuple[int, ...] = (1, 3)
KT_FULL_QEC_CYCLE_INDICES: tuple[int, ...] = (1, 2, 3, 4)
_VALID_QEC_CYCLE_INDICES = frozenset(KT_FULL_QEC_CYCLE_INDICES)


# =============================================================================
# Classical-bit layout and result summaries
# =============================================================================


@dataclass(frozen=True)
class KickbackCultivationRoundBits:
    """Classical bit positions for one cultivation round.

    Positions follow Qiskit's counts convention: bit position 0 is the
    rightmost bit in a flattened counts key.
    """

    folded_root: int
    flag_checks: tuple[int, int, int, int]

    @property
    def all_positions(self) -> tuple[int, ...]:
        """Return the folded root and all helper-measurement positions."""

        return (self.folded_root, *self.flag_checks)

    @property
    def flag_positions(self) -> tuple[int, int, int]:
        """Cultivation-flag bits that must be zero in a retained shot."""

        return self.flag_checks[:3]

    @property
    def folded_root_double_check(self) -> int:
        """Helper-site-7 bit that double-checks the folded root."""

        return self.flag_checks[3]


@dataclass(frozen=True)
class KickbackTomographyBitLayout:
    """Classical-bit positions used by a KT experiment."""

    preparation_round: KickbackCultivationRoundBits | None
    tomography_measurement_round: KickbackCultivationRoundBits
    tomography_check_round: KickbackCultivationRoundBits
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
        if self.preparation_round is not None:
            positions.extend(self.preparation_round.all_positions)
        positions.extend(self.tomography_measurement_round.flag_positions)
        positions.extend(self.tomography_check_round.flag_positions)
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
    retained_shots: int
    rejected_shots: int
    logical_success_shots: int
    logical_error_shots: int
    measure_check_disagreement_shots: int

    @property
    def retainment_rate(self) -> float:
        """Fraction of all shots retained by KT postselection."""

        if self.total_shots == 0:
            return float("nan")
        return self.retained_shots / self.total_shots

    @property
    def logical_error_probability(self) -> float:
        """Conditional KT logical error probability among retained shots."""

        if self.retained_shots == 0:
            return float("nan")
        return self.logical_error_shots / self.retained_shots

    @property
    def fidelity_lower_bound(self) -> float:
        """Lower bound on prepared-state fidelity inferred from KT."""

        if self.retained_shots == 0:
            return float("nan")
        return 1.0 - self.logical_error_probability


# =============================================================================
# KT circuit builder
# =============================================================================


def cultivation_round_bits_from_positions(
    positions: Sequence[int],
) -> KickbackCultivationRoundBits:
    if len(positions) < CULTIVATION_MEASUREMENTS_PER_ROUND:
        raise ValueError("cultivation register must contain at least 5 bit positions.")
    return KickbackCultivationRoundBits(
        folded_root=positions[0],
        flag_checks=(positions[1], positions[2], positions[3], positions[4]),
    )


def normalize_qec_cycle_indices(qec_cycle_indices: Iterable[int]) -> tuple[int, ...]:
    indices = tuple(qec_cycle_indices)
    if any(index not in _VALID_QEC_CYCLE_INDICES for index in indices):
        raise ValueError("qec_cycle_indices may only contain 1, 2, 3, and 4.")
    if len(set(indices)) != len(indices):
        raise ValueError("qec_cycle_indices must not contain duplicates.")
    return tuple(sorted(indices))


def final_stabilizer_bit_count(mode: FinalStabilizerMode) -> int:
    if mode == "both":
        return KT_STABILIZER_BITS_PER_CYCLE
    if mode in {"x", "z"}:
        return KT_STABILIZER_BITS_PER_CYCLE // 2
    if mode == "none":
        return 0
    raise ValueError("final_stabilizer_basis must be 'x', 'z', 'both', or 'none'.")


def build_kickback_tomography_experiment(
    injected_state: InjectedStateName | None = None,
    *,
    injection_azimuth_angle: float | None = None,
    include_preparation_cultivation: bool = True,
    qec_cycle_indices: Iterable[int] = KT_DEFAULT_QEC_CYCLE_INDICES,
    final_stabilizer_basis: FinalStabilizerMode = "x",
    add_barriers: bool = True,
) -> KickbackTomographyExperiment:
    """Build a Steane-patch ``H_XY`` kickback-tomography experiment.

    The protocol order is

    ``encode -> QEC1 -> [prepare cultivation] -> QEC2 -> KT measure
    -> QEC3 -> KT check -> QEC4 -> final stabilizer check``.

    ``injected_state`` and ``injection_azimuth_angle`` are mutually exclusive.
    A supplied angle injects the equatorial state ``Rz(angle)|+⟩`` instead of a
    named state; ``angle=π/4`` gives ``|T⟩``. If neither is supplied, ``|T⟩`` is
    injected. Every cultivation and KT round measures
    ``H_XY = (X + Y) / sqrt(2)``.

    ``qec_cycle_indices`` uses the paper's one-based numbering. The default
    schedule is ``KT_DEFAULT_QEC_CYCLE_INDICES = (1, 3)``; pass
    ``KT_FULL_QEC_CYCLE_INDICES = (1, 2, 3, 4)`` for every Fig. 3 QEC cycle.
    An empty tuple selects the "without QEC cycles" variant. These rounds only
    measure error-detection syndromes. KT postselects on them and does not apply
    corrections or Pauli-frame updates.

    ``final_stabilizer_basis="x"`` matches the final detector block drawn in
    Fig. 3. ``"z"`` selects the equivalent CSS basis, ``"both"`` measures all
    six stabilizers, and ``"none"`` omits the final check.
    """

    if injected_state is not None and injection_azimuth_angle is not None:
        raise ValueError(
            "injected_state and injection_azimuth_angle are mutually exclusive; supply at most one."
        )

    qec_indices = normalize_qec_cycle_indices(qec_cycle_indices)

    patch = QuantumRegister(11, "patch")
    prep_creg = (
        ClassicalRegister(CULTIVATION_MEASUREMENTS_PER_ROUND, "prep_cult")
        if include_preparation_cultivation
        else None
    )
    qec_registers = {
        index: ClassicalRegister(KT_STABILIZER_BITS_PER_CYCLE, f"qec_{index}")
        for index in qec_indices
    }
    kt_measure_creg = ClassicalRegister(CULTIVATION_MEASUREMENTS_PER_ROUND, "kt_measure")
    kt_check_creg = ClassicalRegister(CULTIVATION_MEASUREMENTS_PER_ROUND, "kt_check")
    final_bit_count = final_stabilizer_bit_count(final_stabilizer_basis)
    final_creg = ClassicalRegister(final_bit_count, "final_stab") if final_bit_count else None

    classical_registers: list[ClassicalRegister] = []
    if prep_creg is not None:
        classical_registers.append(prep_creg)
    classical_registers.extend(qec_registers[index] for index in qec_indices)
    classical_registers.extend((kt_measure_creg, kt_check_creg))
    if final_creg is not None:
        classical_registers.append(final_creg)

    circuit = QuantumCircuit(patch, *classical_registers)
    patch_qubits = list(patch)

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        add_barriers=add_barriers,
    )

    if 1 in qec_registers:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(qec_registers[1]),
            add_barriers=add_barriers,
        )

    if include_preparation_cultivation:
        assert prep_creg is not None
        append_h_xy_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(prep_creg),
            add_barriers=add_barriers,
        )

    if 2 in qec_registers:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(qec_registers[2]),
            add_barriers=add_barriers,
        )

    append_h_xy_projection_round(
        circuit,
        patch_qubits=patch_qubits,
        classical_bits=list(kt_measure_creg),
        add_barriers=add_barriers,
    )

    if 3 in qec_registers:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(qec_registers[3]),
            add_barriers=add_barriers,
        )

    append_h_xy_projection_round(
        circuit,
        patch_qubits=patch_qubits,
        classical_bits=list(kt_check_creg),
        add_barriers=add_barriers,
    )

    if 4 in qec_registers:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(qec_registers[4]),
            add_barriers=add_barriers,
        )

    if final_creg is not None:
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(final_creg),
            basis=final_stabilizer_basis,
            add_barriers=add_barriers,
        )

    qec_positions = tuple(
        tuple(circuit.find_bit(bit).index for bit in qec_registers[index]) for index in qec_indices
    )
    preparation_bits = (
        cultivation_round_bits_from_positions(
            tuple(circuit.find_bit(bit).index for bit in prep_creg),
        )
        if prep_creg is not None
        else None
    )
    layout = KickbackTomographyBitLayout(
        preparation_round=preparation_bits,
        tomography_measurement_round=cultivation_round_bits_from_positions(
            tuple(circuit.find_bit(bit).index for bit in kt_measure_creg),
        ),
        tomography_check_round=cultivation_round_bits_from_positions(
            tuple(circuit.find_bit(bit).index for bit in kt_check_creg),
        ),
        qec_cycles=qec_positions,
        final_stabilizers=(
            tuple(circuit.find_bit(bit).index for bit in final_creg)
            if final_creg is not None
            else ()
        ),
    )
    circuit.metadata = {
        **(circuit.metadata or {}),
        "kickback_tomography_bit_layout": {
            "preparation_round": (
                layout.preparation_round.all_positions if layout.preparation_round else None
            ),
            "tomography_measurement_round": layout.tomography_measurement_round.all_positions,
            "tomography_check_round": layout.tomography_check_round.all_positions,
            "qec_cycles": layout.qec_cycles,
            "final_stabilizers": layout.final_stabilizers,
            "qec_cycle_indices": qec_indices,
            "final_stabilizer_basis": final_stabilizer_basis,
        },
    }
    return KickbackTomographyExperiment(circuit=circuit, layout=layout)


# =============================================================================
# Counts parsing and KT estimators
# =============================================================================


def kickback_tomography_shot_is_retained(
    bitstring: str,
    layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool = True,
) -> bool:
    """Return whether one KT shot passes detector postselection.

    By default, the two KT roots are not required to agree. A disagreement is
    retained but excluded from the logical-error numerator by
    :func:`summarize_kickback_tomography_counts`. Set
    ``retain_measure_check_disagreement=False`` to reject disagreements.
    """

    if any(qiskit_bit_value(bitstring, position) for position in layout.detector_positions):
        return False

    measurement_round_root = qiskit_bit_value(
        bitstring,
        layout.tomography_measurement_round.folded_root,
    )
    measurement_round_root_double_check = qiskit_bit_value(
        bitstring,
        layout.tomography_measurement_round.folded_root_double_check,
    )
    if measurement_round_root_double_check != measurement_round_root:
        return False

    check_round_root = qiskit_bit_value(
        bitstring,
        layout.tomography_check_round.folded_root,
    )
    check_round_root_double_check = qiskit_bit_value(
        bitstring,
        layout.tomography_check_round.folded_root_double_check,
    )
    if check_round_root_double_check != check_round_root:
        return False

    if not retain_measure_check_disagreement and measurement_round_root != check_round_root:
        return False

    return True


def summarize_kickback_tomography_counts(
    counts: dict[str, int],
    layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool = True,
) -> KickbackTomographySummary:
    """Summarize KT retainment and logical error from Qiskit counts.

    A logical-error shot has both tomography roots equal to 1 after all
    detectors pass. If the measurement round and check round roots disagree,
    the shot is retained in the retained-shot denominator
    but does not contribute to the logical-error numerator.
    """

    total_shots = sum(counts.values())
    retained_shots = 0
    logical_success_shots = 0
    logical_error_shots = 0
    measure_check_disagreement_shots = 0

    for bitstring, count in counts.items():
        if not kickback_tomography_shot_is_retained(
            bitstring,
            layout,
            retain_measure_check_disagreement=retain_measure_check_disagreement,
        ):
            continue

        retained_shots += count
        measurement_root = qiskit_bit_value(
            bitstring,
            layout.tomography_measurement_round.folded_root,
        )
        check_root = qiskit_bit_value(
            bitstring,
            layout.tomography_check_round.folded_root,
        )
        if measurement_root != check_root:
            measure_check_disagreement_shots += count
        elif measurement_root:
            logical_error_shots += count
        else:
            logical_success_shots += count

    return KickbackTomographySummary(
        total_shots=total_shots,
        retained_shots=retained_shots,
        rejected_shots=total_shots - retained_shots,
        logical_success_shots=logical_success_shots,
        logical_error_shots=logical_error_shots,
        measure_check_disagreement_shots=measure_check_disagreement_shots,
    )
