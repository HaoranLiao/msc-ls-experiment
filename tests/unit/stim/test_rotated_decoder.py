"""Stim integration tests for the shared rotated-surface decoder."""

from __future__ import annotations

from types import ModuleType

import pytest
import stim

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame
from msc.noise.fault_injection_stim import append_pauli_faults
from msc.surface_code import rotated_d3_stim as d3
from msc.surface_code import rotated_d5_stim as d5
from msc.surface_code import rotated_helper_stim as rotated_helper
from msc.surface_code.decoding.rotated_decoder import (
    RotatedDecoder,
    TemporalRotatedDecoder,
)

# ===========================================================================
# Shared constants and helpers (d=3 and d=5)
# ===========================================================================


def _append_single_fault(
    circuit: stim.Circuit,
    code: ModuleType,
    data_qubits: tuple[int, ...],
    pauli: str,
    qubit_label: str,
) -> None:
    """Inject one deterministic Pauli fault as a probability-1 error channel.

    The error-channel form (rather than a deterministic gate) is what makes the
    change detectors fire; see :func:`append_pauli_faults`.
    """
    append_pauli_faults(
        circuit,
        [(pauli, qubit_label)],
        qubit_map=code.label2qubit(data_qubits),
        as_error_channel=True,
    )


def _decoder(code: ModuleType) -> RotatedDecoder:
    return RotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    )


def _expected_syndrome(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return the change-detection syndrome a single data fault should produce.

    Examples (d=3, with ``Z_abde`` on ``{a, b, d, e}`` and ``X_ab`` on
    ``{a, b}``):

    * ``pauli="X"``, ``qubit_label="a"`` -> ``z_deltas["Z_abde"] == 1`` and
      every other Z check plus all X checks stay 0 (an X error is invisible to
      X checks).
    * ``pauli="Y"``, ``qubit_label="d"`` -> both sectors fire on the checks
      supported on ``d``: ``z_deltas["Z_abde"] == z_deltas["Z_dg"] == 1`` and
      ``x_deltas["X_degh"] == 1``.
    """
    z_deltas = {
        name: int(pauli in {"X", "Y"} and qubit_label in support)
        for name, support in code.Z_STABILIZERS.items()
    }
    x_deltas = {
        name: int(pauli in {"Z", "Y"} and qubit_label in support)
        for name, support in code.X_STABILIZERS.items()
    }
    return z_deltas, x_deltas


def _run_single_fault(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
) -> tuple[dict[str, int], dict[str, int], int, dict[str, int], dict[str, int]]:
    data_qubits = tuple(range(len(code.SURFACE_ORDER)))
    z_names = tuple(code.Z_STABILIZERS)
    x_names = tuple(code.X_STABILIZERS)
    z_ancillas = tuple(range(len(data_qubits), len(data_qubits) + len(z_names)))
    x_ancillas = tuple(range(z_ancillas[-1] + 1, z_ancillas[-1] + 1 + len(x_names)))
    circuit = stim.Circuit()
    code.prepare_product_plus(circuit, qubits=data_qubits)
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=data_qubits,
        selected_stabilizer_names=z_names,
        ancillas=z_ancillas,
        round_label="initialize",
        add_ticks=False,
    )
    _append_single_fault(circuit, code, data_qubits, pauli, qubit_label)
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=data_qubits,
        selected_stabilizer_names=z_names + x_names,
        ancillas=z_ancillas + x_ancillas,
        round_label="detect",
        reset_ancillas_initially=True,
        add_ticks=False,
    )
    code.append_logical_measurement(
        circuit,
        "X",
        surface_qubits=data_qubits,
        destructive=True,
    )

    # Annotate the same run with native Stim change detectors: each Z detector
    # compares the detect round with the initialization reference, while each X
    # detector reads the |+>-prepared (known +1) X check directly. The detector
    # sampler must then fire exactly the stabilizers the injected fault flips.
    detect_z_positions = tuple(range(len(z_names), 2 * len(z_names)))
    detect_x_positions = tuple(range(2 * len(z_names), 2 * len(z_names) + len(x_names)))
    rotated_helper.append_stabilizer_detectors(
        circuit,
        rounds=(detect_z_positions,),
        first_round_reference=tuple(range(len(z_names))),
    )
    rotated_helper.append_stabilizer_detectors(circuit, rounds=(detect_x_positions,))

    shot = circuit.compile_sampler(seed=1).sample(1)[0]

    init_z = dict(zip(z_names, map(int, shot[: len(z_names)]), strict=True))
    det_z_start = len(z_names)
    det_x_start = det_z_start + len(z_names)
    det_z = dict(zip(z_names, map(int, shot[det_z_start:det_x_start]), strict=True))
    det_x = dict(
        zip(
            x_names,
            map(int, shot[det_x_start : det_x_start + len(x_names)]),
            strict=True,
        )
    )
    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(init_z)
    stabilizer_frame.record_many({name: 0 for name in x_names})
    deltas = stabilizer_frame.compare_many({**det_z, **det_x}, update=False)
    z_deltas = {name: deltas[name] for name in z_names}
    x_deltas = {name: deltas[name] for name in x_names}

    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    _decoder(code).update_pauli_frame(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        z_deltas=z_deltas,
        x_deltas=x_deltas,
    )
    raw_logical_x = int(shot[-len(code.DEFAULT_LOGICAL_X) :].sum() % 2)
    corrected_logical_x = pauli_frame.interpret_x_readout("L0", raw_logical_x)

    detector_shot = circuit.compile_detector_sampler(seed=1).sample(1)[0]
    detector_z_deltas = dict(zip(z_names, map(int, detector_shot[: len(z_names)]), strict=True))
    detector_x_deltas = dict(zip(x_names, map(int, detector_shot[len(z_names) :]), strict=True))
    return (
        z_deltas,
        x_deltas,
        corrected_logical_x,
        detector_z_deltas,
        detector_x_deltas,
    )


def _run_temporal_single_fault(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
    *,
    rounds: int = 3,
) -> tuple[frozenset[str], frozenset[str], int]:
    """Inject one persistent fault and jointly decode ``rounds`` detector layers.

    Mirrors :func:`_run_single_fault` but repeats the detect round so the
    ``TemporalRotatedDecoder`` receives a spacetime syndrome. The multi-round
    change detectors are sampled natively, so this also checks detector firing
    across rounds before feeding the joint decoder. Returns the decoded X/Z
    error sites and the frame-corrected logical-X readout.
    """
    data_qubits = tuple(range(len(code.SURFACE_ORDER)))
    z_names = tuple(code.Z_STABILIZERS)
    x_names = tuple(code.X_STABILIZERS)
    z_ancillas = tuple(range(len(data_qubits), len(data_qubits) + len(z_names)))
    x_ancillas = tuple(range(z_ancillas[-1] + 1, z_ancillas[-1] + 1 + len(x_names)))

    circuit = stim.Circuit()
    code.prepare_product_plus(circuit, qubits=data_qubits)
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=data_qubits,
        selected_stabilizer_names=z_names,
        ancillas=z_ancillas,
        round_label="initialize",
        add_ticks=False,
    )
    _append_single_fault(circuit, code, data_qubits, pauli, qubit_label)
    for round_index in range(rounds):
        code.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=data_qubits,
            selected_stabilizer_names=z_names + x_names,
            ancillas=z_ancillas + x_ancillas,
            round_label=f"detect{round_index}",
            reset_ancillas_initially=True,
            add_ticks=False,
        )
    code.append_logical_measurement(
        circuit,
        "X",
        surface_qubits=data_qubits,
        destructive=True,
    )

    # Record layout: the init Z round, then `rounds` interleaved Z+X rounds.
    round_width = len(z_names) + len(x_names)
    z_round_positions = [
        tuple(
            range(
                len(z_names) + round_index * round_width,
                len(z_names) + round_index * round_width + len(z_names),
            )
        )
        for round_index in range(rounds)
    ]
    x_round_positions = [
        tuple(
            range(
                len(z_names) + round_index * round_width + len(z_names),
                len(z_names) + (round_index + 1) * round_width,
            )
        )
        for round_index in range(rounds)
    ]
    # Z checks change-detect against the initialization reference; the
    # |+>-prepared X checks read their known +1 value on the first round.
    rotated_helper.append_stabilizer_detectors(
        circuit,
        rounds=z_round_positions,
        first_round_reference=tuple(range(len(z_names))),
    )
    rotated_helper.append_stabilizer_detectors(circuit, rounds=x_round_positions)

    measurement_shot = circuit.compile_sampler(seed=1).sample(1)[0]
    detector_shot = circuit.compile_detector_sampler(seed=1).sample(1)[0]
    z_delta_history = [
        dict(
            zip(
                z_names,
                map(
                    int,
                    detector_shot[round_index * len(z_names) : (round_index + 1) * len(z_names)],
                ),
                strict=True,
            )
        )
        for round_index in range(rounds)
    ]
    x_offset = rounds * len(z_names)
    x_delta_history = [
        dict(
            zip(
                x_names,
                map(
                    int,
                    detector_shot[
                        x_offset
                        + round_index * len(x_names) : x_offset
                        + (round_index + 1) * len(x_names)
                    ],
                ),
                strict=True,
            )
        )
        for round_index in range(rounds)
    ]

    decoder = TemporalRotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    )
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    recovery = decoder.update_pauli_frame(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        z_delta_history=z_delta_history,
        x_delta_history=x_delta_history,
    )
    raw_logical_x = int(measurement_shot[-len(code.DEFAULT_LOGICAL_X) :].sum() % 2)
    corrected_logical_x = pauli_frame.interpret_x_readout("L0", raw_logical_x)
    return recovery.x_error_sites, recovery.z_error_sites, corrected_logical_x


# ===========================================================================
# SET 1: Single-fault decoding (d=3 and d=5)
# ===========================================================================


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
def test_zero_syndrome_decodes_to_identity(code: ModuleType) -> None:
    recovery = _decoder(code).decode(
        z_deltas={name: 0 for name in code.Z_STABILIZERS},
        x_deltas={name: 0 for name in code.X_STABILIZERS},
    )
    assert recovery.is_trivial


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
@pytest.mark.parametrize("pauli", ("X", "Y", "Z"))
def test_single_fault_syndromes_and_frame_corrected_readout(
    code: ModuleType,
    pauli: str,
) -> None:
    for qubit_label in code.SURFACE_ORDER:
        (
            z_deltas,
            x_deltas,
            corrected_logical_x,
            detector_z_deltas,
            detector_x_deltas,
        ) = _run_single_fault(code, pauli, qubit_label)
        expected = _expected_syndrome(code, pauli, qubit_label)
        assert (z_deltas, x_deltas) == expected
        # Native Stim detectors fire exactly where the frame-based deltas do.
        assert (detector_z_deltas, detector_x_deltas) == expected
        assert corrected_logical_x == 0


# ===========================================================================
# SET 2: Temporal multi-round single-fault decoding (d=3 and d=5)
# ===========================================================================


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
@pytest.mark.parametrize("pauli", ("X", "Y", "Z"))
def test_temporal_decoder_corrects_persistent_single_fault(
    code: ModuleType,
    pauli: str,
) -> None:
    for qubit_label in code.SURFACE_ORDER:
        x_error_sites, z_error_sites, corrected_logical_x = _run_temporal_single_fault(
            code,
            pauli,
            qubit_label,
        )
        # A persistent single fault fires one detector event per affected check
        # in the first layer only, so the joint decoder returns a spatial data
        # correction (never a timelike measurement edge) in exactly the
        # anticommuting sector. Its precise site can differ from the injected
        # qubit by a stabilizer, so the fault-tolerant guarantee is the
        # corrected logical readout rather than the literal recovery support.
        assert bool(x_error_sites) == (pauli in {"X", "Y"})
        assert bool(z_error_sites) == (pauli in {"Z", "Y"})
        assert corrected_logical_x == 0
