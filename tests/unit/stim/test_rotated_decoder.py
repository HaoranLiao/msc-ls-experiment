"""Stim integration tests for the shared rotated-surface decoder."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
import stim

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame
from msc.noise.fault_injection_stim import append_pauli_faults
from msc.rotated_surface_code import d3_stim as d3
from msc.rotated_surface_code import d5_stim as d5
from msc.rotated_surface_code import helper_stim as helper
from msc.rotated_surface_code.decoding.decoder import (
    RotatedDecoder,
    SpaceTimeRotatedDecoder,
)

# ===========================================================================
# Single-round decoding (d=3 and d=5)
# ===========================================================================


def _run_single_fault(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
) -> tuple[dict[str, int], dict[str, int], int, dict[str, int], dict[str, int]]:
    """Return frame and native-detector results for one persistent data fault."""
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
    append_pauli_faults(
        circuit,
        [(pauli, qubit_label)],
        qubit_map=code.label2qubit(data_qubits),
        as_error_channel=True,
    )
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=data_qubits,
        selected_stabilizer_names=z_names + x_names,
        ancillas=z_ancillas + x_ancillas,
        round_label="detect",
        add_ticks=False,
    )
    code.append_logical_measurement(
        circuit,
        "X",
        surface_qubits=data_qubits,
        destructive=True,
    )

    # Z detectors compare with initialization; the |+>-prepared X checks use
    # their known +1 values. Both must fire exactly where the fault anticommutes.
    detect_z_positions = tuple(range(len(z_names), 2 * len(z_names)))
    detect_x_positions = tuple(range(2 * len(z_names), 2 * len(z_names) + len(x_names)))
    helper.append_stabilizer_detectors(
        circuit,
        rounds=(detect_z_positions,),
        first_round_reference=tuple(range(len(z_names))),
    )
    helper.append_stabilizer_detectors(circuit, rounds=(detect_x_positions,))

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
    RotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    ).decode_raw_syndromes(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        z_raw=det_z,
        x_raw=det_x,
        stab_frame=stabilizer_frame,
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


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
def test_zero_syndrome_decodes_to_identity(code: ModuleType) -> None:
    recovery = RotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    ).decode(
        z_deltas={name: 0 for name in code.Z_STABILIZERS},
        x_deltas={name: 0 for name in code.X_STABILIZERS},
    )
    assert recovery.is_trivial


@pytest.mark.slow
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
        expected_z_deltas = {
            name: int(pauli in {"X", "Y"} and qubit_label in support)
            for name, support in code.Z_STABILIZERS.items()
        }
        expected_x_deltas = {
            name: int(pauli in {"Z", "Y"} and qubit_label in support)
            for name, support in code.X_STABILIZERS.items()
        }
        expected_deltas = expected_z_deltas, expected_x_deltas
        assert (z_deltas, x_deltas) == expected_deltas
        # Native Stim detectors fire exactly where the frame-based deltas do.
        assert (detector_z_deltas, detector_x_deltas) == expected_deltas
        assert corrected_logical_x == 0


# ===========================================================================
# Spacetime multi-round decoding (d=3 and d=5)
# ===========================================================================


def _run_spacetime_single_fault(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
    *,
    rounds: int = 3,
) -> tuple[frozenset[str], frozenset[str], int]:
    """Inject one persistent fault and jointly decode ``rounds`` detector rounds.

    Mirrors :func:`_run_single_fault` but repeats the detect round so the
    ``SpaceTimeRotatedDecoder`` receives a spacetime syndrome. The multi-round
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
    append_pauli_faults(
        circuit,
        [(pauli, qubit_label)],
        qubit_map=code.label2qubit(data_qubits),
        as_error_channel=True,
    )
    for round_index in range(rounds):
        code.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=data_qubits,
            selected_stabilizer_names=z_names + x_names,
            ancillas=z_ancillas + x_ancillas,
            round_label=f"detect{round_index}",
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
    round_starts = [len(z_names) + round_index * round_width for round_index in range(rounds)]
    z_round_positions = [
        tuple(range(round_start, round_start + len(z_names))) for round_start in round_starts
    ]
    x_round_positions = [
        tuple(range(round_start + len(z_names), round_start + round_width))
        for round_start in round_starts
    ]
    # Z checks change-detect against the initialization reference; the
    # |+>-prepared X checks read their known +1 value on the first round.
    helper.append_stabilizer_detectors(
        circuit,
        rounds=z_round_positions,
        first_round_reference=tuple(range(len(z_names))),
    )
    helper.append_stabilizer_detectors(circuit, rounds=x_round_positions)

    measurement_shot = circuit.compile_sampler(seed=1).sample(1)[0]
    detector_shot = circuit.compile_detector_sampler(seed=1).sample(1)[0]
    x_offset = rounds * len(z_names)
    z_delta_history = [
        dict(zip(z_names, map(int, delta_round), strict=True))
        for delta_round in detector_shot[:x_offset].reshape(rounds, len(z_names))
    ]
    x_delta_history = [
        dict(zip(x_names, map(int, delta_round), strict=True))
        for delta_round in detector_shot[x_offset:].reshape(rounds, len(x_names))
    ]

    decoder = SpaceTimeRotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    )
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    init_z = dict(zip(z_names, measurement_shot[: len(z_names)], strict=True))
    raw_stabilizer_rounds = measurement_shot[
        len(z_names) : len(z_names) + rounds * round_width
    ].reshape(rounds, round_width)
    z_raw_history = [
        dict(zip(z_names, raw_round[: len(z_names)], strict=True))
        for raw_round in raw_stabilizer_rounds
    ]
    x_raw_history = [
        dict(zip(x_names, raw_round[len(z_names) :], strict=True))
        for raw_round in raw_stabilizer_rounds
    ]
    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(init_z)
    stabilizer_frame.record_many(dict.fromkeys(x_names, 0))
    recovery = decoder.decode_raw_syndromes(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        z_raw_history=z_raw_history,
        x_raw_history=x_raw_history,
        stab_frame=stabilizer_frame,
    )
    assert recovery.z_delta_history == tuple(z_delta_history)
    assert recovery.x_delta_history == tuple(x_delta_history)
    raw_logical_x = int(measurement_shot[-len(code.DEFAULT_LOGICAL_X) :].sum() % 2)
    corrected_logical_x = pauli_frame.interpret_x_readout("L0", raw_logical_x)
    return recovery.x_error_sites, recovery.z_error_sites, corrected_logical_x


@pytest.mark.slow
@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
@pytest.mark.parametrize("pauli", ("X", "Y", "Z"))
def test_spacetime_decoder_corrects_persistent_single_fault(
    code: ModuleType,
    pauli: str,
) -> None:
    for qubit_label in code.SURFACE_ORDER:
        x_error_sites, z_error_sites, corrected_logical_x = _run_spacetime_single_fault(
            code,
            pauli,
            qubit_label,
        )
        # A persistent fault fires each affected check in the first delta round.
        # The exact recovery can differ by a stabilizer, so verify its sector and
        # the corrected logical readout rather than its literal physical support.
        assert bool(x_error_sites) == (pauli in {"X", "Y"})
        assert bool(z_error_sites) == (pauli in {"Z", "Y"})
        assert corrected_logical_x == 0


# ===========================================================================
# Spacetime LER batch decoding (d=5)
# ===========================================================================


def test_spacetime_batch_logical_corrections_match_scalar_decoding() -> None:
    """The parity-only batch API agrees with logical parities from ``decode``."""
    decoder = SpaceTimeRotatedDecoder(
        data_qubits=d5.SURFACE_ORDER,
        z_stabilizers=d5.Z_STABILIZERS,
        x_stabilizers=d5.X_STABILIZERS,
        logical_x=d5.DEFAULT_LOGICAL_X,
        logical_z=d5.DEFAULT_LOGICAL_Z,
    )
    rng = np.random.default_rng(1234)
    z_delta_histories = rng.binomial(1, 0.08, (32, 4, len(d5.Z_STABILIZERS))).astype(bool)
    x_delta_histories = rng.binomial(1, 0.08, (32, 4, len(d5.X_STABILIZERS))).astype(bool)

    batch_logical_x_corrections, batch_logical_z_corrections = (
        decoder.decode_logical_corrections_batch(
            z_delta_histories=z_delta_histories,
            x_delta_histories=x_delta_histories,
        )
    )
    recoveries = [
        decoder.decode(
            z_delta_history=[
                dict(zip(d5.Z_STABILIZERS, delta_round, strict=True)) for delta_round in z_history
            ],
            x_delta_history=[
                dict(zip(d5.X_STABILIZERS, delta_round, strict=True)) for delta_round in x_history
            ],
        )
        for z_history, x_history in zip(
            z_delta_histories,
            x_delta_histories,
            strict=True,
        )
    ]

    logical_x = frozenset(d5.DEFAULT_LOGICAL_X)
    logical_z = frozenset(d5.DEFAULT_LOGICAL_Z)
    assert np.array_equal(
        batch_logical_x_corrections,
        [len(recovery.x_error_sites & logical_z) % 2 for recovery in recoveries],
    )
    assert np.array_equal(
        batch_logical_z_corrections,
        [len(recovery.z_error_sites & logical_x) % 2 for recovery in recoveries],
    )
