"""Circuit-level decoding of twirl variants."""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties, PassManager, Target
from qiskit.transpiler.passes import RemoveResetInZeroState

from msc.rotated_surface_code.decoding.circuit_level_decoder import (
    CircuitLevelDecoder,
    DeviceCalibration,
)
from msc.steane_code.state_injection_common import NUM_QUBITS_PATCH_14_SITES
from msc.state_transfer.tomography_qiskit import (
    TOMOGRAPHY_BASES,
    SubmittedTomographyCircuit,
    analyze_ty_tomography_counts,
    build_ty_tomography_circuits,
    resolve_ty_tomography_config,
)


def _counts_from_records(
    decoder: CircuitLevelDecoder,
    records: np.ndarray,
) -> dict[str, int]:
    """Write Stim-ordered measurement rows as Qiskit count keys (bit 0 last)."""
    width = len(decoder.record_clbits)
    characters = np.full((len(records), width), ord("0"), dtype=np.uint8)
    characters[:, width - 1 - np.asarray(decoder.record_clbits)] += records.astype(np.uint8)
    return dict(Counter(row.tobytes().decode("ascii") for row in characters))


# =============================================================================
# Twirl variants
# =============================================================================


def test_each_twirl_variant_is_decoded_with_its_own_circuit() -> None:
    """Analysis decodes every twirl variant's counts with that twirl variant's calibrated circuit.

    Twirl variants add pulses that cancel logically, so they share their
    noiseless record but not their noise model. Check which circuit constructs
    each real decoder directly: different models can agree on sampled outcomes.
    """
    config = resolve_ty_tomography_config(target_distance=5, patch_sites=14, add_barriers=False)
    logical_circuits = build_ty_tomography_circuits(
        target_distance=5,
        patch_sites=14,
        add_barriers=False,
    )

    # An all-to-all device needs no routing; its X pulses past the Steane patch
    # are deliberately noisy.
    num_qubits = logical_circuits["X"].num_qubits
    backend = GenericBackendV2(
        num_qubits,
        basis_gates=["cz", "sx", "x", "rz"],
        coupling_map=[[a, b] for a in range(num_qubits) for b in range(num_qubits) if a != b],
        seed=1,
    )
    target = backend.target
    for qubit in range(num_qubits):
        target.update_instruction_properties(
            "reset",
            (qubit,),
            InstructionProperties(duration=target["measure"][(qubit,)].duration),
        )
    calibration = DeviceCalibration(
        dt=target.dt,
        relaxation_times=tuple((qubit.t1, qubit.t2) for qubit in target.qubit_properties),
        errors={
            (name, qubits): (
                0.05 if name == "x" and qubits[0] >= NUM_QUBITS_PATCH_14_SITES else properties.error
            )
            for name in target.operation_names
            for qubits, properties in target[name].items()
        },
        durations={
            (name, qubits): properties.duration
            for name in target.operation_names
            for qubits, properties in target[name].items()
        },
    )

    twirl_variants_by_basis = {}
    expected_shots_by_basis = {}
    for basis, logical_circuit in logical_circuits.items():
        bit_layout = logical_circuit.metadata["measurement_schema"]
        # Hardware preprocessing drops resets on fresh wires, as the decoder expects.
        first_circuit = transpile(
            PassManager([RemoveResetInZeroState()]).run(logical_circuit),
            backend,
            initial_layout=list(range(num_qubits)),
            optimization_level=0,
            scheduling_method="asap",
        )
        circuits = (first_circuit, _with_late_x_pairs(first_circuit, target))
        decoders = [
            CircuitLevelDecoder(
                circuit,
                calibration=calibration,
                tomography_config=config,
                measurement_bit_layout=bit_layout,
                readout_basis=basis,
                persistent_surface_x_post_selection=True,
                complementary_gap=False,
            )
            for circuit in circuits
        ]
        twirl_variants = tuple(
            SubmittedTomographyCircuit(
                circuit=circuit,
                measurement_bit_layout=bit_layout,
                counts=_counts_from_records(
                    decoder, decoder.circuit.without_noise().compile_sampler(seed=seed).sample(16)
                ),
            )
            for seed, (circuit, decoder) in enumerate(zip(circuits, decoders, strict=True), start=1)
        )
        # These are distinct physical noise models even when their decoded counts agree.
        first_decoder, second_decoder = decoders
        assert first_decoder.circuit != second_decoder.circuit
        twirl_variants_by_basis[basis] = twirl_variants
        expected_shots_by_basis[basis] = np.sum(
            [
                decoder.decode_counts(twirl_variant.counts)
                for decoder, twirl_variant in zip(decoders, twirl_variants, strict=True)
            ],
            axis=0,
        )

    with patch(
        "msc.state_transfer.tomography_qiskit.CircuitLevelDecoder", wraps=CircuitLevelDecoder
    ) as decoder_constructor:
        (summary,) = analyze_ty_tomography_counts(
            {
                basis: dict(
                    sum((Counter(twirl_variant.counts) for twirl_variant in twirl_variants), Counter())
                )
                for basis, twirl_variants in twirl_variants_by_basis.items()
            },
            tomography_config=config,
            persistent_surface_x_post_selection=True,
            complementary_gap_thresholds=(None,),
            submitted_circuits_by_basis=twirl_variants_by_basis,
            calibration=calibration,
        )
    for call, (basis, variant) in zip(
        decoder_constructor.call_args_list,
        ((basis, variant) for basis in TOMOGRAPHY_BASES for variant in twirl_variants_by_basis[basis]),
        strict=True,
    ):
        assert call.args[0] is variant.circuit
        assert call.kwargs["readout_basis"] == basis
        assert call.kwargs["calibration"] is calibration
    for basis, accepted_shots, expectation in (
        ("X", summary.accepted_x_shots, summary.expectation_x),
        ("Z", summary.accepted_z_shots, summary.expectation_z),
    ):
        logical_zero_shots, logical_one_shots = expected_shots_by_basis[basis]
        assert accepted_shots == logical_zero_shots + logical_one_shots
        assert expectation == pytest.approx(
            (logical_zero_shots - logical_one_shots) / accepted_shots
        )


def _with_late_x_pairs(circuit: QuantumCircuit, target: Target) -> QuantumCircuit:
    """Return a twirl-like variant: idles in the last 40% become X-X pairs plus a shorter idle."""
    twirl_variant = circuit.copy_empty_like()
    first_replaced_index = int(0.6 * len(circuit.data))
    for index, instruction in enumerate(circuit.data):
        if index >= first_replaced_index and instruction.operation.name == "delay":
            (qubit,) = instruction.qubits
            x_duration = round(target["x"][(circuit.find_bit(qubit).index,)].duration / target.dt)
            idle_duration = instruction.operation.duration - 2 * x_duration
            if idle_duration >= 0:
                twirl_variant.append(XGate(), [qubit])
                twirl_variant.append(XGate(), [qubit])
                if idle_duration > 0:
                    twirl_variant.append(
                        Delay(idle_duration, unit=instruction.operation.unit), [qubit]
                    )
                continue
        twirl_variant.append(instruction)
    return twirl_variant
