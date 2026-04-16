from research_qec_haoran.decoding import (
    build_d3_unrotated_memory_matching_decoder,
    decode_memory_shots,
    detector_coords_by_index,
)
from research_qec_haoran.noise import (
    inject_pauli_errors_after_measurement_round,
    inject_pauli_errors_before_final_measurement,
)
from research_qec_haoran.surface_code import generate_d3_unrotated_memory_circuit


def test_detector_coordinate_map() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    detector_coords = detector_coords_by_index(circuit)

    assert detector_coords[0] == (0, 1, 0)
    assert detector_coords[23] == (4, 3, 2)


def test_decode_d3_unrotated_memory_shot_without_errors() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    matching = build_d3_unrotated_memory_matching_decoder(rounds=2, basis="Z")

    decoded_shot = decode_memory_shots(
        circuit,
        matching=matching,
        logical_readout_basis="Z",
    )[0]

    assert decoded_shot.stim_observable_index == 0
    assert decoded_shot.logical_readout_basis == "Z"
    assert decoded_shot.triggered_detectors == ()
    assert decoded_shot.triggered_detector_coords == ()
    assert decoded_shot.measured_logical_readout_flip == 0
    assert decoded_shot.decoder_inferred_logical_readout_flip == 0
    assert decoded_shot.post_decoding_logical_readout_mismatch == 0


def test_decode_single_x_error_between_rounds() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    injected = inject_pauli_errors_after_measurement_round(
        circuit,
        pauli="X",
        coords=[(2, 2)],
        measurement_round=1,
    )
    matching = build_d3_unrotated_memory_matching_decoder(rounds=2, basis="Z")

    decoded_shot = decode_memory_shots(
        injected,
        matching=matching,
        logical_readout_basis="Z",
    )[0]

    assert decoded_shot.stim_observable_index == 0
    assert decoded_shot.logical_readout_basis == "Z"
    assert decoded_shot.triggered_detectors == (9, 14)
    assert decoded_shot.triggered_detector_coords == ((2, 1, 1), (2, 3, 1))
    assert decoded_shot.measured_logical_readout_flip == 0
    assert decoded_shot.decoder_inferred_logical_readout_flip == 0
    assert decoded_shot.post_decoding_logical_readout_mismatch == 0


def test_decode_single_x_error_before_final_measurement() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    injected = inject_pauli_errors_before_final_measurement(
        circuit,
        pauli="X",
        coords=[(0, 0)],
    )
    matching = build_d3_unrotated_memory_matching_decoder(rounds=2, basis="Z")

    decoded_shot = decode_memory_shots(
        injected,
        matching=matching,
        logical_readout_basis="Z",
    )[0]

    assert decoded_shot.stim_observable_index == 0
    assert decoded_shot.logical_readout_basis == "Z"
    assert decoded_shot.triggered_detectors == (18,)
    assert decoded_shot.triggered_detector_coords == ((0, 1, 2),)
    assert decoded_shot.measured_logical_readout_flip == 1
    assert decoded_shot.decoder_inferred_logical_readout_flip == 1
    assert decoded_shot.post_decoding_logical_readout_mismatch == 0


def test_weight_three_logical_x_error_is_undetected() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    injected = inject_pauli_errors_before_final_measurement(
        circuit,
        pauli="X",
        coords=[(0, 0), (0, 2), (0, 4)],
    )
    matching = build_d3_unrotated_memory_matching_decoder(rounds=2, basis="Z")

    decoded_shot = decode_memory_shots(
        injected,
        matching=matching,
        logical_readout_basis="Z",
    )[0]

    assert decoded_shot.stim_observable_index == 0
    assert decoded_shot.logical_readout_basis == "Z"
    assert decoded_shot.triggered_detectors == ()
    assert decoded_shot.triggered_detector_coords == ()
    assert decoded_shot.measured_logical_readout_flip == 1
    assert decoded_shot.decoder_inferred_logical_readout_flip == 0
    assert decoded_shot.post_decoding_logical_readout_mismatch == 1
