from research_qec_haoran.surface_code.decoding import (
    PauliFrame,
    build_matching_decoder,
    decode_memory_shots,
    generate_noisy_memory_circuit,
)
from research_qec_haoran.surface_code.decoding.unrotated_d3_memory_stim import (
    _boundary_data_qubit,
    _interpret_stabilizer_sector,
    _interpret_time_label,
    _shared_data_qubit,
    AnnotatedDetector,
    D3ErrorInterpreter,
)
from research_qec_haoran.noise import (
    inject_pauli_errors_after_measurement_round,
    inject_pauli_errors_before_final_measurement,
)
from research_qec_haoran.surface_code import generate_d3_unrotated_memory_circuit
from research_qec_haoran.surface_code.decoding.unrotated_d3_memory_stim import detector_coords_by_index


# ---------------------------------------------------------------------------
# Helper: shared fixtures
# ---------------------------------------------------------------------------

def _circuit_and_matching_decoder(rounds: int = 2, basis: str = "Z"):
    circuit = generate_noisy_memory_circuit(distance=3, rounds=rounds, basis=basis, error_probability=1e-3)
    matching_decoder = build_matching_decoder(circuit)
    return circuit, matching_decoder


def _noiseless_circuit(rounds: int = 2, basis: str = "Z"):
    """Return the noiseless version of the same circuit layout for deterministic fault injection."""
    return generate_d3_unrotated_memory_circuit(rounds=rounds, basis=basis)


# ---------------------------------------------------------------------------
# Unit tests: _interpret_stabilizer_sector
# ---------------------------------------------------------------------------


class TestDetectorSector:
    """Verify stabilizer sector classification from Stim (x, y) coordinates."""

    def test_z_stabilizer_even_x_odd_y(self) -> None:
        assert _interpret_stabilizer_sector(0, 1) == "Z"
        assert _interpret_stabilizer_sector(2, 3) == "Z"
        assert _interpret_stabilizer_sector(4, 1) == "Z"

    def test_x_stabilizer_odd_x_even_y(self) -> None:
        assert _interpret_stabilizer_sector(1, 0) == "X"
        assert _interpret_stabilizer_sector(3, 2) == "X"
        assert _interpret_stabilizer_sector(1, 4) == "X"

    def test_data_qubit_coords_return_unknown(self) -> None:
        # Data qubits have (even, even) or (odd, odd) — neither stabilizer sector
        assert _interpret_stabilizer_sector(0, 0) == "?"
        assert _interpret_stabilizer_sector(1, 1) == "?"


# ---------------------------------------------------------------------------
# Unit tests: _interpret_time_label
# ---------------------------------------------------------------------------


class TestDetectorTimeLabel:
    def test_preparation_boundary(self) -> None:
        assert _interpret_time_label(0, max_t=2) == "preparation-boundary"

    def test_final_boundary(self) -> None:
        assert _interpret_time_label(2, max_t=2) == "final-boundary"

    def test_bulk(self) -> None:
        assert _interpret_time_label(1, max_t=2) == "bulk"
        assert _interpret_time_label(1, max_t=3) == "bulk"
        assert _interpret_time_label(2, max_t=3) == "bulk"


# ---------------------------------------------------------------------------
# Unit tests: _shared_data_qubit
# ---------------------------------------------------------------------------


class TestSharedDataQubit:
    def test_adjacent_z_stabilizers_share_data_qubit(self) -> None:
        # Z-stabilizers at (2,1) and (2,3) share data qubit (2,2)
        assert _shared_data_qubit(2, 1, 2, 3) == (2, 2)

    def test_adjacent_z_stabilizers_share_boundary_data_qubit(self) -> None:
        # Z-stabilizers at (0,1) and (0,3) share data qubit (0,2)
        assert _shared_data_qubit(0, 1, 0, 3) == (0, 2)

    def test_non_adjacent_stabilizers_return_none(self) -> None:
        # (0,1) and (4,1) midpoint is (2,1) which is a stabilizer, not a data qubit
        assert _shared_data_qubit(0, 1, 4, 1) is None


# ---------------------------------------------------------------------------
# Unit tests: _boundary_data_qubit
# ---------------------------------------------------------------------------


class TestBoundaryDataQubit:
    def test_z_stabilizer_at_01_maps_to_00(self) -> None:
        # Z-stabilizer at (0,1): boundary data qubit is (0,0) on the x=0 edge
        assert _boundary_data_qubit(0, 1) == (0, 0)

    def test_z_stabilizer_at_03_maps_to_04(self) -> None:
        # Z-stabilizer at (0,3): boundary data qubit is (0,4) on the y=4 edge
        assert _boundary_data_qubit(0, 3) == (0, 4)

    def test_z_stabilizer_at_41_maps_to_40(self) -> None:
        # Z-stabilizer at (4,1): boundary data qubit is (4,0) on the x=4 edge
        assert _boundary_data_qubit(4, 1) == (4, 0)

    def test_z_stabilizer_at_43_maps_to_44(self) -> None:
        # Z-stabilizer at (4,3): boundary data qubit is (4,4)
        assert _boundary_data_qubit(4, 3) == (4, 4)

    def test_x_stabilizer_at_10_maps_to_00(self) -> None:
        # X-stabilizer at (1,0): boundary data qubit is (0,0) on the y=0 edge
        assert _boundary_data_qubit(1, 0) == (0, 0)

    def test_x_stabilizer_at_14_maps_to_04(self) -> None:
        # X-stabilizer at (1,4): boundary data qubit is (0,4)
        assert _boundary_data_qubit(1, 4) == (0, 4)


# ---------------------------------------------------------------------------
# Unit tests: _interpret_error_description
# ---------------------------------------------------------------------------


class TestInferErrorDescription:
    """Verify that error descriptions correctly name the Pauli type and data qubit."""

    def test_bulk_x_error_names_correct_pauli_and_qubit(self) -> None:
        # Two Z-stabilizer detectors paired → must say "X error", not "Z error"
        evt_a = AnnotatedDetector(detector_index=9, coord=(2, 1, 1), sector="Z", time_label="bulk")
        evt_b = AnnotatedDetector(detector_index=14, coord=(2, 3, 1), sector="Z", time_label="bulk")
        desc = D3ErrorInterpreter.interpret_error_description(evt_a, evt_b)
        assert desc == "X error on data qubit (2,2), t=1"

    def test_bulk_z_error_from_x_stabilizers(self) -> None:
        # Two X-stabilizer detectors paired → must say "Z error"
        evt_a = AnnotatedDetector(detector_index=6, coord=(1, 0, 1), sector="X", time_label="bulk")
        evt_b = AnnotatedDetector(detector_index=7, coord=(3, 0, 1), sector="X", time_label="bulk")
        desc = D3ErrorInterpreter.interpret_error_description(evt_a, evt_b)
        assert desc == "Z error on data qubit (2,0), t=1"

    def test_boundary_x_error_names_data_qubit(self) -> None:
        # Z-stabilizer at (0,1) matched to boundary → "X error on data qubit (0,0)"
        evt = AnnotatedDetector(detector_index=18, coord=(0, 1, 2), sector="Z", time_label="final-boundary")
        desc = D3ErrorInterpreter.interpret_error_description(evt, "boundary")
        assert desc == "X error on data qubit (0,0), t=2"

    def test_boundary_z_error_from_x_stabilizer(self) -> None:
        # X-stabilizer at (1,0) matched to boundary → "Z error on data qubit (0,0)"
        evt = AnnotatedDetector(detector_index=6, coord=(1, 0, 0), sector="X", time_label="preparation-boundary")
        desc = D3ErrorInterpreter.interpret_error_description(evt, "boundary")
        assert desc == "Z error on data qubit (0,0), t=0"

    def test_boundary_order_does_not_matter(self) -> None:
        evt = AnnotatedDetector(detector_index=18, coord=(0, 1, 2), sector="Z", time_label="final-boundary")
        assert D3ErrorInterpreter.interpret_error_description("boundary", evt) == D3ErrorInterpreter.interpret_error_description(evt, "boundary")

    def test_measurement_error_description(self) -> None:
        # Same stabilizer at different times → measurement error
        evt_a = AnnotatedDetector(detector_index=0, coord=(0, 1, 0), sector="Z", time_label="preparation-boundary")
        evt_b = AnnotatedDetector(detector_index=8, coord=(0, 1, 1), sector="Z", time_label="bulk")
        desc = D3ErrorInterpreter.interpret_error_description(evt_a, evt_b)
        assert "measurement error" in desc
        assert "(0,1)" in desc

    def test_cross_sector_is_y_error(self) -> None:
        # One Z-stabilizer paired with one X-stabilizer at same time → "Y error"
        evt_a = AnnotatedDetector(detector_index=0, coord=(0, 1, 0), sector="Z", time_label="bulk")
        evt_b = AnnotatedDetector(detector_index=1, coord=(1, 0, 0), sector="X", time_label="bulk")
        desc = D3ErrorInterpreter.interpret_error_description(evt_a, evt_b)
        assert desc.startswith("Y error")


# ---------------------------------------------------------------------------
# Unit tests: AnnotatedDetector.__str__ and AnnotatedCorrection.__str__
# ---------------------------------------------------------------------------


class TestStringRepresentations:
    def test_detection_event_str(self) -> None:
        evt = AnnotatedDetector(detector_index=9, coord=(2, 1, 1), sector="Z", time_label="bulk")
        s = str(evt)
        assert "D9" in s
        assert "Z-stabilizer" in s
        assert "(2,1)" in s
        assert "t=1" in s
        assert "[bulk]" in s

    def test_detection_event_str_does_not_say_x_for_z_sector(self) -> None:
        evt = AnnotatedDetector(detector_index=0, coord=(0, 1, 0), sector="Z", time_label="preparation-boundary")
        s = str(evt)
        assert "Z-stabilizer" in s
        assert "X-stabilizer" not in s


# ---------------------------------------------------------------------------
# Integration tests: full decode pipeline
# ---------------------------------------------------------------------------


def test_detector_coordinate_map() -> None:
    circuit = generate_noisy_memory_circuit(distance=3, rounds=2, basis="Z", error_probability=1e-3)
    detector_coords = detector_coords_by_index(circuit)

    assert detector_coords[0] == (0, 1, 0)
    assert detector_coords[23] == (4, 3, 2)


def test_decode_d3_unrotated_memory_shot_without_errors() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()

    decoded_shot = decode_memory_shots(
        circuit, matching_decoder=matching_decoder, logical_readout_basis="Z",
    )[0]

    assert decoded_shot.logical_readout_basis == "Z"
    assert decoded_shot.detection_events == ()
    assert decoded_shot.actual_logical_flip is False
    assert decoded_shot.decoder_inferred_logical_flip is False
    assert decoded_shot.residual_logical_error is False


def test_decode_no_error_as_dict_messages() -> None:
    """Verify as_dict produces correct human-readable strings for the no-error case."""
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    d = decode_memory_shots(noiseless, matching_decoder=matching_decoder, logical_readout_basis="Z")[0].as_dict()

    assert d["detection_events"] == "none — no detectors fired"
    assert d["decoder_corrections"] == "none — decoder inferred no errors"
    assert "no errors occurred" in d["error x correction product"]
    assert d["decoding_succeeded"] is True


def test_decode_single_x_error_between_rounds() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_after_measurement_round(
        noiseless, pauli="X", coords=[(2, 2)], measurement_round=1,
    )

    shot = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]

    assert len(shot.detection_events) == 2
    assert shot.detection_events[0].coord == (2, 1, 1)
    assert shot.detection_events[1].coord == (2, 3, 1)
    # Both must be Z-sector (detecting X errors)
    assert shot.detection_events[0].sector == "Z"
    assert shot.detection_events[1].sector == "Z"
    assert shot.actual_logical_flip is False
    assert shot.decoder_inferred_logical_flip is False
    assert shot.residual_logical_error is False
    # Correction description must say "X error on data qubit (2,2)"
    assert len(shot.decoder_corrections) == 1
    assert shot.decoder_corrections[0].inferred_error_type == "X error on data qubit (2,2), t=1"
    assert shot.decoder_corrections[0].flips_logical is False


def test_decode_single_x_error_between_rounds_as_dict() -> None:
    """Verify as_dict for a corrected non-boundary error."""
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_after_measurement_round(
        noiseless, pauli="X", coords=[(2, 2)], measurement_round=1,
    )
    d = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0].as_dict()

    assert d["decoding_succeeded"] is True
    assert "no frame update needed" in d["error x correction product"]
    assert isinstance(d["detection_events"], list)
    assert len(d["detection_events"]) == 2


def test_decode_single_x_error_before_final_measurement() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(0, 0)],
    )

    shot = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]

    assert len(shot.detection_events) == 1
    assert shot.detection_events[0].coord == (0, 1, 2)
    assert shot.detection_events[0].sector == "Z"
    assert shot.detection_events[0].time_label == "final-boundary"
    assert shot.actual_logical_flip is True
    assert shot.decoder_inferred_logical_flip is True
    assert shot.residual_logical_error is False
    # Correction must say "X error on data qubit (0,0)" and flip L0
    assert len(shot.decoder_corrections) == 1
    assert shot.decoder_corrections[0].inferred_error_type == "X error on data qubit (0,0), t=2"
    assert shot.decoder_corrections[0].flips_logical is True


def test_decode_single_x_error_before_final_as_dict() -> None:
    """Verify as_dict for a boundary error that flips L0."""
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(0, 0)],
    )
    d = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0].as_dict()

    assert d["decoding_succeeded"] is True
    assert "Pauli frame update" in d["error x correction product"]
    assert d["actual_logical_flip"] is True
    assert d["decoder_inferred_logical_flip"] is True


def test_weight_three_logical_x_error_is_undetected() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(0, 0), (0, 2), (0, 4)],
    )

    shot = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]

    assert shot.detection_events == ()
    assert shot.decoder_corrections == ()
    assert shot.actual_logical_flip is True
    assert shot.decoder_inferred_logical_flip is False
    assert shot.residual_logical_error is True


def test_weight_three_logical_x_as_dict() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(0, 0), (0, 2), (0, 4)],
    )
    d = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0].as_dict()

    assert d["decoding_succeeded"] is False
    assert "FLIP" in d["error x correction product"]


# ---------------------------------------------------------------------------
# Integration tests: weight-2 decoder miscorrection
# ---------------------------------------------------------------------------


def test_weight_two_x_error_00_02_causes_logical_error() -> None:
    """X on (0,0)+(0,2): decoder miscorrects, completing the logical X string."""
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(0, 0), (0, 2)],
    )

    shot = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]

    # One detector fires (the unpaired stabilizer at (0,3))
    assert len(shot.detection_events) == 1
    assert shot.detection_events[0].sector == "Z"
    assert shot.residual_logical_error is True
    # Decoder matches to boundary and infers X error on (0,4)
    assert len(shot.decoder_corrections) == 1
    assert "X error on data qubit (0,4)" in shot.decoder_corrections[0].inferred_error_type


def test_weight_two_x_error_02_04_causes_logical_error() -> None:
    """X on (0,2)+(0,4): symmetric miscorrection."""
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(0, 2), (0, 4)],
    )

    shot = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]

    assert len(shot.detection_events) == 1
    assert shot.detection_events[0].sector == "Z"
    assert shot.residual_logical_error is True
    assert len(shot.decoder_corrections) == 1
    assert "X error on data qubit (0,0)" in shot.decoder_corrections[0].inferred_error_type


# ---------------------------------------------------------------------------
# Integration tests: X error on non-logical-Z data qubit at boundary
# ---------------------------------------------------------------------------


def test_x_error_on_non_logical_boundary_qubit() -> None:
    """X on (4,4) — boundary qubit NOT in logical-Z support. Should be corrected."""
    circuit, matching_decoder = _circuit_and_matching_decoder()
    noiseless = _noiseless_circuit()
    injected = inject_pauli_errors_before_final_measurement(
        noiseless, pauli="X", coords=[(4, 4)],
    )

    shot = decode_memory_shots(injected, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]

    assert len(shot.detection_events) == 1
    assert shot.actual_logical_flip is False
    assert shot.decoder_inferred_logical_flip is False
    assert shot.residual_logical_error is False


# ---------------------------------------------------------------------------
# Integration tests: logical observable qubit coords
# ---------------------------------------------------------------------------


def test_logical_observable_qubits_for_z_basis() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder(basis="Z")
    shot = decode_memory_shots(circuit, matching_decoder=matching_decoder, logical_readout_basis="Z")[0]
    # Logical Z for unrotated d=3: top row y=0 → (0,0), (2,0), (4,0)
    assert set(shot.logical_observable_qubits) == {(0, 0), (2, 0), (4, 0)}


def test_logical_observable_qubits_for_x_basis() -> None:
    circuit, matching_decoder = _circuit_and_matching_decoder(basis="X")
    shot = decode_memory_shots(circuit, matching_decoder=matching_decoder, logical_readout_basis="X")[0]
    # Logical X for unrotated d=3: left column x=0 → (0,0), (0,2), (0,4)
    assert set(shot.logical_observable_qubits) == {(0, 0), (0, 2), (0, 4)}


# ---------------------------------------------------------------------------
# Unit tests: PauliFrame
# ---------------------------------------------------------------------------


class TestPauliFrame:
    def test_initial_state_is_identity(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        assert frame.current_byproduct("L0") == "I"

    def test_x_update(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=True)
        assert frame.current_byproduct("L0") == "X"

    def test_z_update(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_z("L0", flip=True)
        assert frame.current_byproduct("L0") == "Z"

    def test_xz_gives_y(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=True)
        frame.update_z("L0", flip=True)
        assert frame.current_byproduct("L0") == "Y"

    def test_double_x_cancels(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=True)
        frame.update_x("L0", flip=True)
        assert frame.current_byproduct("L0") == "I"

    def test_flip_false_is_noop(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=False)
        assert frame.current_byproduct("L0") == "I"

    def test_interpret_z_readout_with_x_flip(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=True)
        # X anticommutes with Z → flips the readout
        assert frame.interpret_z_readout("L0", uncorrected_bit=0) == 1
        assert frame.interpret_z_readout("L0", uncorrected_bit=1) == 0

    def test_interpret_z_readout_with_z_flip(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_z("L0", flip=True)
        # Z commutes with Z → no effect
        assert frame.interpret_z_readout("L0", uncorrected_bit=0) == 0
        assert frame.interpret_z_readout("L0", uncorrected_bit=1) == 1

    def test_interpret_x_readout_with_z_flip(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_z("L0", flip=True)
        # Z anticommutes with X → flips the readout
        assert frame.interpret_x_readout("L0", uncorrected_bit=0) == 1

    def test_interpret_x_readout_with_x_flip(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=True)
        # X commutes with X → no effect
        assert frame.interpret_x_readout("L0", uncorrected_bit=0) == 0

    def test_reset(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.update_x("L0", flip=True)
        frame.update_z("L0", flip=True)
        frame.reset("L0")
        assert frame.current_byproduct("L0") == "I"

    def test_as_dict(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        frame.register("L1")
        frame.update_x("L0", flip=True)
        assert frame.as_dict() == {"L0": "X", "L1": "I"}

    def test_repr(self) -> None:
        frame = PauliFrame()
        frame.register("L0")
        assert "L0: I" in repr(frame)
