import stim

from research_qec_haoran.noise import (
    inject_pauli_errors_after_instruction_occurrence,
    inject_pauli_errors_after_measurement_round,
    inject_pauli_errors_before_final_measurement,
)
from research_qec_haoran.surface_code import generate_d3_unrotated_memory_circuit


def test_inject_pauli_errors_after_measurement_round() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    injected = inject_pauli_errors_after_measurement_round(
        circuit,
        pauli="X",
        coords=[(2, 2)],
        measurement_round=1,
    )

    assert "MR 1 3 5 7 9 11 13 15 17 19 21 23\nX_ERROR(1) 12" in str(injected)


def test_inject_pauli_errors_after_instruction_occurrence() -> None:
    circuit = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        M 0
        TICK
        M 1
        """
    )

    injected = inject_pauli_errors_after_instruction_occurrence(
        circuit,
        pauli="Z",
        coords=[(1, 0)],
        instruction_name="M",
        occurrence=2,
    )

    assert str(injected).endswith("M 1\nZ_ERROR(1) 1")


def test_inject_pauli_errors_before_final_measurement_uses_final_block() -> None:
    circuit = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        M 0
        TICK
        MX 0
        MY 1
        """
    )

    injected = inject_pauli_errors_before_final_measurement(
        circuit,
        pauli="X",
        coords=[(0, 0)],
    )

    expected = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        M 0
        TICK
        X_ERROR(1) 0
        MX 0
        MY 1
        """
    )
    assert injected == expected
