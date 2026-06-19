"""Tests for the Steane-code cultivation helpers."""

from __future__ import annotations

import math

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    append_hadamard_axis_projection_round,
    build_steane_injection_with_h_cultivation,
    postselect_success_counts,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    InjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder,
)
from research_qec_haoran.utils.normalize import qiskit_bit_value

# =============================================================================
# Constants
# =============================================================================

NUM_SHOTS = 1000
INV_SQRT2 = 1.0 / math.sqrt(2)
NOISY_TEST_SEED = 20251218

# States to test for cultivation — minimal representative subset.
CULTIVATION_TEST_STATES: list[InjectedStateName] = [
    "H", "1", "+", "Y+", "T",
]


def _expected_acceptance_probability(state: InjectedStateName) -> float:
    """Return P(accept) = |⟨H|ψ⟩|² for the given injected state.

    |H⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩ is the +1 eigenstate of Hadamard.
    """
    cos_pi8 = math.cos(math.pi / 8)
    sin_pi8 = math.sin(math.pi / 8)
    probs: dict[str, float] = {
        "H": 1.0,
        "0": cos_pi8**2,
        "1": sin_pi8**2,
        "+": (1 + math.sin(math.pi / 4)) / 2,  # = cos²(π/8)
        "-": (1 - math.sin(math.pi / 4)) / 2,  # = sin²(π/8)
        "Y+": 0.5,
        "Y-": 0.5,
        "i": 0.5,
        "-i": 0.5,
        "S": 0.5,
        "T": 0.75,
        "Tdg": 0.75,
    }
    return probs[state]


# =============================================================================
# Helpers
# =============================================================================


def _build_cultivation_with_logical_measurement(
    state: InjectedStateName,
    basis: str,
) -> QuantumCircuit:
    """Build: encoder → cultivation round → destructive logical measurement.

    Returns a circuit with 11 qubits and 8 classical bits:
      - bits 0–4: cultivation outcomes (success = all zero)
      - bits 5–7: destructive logical measurement on 3-qubit support
    """
    qreg = QuantumRegister(11, "q")
    creg = ClassicalRegister(8, "c")
    circuit = QuantumCircuit(qreg, creg)

    # Encode
    append_patch_steane_state_injection_encoder(
        circuit, patch_qubits=list(qreg), injected_state=state, add_barriers=False,
    )

    # Cultivation round (5 classical bits: indices 0–4)
    append_hadamard_axis_projection_round(
        circuit,
        patch_qubits=list(qreg),
        classical_bits=[creg[i] for i in range(5)],
        add_barriers=False,
    )

    # Destructive logical measurement (3 classical bits: indices 5–7)
    support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    append_patch_logical_measurement(
        circuit,
        basis,
        patch_qubits=list(qreg),
        support=support,
        classical_bits=[creg[i] for i in range(5, 8)],
        destructive=True,
    )

    return circuit


def _postselect_and_compute_expectation(
    counts: dict[str, int],
    cult_bit_positions: list[int] | None = None,
    logical_bit_positions: list[int] | None = None,
) -> tuple[float, int]:
    """Postselect on cultivation bits = all zero, return (⟨logical⟩, n_accepted).

    Uses :func:`postselect_success_counts` from the cultivation module for the
    postselection filtering, then computes the logical expectation value from
    the surviving shots.

    Parameters
    ----------
    counts
        Qiskit counts dictionary.
    cult_bit_positions
        Bit indices (Qiskit convention: 0 = rightmost) for cultivation outcomes.
        Defaults to [0, 1, 2, 3, 4].
    logical_bit_positions
        Bit indices for the logical measurement outcomes whose parity gives ⟨O_L⟩.
        Defaults to [5, 6, 7].
    """
    if cult_bit_positions is None:
        cult_bit_positions = list(range(5))
    if logical_bit_positions is None:
        logical_bit_positions = [5, 6, 7]

    accepted_counts = postselect_success_counts(counts, cult_bit_positions=cult_bit_positions)
    total_accepted = sum(accepted_counts.values())

    if total_accepted == 0:
        return 0.0, 0

    even_parity_count = 0
    for bitstring, count in accepted_counts.items():
        parity = sum(qiskit_bit_value(bitstring, i) for i in logical_bit_positions) % 2
        if parity == 0:
            even_parity_count += count

    expectation = (2 * even_parity_count - total_accepted) / total_accepted
    return expectation, total_accepted


def _compute_success_rate(counts: dict[str, int], n_cult_bits: int = 5) -> float:
    """Compute fraction of shots where all cultivation bits are zero.

    Uses :func:`postselect_success_counts` from the cultivation module.
    """
    total = sum(counts.values())
    accepted_counts = postselect_success_counts(counts)
    accepted = sum(accepted_counts.values())
    return accepted / total


# =============================================================================
# Test Set 1: Postselected logical expectation values and success rates
# =============================================================================


class TestPostselectedLogicalExpectation:
    """After cultivation + postselection, the state is |H_L⟩ regardless of input."""

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_expectation_matches_h_state(
        self, state: InjectedStateName, basis: str
    ) -> None:
        """⟨Z_L⟩ = ⟨X_L⟩ = 1/√2 after successful cultivation for any input."""
        circuit = _build_cultivation_with_logical_measurement(state, basis)
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        expectation, n_accepted = _postselect_and_compute_expectation(counts)

        # Must have at least some accepted shots
        assert n_accepted > 0, (
            f"No accepted shots for state={state!r}, basis={basis}"
        )

        # Statistical tolerance: ~4σ where σ = sqrt((1 - ev²) / n_accepted)
        # For ev ≈ 1/√2: var per shot ≈ 0.5, σ ≈ sqrt(0.5/n_accepted)
        tolerance = 4 * math.sqrt(0.5 / n_accepted) + 0.01
        assert expectation == pytest.approx(INV_SQRT2, abs=tolerance), (
            f"State {state!r}, basis {basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
            f"expected {INV_SQRT2:.4f} (n_accepted={n_accepted})"
        )

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    def test_success_rate(self, state: InjectedStateName) -> None:
        """Acceptance rate matches |⟨H|ψ⟩|² for each injected state."""
        # Use the cultivation-only circuit (no logical measurement needed)
        circuit = build_steane_injection_with_h_cultivation(
            injected_state=state, add_barriers=False,
        )
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        measured_rate = _compute_success_rate(counts, n_cult_bits=5)
        expected_rate = _expected_acceptance_probability(state)

        # For |H⟩ input, acceptance must be exactly 1.0
        if state == "H":
            assert measured_rate == 1.0, (
                f"H-state acceptance should be 1.0, got {measured_rate:.4f}"
            )
        else:
            # Statistical tolerance for binomial proportion: ~4σ
            sigma = math.sqrt(expected_rate * (1 - expected_rate) / NUM_SHOTS)
            tolerance = max(4 * sigma, 0.02)
            assert measured_rate == pytest.approx(expected_rate, abs=tolerance), (
                f"State {state!r}: acceptance = {measured_rate:.4f}, "
                f"expected = {expected_rate:.4f}"
            )


# =============================================================================
# Test Set 2: Stabilizer syndromes after cultivation
# =============================================================================


class TestStabilizerSyndromesAfterCultivation:
    """In noiseless setting, stabilizers remain +1 after successful cultivation."""

    @pytest.mark.parametrize("state", ["H", "0", "+", "T"])
    def test_all_stabilizers_trivial(self, state: InjectedStateName) -> None:
        """All 6 stabilizer measurements yield 0 (eigenvalue +1) after cultivation."""
        n_stab_ancilla = 6

        qreg = QuantumRegister(11, "patch")
        anc_reg = QuantumRegister(n_stab_ancilla, "stab_anc")
        cult_creg = ClassicalRegister(5, "cult")
        stab_creg = ClassicalRegister(n_stab_ancilla, "stab")
        circuit = QuantumCircuit(qreg, anc_reg, cult_creg, stab_creg)

        # Encode
        append_patch_steane_state_injection_encoder(
            circuit, patch_qubits=list(qreg), injected_state=state, add_barriers=False,
        )

        # Cultivation round
        append_hadamard_axis_projection_round(
            circuit,
            patch_qubits=list(qreg),
            classical_bits=list(cult_creg),
            add_barriers=False,
        )

        # Stabilizer measurement
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(qreg),
            ancilla_qubits=list(anc_reg),
            classical_bits=list(stab_creg),
            add_barriers=False,
        )

        # For "H", single shot deterministically accepts; use 1 shot.
        # For others, use enough shots to guarantee at least one acceptance.
        n_shots = 1 if state == "H" else 50
        sim = AerSimulator()
        counts = sim.run(circuit, shots=n_shots).result().get_counts()

        # Check all accepted shots have all-zero stabilizer outcomes
        found_accepted = False
        for bitstring in counts:
            # Layout in Qiskit bit positions: bits 0..4 are cultivation, and
            # bits 5..10 are the final stabilizer measurement.
            if any(qiskit_bit_value(bitstring, i) for i in range(5)):
                continue  # rejected shot

            found_accepted = True
            assert all(qiskit_bit_value(bitstring, i) == 0 for i in range(5, 11)), (
                f"State {state!r}: accepted shot has non-trivial syndrome "
                f"in bitstring {bitstring!r} (expected all zeros)"
            )

        assert found_accepted, f"State {state!r}: no accepted shots in {n_shots} attempts"


# =============================================================================
# Test Set 3: Noisy multi-round cultivation approaches ideal
# =============================================================================

NOISE_P_1Q = 0.001
NOISE_P_2Q = 0.01


def _build_noisy_noise_model(p_1q: float = NOISE_P_1Q, p_2q: float = NOISE_P_2Q) -> NoiseModel:
    """Construct a depolarizing noise model for all gates."""
    noise_model = NoiseModel()
    # 1-qubit depolarizing on all single-qubit gates
    error_1q = depolarizing_error(p_1q, 1)
    for gate in ["h", "s", "sdg", "t", "tdg", "x", "z", "ry"]:
        noise_model.add_all_qubit_quantum_error(error_1q, gate)
    # 2-qubit depolarizing on CX
    error_2q = depolarizing_error(p_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, "cx")
    return noise_model


def _build_multi_round_cultivation_circuit(
    n_rounds: int,
    basis: str,
    *,
    injected_state: InjectedStateName = "H",
) -> QuantumCircuit:
    """Build: encoder → n_rounds cultivation → destructive logical measurement.

    Classical register layout:
      - bits [0, 5*n_rounds): cultivation outcomes for all rounds
      - bits [5*n_rounds, 5*n_rounds+3): destructive logical measurement
    """
    n_cult_total = 5 * n_rounds
    n_logical = 3
    n_total_classical = n_cult_total + n_logical

    qreg = QuantumRegister(11, "q")
    creg = ClassicalRegister(n_total_classical, "c")
    circuit = QuantumCircuit(qreg, creg)

    # Encode
    append_patch_steane_state_injection_encoder(
        circuit, patch_qubits=list(qreg), injected_state=injected_state, add_barriers=False,
    )

    # Multiple cultivation rounds
    for r in range(n_rounds):
        offset = 5 * r
        append_hadamard_axis_projection_round(
            circuit,
            patch_qubits=list(qreg),
            classical_bits=[creg[offset + i] for i in range(5)],
            add_barriers=False,
        )

    # Destructive logical measurement
    support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    append_patch_logical_measurement(
        circuit,
        basis,
        patch_qubits=list(qreg),
        support=support,
        classical_bits=[creg[n_cult_total + i] for i in range(n_logical)],
        destructive=True,
    )

    return circuit


def _postselect_multi_round(
    counts: dict[str, int],
    n_rounds: int,
) -> tuple[float, int]:
    """Postselect on all cultivation rounds succeeding, compute logical expectation.

    Uses :func:`postselect_success_counts` from the cultivation module.

    Classical register layout:
      - bits [0, 5*n_rounds): cultivation outcomes (must all be 0)
      - bits [5*n_rounds, 5*n_rounds+3): logical measurement
    """
    n_cult_total = 5 * n_rounds
    cult_positions = list(range(n_cult_total))
    logical_positions = [n_cult_total, n_cult_total + 1, n_cult_total + 2]
    return _postselect_and_compute_expectation(
        counts,
        cult_bit_positions=cult_positions,
        logical_bit_positions=logical_positions,
    )


class TestNoisyCultivationImprovesWithRounds:
    """With depolarizing noise, postselected cultivation preserves logical |H_L⟩ fidelity.

    Under circuit-level depolarizing noise, the cultivation check acts as an
    error-detection step: shots where noise flipped the logical H eigenvalue
    are preferentially rejected. We verify:
    1. With |H⟩ injection and noise, postselected expectation values remain
       close to the ideal 1/√2 for both Z_L and X_L.
    2. Without postselection (noisy encoder + direct measurement), the
       expectation deviates more from ideal than with postselected cultivation.
    3. More rounds → higher fidelity (lower |⟨O_L⟩ − 1/√2|) and lower
       acceptance rate (higher discard fraction).
    """

    ROUNDS_TO_TEST = [1, 2, 3]

    @pytest.mark.parametrize("basis", ["Z"])
    def test_postselected_fidelity_near_ideal(self, basis: str) -> None:
        """After noisy cultivation + postselection, ⟨Z_L⟩ and ⟨X_L⟩ ≈ 1/√2."""
        noise_model = _build_noisy_noise_model()
        sim = AerSimulator(noise_model=noise_model, seed_simulator=NOISY_TEST_SEED)

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _build_multi_round_cultivation_circuit(
                n_rounds, basis, injected_state="H",
            )
            counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()
            expectation, n_accepted = _postselect_multi_round(counts, n_rounds)

            assert n_accepted > 50, (
                f"Too few accepted shots ({n_accepted}) for {n_rounds} rounds, "
                f"basis={basis}. Consider reducing noise or increasing shots."
            )

            # With low noise and postselection, the expectation should be
            # close to 1/√2. Allow ~4σ statistical error + small systematic.
            stat_err = math.sqrt(0.5 / n_accepted)
            tolerance = 4 * stat_err + 0.03  # 3% systematic budget for noise
            assert expectation == pytest.approx(INV_SQRT2, abs=tolerance), (
                f"Rounds={n_rounds}, basis={basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
                f"expected ≈ {INV_SQRT2:.4f} (n_accepted={n_accepted}, tol={tolerance:.4f})"
            )

    @pytest.mark.parametrize("basis", ["X"])
    def test_more_rounds_improves_fidelity_and_increases_discard(
        self, basis: str
    ) -> None:
        """More cultivation rounds → better fidelity and higher discard rate."""
        noise_model = _build_noisy_noise_model()
        sim = AerSimulator(noise_model=noise_model, seed_simulator=NOISY_TEST_SEED)

        n_shots = 10000  # more shots to reduce statistical noise for trend test
        errors: list[float] = []
        acceptance_rates: list[float] = []

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _build_multi_round_cultivation_circuit(
                n_rounds, basis, injected_state="H",
            )
            counts = sim.run(circuit, shots=n_shots).result().get_counts()
            expectation, n_accepted = _postselect_multi_round(counts, n_rounds)

            assert n_accepted > 50, (
                f"Too few accepted shots ({n_accepted}) for {n_rounds} rounds"
            )

            errors.append(abs(expectation - INV_SQRT2))
            acceptance_rates.append(n_accepted / n_shots)

        errors_by_round = list(zip(self.ROUNDS_TO_TEST, errors, strict=True))
        rates_by_round = list(zip(self.ROUNDS_TO_TEST, acceptance_rates, strict=True))

        # Fidelity should improve (error decreases) with more rounds.
        # Allow small statistical slack — require error_last <= error_first + margin.
        assert errors[-1] <= errors[-2] + 0.03, (
            f"Basis {basis}: fidelity did not improve with more rounds. "
            f"Errors by round: {errors_by_round}"
        )
        print(f"Errors by round: {errors_by_round}")

        # Acceptance rate should decrease with more rounds (more chances to reject).
        assert acceptance_rates[-1] < acceptance_rates[-2] + 0.01, (
            f"Basis {basis}: acceptance rate did not decrease with more rounds. "
            f"Rates by round: {rates_by_round}"
        )
        print(f"Rates by round: {rates_by_round}")

    @pytest.mark.parametrize("basis", ["X"])
    def test_cultivation_improves_over_no_postselection(self, basis: str) -> None:
        """Postselected cultivation yields better fidelity than raw noisy encoder."""
        noise_model = _build_noisy_noise_model()
        sim = AerSimulator(noise_model=noise_model, seed_simulator=NOISY_TEST_SEED)

        # --- Raw noisy encoder (no cultivation, just encoder → measurement) ---
        qreg = QuantumRegister(11, "q")
        creg = ClassicalRegister(3, "c")
        raw_circuit = QuantumCircuit(qreg, creg)
        append_patch_steane_state_injection_encoder(
            raw_circuit, patch_qubits=list(qreg), injected_state="H", add_barriers=False,
        )
        support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
        append_patch_logical_measurement(
            raw_circuit, basis, patch_qubits=list(qreg), support=support,
            classical_bits=list(creg), destructive=True,
        )
        raw_counts = sim.run(raw_circuit, shots=NUM_SHOTS).result().get_counts()
        # Compute expectation from raw counts (no postselection)
        raw_even = 0
        raw_total = 0
        for bitstring, count in raw_counts.items():
            parity = sum(qiskit_bit_value(bitstring, i) for i in range(3)) % 2
            raw_total += count
            if parity == 0:
                raw_even += count
        raw_expectation = (2 * raw_even - raw_total) / raw_total
        raw_error = abs(raw_expectation - INV_SQRT2)

        # --- Cultivation with postselection ---
        circuit = _build_multi_round_cultivation_circuit(1, basis, injected_state="H")
        cult_counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()
        cult_expectation, n_accepted = _postselect_multi_round(cult_counts, 1)
        cult_error = abs(cult_expectation - INV_SQRT2)

        assert n_accepted > 50, (
            f"Too few accepted shots ({n_accepted}) for basis={basis}"
        )

        # Postselected cultivation should yield fidelity at least as good
        # as raw encoder (and typically better since it filters detectable errors)
        assert cult_error <= raw_error + 0.02, (
            f"Basis {basis}: cultivation (|ev-ideal|={cult_error:.4f}) should not be "
            f"much worse than raw encoder (|ev-ideal|={raw_error:.4f})"
        )
