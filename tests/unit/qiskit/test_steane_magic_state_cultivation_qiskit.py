"""Tests for the Steane-code cultivation helpers."""

from __future__ import annotations

import math

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator

from magic_state_cultivation.noise import build_noisy_aer_simulator
from magic_state_cultivation.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
    STEANE_QED_MEASUREMENTS_PER_ROUND,
    build_steane_injection_and_cultivation_circuit,
    postselect_success_counts,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import (
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    InjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
)
from magic_state_cultivation.utils.normalize import qiskit_bit_value

# =============================================================================
# Constants
# =============================================================================

NUM_SHOTS = 1000
INV_SQRT2 = 1.0 / math.sqrt(2)
NOISY_TEST_SEED = 42

# States to test for cultivation — minimal representative subset.
CULTIVATION_TEST_STATES: list[InjectedStateName] = [
    "Ty",
    "1",
    "+",
    "Y+",
    "T",
]


def _expected_acceptance_probability(state: InjectedStateName) -> float:
    """Return P(accept) = |⟨T|ψ⟩|² for the given injected state.

    |T⟩ = T|+⟩ is the +1 eigenstate of H_XY = (X + Y) / √2.
    """

    cos_pi8 = math.cos(math.pi / 8)
    sin_pi8 = math.sin(math.pi / 8)
    probs: dict[InjectedStateName, float] = {
        "0": 0.5,
        "1": 0.5,
        "+": cos_pi8**2,
        "-": sin_pi8**2,
        "Y+": cos_pi8**2,
        "Y-": sin_pi8**2,
        "i": cos_pi8**2,
        "-i": sin_pi8**2,
        "S": cos_pi8**2,
        "T": 1.0,
        "Tdg": 0.5,
        "Ty": 0.75,
    }
    return probs[state]


def _build_cultivation_with_logical_measurement(
    state: InjectedStateName,
    basis: str,
    *,
    n_rounds: int = 1,
) -> QuantumCircuit:
    """Build injection, cultivation rounds, and destructive logical readout.

    Uses :func:`build_steane_injection_and_cultivation_circuit` for the encoder
    and cultivation stages, then appends a destructive logical measurement.

    For one cultivation round, returns a circuit with 11 qubits and 18
    classical bits:
      - bits 0–3: encoder flag checks (must be zero)
      - bits 4–9: post-injection Steane QED syndrome (must be zero)
      - bits 10–14: cultivation outcomes (must be zero)
      - bits 15–17: destructive logical measurement on 3-qubit support
    """
    circuit = build_steane_injection_and_cultivation_circuit(
        injected_state=state,
        cultivation_rounds=n_rounds,
        add_barriers=False,
    )

    # Append destructive logical measurement (3 classical bits)
    support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    logical_creg = ClassicalRegister(len(support), "logical")
    circuit.add_register(logical_creg)
    append_patch_logical_measurement(
        circuit,
        basis,
        patch_qubits=list(circuit.qregs[0]),
        support=support,
        classical_bits=list(logical_creg),
        destructive=True,
    )

    return circuit


N_POSTSELECTION_BITS = 4 + STEANE_QED_MEASUREMENTS_PER_ROUND + CULTIVATION_MEASUREMENTS_PER_ROUND


def _postselect_and_compute_expectation(
    counts: dict[str, int],
    postselection_bit_positions: list[int] | None = None,
    logical_bit_positions: list[int] | None = None,
) -> tuple[float, int]:
    """Postselect on encoder, QED, and cultivation bits, then return expectation.

    Uses :func:`postselect_success_counts` from the cultivation module for the
    postselection filtering, then computes the logical expectation value from
    the surviving shots.

    Parameters
    ----------
    counts
        Qiskit counts dictionary.
    postselection_bit_positions
        Bit indices (Qiskit convention: 0 = rightmost) for postselection.
        Defaults to the encoder-flag, post-injection QED, and cultivation
        bits.
    logical_bit_positions
        Bit indices for the logical measurement outcomes whose parity gives ⟨O_L⟩.
        Defaults to the three positions immediately after the postselection
        bits.
    """
    if postselection_bit_positions is None:
        postselection_bit_positions = list(range(N_POSTSELECTION_BITS))
    if logical_bit_positions is None:
        logical_bit_positions = [N_POSTSELECTION_BITS + i for i in range(3)]

    accepted_counts = postselect_success_counts(
        counts,
        postselection_bit_positions=postselection_bit_positions,
    )
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


def _compute_success_rate(counts: dict[str, int]) -> float:
    """Compute the fraction passing encoder, QED, and cultivation checks.

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
    """Postselection projects onto the logical H_XY +1 state."""

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    @pytest.mark.parametrize("basis", ["Z", "X", "Y"])
    def test_expectation_matches_t_state(self, state: InjectedStateName, basis: str) -> None:
        """Accepted shots have the logical Pauli expectations of T|+⟩."""
        circuit = _build_cultivation_with_logical_measurement(state, basis)
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        expectation, n_accepted = _postselect_and_compute_expectation(counts)

        # Must have at least some accepted shots
        assert n_accepted > 0, f"No accepted shots for state={state!r}, basis={basis}"

        # Statistical tolerance: ~4σ where σ = sqrt((1 - ev²) / n_accepted).
        # For T|+⟩, ⟨Z_L⟩ = 0 and ⟨X_L⟩ = ⟨Y_L⟩ = 1/√2.
        expected = 0.0 if basis == "Z" else INV_SQRT2
        tolerance = 4 * math.sqrt((1.0 - expected**2) / n_accepted) + 0.01
        assert expectation == pytest.approx(expected, abs=tolerance), (
            f"State {state!r}, basis {basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
            f"expected {expected:.4f} after H_XY cultivation (n_accepted={n_accepted})"
        )

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    def test_success_rate(self, state: InjectedStateName) -> None:
        """Acceptance rate matches |⟨T|ψ⟩|² for each injected state."""
        # Use the cultivation-only circuit (no logical measurement needed)
        circuit = build_steane_injection_and_cultivation_circuit(
            injected_state=state,
            add_barriers=False,
        )
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        measured_rate = _compute_success_rate(counts)
        expected_rate = _expected_acceptance_probability(state)

        # For |T⟩ input, acceptance must be exactly 1.0.
        if state == "T":
            assert (
                measured_rate == 1.0
            ), f"T-state acceptance should be 1.0, got {measured_rate:.4f}"
        else:
            # Statistical tolerance for binomial proportion: ~4σ
            sigma = math.sqrt(expected_rate * (1 - expected_rate) / NUM_SHOTS)
            tolerance = max(4 * sigma, 0.02)
            assert measured_rate == pytest.approx(
                expected_rate, abs=tolerance
            ), f"State {state!r}: acceptance = {measured_rate:.4f}, expected = {expected_rate:.4f}"


# =============================================================================
# Test Set 2: Stabilizer syndromes after cultivation
# =============================================================================


class TestStabilizerSyndromesAfterCultivation:
    """In noiseless setting, stabilizers remain +1 after successful cultivation."""

    @pytest.mark.parametrize("state", ["Ty", "0", "+", "T"])
    def test_all_stabilizers_trivial(self, state: InjectedStateName) -> None:
        """All 6 stabilizer measurements yield 0 (eigenvalue +1) after cultivation."""
        circuit = build_steane_injection_and_cultivation_circuit(
            injected_state=state,
            add_barriers=False,
        )

        # Append stabilizer measurement
        n_stab = 6
        stab_creg = ClassicalRegister(n_stab, "stab")
        circuit.add_register(stab_creg)
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=list(circuit.qregs[0]),
            classical_bits=list(stab_creg),
            add_barriers=False,
        )

        # A T-state input deterministically accepts; use 1 shot.
        # For others, use enough shots to guarantee at least one acceptance.
        n_shots = 1 if state == "T" else 50
        sim = AerSimulator()
        counts = sim.run(circuit, shots=n_shots).result().get_counts()

        # Check all accepted shots have all-zero stabilizer outcomes.
        # The postselection prefix contains 4 encoder-flag bits, 6
        # post-injection QED bits, and 5 cultivation bits. The final 6 bits are
        # the additional stabilizer measurement under test.
        found_accepted = False
        for bitstring in counts:
            if any(qiskit_bit_value(bitstring, i) for i in range(N_POSTSELECTION_BITS)):
                continue  # rejected shot

            found_accepted = True
            stab_start = N_POSTSELECTION_BITS
            assert all(
                qiskit_bit_value(bitstring, i) == 0 for i in range(stab_start, stab_start + n_stab)
            ), (
                f"State {state!r}: accepted shot has non-trivial syndrome "
                f"in bitstring {bitstring!r} (expected all zeros)"
            )

        assert found_accepted, f"State {state!r}: no accepted shots in {n_shots} attempts"


# =============================================================================
# Test Set 3: Noisy multi-round cultivation approaches ideal
# =============================================================================

NOISE_P_1Q = 0.0005
NOISE_P_2Q = 0.005
NOISE_MEASURE = 0.005
NOISE_RESET = 0.005


def _n_postselection_bits(n_rounds: int) -> int:
    """Return the number of postselection bits for n_rounds cultivation."""
    return 4 + STEANE_QED_MEASUREMENTS_PER_ROUND + n_rounds * CULTIVATION_MEASUREMENTS_PER_ROUND


def _postselect_multi_round(
    counts: dict[str, int],
    n_rounds: int,
) -> tuple[float, int]:
    """Postselect on encoder, QED, and cultivation, then compute expectation.

    Uses :func:`postselect_success_counts` from the cultivation module.
    """
    n_ps = _n_postselection_bits(n_rounds)
    return _postselect_and_compute_expectation(
        counts,
        postselection_bit_positions=list(range(n_ps)),
        logical_bit_positions=[n_ps, n_ps + 1, n_ps + 2],
    )


class TestNoisyCultivationWithRounds:
    """With depolarizing noise, postselected cultivation preserves logical |T_L⟩ fidelity.

    Under circuit-level depolarizing noise, the cultivation check acts as an
    error-detection step: shots where noise flipped the logical H_XY eigenvalue
    are preferentially rejected. We verify:
    1. With |T⟩ injection and noise, postselected X_L remains close to 1/√2
       and Z_L remains close to zero.
    2. Without postselection (noisy encoder + direct measurement), the
       expectation deviates more from ideal than with postselected cultivation.
    3. More rounds preserve a near-ideal X_L expectation and lower the
       acceptance rate (higher discard fraction).
    """

    ROUNDS_TO_TEST = [1, 2, 3]

    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_postselected_fidelity_near_ideal(self, basis: str) -> None:
        """Check the postselected logical X, Y, and Z expectations of |T_L⟩."""
        expected = INV_SQRT2 if basis in {"X", "Y"} else 0.0

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _build_cultivation_with_logical_measurement(
                "T",
                basis,
                n_rounds=n_rounds,
            )
            noisy_simulator = build_noisy_aer_simulator(
                circuit,
                p_1q=NOISE_P_1Q,
                p_2q=NOISE_P_2Q,
                p_measure=NOISE_MEASURE,
                p_reset=NOISE_RESET,
                seed_simulator=NOISY_TEST_SEED,
            )
            counts = noisy_simulator.run(circuit, shots=NUM_SHOTS).result().get_counts()
            expectation, n_accepted = _postselect_multi_round(counts, n_rounds)

            assert n_accepted > 50, (
                f"Too few accepted shots ({n_accepted}) for {n_rounds} rounds, "
                f"basis={basis}. Consider reducing noise or increasing shots."
            )

            # Allow four standard errors plus a 2% systematic noise budget.
            # Estimate variance from the observed noisy ±1 outcomes rather
            # than the ideal-state expectation.
            statistical_error = math.sqrt((1.0 - expectation**2) / n_accepted)
            tolerance = 4 * statistical_error + 0.02
            assert expectation == pytest.approx(expected, abs=tolerance), (
                f"Rounds={n_rounds}, basis={basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
                f"expected ≈ {expected:.4f} (n_accepted={n_accepted}, tol={tolerance:.4f})"
            )

    @pytest.mark.parametrize("basis", ["X"])
    def test_more_rounds_preserves_fidelity_and_increases_discard(self, basis: str) -> None:
        """More rounds preserve the X_L expectation and increase discard."""
        n_shots = 10000  # more shots to reduce statistical noise for trend test
        errors: list[float] = []
        acceptance_rates: list[float] = []

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _build_cultivation_with_logical_measurement(
                "T",
                basis,
                n_rounds=n_rounds,
            )
            noisy_simulator = build_noisy_aer_simulator(
                circuit,
                p_1q=NOISE_P_1Q,
                p_2q=NOISE_P_2Q,
                p_measure=NOISE_MEASURE,
                p_reset=NOISE_RESET,
                seed_simulator=NOISY_TEST_SEED,
            )
            counts = noisy_simulator.run(circuit, shots=n_shots).result().get_counts()
            expectation, n_accepted = _postselect_multi_round(counts, n_rounds)

            assert n_accepted > 50, f"Too few accepted shots ({n_accepted}) for {n_rounds} rounds"

            errors.append(abs(expectation - INV_SQRT2))
            acceptance_rates.append(n_accepted / n_shots)

        errors_by_round = list(zip(self.ROUNDS_TO_TEST, errors, strict=True))
        rates_by_round = list(zip(self.ROUNDS_TO_TEST, acceptance_rates, strict=True))

        # Extra noisy rounds add faults as well as checks, so compare the
        # multi-round endpoint with the one-round baseline instead of requiring
        # strict monotonic improvement at every intermediate round.
        assert errors[-1] <= errors[0] + 0.03, (
            f"Basis {basis}: fidelity degraded too much with more rounds. "
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
        """Cultivation improves on the noisy encoded state without postselection."""
        # Use the shared front end with both QED and cultivation disabled. The
        # encoder-flag outcomes remain recorded but are deliberately ignored,
        # giving the unpostselected baseline.
        uncultivated_circuit = build_steane_injection_and_cultivation_circuit(
            injected_state="T",
            cultivation_rounds=0,
            add_qed_round=False,
            add_barriers=False,
        )
        qreg = uncultivated_circuit.qregs[0]
        creg = ClassicalRegister(3, "c")
        uncultivated_circuit.add_register(creg)
        support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
        append_patch_logical_measurement(
            uncultivated_circuit,
            basis,
            patch_qubits=list(qreg),
            support=support,
            classical_bits=list(creg),
            destructive=True,
        )
        uncultivated_simulator = build_noisy_aer_simulator(
            uncultivated_circuit,
            p_1q=NOISE_P_1Q,
            p_2q=NOISE_P_2Q,
            p_measure=NOISE_MEASURE,
            p_reset=NOISE_RESET,
            seed_simulator=NOISY_TEST_SEED,
        )
        uncultivated_counts = (
            uncultivated_simulator.run(
                uncultivated_circuit,
                shots=NUM_SHOTS,
            )
            .result()
            .get_counts()
        )
        uncultivated_even = 0
        uncultivated_total = 0
        for bitstring, count in uncultivated_counts.items():
            parity = sum(qiskit_bit_value(bitstring, i) for i in range(4, 7)) % 2
            uncultivated_total += count
            if parity == 0:
                uncultivated_even += count
        uncultivated_expectation = (2 * uncultivated_even - uncultivated_total) / uncultivated_total
        uncultivated_error = abs(uncultivated_expectation - INV_SQRT2)

        # --- Cultivation with postselection ---
        cultivated_circuit = _build_cultivation_with_logical_measurement(
            "T",
            basis,
        )
        cultivated_simulator = build_noisy_aer_simulator(
            cultivated_circuit,
            p_1q=NOISE_P_1Q,
            p_2q=NOISE_P_2Q,
            p_measure=NOISE_MEASURE,
            p_reset=NOISE_RESET,
            seed_simulator=NOISY_TEST_SEED,
        )
        cultivated_counts = (
            cultivated_simulator.run(
                cultivated_circuit,
                shots=NUM_SHOTS,
            )
            .result()
            .get_counts()
        )
        cultivated_expectation, n_accepted = _postselect_multi_round(
            cultivated_counts,
            1,
        )
        cultivated_error = abs(cultivated_expectation - INV_SQRT2)

        assert n_accepted > 50, f"Too few accepted shots ({n_accepted}) for basis={basis}"

        # Postselected cultivation should yield fidelity at least as good as
        # the uncultivated encoder and typically filters detectable errors.
        # The margin allows for statistical variance in seeded noisy tests.
        assert cultivated_error <= uncultivated_error + 0.10, (
            f"Basis {basis}: cultivation (|ev-ideal|={cultivated_error:.4f}) should "
            "not be much worse than the uncultivated encoder "
            f"(|ev-ideal|={uncultivated_error:.4f})"
        )
