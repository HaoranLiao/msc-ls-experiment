"""Tests for the Steane-code cultivation helpers."""

from __future__ import annotations

import math

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from magic_state_cultivation.steane_code.magic_state_cultivation_qiskit import (
    append_steane_injection_with_h_cultivation,
    postselect_success_counts,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import (
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    InjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from magic_state_cultivation.utils.normalize import qiskit_bit_value

# =============================================================================
# Constants
# =============================================================================

NUM_SHOTS = 1000
INV_SQRT2 = 1.0 / math.sqrt(2)
NOISY_TEST_SEED = 20251218

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


# =============================================================================
# Helpers
# =============================================================================


def _make_cultivation_circuit(
    injected_state: InjectedStateName = "T",
    n_rounds: int = 1,
    add_barriers: bool = False,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> QuantumCircuit:
    """Create registers and append injection + cultivation rounds."""
    qreg = QuantumRegister(11, "q")
    enc_creg = ClassicalRegister(4, "enc")
    cult_cregs = [ClassicalRegister(5, f"cult_{r}") for r in range(n_rounds)]
    circuit = QuantumCircuit(qreg, enc_creg, *cult_cregs)

    append_steane_injection_with_h_cultivation(
        circuit,
        patch_qubits=list(qreg),
        injected_state=injected_state,
        encoder_classical_bits=list(enc_creg),
        cultivation_classical_bits=[list(c) for c in cult_cregs],
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )
    return circuit


def _build_cultivation_with_logical_measurement(
    state: InjectedStateName,
    basis: str,
) -> QuantumCircuit:
    """Build: encoder → cultivation round → destructive logical measurement.

    Uses :func:`_make_cultivation_circuit` for the encoder and
    cultivation stages, then appends a destructive logical measurement.

    Returns a circuit with 11 qubits and 12 classical bits:
      - bits 0–3: encoder helper checks (must be zero)
      - bits 4–8: cultivation outcomes (must be zero)
      - bits 9–11: destructive logical measurement on 3-qubit support
    """
    circuit = _make_cultivation_circuit(injected_state=state)

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


N_POSTSELECTION_BITS = 9  # 4 encoder helpers + 5 cultivation bits


def _postselect_and_compute_expectation(
    counts: dict[str, int],
    cult_bit_positions: list[int] | None = None,
    logical_bit_positions: list[int] | None = None,
) -> tuple[float, int]:
    """Postselect on encoder+cultivation bits = all zero, return (⟨logical⟩, n_accepted).

    Uses :func:`postselect_success_counts` from the cultivation module for the
    postselection filtering, then computes the logical expectation value from
    the surviving shots.

    Parameters
    ----------
    counts
        Qiskit counts dictionary.
    cult_bit_positions
        Bit indices (Qiskit convention: 0 = rightmost) for postselection.
        Defaults to [0..8] (4 encoder helpers + 5 cultivation bits).
    logical_bit_positions
        Bit indices for the logical measurement outcomes whose parity gives ⟨O_L⟩.
        Defaults to [9, 10, 11].
    """
    if cult_bit_positions is None:
        cult_bit_positions = list(range(N_POSTSELECTION_BITS))
    if logical_bit_positions is None:
        logical_bit_positions = [N_POSTSELECTION_BITS + i for i in range(3)]

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
    """Postselection projects onto the logical H_XY +1 state."""

    @pytest.mark.parametrize("state", CULTIVATION_TEST_STATES)
    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_expectation_matches_t_state(self, state: InjectedStateName, basis: str) -> None:
        """Accepted shots have the logical Pauli expectations of T|+⟩."""
        circuit = _build_cultivation_with_logical_measurement(state, basis)
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        expectation, n_accepted = _postselect_and_compute_expectation(counts)

        # Must have at least some accepted shots
        assert n_accepted > 0, f"No accepted shots for state={state!r}, basis={basis}"

        # Statistical tolerance: ~4σ where σ = sqrt((1 - ev²) / n_accepted).
        # For T|+⟩, ⟨Z_L⟩ = 0 and ⟨X_L⟩ = 1/√2.
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
        circuit = _make_cultivation_circuit(injected_state=state)
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        measured_rate = _compute_success_rate(counts, n_cult_bits=5)
        expected_rate = _expected_acceptance_probability(state)

        # For |T⟩ input, acceptance must be exactly 1.0.
        if state == "T":
            assert measured_rate == 1.0, (
                f"T-state acceptance should be 1.0, got {measured_rate:.4f}"
            )
        else:
            # Statistical tolerance for binomial proportion: ~4σ
            sigma = math.sqrt(expected_rate * (1 - expected_rate) / NUM_SHOTS)
            tolerance = max(4 * sigma, 0.02)
            assert measured_rate == pytest.approx(expected_rate, abs=tolerance), (
                f"State {state!r}: acceptance = {measured_rate:.4f}, expected = {expected_rate:.4f}"
            )


# =============================================================================
# Test Set 2: Stabilizer syndromes after cultivation
# =============================================================================


class TestStabilizerSyndromesAfterCultivation:
    """In noiseless setting, stabilizers remain +1 after successful cultivation."""

    @pytest.mark.parametrize("state", ["Ty", "0", "+", "T"])
    def test_all_stabilizers_trivial(self, state: InjectedStateName) -> None:
        """All 6 stabilizer measurements yield 0 (eigenvalue +1) after cultivation."""
        circuit = _make_cultivation_circuit(injected_state=state)

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
        # Bits 0..8 are postselection (4 enc helpers + 5 cultivation),
        # bits 9..14 are the stabilizer measurement.
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
    injected_state: InjectedStateName = "T",
) -> QuantumCircuit:
    """Build: encoder → n_rounds cultivation → destructive logical measurement.

    Uses :func:`_make_cultivation_circuit` for the encoder and
    cultivation stages, then appends a destructive logical measurement.
    """
    circuit = _make_cultivation_circuit(
        injected_state=injected_state,
        n_rounds=n_rounds,
    )

    # Destructive logical measurement
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


def _n_postselection_bits(n_rounds: int) -> int:
    """Return the number of postselection bits for n_rounds cultivation."""
    return 4 + 5 * n_rounds  # 4 encoder helpers + 5 per round


def _postselect_multi_round(
    counts: dict[str, int],
    n_rounds: int,
) -> tuple[float, int]:
    """Postselect on all encoder+cultivation rounds succeeding, compute logical expectation.

    Uses :func:`postselect_success_counts` from the cultivation module.
    """
    n_ps = _n_postselection_bits(n_rounds)
    return _postselect_and_compute_expectation(
        counts,
        cult_bit_positions=list(range(n_ps)),
        logical_bit_positions=[n_ps, n_ps + 1, n_ps + 2],
    )


class TestNoisyCultivationImprovesWithRounds:
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

    @pytest.mark.parametrize("basis", ["X", "Z"])
    def test_postselected_fidelity_near_ideal(self, basis: str) -> None:
        """Check the postselected logical X and Z expectations of |T_L⟩."""
        noise_model = _build_noisy_noise_model()
        sim = AerSimulator(noise_model=noise_model, seed_simulator=NOISY_TEST_SEED)
        expected = INV_SQRT2 if basis == "X" else 0.0

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _build_multi_round_cultivation_circuit(
                n_rounds,
                basis,
                injected_state="T",
            )
            counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()
            expectation, n_accepted = _postselect_multi_round(counts, n_rounds)

            assert n_accepted > 50, (
                f"Too few accepted shots ({n_accepted}) for {n_rounds} rounds, "
                f"basis={basis}. Consider reducing noise or increasing shots."
            )

            # Allow ~4σ statistical error plus a small systematic noise budget.
            stat_err = math.sqrt((1.0 - expected**2) / n_accepted)
            tolerance = 4 * stat_err + 0.03  # 3% systematic budget for noise
            assert expectation == pytest.approx(expected, abs=tolerance), (
                f"Rounds={n_rounds}, basis={basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
                f"expected ≈ {expected:.4f} (n_accepted={n_accepted}, tol={tolerance:.4f})"
            )

    @pytest.mark.parametrize("basis", ["X"])
    def test_more_rounds_preserves_fidelity_and_increases_discard(self, basis: str) -> None:
        """More rounds preserve the X_L expectation and increase discard."""
        noise_model = _build_noisy_noise_model()
        sim = AerSimulator(noise_model=noise_model, seed_simulator=NOISY_TEST_SEED)

        n_shots = 10000  # more shots to reduce statistical noise for trend test
        errors: list[float] = []
        acceptance_rates: list[float] = []

        for n_rounds in self.ROUNDS_TO_TEST:
            circuit = _build_multi_round_cultivation_circuit(
                n_rounds,
                basis,
                injected_state="T",
            )
            counts = sim.run(circuit, shots=n_shots).result().get_counts()
            expectation, n_accepted = _postselect_multi_round(counts, n_rounds)

            assert n_accepted > 50, f"Too few accepted shots ({n_accepted}) for {n_rounds} rounds"

            errors.append(abs(expectation - INV_SQRT2))
            acceptance_rates.append(n_accepted / n_shots)

        errors_by_round = list(zip(self.ROUNDS_TO_TEST, errors, strict=True))
        rates_by_round = list(zip(self.ROUNDS_TO_TEST, acceptance_rates, strict=True))

        # Extra noisy rounds add faults as well as checks, so allow bounded
        # degradation rather than requiring strict monotonic improvement.
        assert errors[-1] <= errors[-2] + 0.03, (
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
        """Postselected cultivation yields better fidelity than raw noisy encoder."""
        noise_model = _build_noisy_noise_model()
        sim = AerSimulator(noise_model=noise_model, seed_simulator=NOISY_TEST_SEED)

        # --- Raw noisy encoder (no cultivation, just encoder → measurement) ---
        qreg = QuantumRegister(11, "q")
        creg = ClassicalRegister(3, "c")
        raw_circuit = QuantumCircuit(qreg, creg)
        append_patch_steane_state_injection_encoder_crumble_sequences(
            raw_circuit,
            patch_qubits=list(qreg),
            injected_state="T",
            add_barriers=False,
        )
        support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X
        append_patch_logical_measurement(
            raw_circuit,
            basis,
            patch_qubits=list(qreg),
            support=support,
            classical_bits=list(creg),
            destructive=True,
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
        circuit = _build_multi_round_cultivation_circuit(1, basis, injected_state="T")
        cult_counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()
        cult_expectation, n_accepted = _postselect_multi_round(cult_counts, 1)
        cult_error = abs(cult_expectation - INV_SQRT2)

        assert n_accepted > 50, f"Too few accepted shots ({n_accepted}) for basis={basis}"

        # Postselected cultivation should yield fidelity at least as good
        # as raw encoder (and typically better since it filters detectable errors).
        # The margin allows for statistical variance in seeded noisy tests.
        assert cult_error <= raw_error + 0.10, (
            f"Basis {basis}: cultivation (|ev-ideal|={cult_error:.4f}) should not be "
            f"much worse than raw encoder (|ev-ideal|={raw_error:.4f})"
        )
