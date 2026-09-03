"""Tests for the Steane-code cultivation helpers."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator

from msc.noise import build_noisy_aer_simulator
from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    STEANE_QED_MEASUREMENTS_PER_ROUND,
)
from msc.steane_code.magic_state_cultivation_qiskit import (
    build_11_site_patch_injection_and_cultivation_circuit,
    build_14_site_patch_injection_and_cultivation_circuit,
    postselect_success_counts,
)
from msc.steane_code.state_injection_common import (
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_DEFAULT_LOGICAL_X,
    PATCH_14_SITE_DEFAULT_LOGICAL_Z,
)
from msc.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_11_site_patch_logical_measurement,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_logical_measurement,
    append_14_site_patch_superdense_stabilizer_measurement,
)
from msc.utils.normalize import qiskit_bit_value

# =============================================================================
# Constants
# =============================================================================

NUM_SHOTS = 1000
INV_SQRT2 = 1.0 / math.sqrt(2)
NOISY_TEST_SEED = 42


# =============================================================================
# Per-patch cultivation configuration
# =============================================================================


@dataclass(frozen=True)
class PatchCultivationSuite:
    """Bundles the pieces that differ between the 11-site and 14-site patches."""

    name: str
    patch_sites: int
    build_circuit: Callable[..., QuantumCircuit]
    append_logical_measurement: Callable[..., None]
    append_stabilizer_measurement: Callable[..., None]
    default_logical_x: tuple[int, ...]
    default_logical_z: tuple[int, ...]
    encoder_flag_bits: int
    qed_measurements_per_round: int
    cultivation_measurements_per_round: int
    accepted_test_states: tuple[InjectedStateName, ...]
    success_rate_test_states: tuple[InjectedStateName, ...]
    stabilizer_check_states: tuple[InjectedStateName, ...]


ELEVEN_SITE = PatchCultivationSuite(
    name="11-site",
    patch_sites=11,
    build_circuit=functools.partial(
        build_11_site_patch_injection_and_cultivation_circuit,
        skip_cx_1_4=False,
    ),
    append_logical_measurement=append_11_site_patch_logical_measurement,
    append_stabilizer_measurement=append_11_site_patch_stabilizer_measurement,
    default_logical_x=PATCH_11_SITE_DEFAULT_LOGICAL_X,
    default_logical_z=PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    encoder_flag_bits=len(PATCH_11_SITE_HELPER_SITES),
    qed_measurements_per_round=STEANE_QED_MEASUREMENTS_PER_ROUND,
    cultivation_measurements_per_round=PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    accepted_test_states=("Ty", "1", "+", "Y+", "T"),
    success_rate_test_states=("Ty", "1", "+", "Y+", "T"),
    stabilizer_check_states=("Ty", "0", "+", "T"),
)

FOURTEEN_SITE = PatchCultivationSuite(
    name="14-site",
    patch_sites=14,
    build_circuit=build_14_site_patch_injection_and_cultivation_circuit,
    append_logical_measurement=append_14_site_patch_logical_measurement,
    append_stabilizer_measurement=append_14_site_patch_superdense_stabilizer_measurement,
    default_logical_x=PATCH_14_SITE_DEFAULT_LOGICAL_X,
    default_logical_z=PATCH_14_SITE_DEFAULT_LOGICAL_Z,
    encoder_flag_bits=len(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES),
    qed_measurements_per_round=PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    cultivation_measurements_per_round=PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    accepted_test_states=("T",),
    success_rate_test_states=("T",),
    stabilizer_check_states=("T",),
)

PATCH_SUITES: tuple[PatchCultivationSuite, ...] = (ELEVEN_SITE, FOURTEEN_SITE)
PATCH_IDS: tuple[str, ...] = tuple(patch.name for patch in PATCH_SUITES)

PATCH_ACCEPTED_STATE_CASES: tuple[tuple[PatchCultivationSuite, InjectedStateName], ...] = tuple(
    (patch, state) for patch in PATCH_SUITES for state in patch.accepted_test_states
)
PATCH_ACCEPTED_STATE_IDS: tuple[str, ...] = tuple(
    f"{patch.name}-{state}" for patch, state in PATCH_ACCEPTED_STATE_CASES
)

PATCH_SUCCESS_RATE_STATE_CASES: tuple[tuple[PatchCultivationSuite, InjectedStateName], ...] = tuple(
    (patch, state) for patch in PATCH_SUITES for state in patch.success_rate_test_states
)
PATCH_SUCCESS_RATE_STATE_IDS: tuple[str, ...] = tuple(
    f"{patch.name}-{state}" for patch, state in PATCH_SUCCESS_RATE_STATE_CASES
)


def _build_cultivation_with_logical_measurement(
    patch: PatchCultivationSuite,
    state: InjectedStateName,
    basis: str,
    *,
    n_rounds: int = 1,
) -> QuantumCircuit:
    """Build injection, cultivation rounds, and destructive logical readout."""
    circuit = patch.build_circuit(
        state,
        cultivation_rounds=n_rounds,
        add_qed_round=True,
        add_barriers=False,
    )

    # Append destructive logical measurement (3 classical bits)
    support = patch.default_logical_z if basis == "Z" else patch.default_logical_x
    logical_creg = ClassicalRegister(len(support), "logical")
    circuit.add_register(logical_creg)
    patch.append_logical_measurement(
        circuit,
        basis,
        patch_qubits=list(circuit.qregs[0]),
        support=support,
        bits=list(logical_creg),
        destructive=True,
    )

    return circuit


def _n_postselection_bits(patch: PatchCultivationSuite, n_rounds: int) -> int:
    """Return encoder-flag, QED, and cultivation postselection bits."""
    return (
        patch.encoder_flag_bits
        + patch.qed_measurements_per_round
        + n_rounds * patch.cultivation_measurements_per_round
    )


def _postselect_and_compute_expectation(
    counts: dict[str, int],
    circuit: QuantumCircuit,
) -> tuple[float, int]:
    """Postselect on encoder, QED, and cultivation bits, then return expectation.

    Uses :func:`postselect_success_counts` from the cultivation module for the
    postselection filtering. The circuit's final classical register contains
    the logical readout; every preceding bit is postselection data.

    Parameters
    ----------
    counts
        Qiskit counts dictionary.
    circuit
        Executed circuit with the logical readout as its final classical
        register.
    """
    logical_bit_count = len(circuit.cregs[-1])
    logical_bit_start = circuit.num_clbits - logical_bit_count

    accepted_counts = postselect_success_counts(
        counts,
        postselection_bit_positions=range(logical_bit_start),
    )
    total_accepted = sum(accepted_counts.values())

    if total_accepted == 0:
        return 0.0, 0

    even_parity_count = 0
    for bitstring, count in accepted_counts.items():
        parity = (
            sum(
                qiskit_bit_value(bitstring, position)
                for position in range(logical_bit_start, circuit.num_clbits)
            )
            % 2
        )
        if parity == 0:
            even_parity_count += count

    expectation = (2 * even_parity_count - total_accepted) / total_accepted
    return expectation, total_accepted


# =============================================================================
# Test Set 1: Postselected logical expectation values and success rates
# =============================================================================


class TestPostselectedLogicalExpectation:
    """Postselection projects onto the logical H_XY +1 state."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("patch", "state"), PATCH_ACCEPTED_STATE_CASES, ids=PATCH_ACCEPTED_STATE_IDS
    )
    @pytest.mark.parametrize("basis", ["Z", "X", "Y"])
    def test_expectation_matches_t_state(
        self,
        patch: PatchCultivationSuite,
        state: InjectedStateName,
        basis: str,
    ) -> None:
        """Accepted shots have the logical Pauli expectations of T|+⟩."""
        circuit = _build_cultivation_with_logical_measurement(patch, state, basis)
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        expectation, n_accepted = _postselect_and_compute_expectation(counts, circuit)

        # Must have at least some accepted shots
        assert (
            n_accepted > 0
        ), f"No accepted shots for patch={patch.name}, state={state!r}, basis={basis}"

        # Statistical tolerance: ~4σ where σ = sqrt((1 - ev²) / n_accepted).
        # For T|+⟩, ⟨Z_L⟩ = 0 and ⟨X_L⟩ = ⟨Y_L⟩ = 1/√2.
        expected = 0.0 if basis == "Z" else INV_SQRT2
        tolerance = 4 * math.sqrt((1.0 - expected**2) / n_accepted) + 0.01
        assert expectation == pytest.approx(expected, abs=tolerance), (
            f"{patch.name} state {state!r}, basis {basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
            f"expected {expected:.4f} after H_XY cultivation (n_accepted={n_accepted})"
        )

    @pytest.mark.parametrize(
        ("patch", "state"), PATCH_SUCCESS_RATE_STATE_CASES, ids=PATCH_SUCCESS_RATE_STATE_IDS
    )
    def test_success_rate(
        self,
        patch: PatchCultivationSuite,
        state: InjectedStateName,
    ) -> None:
        """Acceptance rate matches |⟨T|ψ⟩|² for each injected state."""
        # Use the cultivation-only circuit (no logical measurement needed)
        circuit = patch.build_circuit(
            state,
            add_qed_round=True,
            add_barriers=False,
        )
        sim = AerSimulator()
        counts = sim.run(circuit, shots=NUM_SHOTS).result().get_counts()

        # Measured success rate: fraction passing encoder, QED, and cultivation checks.
        total = sum(counts.values())
        accepted_counts = postselect_success_counts(counts)
        measured_rate = sum(accepted_counts.values()) / total

        # Expected P(accept) = |⟨T|ψ⟩|², where |T⟩ = T|+⟩ is the +1 eigenstate
        # of H_XY = (X + Y) / √2.
        cos_pi8 = math.cos(math.pi / 8)
        sin_pi8 = math.sin(math.pi / 8)
        expected_rates: dict[InjectedStateName, float] = {
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
        expected_rate = expected_rates[state]

        # For |T⟩ input, acceptance must be exactly 1.0.
        if state == "T":
            assert (
                measured_rate == 1.0
            ), f"{patch.name} T-state acceptance should be 1.0, got {measured_rate:.4f}"
        else:
            # Statistical tolerance for binomial proportion: ~4σ
            sigma = math.sqrt(expected_rate * (1 - expected_rate) / NUM_SHOTS)
            tolerance = max(4 * sigma, 0.02)
            assert measured_rate == pytest.approx(expected_rate, abs=tolerance), (
                f"{patch.name} state {state!r}: acceptance = {measured_rate:.4f}, "
                f"expected = {expected_rate:.4f}"
            )

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_builder_allocates_every_postselection_bit(
        self,
        patch: PatchCultivationSuite,
    ) -> None:
        circuit = patch.build_circuit("T", add_qed_round=True, add_barriers=False)
        counts = AerSimulator().run(circuit, shots=10).result().get_counts()

        expected_bits = _n_postselection_bits(patch, n_rounds=1)
        assert circuit.num_qubits == patch.patch_sites
        assert circuit.num_clbits == expected_bits
        assert sum(postselect_success_counts(counts).values()) == 10


# =============================================================================
# Test Set 2: Stabilizer syndromes after cultivation
# =============================================================================


class TestStabilizerSyndromesAfterCultivation:
    """In noiseless setting, stabilizers remain +1 after successful cultivation."""

    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    def test_all_stabilizers_trivial(self, patch: PatchCultivationSuite) -> None:
        """All stabilizer results and the 14-site D2-copy flag are zero."""
        for state in patch.stabilizer_check_states:
            circuit = patch.build_circuit(state, add_qed_round=True, add_barriers=False)

            stab_creg = ClassicalRegister(patch.qed_measurements_per_round, "stab")
            circuit.add_register(stab_creg)
            measurement_destinations = (
                {
                    "stabilizer_bits": list(stab_creg)[:STEANE_QED_MEASUREMENTS_PER_ROUND],
                    "d2_copy_flag_bit": stab_creg[-1],
                }
                if patch.patch_sites == 14
                else {"bits": list(stab_creg)}
            )
            patch.append_stabilizer_measurement(
                circuit,
                patch_qubits=list(circuit.qregs[0]),
                **measurement_destinations,
                add_barriers=False,
            )

            # A T-state input deterministically accepts; use 1 shot.
            # For others, use enough shots to guarantee at least one acceptance.
            n_shots = 1 if state == "T" else 50
            counts = AerSimulator().run(circuit, shots=n_shots).result().get_counts()

            # Check all accepted shots have all-zero stabilizer outcomes.
            postselection_bit_count = circuit.num_clbits - len(stab_creg)
            found_accepted = False
            for bitstring in counts:
                if any(
                    qiskit_bit_value(bitstring, position)
                    for position in range(postselection_bit_count)
                ):
                    continue  # rejected shot

                found_accepted = True
                assert all(
                    qiskit_bit_value(bitstring, position) == 0
                    for position in range(postselection_bit_count, circuit.num_clbits)
                ), (
                    f"{patch.name} state {state!r}: accepted shot has non-trivial syndrome "
                    f"in bitstring {bitstring!r} (expected all zeros)"
                )

            assert (
                found_accepted
            ), f"{patch.name} state {state!r}: no accepted shots in {n_shots} attempts"


# =============================================================================
# Test Set 3: Noisy cultivation approaches ideal
# =============================================================================


class TestNoisyCultivation:
    """With depolarizing noise, postselected cultivation preserves logical |T_L⟩ fidelity.

    Under circuit-level depolarizing noise, the cultivation check acts as an
    error-detection step: shots where noise flipped the logical H_XY eigenvalue
    are preferentially rejected.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("patch", PATCH_SUITES, ids=PATCH_IDS)
    @pytest.mark.parametrize("basis", ["X", "Y", "Z"])
    def test_postselected_fidelity_near_ideal(
        self,
        patch: PatchCultivationSuite,
        basis: str,
    ) -> None:
        """Check the postselected logical X, Y, and Z expectations of |T_L⟩."""
        expected = INV_SQRT2 if basis in {"X", "Y"} else 0.0

        circuit = _build_cultivation_with_logical_measurement(patch, "T", basis)
        noisy_simulator = build_noisy_aer_simulator(
            circuit,
            seed_simulator=NOISY_TEST_SEED,
        )
        counts = noisy_simulator.run(circuit, shots=NUM_SHOTS).result().get_counts()
        expectation, n_accepted = _postselect_and_compute_expectation(counts, circuit)

        assert (
            n_accepted > 50
        ), f"Too few accepted shots ({n_accepted}) for patch={patch.name}, basis={basis}."

        # Allow four standard errors plus a 2% systematic noise budget.
        # Estimate variance from the observed noisy ±1 outcomes rather
        # than the ideal-state expectation.
        statistical_error = math.sqrt((1.0 - expectation**2) / n_accepted)
        tolerance = 4 * statistical_error + 0.02
        assert expectation == pytest.approx(expected, abs=tolerance), (
            f"{patch.name} basis={basis}: ⟨{basis}_L⟩ = {expectation:.4f}, "
            f"expected ≈ {expected:.4f} (n_accepted={n_accepted}, tol={tolerance:.4f})"
        )
