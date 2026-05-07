"""Tests for T-state injection on the d=3 unrotated surface code."""

from __future__ import annotations

import pytest

from research_qec_haoran.state_injection import build_t_state_injection_circuit


class TestBuildTStateInjectionCircuit:

    def test_basic_structure(self) -> None:
        c = build_t_state_injection_circuit(rounds=3)
        assert c.num_qubits == 25
        assert c.num_observables == 1  # inherited from the memory-X circuit

    def test_uses_rx_init(self) -> None:
        """Memory-X initialises data qubits with RX (|+_L⟩)."""
        c = build_t_state_injection_circuit(rounds=1)
        assert "RX" in str(c)

    def test_collapse_uncollapse_present(self) -> None:
        c = build_t_state_injection_circuit(rounds=1)
        text = str(c)
        assert "CX 2 0" in text
        assert "CX 4 0" in text

    def test_noise_channel_added(self) -> None:
        c = build_t_state_injection_circuit(rounds=1, noise=1e-3)
        assert "DEPOLARIZE1(0.001) 0" in str(c)

    def test_no_noise_by_default(self) -> None:
        c = build_t_state_injection_circuit(rounds=1)
        assert "DEPOLARIZE1" not in str(c)
