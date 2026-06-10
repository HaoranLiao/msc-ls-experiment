"""Tests for the Steane-code cultivation helpers."""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    DATA_SITES,
    FOLDED_MEASUREMENT_SITE,
    HELPER_SITES,
    append_fold_network,
    append_unfold_network,
    apply_ty,
    apply_ty_dagger,
    build_steane_injection_with_h_cultivation,
    postselect_success_counts,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    append_patch_steane_state_injection_encoder,
)


