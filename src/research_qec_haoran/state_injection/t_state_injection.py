r"""Non-fault-tolerant T-state injection for the d=3 unrotated surface code.

Protocol
--------

1. Prepare the stabilised logical ``|+_L\rangle`` state using the existing
   ``generate_d3_unrotated_memory_circuit(rounds, basis="X")``.

2. Collapse the weight-3 logical Z onto qubit ``q0 = (0,0)`` via::

       CX  q1, q0        # q1 = (2,0)
       CX  q2, q0        # q2 = (4,0)

3. Apply physical ``T = Rz(π/4)`` on ``q0`` (non-fault-tolerant).
   Stim has no native T gate; the rotation is implicit and optionally
   modelled by a depolarising channel.

4. Un-collapse::

       CX  q2, q0
       CX  q1, q0

After this the patch holds ``|T_L\rangle``.  No post-injection
stabilizer extraction is appended — the caller can add that separately
if needed.

Public API
----------
- ``build_t_state_injection_circuit``
"""

from __future__ import annotations

import stim

from research_qec_haoran.surface_code.unrotated_d3 import (
    generate_d3_unrotated_memory_circuit,
)

# ── Stim qubit-index helpers ─────────────────────────────────────────
# index = x + 5*y on Stim's 5×5 grid for the d=3 unrotated code.

_Q0 = 0   # (0, 0) — injection target
_Q1 = 2   # (2, 0)
_Q2 = 4   # (4, 0)


def build_t_state_injection_circuit(
    *,
    rounds: int = 3,
    noise: float = 0.0,
) -> stim.Circuit:
    r"""Build a T-state injection circuit on the d=3 unrotated surface code.

    Parameters
    ----------
    rounds
        Number of stabilizer-extraction rounds used to establish
        ``|+_L\rangle`` before injection (≥ 1).
    noise
        If > 0, a ``DEPOLARIZE1`` channel is placed on the injection
        qubit ``q0`` at the T-rotation step, modelling imperfect
        non-fault-tolerant state preparation.

    Returns
    -------
    stim.Circuit
        The circuit that prepares ``|+_L\rangle``, collapses logical Z
        onto one qubit, applies the (non-FT) T rotation, and
        un-collapses.
    """
    # Phase 1: stabilised |+_L⟩
    c = generate_d3_unrotated_memory_circuit(rounds=rounds, basis="X")

    # Phase 2: collapse logical Z onto q0
    c.append("TICK")
    c.append("CX", [_Q1, _Q0])
    c.append("TICK")
    c.append("CX", [_Q2, _Q0])
    c.append("TICK")

    # Phase 3: T rotation (non-FT)
    if noise > 0:
        c.append("DEPOLARIZE1", [_Q0], noise)
    c.append("TICK")

    # Phase 4: un-collapse
    c.append("CX", [_Q2, _Q0])
    c.append("TICK")
    c.append("CX", [_Q1, _Q0])
    c.append("TICK")

    return c
