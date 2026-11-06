from __future__ import annotations

import stim

from msc.noise.fault_injection_stim import add_si1000_noise, schedule_si1000_moments


def test_si1000_moments_batch_measurements_and_record_them_in_moment_order() -> None:
    """Pin one kind per moment, measurements deferred while a gate is ready, and record order.

    The first ``M 0`` waits for ``CX 1 2`` and is measured with ``M 1``, so the output records
    ``M 0, M 1, M 0`` and the detector on ``M 1`` is re-targeted. The virtual ``S 1`` takes no
    moment and precedes ``q1``'s next gate; the second ``R 0`` gets its own moment.
    """
    circuit = stim.Circuit(
        """
        R 0 1
        H 0
        M 0
        R 0
        H 0
        M 0
        H 1
        S 1
        CX 1 2
        M 1
        DETECTOR rec[-1]
        """
    )
    scheduled_circuit, measurement_order = schedule_si1000_moments(circuit)

    assert measurement_order == [0, 2, 1]
    assert scheduled_circuit == stim.Circuit(
        """
        R 0 1
        TICK
        H 0 1
        TICK
        S 1
        CX 1 2
        TICK
        M 0 1
        TICK
        R 0
        TICK
        H 0
        TICK
        M 0
        TICK
        DETECTOR rec[-2]
        """
    )


def test_si1000_noise_follows_google_convention_per_moment() -> None:
    """Pin noise per moment kind; idle in measurement moments is ``p/10`` once in use, else ``2p``.

    ``q2`` is unused until the last moment, so it idles at ``2p`` in the first measurement
    moment while the used ``q0`` gets ``p/10``; the virtual ``S 1`` leaves ``q1`` idle.
    """
    circuit = stim.Circuit(
        """
        R 0 1
        TICK
        H 0
        S 1
        TICK
        CZ 0 1
        TICK
        M 1
        TICK
        R 1
        TICK
        M 0 2
        """
    )

    assert add_si1000_noise(circuit, p=0.001) == stim.Circuit(
        """
        R 0 1
        X_ERROR(0.002) 0 1
        DEPOLARIZE1(0.002) 2
        TICK
        H 0
        S 1
        DEPOLARIZE1(0.0001) 0 1 2
        TICK
        CZ 0 1
        DEPOLARIZE2(0.001) 0 1
        DEPOLARIZE1(0.0001) 2
        TICK
        M(0.005) 1
        DEPOLARIZE1(0.0001) 0
        DEPOLARIZE1(0.002) 2
        TICK
        R 1
        X_ERROR(0.002) 1
        DEPOLARIZE1(0.002) 0 2
        TICK
        M(0.005) 0 2
        DEPOLARIZE1(0.0001) 1
        """
    )
