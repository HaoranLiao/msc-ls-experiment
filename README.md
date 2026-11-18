# msc-ls-experiment

Code for *Magic state cultivation with lattice-surgery escape on superconducting hardware*. It
implements magic state cultivation with lattice-surgery escape, proposed by Hirano et al.
([arXiv:2510.24615](https://arxiv.org/abs/2510.24615)), for execution on hardware: Steane-code
state injection and cultivation, followed by lattice-surgery transfer to a rotated surface code,
built in Stim for simulation and in Qiskit for IBM hardware. Beyond the original protocol, it adds
an arbitrary-state injection encoder routed for hardware connectivity, a minimal post-injection
check set and a minimal Steane-check schedule during the merge that keep the fault distance,
fault-distance certification of the Stim circuits, and circuit-level decoding of hardware counts
with error models calibrated to the device, by matching or Tesseract with complementary-gap
postselection.

## Citation

If you reuse this code, please cite our paper:

> Liao et al., *Magic state cultivation with lattice-surgery escape on superconducting hardware*,
> [arXiv:2611.22035](https://arxiv.org/abs/2611.22035) (2026).

```bibtex
@article{liao2026magic,
  title         = {Magic state cultivation with lattice-surgery escape on superconducting hardware},
  author        = {Liao, Haoran and Schutzman, Michael and Munizzi, William and Mundada, Pranav and Hartnett, Gavin and Baum, Yuval},
  year          = {2026},
  journal       = {arXiv:2611.22035},
  url           = {https://arxiv.org/pdf/2611.22035},
}
```

## Layout

- `src/msc/steane_code/`: Steane state injection, cultivation, kickback tomography, and lookup decoder
- `src/msc/rotated_surface_code/`: rotated d=3 and d=5 patches, phenomenological and circuit-level decoders
- `src/msc/state_transfer/`: Steane-to-rotated-surface-code lattice-surgery transfer, Stim logical-error-rate estimates, Ty tomography, and memory experiment
- `src/msc/decoding/`, `src/msc/noise/`, `src/msc/utils/`: Pauli frames, fault injection and noise models, circuit helpers
- `tests/unit/`: Stim and Qiskit unit tests

## Install and test

```bash
pip install -e ".[dev]"
pytest
```

Tests marked `slow` run exhaustive sweeps; skip them with `pytest -m "not slow"`.

## License

Apache License 2.0.
