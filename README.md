# AspectQ: Aspect-Oriented Weaving of Crosscutting Concerns into Quantum Programs

AspectQ brings the aspect-oriented programming (AOP) model to gate-based
quantum programs written with Qiskit. Concerns such as error mitigation,
observability, security policy, provenance and usage metering are written once
as *aspects* and woven into programs instead of being scattered through every
driver script.

* **Quantum join-point model**: gates, two-qubit gates, ASAP layers,
  measurements, circuit entry/exit and runs, with quantum-specific context.
* **Pointcut language**: designators such as `two_qubit_gates()`,
  `on_qubits(S)`, `layer()`, `before_measure()`, `run_scope()`, `tagged(t)`,
  `shots_at_least(n)` and `backend_matches(s)`, combined with `&`, `|` and `~`.
* **Two weaving phases**: *structural* advice is woven into the circuit at
  compile time (on the logical circuit or at any stage of a Qiskit staged pass
  manager via `AspectQPass`); *runtime* advice forms a `proceed` chain around
  every execution, in which before, around and after advice nest by
  AspectJ-style precedence. `SamplerBackend` runs the runtime phase through a
  SamplerV2 primitive (needed for IBM hardware).
* **Behavior preservation**: structural weaving visits operations in a
  topological (layered) order and keeps registers and global phase, so
  directive-only advice leaves the circuit's semantics unchanged
  (Proposition 1 in the paper, checked by `tests/test_weaving.py`).
  Circuits with classical variables (`Store`) are rejected.

## Aspects shipped with the library

| Aspect | Phase | What it does |
|---|---|---|
| `ErrorMitigationAspect` | runtime, around | zero-noise extrapolation by global folding or pointcut-selected local folding (linear, Richardson, exponential) |
| `ReadoutMitigationAspect` | runtime, around | tensored readout-error mitigation; compose it *inside* ZNE |
| `ResourceQuotaAspect` | runtime, around | meters executions and shots, enforces a budget |
| `ObservabilityAspect` | structural, analysis | width, depth, layers, gate histogram; optional layer probes |
| `SecurityPolicyAspect` | structural, analysis | gate allow-list, restricted (physical) qubits, width/depth caps |
| `ComplianceAuditAspect` | structural, around | SHA-256 hash-chained provenance, optional HMAC and anchored head |
| `GateFoldingAspect` | structural, after | appends (G^dagger G)^n after each gate a pointcut selects |

## Quick start

```python
from aspectq import (Weaver, WeaveContext, ObservabilityAspect, SecurityPolicyAspect,
                     ComplianceAuditAspect, ErrorMitigationAspect, ReadoutMitigationAspect,
                     ResourceQuotaAspect, attach)

weaver = Weaver([                         # list order = precedence
    ObservabilityAspect(),
    SecurityPolicyAspect(allowed_gates={"rz", "sx", "x", "cz"}, max_depth=200),
    ComplianceAuditAspect(),
    ErrorMitigationAspect(obs_qubits=[0]),
    ReadoutMitigationAspect(),            # inside ZNE: corrects every scaled run
    ResourceQuotaAspect(max_shots=10**6), # innermost: meters what reaches the backend
])
woven, wc = weaver.weave(circuit)         # structural phase
result = weaver.run(measured, backend, shots=16384, wc=wc)   # runtime phase
print(wc.metrics["zne"], wc.violations, wc.audit[-1]["hash"])

# structural aspects can also run inside the transpiler, on physical qubits
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
attach(pm, Weaver([SecurityPolicyAspect(restricted_qubits=bad_qubits)]),
       stage="post_optimization")
```

`python examples/quickstart.py` runs a complete example on the IBM Torino
calibration snapshot (`FakeTorino`), including a user-defined aspect.

## Repository layout

```
aspectq/
  core.py          join points, pointcuts, Aspect, two-phase Weaver
  passes.py        AspectQPass and attach() for Qiskit staged pass managers
  aspects.py       the six aspects
  qutils.py        shared domain functions (layering, folding, extrapolation, digests)
  designs/         tangled, procedural and woven drivers used in RQ1
  metrics.py       separation-of-concerns metrics (CDC, CDO, CDLOC, LOCC, CL)
  benchmarks.py    benchmark programs
  simulation.py    noise models and an exact (density-matrix) backend
experiments/
  run_all.py           RQ1-RQ4: results/results.json and figures/
  crosscheck_mitiq.py  cross-validation of the ZNE aspect against Mitiq
  make_macros.py       numbers and tables for the paper (paper/generated/)
  capture_terminal.py  terminal captures used in the paper's appendix
examples/quickstart.py
execution.py (in aspectq/)  SamplerBackend adapter for SamplerV2 primitives
tests/                 tests (semantics, aspects, pass manager, sampler, designs)
```

## Reproducing the paper

```bash
pip install -r requirements-lock.txt      # exact versions of the reported run
python -m pytest                          # test suite
python experiments/run_all.py --quick     # reduced run, about 40 s
python experiments/run_all.py             # full evaluation, about 10 min
python experiments/make_macros.py         # writes ../paper/generated/
docker run --rm -v "$PWD":/w -w /w python:3.12-slim bash -lc \
  "pip install mitiq==1.1.0 qiskit==2.6.0 qiskit-aer==0.17.2 ply && python experiments/crosscheck_mitiq.py"
```

All runs are simulator-based and seeded (seed 20260723); apart from timings,
every number is reproducible. Main results of the reported run:

| Question | Result |
|---|---|
| RQ1 modularity | vs. tangled driver: concern code in the driver 67 -> 10 lines, edit sites 11 -> 5; a well-factored procedural design matches these line metrics with less code, so AspectQ's advantages are composition by precedence and stage-aware weaving |
| RQ2 preservation | 8,115 / 8,115 checks (operator equality, DAG isomorphism, folding, transpiler stages); precedence decides what the quota aspect meters and whether readout correction reaches the extrapolated value |
| RQ3 overhead | about 7.8 us per operation, linear; 0.26-2.65x a level-1 transpilation of the logical circuit; x2.1-9.7 when woven inside the transpiler; 0.3 us per runtime aspect |
| RQ4 mitigation | ZNE: MAE -70.0% [65.8, 73.9] (D1), -79.5% (D2); Torino noise model: ZNE + readout -68.8% vs ZNE alone -50.1%; local 2q folding matches global folding with 62% of the operations |
| RQ4 cross-check | matches Mitiq 1.1.0 to 2.4e-10 in 119 of 120 comparisons (the other is a documented sign-change fallback) |
| RQ4 policy / audit | whether a job touches a poorly calibrated qubit varies from 23% to 100% with the optimization level, so it is decidable only after layout; the hash chain needs an anchored head against truncation and rewrites |

Requirements: Python 3.11+ and Qiskit 2.x (tested with Python 3.13.15,
Qiskit 2.6.0, Aer 0.17.2).
