# AspectQ — Aspect-Oriented Weaving of Crosscutting Concerns into Quantum Programs

A research prototype and paper introducing the first aspect-oriented programming
(AOP) framework for quantum software. AspectQ modularizes four crosscutting
concerns of NISQ software — **error mitigation, observability, security policy,
and tamper-evident compliance audit** — as reusable *aspects* woven into quantum
programs via a two-phase (compile-time / runtime) weaver.

## Idea in one paragraph
Around every useful quantum circuit, developers hand-inline error mitigation,
logging, policy checks, and audit code. As in classical software, this scatters
and tangles each concern. AspectQ adapts AOP (join points, pointcuts, advice,
weaving) to quantum programs with two quantum-specific twists: a **Quantum
Join-Point Model** (gates, layers, measurements, circuit/run boundaries) and a
**two-phase weaver** that separates *structural* advice (woven over the circuit
DAG at compile time) from *runtime* advice (woven around shot-based execution as
classical post-processing) — a separation forced by measurement collapse.

## Repository layout
```
AspectQ/
├── aspectq/                 # the framework (library)
│   ├── core.py              # join-point model, pointcut language, advice, two-phase weaver
│   ├── aspects.py           # 4 reusable aspects (ZNE, observability, security, audit)
│   ├── baseline.py          # tangled baseline (same concerns, hand-inlined) for RQ1
│   └── metrics.py           # separation-of-concerns metrics (CDC/CDO/CDLOC/CL)
├── experiments/
│   └── run_all.py           # regenerates every number + figure in the paper
├── results/results.json     # machine-readable results
├── figures/                 # generated figures (.pdf + .png)
├── requirements.txt
└── README.md
```

## Reproduce the results
```bash
pip install -r requirements.txt
cd AspectQ
python experiments/run_all.py        # ~1s; writes results/results.json + figures/
```
Everything is seeded (`seed=20260723`) and simulator-only, so runs are
deterministic. Key results:

| RQ | Result |
|----|--------|
| RQ1 Modularity | concern scattering (CDLOC) and change effort (CL) drop 15 → 4 (3.75×); behavior provably unchanged (identical violations, 2-qubit accounting, mitigated value) |
| RQ2 Overhead | 0.24–1.40 ms structural weaving for 2–12 qubits; 0 added two-qubit gates |
| RQ3 Mitigation | woven ZNE aspect: **57.5%** mean error reduction (depths 2–7) |
| RQ3 Security | planted-violation detection precision = recall = **1.0** |
| RQ3 Audit | **100%** (8/8) tamper detection on the provenance chain |


## Notes & honest limitations
Experiments are simulator-based (Qiskit Aer, depolarizing noise) on
hardware-efficient ansätze up to 12 qubits. ZNE gains depend on the noise
regime (fixed and reported); at very small signal (depth 7) extrapolation
variance can slightly hurt — a limitation of ZNE, not of the weaving mechanism.
Hardware validation and additional aspects are future work.
