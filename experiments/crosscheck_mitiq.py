"""
Cross-check: AspectQ's woven ZNE against Mitiq's ``execute_with_zne``.

Both pipelines use global unitary folding with scale factors (1, 3, 5) and the
same extrapolation models (exponential with asymptote 0, Richardson, linear).
Expectation values come from the exact (density-matrix) noisy channel, so the
comparison has no shot noise.  Mitiq needs its own environment; see
experiments/README.md for the container command.
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mitiq  # noqa: E402
import qiskit  # noqa: E402
from mitiq import zne  # noqa: E402
from mitiq.zne.inference import ExpFactory, LinearFactory, RichardsonFactory  # noqa: E402
from mitiq.zne.scaling import fold_global as mitiq_fold  # noqa: E402

from aspectq import ErrorMitigationAspect, Weaver, WeaveContext  # noqa: E402
from aspectq.benchmarks import hardware_efficient_ansatz  # noqa: E402
from aspectq.qutils import parity_expectation  # noqa: E402
from aspectq.simulation import ExactBackend, depolarizing_noise  # noqa: E402

SEED = 20260723
SCALES = [1, 3, 5]


def main():
    backend = ExactBackend(depolarizing_noise(1e-3, 1e-2))

    def executor(circ) -> float:
        c = circ.copy()
        c.measure_all()
        return parity_expectation(backend.run(c, shots=1).result().get_counts(), [0])

    rows = []
    for depth in range(1, 9):
        for k in range(5):
            qc = hardware_efficient_ansatz(4, depth, seed=SEED + 1000 * depth + k)
            meas = qc.copy()
            meas.measure_all()
            row = {"depth": depth, "instance": k}
            for method, factory in (("exponential", ExpFactory(SCALES, asymptote=0.0)),
                                    ("richardson", RichardsonFactory(SCALES)),
                                    ("linear", LinearFactory(SCALES))):
                wc = WeaveContext()
                Weaver([ErrorMitigationAspect(scale_factors=tuple(SCALES), obs_qubits=[0],
                                              extrapolation=method)]).run(
                    meas, backend, shots=1, wc=wc)
                ours = wc.metrics["zne"]["mitigated"]
                theirs = float(zne.execute_with_zne(qc, executor, factory=factory,
                                                    scale_noise=mitiq_fold))
                row[method] = {"aspectq": ours, "mitiq": theirs, "abs_diff": abs(ours - theirs),
                               "aspectq_method": wc.metrics["zne"]["method"],
                               "scaled_expvals": wc.metrics["zne"]["expvals"]}
            folded = mitiq_fold(qc, 3)
            row["mitiq_fold3_ops"] = dict(folded.count_ops())
            rows.append(row)
    methods = ("exponential", "richardson", "linear")
    worst = {m: max(r[m]["abs_diff"] for r in rows) for m in methods}
    match = {m: sum(r[m]["abs_diff"] < 1e-8 for r in rows) for m in methods}
    same_model = [r for r in rows if r["exponential"]["aspectq_method"] == "exponential"]
    worst_same = max(r["exponential"]["abs_diff"] for r in same_model)
    out = {"mitiq": mitiq.__version__, "qiskit": qiskit.__version__,
           "python": platform.python_version(), "instances": len(rows),
           "max_abs_diff": worst, "matches_1e-8": match,
           "exponential_same_model": {"instances": len(same_model),
                                      "max_abs_diff": worst_same},
           "rows": rows}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "crosscheck_mitiq.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"Mitiq {mitiq.__version__} | Qiskit {qiskit.__version__} | Python {platform.python_version()}")
    print(f"{len(rows)} instances (HEA, 4 qubits, depths 1-8), exact noisy channel")
    for m, v in worst.items():
        print(f"  {m:<12} max |AspectQ - Mitiq| = {v:.2e}   agree to 1e-8: {match[m]}/{len(rows)}")
    print(f"  exponential, excluding sign-change fallbacks: {len(same_model)} instances, "
          f"max |diff| = {worst_same:.2e}")


if __name__ == "__main__":
    main()
