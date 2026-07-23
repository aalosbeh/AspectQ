"""
AspectQ evaluation driver.  Runs three research questions on a noisy simulator
and emits results.json plus paper-ready figures.

  RQ1  Modularity  : does AspectQ reduce scattering/tangling vs a tangled baseline?
  RQ2  Overhead    : what is the compile-time weaving and runtime sampling cost?
  RQ3  Effectiveness: do the woven aspects actually work (mitigation, policy,
                      observability, tamper-evident audit)?

Everything is simulator-only, seeded, and reproducible.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspectq import (Weaver, ObservabilityAspect, SecurityPolicyAspect,
                     ComplianceAuditAspect, ErrorMitigationAspect,
                     any_join_point)
from aspectq.baseline import run_program_tangled
from aspectq import metrics as m

SEED = 20260723
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG1 = os.path.join(ROOT, "figures")
FIG2 = os.path.join(ROOT, "paper", "figures")
RESULTS = os.path.join(ROOT, "results")
for d in (FIG1, FIG2, RESULTS):
    os.makedirs(d, exist_ok=True)

ONE_Q = ["ry", "rx", "rz", "h", "x", "sx"]
TWO_Q = ["cx", "cz"]


# --------------------------------------------------------------------------- #
# circuit + noise helpers
# --------------------------------------------------------------------------- #
def hardware_efficient_ansatz(n_qubits: int, n_layers: int, seed: int = SEED
                              ) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=f"hea_{n_qubits}q_{n_layers}L")
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(rng.uniform(0, np.pi)), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def ideal_parity(circuit: QuantumCircuit, obs_qubits: list[int]) -> float:
    n = circuit.num_qubits
    label = ["I"] * n
    for q in obs_qubits:
        label[n - 1 - q] = "Z"
    return float(Statevector(circuit).expectation_value(Pauli("".join(label))).real)


def noisy_backend(p1=0.004, p2=0.04) -> AerSimulator:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ONE_Q)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_Q)
    be = AerSimulator(noise_model=nm)
    be.set_options(seed_simulator=SEED)
    return be


# --------------------------------------------------------------------------- #
# RQ1 -- modularity
# --------------------------------------------------------------------------- #
def rq1_modularity() -> dict:
    base_path = os.path.join(ROOT, "aspectq", "baseline.py")
    asp_path = os.path.join(ROOT, "aspectq", "aspects.py")
    comp = m.compare(base_path, asp_path)

    # functional-equivalence check: tangled vs woven produce identical results
    be = noisy_backend()
    qc = hardware_efficient_ansatz(4, 4)
    allowed = set(ONE_Q + TWO_Q)
    restricted = {3}

    tangled = run_program_tangled(
        qc.copy(), be, shots=8192, allowed_gates=allowed,
        restricted_qubits=restricted, max_depth=100, audit_chain=[])

    weaver = Weaver([
        ObservabilityAspect(),
        SecurityPolicyAspect(allowed_gates=allowed, restricted_qubits=restricted,
                             max_depth=100),
        ComplianceAuditAspect(applied_aspects=["Observability", "SecurityPolicy",
                                               "ErrorMitigation"]),
        ErrorMitigationAspect(obs_qubits=[0]),
    ])
    woven, wc = weaver.weave(qc.copy())
    run_qc = qc.copy(); run_qc.measure_all()
    res = weaver.run(run_qc, be, shots=8192, wc=wc)

    equiv = {
        "violation_count_match": len(tangled["violations"]) == len(wc.violations),
        "violation_count": len(wc.violations),
        "twoq_match": (tangled["observability"]["two_qubit_gates"]
                       == wc.metrics["observability"]["two_qubit_gates"]),
        "mitigated_abs_diff": abs(tangled["expectation_mitigated"]
                                  - res["expectation_mitigated"]),
    }
    comp["functional_equivalence"] = equiv
    return comp


# --------------------------------------------------------------------------- #
# RQ2 -- overhead
# --------------------------------------------------------------------------- #
def rq2_overhead() -> dict:
    allowed = set(ONE_Q + TWO_Q)
    sizes = [2, 4, 6, 8, 10, 12]
    weave_ms, depths, sizes_ops = [], [], []
    for n in sizes:
        qc = hardware_efficient_ansatz(n, 5)
        weaver = Weaver([
            ObservabilityAspect(),
            SecurityPolicyAspect(allowed_gates=allowed, max_qubits=64,
                                 max_depth=10_000),
            ComplianceAuditAspect(applied_aspects=["Observability",
                                                   "SecurityPolicy"]),
        ])
        # median of a few weaves to stabilise the timing
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            woven, wc = weaver.weave(qc.copy())
            times.append((time.perf_counter() - t0) * 1e3)
        weave_ms.append(float(np.median(times)))
        depths.append(int(qc.depth()))
        sizes_ops.append(int(qc.size()))

    # structural overhead of observability probes (added barriers, 0 gates)
    qc = hardware_efficient_ansatz(6, 5)
    base_depth, base_ops = qc.depth(), qc.size()
    w_probe = Weaver([ObservabilityAspect(insert_probes=True)])
    woven, wc = w_probe.weave(qc.copy())
    probe_overhead = {
        "base_depth": int(base_depth), "woven_depth": int(woven.depth()),
        "base_ops": int(base_ops),
        "barriers_added": int(wc.metrics["observability"]["probes_inserted"]),
        "two_qubit_gates_added": int(woven.count_ops().get("cx", 0)
                                     - qc.count_ops().get("cx", 0)),
    }

    # runtime sampling overhead of ZNE
    zne_overhead = {"scale_factors": 3, "sampling_multiplier": 3}

    # figure: weave time vs qubits
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(sizes, weave_ms, "o-", color="#2b6cb0", lw=2)
    plt.xlabel("Number of qubits"); plt.ylabel("Structural weave time (ms)")
    plt.title("AspectQ compile-time weaving overhead")
    plt.grid(alpha=0.3); plt.tight_layout()
    for d in (FIG1, FIG2):
        plt.savefig(os.path.join(d, "fig_overhead.pdf"))
        plt.savefig(os.path.join(d, "fig_overhead.png"), dpi=150)
    plt.close()

    return {"sizes": sizes, "weave_ms": weave_ms, "depths": depths,
            "ops": sizes_ops, "probe_overhead": probe_overhead,
            "zne_overhead": zne_overhead}


# --------------------------------------------------------------------------- #
# RQ3 -- effectiveness
# --------------------------------------------------------------------------- #
def rq3a_error_mitigation() -> dict:
    be = noisy_backend()
    depths = [2, 3, 4, 5, 6, 7]
    ideal, unmit, mit = [], [], []
    for L in depths:
        qc = hardware_efficient_ansatz(4, L)
        ideal_val = ideal_parity(qc, [0])
        weaver = Weaver([ErrorMitigationAspect(scale_factors=(1, 3, 5),
                                               obs_qubits=[0])])
        run_qc = qc.copy(); run_qc.measure_all()
        res = weaver.run(run_qc, be, shots=16384)
        ideal.append(ideal_val)
        unmit.append(res["expectation_unmitigated"])
        mit.append(res["expectation_mitigated"])

    err_unmit = [abs(u - i) for u, i in zip(unmit, ideal)]
    err_mit = [abs(mm - i) for mm, i in zip(mit, ideal)]
    reduction = [(eu - em) / eu * 100 if eu > 1e-9 else 0.0
                 for eu, em in zip(err_unmit, err_mit)]

    plt.figure(figsize=(5.2, 3.4))
    plt.plot(depths, err_unmit, "s--", color="#c53030", lw=2,
             label="Unmitigated (noisy)")
    plt.plot(depths, err_mit, "o-", color="#2f855a", lw=2,
             label="ZNE aspect (woven)")
    plt.xlabel("Ansatz depth (layers)")
    plt.ylabel(r"Absolute error in $\langle Z_0\rangle$")
    plt.title("Woven error-mitigation aspect vs unmitigated")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    for d in (FIG1, FIG2):
        plt.savefig(os.path.join(d, "fig_zne.pdf"))
        plt.savefig(os.path.join(d, "fig_zne.png"), dpi=150)
    plt.close()

    return {"depths": depths, "ideal": ideal, "unmitigated": unmit,
            "mitigated": mit, "err_unmitigated": err_unmit,
            "err_mitigated": err_mit, "pct_reduction": reduction,
            "mean_pct_reduction": float(np.mean(reduction))}


def _make_security_suite():
    """Return (circuits, expected_violation_flags).

    Conforming circuits use only qubits 0-2 (the restricted qubit is 3), only
    allow-listed gates, and stay under the depth cap.  Three violation classes
    are planted: forbidden gate, restricted-qubit access, and depth-cap breach.
    """
    suite = []
    for k in range(12):                       # conforming: 3 qubits, shallow
        suite.append((hardware_efficient_ansatz(3, 2, seed=SEED + k), False))
    for k in range(3):                        # forbidden gate ('t' not allowed)
        qc = hardware_efficient_ansatz(3, 2, seed=SEED + 100 + k)
        qc.t(0); suite.append((qc, True))
    for k in range(3):                        # touches restricted qubit 3
        qc = hardware_efficient_ansatz(4, 2, seed=SEED + 200 + k)
        suite.append((qc, True))
    for k in range(2):                        # exceeds depth cap
        qc = hardware_efficient_ansatz(3, 40, seed=SEED + 300 + k)
        suite.append((qc, True))
    return suite


def rq3b_security() -> dict:
    allowed = set(ONE_Q + TWO_Q)             # note: 't' deliberately excluded
    suite = _make_security_suite()
    tp = fp = tn = fn = 0
    for qc, is_bad in suite:
        weaver = Weaver([SecurityPolicyAspect(
            allowed_gates=allowed, restricted_qubits={3}, max_depth=30)])
        # restricted qubit 3 only counts for the planted x(3) group, so scope it:
        _, wc = weaver.weave(qc.copy())
        flagged = len(wc.violations) > 0
        tp += is_bad and flagged
        fp += (not is_bad) and flagged
        tn += (not is_bad) and (not flagged)
        fn += is_bad and (not flagged)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"n_circuits": len(suite), "planted_violations": sum(b for _, b in suite),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall}


def rq3c_observability() -> dict:
    qc = hardware_efficient_ansatz(6, 5)
    weaver = Weaver([ObservabilityAspect()])
    _, wc = weaver.weave(qc.copy())
    obs = wc.metrics["observability"]
    ground_truth_2q = qc.count_ops().get("cx", 0)
    return {"reported_two_qubit_gates": obs["two_qubit_gates"],
            "ground_truth_two_qubit_gates": int(ground_truth_2q),
            "match": obs["two_qubit_gates"] == ground_truth_2q,
            "num_layers": obs["num_layers"],
            "gate_histogram": obs["gate_histogram"],
            "weave_time_ms": wc.metrics["weave_time_ms"]}


def rq3d_compliance() -> dict:
    suite = [hardware_efficient_ansatz(4, 3, seed=SEED + k) for k in range(8)]
    audit = ComplianceAuditAspect(applied_aspects=["Observability",
                                                   "SecurityPolicy",
                                                   "ErrorMitigation"])
    weaver = Weaver([audit])
    for qc in suite:
        weaver.weave(qc)
    ok_clean, broken_clean = ComplianceAuditAspect.verify(audit.chain)

    # tamper with every position in turn and confirm detection
    detections = 0
    for i in range(len(audit.chain)):
        import copy
        tampered = copy.deepcopy(audit.chain)
        tampered[i]["payload"]["circuit"] = "MALICIOUSLY_SWAPPED"
        ok, idx = ComplianceAuditAspect.verify(tampered)
        if (not ok) and idx == i:
            detections += 1
    return {"chain_length": len(audit.chain), "clean_verify_ok": ok_clean,
            "tamper_trials": len(audit.chain), "tamper_detected": detections,
            "detection_rate": detections / len(audit.chain)}


# --------------------------------------------------------------------------- #
# modularity figure
# --------------------------------------------------------------------------- #
def modularity_figure(comp: dict):
    totals = comp["totals"]
    labels = ["CDC", "CDO", "CDLOC", "CL"]      # LOCC shown in the table, not here
    base = [totals["baseline"][k] for k in labels]
    asp = [totals["aspectq"][k] for k in labels]
    x = np.arange(len(labels)); w = 0.38
    plt.figure(figsize=(5.6, 3.4))
    b1 = plt.bar(x - w / 2, base, w, label="Tangled baseline", color="#c53030")
    b2 = plt.bar(x + w / 2, asp, w, label="AspectQ", color="#2b6cb0")
    plt.bar_label(b1, padding=2, fontsize=8); plt.bar_label(b2, padding=2, fontsize=8)
    plt.xticks(x, labels)
    plt.ylabel("Count (lower = better modularity)")
    plt.title("Separation-of-concerns metrics (all 4 concerns)")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    for d in (FIG1, FIG2):
        plt.savefig(os.path.join(d, "fig_modularity.pdf"))
        plt.savefig(os.path.join(d, "fig_modularity.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
def main():
    t0 = time.perf_counter()
    results = {"seed": SEED}
    print("RQ1 modularity ..."); results["rq1_modularity"] = rq1_modularity()
    modularity_figure(results["rq1_modularity"])
    print("RQ2 overhead ...");    results["rq2_overhead"] = rq2_overhead()
    print("RQ3a mitigation ..."); results["rq3a_error_mitigation"] = rq3a_error_mitigation()
    print("RQ3b security ...");   results["rq3b_security"] = rq3b_security()
    print("RQ3c observability ..."); results["rq3c_observability"] = rq3c_observability()
    print("RQ3d compliance ..."); results["rq3d_compliance"] = rq3d_compliance()
    results["wall_time_s"] = time.perf_counter() - t0

    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- concise summary --------------------------------------------------
    t = results["rq1_modularity"]["totals"]
    print("\n==================  SUMMARY  ==================")
    print(f"RQ1  CDLOC (fragments): baseline={t['baseline']['CDLOC']}  "
          f"aspectq={t['aspectq']['CDLOC']}   "
          f"CL: baseline={t['baseline']['CL']} aspectq={t['aspectq']['CL']}")
    print(f"     functional-equivalence: {results['rq1_modularity']['functional_equivalence']}")
    print(f"RQ2  weave time {results['rq2_overhead']['weave_ms'][0]:.2f}-"
          f"{results['rq2_overhead']['weave_ms'][-1]:.2f} ms for "
          f"{results['rq2_overhead']['sizes'][0]}-{results['rq2_overhead']['sizes'][-1]} qubits; "
          f"probe gate overhead (2q)={results['rq2_overhead']['probe_overhead']['two_qubit_gates_added']}")
    print(f"RQ3a mean error reduction (ZNE aspect): "
          f"{results['rq3a_error_mitigation']['mean_pct_reduction']:.1f}%")
    print(f"RQ3b security precision/recall: "
          f"{results['rq3b_security']['precision']:.2f}/"
          f"{results['rq3b_security']['recall']:.2f} "
          f"({results['rq3b_security']['tp']} TP, {results['rq3b_security']['fp']} FP)")
    print(f"RQ3c observability 2q-count match: {results['rq3c_observability']['match']}")
    print(f"RQ3d tamper detection: {results['rq3d_compliance']['tamper_detected']}/"
          f"{results['rq3d_compliance']['tamper_trials']}")
    print(f"wall time: {results['wall_time_s']:.1f}s")
    print("===============================================")


if __name__ == "__main__":
    main()
