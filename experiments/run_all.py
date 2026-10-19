"""
AspectQ evaluation driver.  Regenerates every number, table and figure of the
paper and writes results/results.json and figures/*.{pdf,png}.

  RQ1  Modularity     -- tangled vs procedural vs AspectQ designs (5 concerns)
  RQ2  Preservation   -- does weaving keep program semantics; does composition
                         order behave as specified?
  RQ3  Overhead       -- structural weaving cost vs circuit size and vs
                         transpilation; runtime-weaver cost; probe cost
  RQ4  Effectiveness  -- mitigation (3 noise models), policy, observability,
                         tamper-evident audit

Everything is simulator-based and seeded.  ``--quick`` runs a reduced
configuration (used by CI and for the screenshots in the appendix).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np
import qiskit
import qiskit_aer
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aspectq import (Aspect, ComplianceAuditAspect,  # noqa: E402
                     ErrorMitigationAspect, ObservabilityAspect, Phase,
                     ReadoutMitigationAspect, ResourceQuotaAspect,
                     SecurityPolicyAspect, Weaver, WeaveContext, attach,
                     fold_global, run_scope, two_qubit_gates)
from aspectq import metrics as M  # noqa: E402
from aspectq.benchmarks import hardware_efficient_ansatz, workload_suite  # noqa: E402
from aspectq.designs import ConcernConfig  # noqa: E402
from aspectq.designs.procedural import (check_policy,  # noqa: E402
                                        run_program_procedural)
from aspectq.designs.tangled import run_program_tangled  # noqa: E402
from aspectq.designs.woven import run_program_woven  # noqa: E402
from aspectq.qutils import extrapolate, is_two_qubit_gate  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from aspectq.simulation import (ExactBackend, depolarizing_backend,  # noqa: E402
                                depolarizing_noise)

SEED = 20260723
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
ALLOWED = {"ry", "rx", "rz", "h", "x", "sx", "cx", "cz", "rzz", "p", "u"}

# validated categorical slots (blue, orange, aqua) + recessive gray
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"


def log(msg: str) -> None:
    print(msg, flush=True)


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def ideal_z(circuit: QuantumCircuit, qubits) -> float:
    label = ["I"] * circuit.num_qubits
    for q in qubits:
        label[circuit.num_qubits - 1 - q] = "Z"
    return float(Statevector(circuit).expectation_value(Pauli("".join(label))).real)


def mean_ci(values, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# =========================================================================== #
# RQ1 -- modularity
# =========================================================================== #
def _cfg(obs) -> ConcernConfig:
    return ConcernConfig(allowed_gates=ALLOWED, restricted_qubits=frozenset({3}),
                         max_qubits=4, max_depth=40, obs_qubits=tuple(obs),
                         max_shots=10 ** 7)


def _report_woven(wc: WeaveContext) -> dict:
    return {"violations": sorted(wc.violations),
            "observability": wc.metrics["observability"],
            "zne": {k: wc.metrics["zne"][k] for k in ("expvals", "mitigated", "method")},
            "audit": wc.audit[-1], "quota": wc.metrics["quota"]}


def _run_design(name, circ, backend, shots, obs):
    cfg = _cfg(obs)
    if name == "aspectq":
        wc = WeaveContext()
        value = run_program_woven(circ, backend, shots, cfg, wc)["expectation"]
        return value, _report_woven(wc)
    ctx: dict = {}
    fn = run_program_tangled if name == "tangled" else run_program_procedural
    value = fn(circ, backend, shots, cfg, ctx)["expectation"]
    ctx["violations"] = sorted(ctx["violations"])
    return value, ctx


def rq1(args) -> dict:
    log("RQ1  modularity: metrics + three-design equivalence")
    metrics = M.compare()
    exact = ExactBackend(depolarizing_noise(1e-3, 1e-2))
    rows = []
    for circ, obs in workload_suite(SEED):
        runs = {d: _run_design(d, circ, exact, 8192, obs)
                for d in ("tangled", "procedural", "aspectq")}
        ref_val, ref = runs["tangled"]
        row = {"program": circ.name, "qubits": circ.num_qubits,
               "depth": circ.depth(), "ops": circ.size(),
               "violations": len(ref["violations"]),
               "max_abs_diff_expectation": max(abs(v - ref_val) for v, _ in runs.values()),
               "max_abs_diff_mitigated": max(abs(r["zne"]["mitigated"] - ref["zne"]["mitigated"])
                                             for _, r in runs.values())}
        for key in ("violations", "observability", "audit", "quota"):
            row[f"identical_{key}"] = all(
                json.dumps(r[key], sort_keys=True) == json.dumps(ref[key], sort_keys=True)
                for _, r in runs.values())
        rows.append(row)

    # sampled run: tangled and procedural execute identical circuits; the woven
    # design executes an equivalent re-linearised circuit, so it agrees in law
    sampled = []
    for circ, obs in workload_suite(SEED):
        be = depolarizing_backend(1e-3, 1e-2, seed=SEED)
        vals = {d: _run_design(d, circ, be, 16384, obs) for d in ("tangled", "procedural", "aspectq")}
        t, a = vals["tangled"][1]["zne"]["mitigated"], vals["aspectq"][1]["zne"]["mitigated"]
        sampled.append({"program": circ.name,
                        "tangled_eq_procedural": vals["tangled"][1]["zne"] == vals["procedural"][1]["zne"],
                        "abs_diff_woven": abs(t - a)})
    return {"metrics": metrics, "equivalence_exact": rows, "equivalence_sampled": sampled}


def fig_modularity(res: dict) -> None:
    style()
    designs = [("tangled", "Tangled", C_GRAY), ("procedural", "Procedural", C_AQUA),
               ("aspectq", "AspectQ", C_BLUE)]
    panels = [("CL", "Change locality (edit sites)"), ("FRAG", "Concern fragments"),
              ("CDLOC", "CDLOC (transition points)"), ("driver", "Concern LOC in driver")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, (key, title) in zip(axes, panels):
        vals = [res["metrics"][d]["driver"]["concern_lines"] if key == "driver"
                else res["metrics"][d]["totals"][key] for d, _, _ in designs]
        bars = ax.bar(range(3), vals, width=0.62, color=[c for _, _, c in designs],
                      label=[n for _, n, _ in designs])
        ax.bar_label(bars, padding=1.5, fontsize=7, color=INK)
        ax.set_xticks([])
        ax.set_title(title, color=INK)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in designs]
    fig.legend(handles, [n for _, n, _ in designs], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    save(fig, "fig_modularity")


# =========================================================================== #
# RQ2 -- behaviour preservation and composition semantics
# =========================================================================== #
def _structural_subsets():
    makers = [("obs", lambda: ObservabilityAspect(insert_probes=True)),
              ("sec", lambda: SecurityPolicyAspect(allowed_gates={"h", "cx"},
                                                   restricted_qubits={0}, max_depth=5)),
              ("aud", lambda: ComplianceAuditAspect())]
    for r in range(len(makers) + 1):
        for combo in itertools.combinations(makers, r):
            yield "+".join(n for n, _ in combo) or "none", [mk() for _, mk in combo]


def program_order_probes(circuit: QuantumCircuit) -> QuantumCircuit:
    """Ablation: insert a probe barrier in *program order* whenever a new ASAP
    layer index first appears (the naive placement)."""
    out = circuit.copy_empty_like()
    qt = [0] * circuit.num_qubits
    last = -1
    for instr in circuit.data:
        q = [circuit.find_bit(b).index for b in instr.qubits]
        if getattr(instr.operation, "_directive", False):
            out.append(instr.operation, instr.qubits, instr.clbits)
            continue
        lay = max((qt[i] for i in q), default=0)
        for i in q:
            qt[i] = lay + 1
        if lay > last:
            out.barrier()
            last = lay
        out.append(instr.operation, instr.qubits, instr.clbits)
    return out


def rq2(args) -> dict:
    log("RQ2  behaviour preservation + composition")
    n = 30 if args.quick else 300
    rng = np.random.default_rng(SEED)
    unitary_pass = unitary_total = dag_pass = dag_total = 0
    depth_neutral = 0
    naive_increase = []
    obs_exact = 0
    circuits = []
    for i in range(n):
        w, d = int(rng.integers(2, 7)), int(rng.integers(2, 13))
        circuits.append(("u", random_circuit(w, d, max_operands=3, seed=SEED + i)))
        circuits.append(("m", random_circuit(w, d, max_operands=3, measure=True,
                                             reset=True, seed=SEED + 10_000 + i)))
    for kind, circ in circuits:
        ref_op = Operator(circ) if kind == "u" else None
        ref_dag = circuit_to_dag(RemoveBarriers()(circ))
        for _, aspects in _structural_subsets():
            woven, wc = Weaver(aspects).weave(circ)
            if kind == "u":
                unitary_total += 1
                unitary_pass += Operator(woven) == ref_op
            dag_total += 1
            dag_pass += circuit_to_dag(RemoveBarriers()(woven)) == ref_dag
        woven, wc = Weaver([ObservabilityAspect(insert_probes=True)]).weave(circ)
        depth_neutral += woven.depth() == circ.depth()
        naive = program_order_probes(circ)
        if circ.depth():
            naive_increase.append((naive.depth() - circ.depth()) / circ.depth())
        obs = wc.metrics["observability"]
        ops = [i for i in circ.data if not getattr(i.operation, "_directive", False)]
        hist: dict = {}
        for i in ops:
            hist[i.operation.name] = hist.get(i.operation.name, 0) + 1
        obs_exact += (obs["num_layers"] == circ.depth()
                      and obs["two_qubit_gates"] == sum(1 for i in ops if is_two_qubit_gate(i))
                      and obs["gate_histogram"] == hist)

    fold_pass = fold_total = 0
    for i in range(n):
        circ = random_circuit(int(rng.integers(2, 6)), int(rng.integers(2, 10)),
                              max_operands=3, seed=SEED + 20_000 + i)
        ref = Operator(circ)
        for lam in (3, 5, 7):
            f = fold_global(circ, lam)
            fold_total += 1
            fold_pass += (Operator(f) == ref) and f.size() == lam * circ.size()

    hea = hardware_efficient_ansatz(6, 5, seed=SEED)
    hea_naive = program_order_probes(hea)
    hea_woven, hea_wc = Weaver([ObservabilityAspect(insert_probes=True)]).weave(hea)

    # transpiler-stage weaving on a calibrated device model
    dev = FakeTorino()
    stage_rows = []
    for circ, _ in workload_suite(SEED):
        for stage in ("post_layout", "post_routing", "post_optimization"):
            pm0 = generate_preset_pass_manager(optimization_level=2, backend=dev,
                                               seed_transpiler=SEED)
            plain = pm0.run(circ)
            pm1 = generate_preset_pass_manager(optimization_level=2, backend=dev,
                                               seed_transpiler=SEED)
            attach(pm1, Weaver([ObservabilityAspect(),
                                SecurityPolicyAspect(max_qubits=133),
                                ComplianceAuditAspect()]), stage=stage)
            woven = pm1.run(circ)
            ok = circuit_to_dag(woven) == circuit_to_dag(plain)
            if stage == "post_optimization":
                pm2 = generate_preset_pass_manager(optimization_level=2, backend=dev,
                                                   seed_transpiler=SEED)
                attach(pm2, Weaver([ObservabilityAspect(insert_probes=True)]), stage=stage)
                probed = pm2.run(circ)
                ok = ok and circuit_to_dag(RemoveBarriers()(probed)) == circuit_to_dag(plain)
            stage_rows.append({"program": circ.name, "stage": stage, "identical": bool(ok)})

    # composition: precedence of around advice
    comp = {}
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    exact = ExactBackend(depolarizing_noise(1e-3, 1e-2))
    for order, aspects in (("zne>quota", [ErrorMitigationAspect(obs_qubits=[0, 1]),
                                          ResourceQuotaAspect()]),
                           ("quota>zne", [ResourceQuotaAspect(),
                                          ErrorMitigationAspect(obs_qubits=[0, 1])])):
        wc = WeaveContext()
        Weaver(aspects).run(bell, exact, shots=4096, wc=wc)
        comp[order] = {"metered_executions": wc.metrics["quota"]["executions"],
                       "metered_shots": wc.metrics["quota"]["shots"],
                       "backend_executions": wc.metrics["runtime"]["executions"]}
    # readout mitigation must sit inside ZNE
    ro_rows = []
    n_ro = 5 if args.quick else 20
    for k in range(n_ro):
        circ = hardware_efficient_ansatz(4, 4, seed=SEED + 500 + k)
        ideal = ideal_z(circ, [0])
        meas = circ.copy()
        meas.measure_all()
        be = depolarizing_backend(1e-3, 1e-2, readout=0.04, seed=SEED + k)
        row = {"ideal": ideal}
        for order, mk in (("zne>ro", lambda: [ErrorMitigationAspect(), ReadoutMitigationAspect()]),
                          ("ro>zne", lambda: [ReadoutMitigationAspect(), ErrorMitigationAspect()]),
                          ("zne", lambda: [ErrorMitigationAspect()])):
            wc = WeaveContext()
            res = Weaver(mk()).run(meas, be, shots=16384, wc=wc)
            row[order] = abs(res["expectation_mitigated"] - ideal)
        ro_rows.append(row)
    comp["readout_order"] = {k: mean_ci([r[k] for r in ro_rows]) for k in ("zne", "zne>ro", "ro>zne")}

    return {"n_circuits": len(circuits), "aspect_subsets": 8,
            "unitary_checks": [int(unitary_pass), unitary_total],
            "dag_checks": [int(dag_pass), dag_total],
            "folding_checks": [int(fold_pass), fold_total],
            "probe_depth_neutral": [int(depth_neutral), len(circuits)],
            "naive_probe_depth_increase": mean_ci(naive_increase),
            "naive_probe_depth_increase_max": float(np.max(naive_increase)),
            "hea6x5_depth": {"base": hea.depth(), "layered_probes": hea_woven.depth(),
                             "program_order_probes": hea_naive.depth(),
                             "probes": hea_wc.metrics["observability"]["probes_inserted"]},
            "observability_exact": [int(obs_exact), len(circuits)],
            "transpiler_stages": stage_rows,
            "composition": comp}


# =========================================================================== #
# RQ3 -- overhead
# =========================================================================== #
def _median_time(fn, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def rq3(args) -> dict:
    log("RQ3  overhead")
    widths = [4, 8, 16] if args.quick else [4, 8, 16, 32, 64, 128]
    layers = [5, 20] if args.quick else [5, 20, 80]
    dev = FakeTorino()
    rows = []
    for n in widths:
        for L in layers:
            qc = hardware_efficient_ansatz(n, L, seed=SEED)
            full = lambda: Weaver([ObservabilityAspect(),
                                   SecurityPolicyAspect(allowed_gates=ALLOWED,
                                                        restricted_qubits={n - 1},
                                                        max_depth=10 ** 6),
                                   ComplianceAuditAspect()]).weave(qc)
            empty = lambda: Weaver([]).weave(qc)
            row = {"qubits": n, "layers": L, "ops": qc.size(), "depth": qc.depth(),
                   "weave_empty_ms": _median_time(empty, 7),
                   "weave_full_ms": _median_time(full, 7)}
            if L <= 20 and n <= 128:
                pm = generate_preset_pass_manager(optimization_level=1, backend=dev,
                                                  seed_transpiler=SEED)
                row["transpile_ms"] = _median_time(lambda: pm.run(qc), 7)
                pm_w = generate_preset_pass_manager(optimization_level=1, backend=dev,
                                                    seed_transpiler=SEED)
                attach(pm_w, Weaver([ObservabilityAspect(),
                                     SecurityPolicyAspect(max_qubits=dev.num_qubits),
                                     ComplianceAuditAspect()]), stage="post_optimization")
                row["transpile_woven_ms"] = _median_time(lambda: pm_w.run(qc), 7)
            rows.append(row)
    ops = np.array([r["ops"] for r in rows], float)
    full = np.array([r["weave_full_ms"] for r in rows])
    b, a = np.polyfit(ops, full, 1)
    pred = a + b * ops
    r2 = 1 - np.sum((full - pred) ** 2) / np.sum((full - full.mean()) ** 2)
    ratios = [r["weave_full_ms"] / r["transpile_ms"] for r in rows if "transpile_ms" in r]
    pass_ratios = [r["transpile_woven_ms"] / r["transpile_ms"] for r in rows
                   if "transpile_ms" in r]

    # dispatch cost of the runtime chain: pass-through aspects on a null backend
    class PassThrough(Aspect):
        phase = Phase.RUNTIME

    class NullBackend:
        def run(self, circ, shots=1):
            return _NullJob()

    class _NullJob:
        def result(self):
            return self

        def get_counts(self):
            return {"0": 1}

    bell = QuantumCircuit(1)
    bell.measure_all()
    chain_rows = []
    for k in (0, 1, 2, 4, 8, 16):
        weaver = Weaver([PassThrough(run_scope()) for _ in range(k)])
        chain_rows.append({"aspects": k, "run_us": 1e3 * _median_time(
            lambda: [weaver.run(bell, NullBackend(), shots=1) for _ in range(100)], 7) / 100})
    kk = np.array([c["aspects"] for c in chain_rows], float)
    tt = np.array([c["run_us"] for c in chain_rows])
    per_aspect_us = float(np.polyfit(kk, tt, 1)[0])

    # probe cost after transpilation (barriers block cross-layer optimisation)
    probe_rows = []
    for n in ([4, 8] if args.quick else [4, 8, 16]):
        qc = hardware_efficient_ansatz(n, 5, seed=SEED)
        probed, _ = Weaver([ObservabilityAspect(insert_probes=True)]).weave(qc)
        res = {"qubits": n}
        for tag, c in (("plain", qc), ("probed", probed)):
            pm = generate_preset_pass_manager(optimization_level=3, backend=dev,
                                              seed_transpiler=SEED,
                                              scheduling_method="alap")
            t = pm.run(c)
            res[f"{tag}_depth"] = t.depth()
            res[f"{tag}_2q"] = int(t.count_ops().get("cz", 0))
            res[f"{tag}_duration_us"] = float(t.estimate_duration(dev.target, unit="s") * 1e6)
        probe_rows.append(res)
    return {"rows": rows, "fit": {"us_per_op": float(b * 1e3), "intercept_ms": float(a),
                                  "r2": float(r2)},
            "weave_over_transpile": {"min": float(min(ratios)), "max": float(max(ratios)),
                                     "median": float(np.median(ratios))},
            "pass_in_pipeline": {"min": float(min(pass_ratios)), "max": float(max(pass_ratios)),
                                 "median": float(np.median(pass_ratios))},
            "runtime_chain": {"rows": chain_rows, "us_per_aspect": per_aspect_us},
            "probe_after_transpile": probe_rows}


def fig_overhead(res: dict) -> None:
    style()
    rows = res["rows"]
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    o = np.array([r["ops"] for r in rows])
    idx = np.argsort(o)
    ax.loglog(o[idx], np.array([r["weave_full_ms"] for r in rows])[idx], "o-",
              color=C_BLUE, label="AspectQ weave (3 aspects)")
    ax.loglog(o[idx], np.array([r["weave_empty_ms"] for r in rows])[idx], "s--",
              color=C_AQUA, label="AspectQ weave (no aspects)")
    tr = sorted((r["ops"], r["transpile_ms"], r["transpile_woven_ms"]) for r in rows
                if "transpile_ms" in r)
    ax.loglog([t[0] for t in tr], [t[1] for t in tr], "^", color=C_ORANGE,
              label="Qiskit transpile, level 1")
    ax.loglog([t[0] for t in tr], [t[2] for t in tr], "v", color=C_GRAY,
              label="level 1 + AspectQPass (3 aspects)")
    ax.set_xlabel("Operations in circuit")
    ax.set_ylabel("Median time (ms)")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3,
              handlelength=1.8, columnspacing=1.0)
    save(fig, "fig_overhead")


# =========================================================================== #
# RQ4 -- effectiveness
# =========================================================================== #
def _zne_instance(qc, backend, exact, shots):
    meas = qc.copy()
    meas.measure_all()
    wc = WeaveContext()
    Weaver([ErrorMitigationAspect(obs_qubits=[0])]).run(meas, backend, shots=shots, wc=wc)
    z = wc.metrics["zne"]
    out = {"unmitigated": z["unmitigated"], "expvals": z["expvals"],
           "stderr": z["mitigated_stderr"],
           "weaver_overhead_ms": wc.metrics["runtime"]["weaver_overhead_ms"]}
    for m in ("exponential", "richardson", "linear"):
        out[m] = extrapolate((1, 3, 5), z["expvals"], m)[0]
    if exact is not None:
        wce = WeaveContext()
        Weaver([ErrorMitigationAspect(obs_qubits=[0])]).run(meas, exact, shots=1, wc=wce)
        out["exact_unmitigated"] = wce.metrics["zne"]["unmitigated"]
        out["exact_mitigated"] = wce.metrics["zne"]["mitigated"]
    return out


def rq4_mitigation(args) -> dict:
    log("RQ4a error mitigation under three noise models")
    shots = 16384
    inst = 4 if args.quick else 30
    depths = [2, 4] if args.quick else list(range(1, 9))
    models = {"D1": (1e-3, 1e-2), "D2": (4e-3, 4e-2)}
    out = {}
    for name, (p1, p2) in models.items():
        recs = []
        for L in depths:
            for k in range(inst):
                qc = hardware_efficient_ansatz(4, L, seed=SEED + 1000 * L + k)
                be = depolarizing_backend(p1, p2, seed=SEED + 1000 * L + k)
                r = _zne_instance(qc, be, ExactBackend(depolarizing_noise(p1, p2)), shots)
                r.update({"depth": L, "instance": k, "ideal": ideal_z(qc, [0])})
                recs.append(r)
        out[name] = {"p1": p1, "p2": p2, "records": recs}
        log(f"     {name} done ({len(recs)} instances)")

    # noise-strength sweep at fixed depth (p1 = p2 / 10)
    sweep_p2 = [0.005, 0.02, 0.04] if args.quick else [0.0025, 0.005, 0.01, 0.02, 0.04, 0.06]
    recs = []
    for pi, p2 in enumerate(sweep_p2):
        for k in range(3 if args.quick else 20):
            qc = hardware_efficient_ansatz(4, 4, seed=SEED + 50_000 + k)
            be = depolarizing_backend(p2 / 10, p2, seed=SEED + 100_000 + 1000 * pi + k)
            r = _zne_instance(qc, be, None, shots)
            r.update({"p2": p2, "depth": 4, "instance": k, "ideal": ideal_z(qc, [0])})
            recs.append(r)
    out["SWEEP"] = {"depth": 4, "records": recs}
    log(f"     noise sweep done ({len(recs)} instances)")

    # pointcut-driven local folding: fold only two-qubit gates (D1 noise)
    recs = []
    for L in ([2, 4] if args.quick else [2, 4, 6]):
        for k in range(4 if args.quick else 30):
            qc = hardware_efficient_ansatz(4, L, seed=SEED + 1000 * L + k)
            meas = qc.copy()
            meas.measure_all()
            n2 = sum(1 for i in qc.data if is_two_qubit_gate(i))
            row = {"depth": L, "instance": k, "ideal": ideal_z(qc, [0]),
                   "ops": qc.size(), "two_qubit": n2}
            for tag, fold_pc in (("global", None), ("local2q", two_qubit_gates())):
                be = depolarizing_backend(1e-3, 1e-2, seed=SEED + 300_000 + 1000 * L + k)
                wc = WeaveContext()
                Weaver([ErrorMitigationAspect(obs_qubits=[0], fold_pointcut=fold_pc)]).run(
                    meas, be, shots=shots, wc=wc)
                row[f"{tag}_unmitigated"] = wc.metrics["zne"]["unmitigated"]
                row[f"{tag}_mitigated"] = wc.metrics["zne"]["mitigated"]
            row["global_executed_ops"] = 9 * qc.size()
            row["local2q_executed_ops"] = 3 * qc.size() + 6 * n2
            recs.append(row)
    out["LOCALFOLD"] = {"noise": "D1", "records": recs}
    log(f"     local folding done ({len(recs)} instances)")

    # device model: IBM Torino calibration snapshot (FakeTorino)
    dev = FakeTorino()
    sim = AerSimulator.from_backend(dev)
    basis = list(dev.operation_names)
    ro_err = [dev.target["measure"][(q,)].error for q in range(dev.num_qubits)]
    bad = {q for q, e in enumerate(ro_err) if e > 0.05}
    ft = lambda c: transpile(c, basis_gates=basis, optimization_level=0)
    dev_depths = [2, 4] if args.quick else list(range(1, 7))
    for level, n_inst in ((2, 3 if args.quick else 20), (1, 2 if args.quick else 10)):
        recs = []
        for L in dev_depths:
            for k in range(n_inst):
                qc = hardware_efficient_ansatz(4, L, seed=SEED + 1000 * L + k)
                pm = generate_preset_pass_manager(optimization_level=level, backend=dev,
                                                  seed_transpiler=SEED + k)
                policy = SecurityPolicyAspect(restricted_qubits=bad)
                attach(pm, Weaver([policy]), stage="post_optimization")
                phys = pm.run(qc)
                flagged = len(pm.property_set["aspectq_context"].violations) > 0
                p0 = phys.layout.final_index_layout()[0]
                creg = ClassicalRegister(1, "m")
                phys.add_register(creg)
                phys.measure(p0, creg[0])
                sim.set_options(seed_simulator=SEED + 200_000 + 10_000 * level + 1000 * L + k)
                wc1 = WeaveContext()
                Weaver([ErrorMitigationAspect(obs_qubits=[p0], fold_transform=ft)]).run(
                    phys, sim, shots=shots, wc=wc1)
                wc2 = WeaveContext()
                Weaver([ErrorMitigationAspect(obs_qubits=[p0], fold_transform=ft),
                        ReadoutMitigationAspect()]).run(phys, sim, shots=shots, wc=wc2)
                recs.append({"depth": L, "instance": k, "ideal": ideal_z(qc, [0]),
                             "physical_q0": int(p0), "readout_error_q0": ro_err[p0],
                             "policy_flagged": bool(flagged),
                             "unmitigated": wc1.metrics["zne"]["unmitigated"],
                             "zne": wc1.metrics["zne"]["mitigated"],
                             "readout": wc2.metrics["zne"]["unmitigated"],
                             "zne_readout": wc2.metrics["zne"]["mitigated"],
                             "weaver_overhead_ms": wc1.metrics["runtime"]["weaver_overhead_ms"]})
        out[f"DEV_L{level}"] = {"backend": "FakeTorino", "optimization_level": level,
                                "restricted_qubits": len(bad), "records": recs}
        log(f"     DEV level {level} done ({len(recs)} instances)")
    return out


def summarise_mitigation(raw: dict) -> dict:
    summ = {}
    for name, block in raw.items():
        recs = block["records"]
        if name == "SWEEP":
            per_p = {}
            for p2 in sorted({r["p2"] for r in recs}):
                sub = [r for r in recs if r["p2"] == p2]
                per_p[p2] = {m: mean_ci([abs(r[m] - r["ideal"]) for r in sub])
                             for m in ("unmitigated", "exponential", "richardson")}
            summ[name] = {"per_p2": per_p, "depth": block["depth"], "n": len(recs)}
            continue
        if name == "LOCALFOLD":
            keys = ("global_unmitigated", "global_mitigated", "local2q_mitigated")
            per_depth = {}
            for L in sorted({r["depth"] for r in recs}):
                sub = [r for r in recs if r["depth"] == L]
                per_depth[L] = {k: mean_ci([abs(r[k] - r["ideal"]) for r in sub]) for k in keys}
            pooled = {k: mean_ci([abs(r[k] - r["ideal"]) for r in recs]) for k in keys}
            e_g = np.array([abs(r["global_mitigated"] - r["ideal"]) for r in recs])
            e_l = np.array([abs(r["local2q_mitigated"] - r["ideal"]) for r in recs])
            summ[name] = {"per_depth": per_depth, "pooled": pooled, "n": len(recs),
                          "wilcoxon_two_sided_p": float(wilcoxon(e_g, e_l).pvalue),
                          "executed_ops_ratio": float(np.mean(
                              [r["local2q_executed_ops"] / r["global_executed_ops"]
                               for r in recs]))}
            continue
        is_dev = name.startswith("DEV")
        methods = (["unmitigated", "zne", "readout", "zne_readout"] if is_dev
                   else ["unmitigated", "exponential", "richardson", "linear"])
        per_depth = {}
        for L in sorted({r["depth"] for r in recs}):
            sub = [r for r in recs if r["depth"] == L]
            per_depth[L] = {m: mean_ci([abs(r[m] - r["ideal"]) for r in sub]) for m in methods}
            if not is_dev:
                per_depth[L]["exact_mitigated"] = mean_ci(
                    [abs(r["exact_mitigated"] - r["ideal"]) for r in sub])
        pooled = {m: mean_ci([abs(r[m] - r["ideal"]) for r in recs]) for m in methods}
        best = "zne_readout" if is_dev else "exponential"
        improved = np.mean([abs(r[best] - r["ideal"]) < abs(r["unmitigated"] - r["ideal"])
                            for r in recs])
        e_un = np.array([abs(r["unmitigated"] - r["ideal"]) for r in recs])
        e_best = np.array([abs(r[best] - r["ideal"]) for r in recs])
        stat = wilcoxon(e_un, e_best, alternative="greater")
        a12 = float(np.mean([(u > b) + 0.5 * (u == b) for u in e_un for b in e_best]))
        boot = np.random.default_rng(SEED).integers(0, len(recs), size=(2000, len(recs)))
        red = 100 * (1 - e_best[boot].mean(axis=1) / e_un[boot].mean(axis=1))
        s = {"per_depth": per_depth, "pooled": pooled,
             "reduction_pct": 100 * (1 - pooled[best][0] / pooled["unmitigated"][0]),
             "reduction_pct_ci": [float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))],
             "wilcoxon_p": float(stat.pvalue), "a12": a12,
             "fraction_improved": float(improved),
             "weaver_overhead_ms_median": float(np.median([r["weaver_overhead_ms"] for r in recs])),
             "n": len(recs)}
        if is_dev:
            s["reduction_pct_zne_only"] = 100 * (1 - pooled["zne"][0] / pooled["unmitigated"][0])
            s["policy_flagged_fraction"] = float(np.mean([r["policy_flagged"] for r in recs]))
            s["median_readout_error_q0"] = float(np.median([r["readout_error_q0"] for r in recs]))
        else:
            s["mean_stderr"] = float(np.mean([r["stderr"] for r in recs]))
        summ[name] = s
    return summ


def fig_zne(summ: dict) -> None:
    style()
    panels = [("D1", r"Depolarizing $p_1{=}10^{-3}, p_2{=}10^{-2}$"),
              ("D2", r"Depolarizing $p_1{=}4{\times}10^{-3}, p_2{=}4{\times}10^{-2}$"),
              ("DEV_L2", "FakeTorino noise model, transpiled at level 2")]
    fig, grid = plt.subplots(2, 2, figsize=(7.0, 4.3))
    axes = grid.ravel()
    for ax, (name, title) in zip(axes[:3], panels):
        pd = summ[name]["per_depth"]
        depths = sorted(pd, key=int)
        x = [int(d) for d in depths]
        series = ([("unmitigated", "Unmitigated", C_GRAY, "s--"),
                   ("zne", "ZNE", C_BLUE, "o-"),
                   ("readout", "Readout", C_AQUA, "v-."),
                   ("zne_readout", "ZNE + readout", C_ORANGE, "D-")]
                  if name.startswith("DEV") else
                  [("unmitigated", "Unmitigated", C_GRAY, "s--"),
                   ("exponential", "ZNE (exponential)", C_BLUE, "o-"),
                   ("richardson", "ZNE (Richardson)", C_ORANGE, "D:")])
        for key, label, col, fmt in series:
            mu = [pd[d][key][0] for d in depths]
            lo = [pd[d][key][1] for d in depths]
            hi = [pd[d][key][2] for d in depths]
            ax.plot(x, mu, fmt, color=col, label=label)
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Ansatz layers")
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", ncol=2 if len(series) > 3 else 1)
    sw = summ["SWEEP"]["per_p2"]
    ps = sorted(sw, key=float)
    x = [float(p) for p in ps]
    ax = axes[3]
    for key, label, col, fmt in (("unmitigated", "Unmitigated", C_GRAY, "s--"),
                                 ("exponential", "ZNE (exponential)", C_BLUE, "o-"),
                                 ("richardson", "ZNE (Richardson)", C_ORANGE, "D:")):
        ax.plot(x, [sw[p][key][0] for p in ps], fmt, color=col, label=label)
        ax.fill_between(x, [sw[p][key][1] for p in ps], [sw[p][key][2] for p in ps],
                        color=col, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{p:g}" for p in x])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"Noise sweep, {summ['SWEEP']['depth']} layers ($p_1 = p_2/10$)", color=INK)
    ax.set_xlabel(r"Two-qubit depolarizing parameter $p_2$")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    for a, tag in zip(axes, "abcd"):
        a.text(-0.12, 1.04, f"({tag})", transform=a.transAxes, fontsize=8,
               fontweight="bold", color=INK)
    for a in (axes[0], axes[2]):
        a.set_ylabel(r"Mean $|\langle Z_0\rangle - \langle Z_0\rangle_{\rm ideal}|$")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, "fig_zne")


def _random_clean(rng, n_used: int, max_layers: int) -> QuantumCircuit:
    """Random circuit over allow-listed gates on qubits 0..n_used-1.

    Qubit n_used is the restricted qubit and qubit n_used+1 an idle spare.
    """
    qc = QuantumCircuit(n_used + 2)
    for _ in range(int(rng.integers(2, max_layers))):
        for q in range(n_used):
            g = rng.choice(["h", "x", "sx", "rz", "ry", "rx"])
            if g in ("rz", "ry", "rx"):
                getattr(qc, g)(float(rng.uniform(0, np.pi)), q)
            else:
                getattr(qc, g)(q)
        a, b = rng.choice(n_used, size=2, replace=False)
        getattr(qc, rng.choice(["cx", "cz"]))(int(a), int(b))
    return qc


def _insert(circ: QuantumCircuit, pos: int, name: str, qubits) -> QuantumCircuit:
    """Copy of ``circ`` with gate ``name`` on ``qubits`` inserted at ``pos``."""
    out = circ.copy_empty_like()
    for i, instr in enumerate(circ.data):
        if i == pos:
            getattr(out, name)(*qubits)
        out.append(instr.operation, instr.qubits, instr.clbits)
    if pos >= len(circ.data):
        getattr(out, name)(*qubits)
    return out


def rq4_policy(args) -> dict:
    log("RQ4b security policy: mutation suite + physical-layer check")
    rng = np.random.default_rng(SEED)
    allowed = {"h", "x", "sx", "rz", "ry", "rx", "cx", "cz"}
    n_clean = 40 if args.quick else 200
    classes = ["forbidden_gate", "restricted_gate", "restricted_measure",
               "depth_cap", "width_cap"]
    cap_depth = 30
    suite = []
    for i in range(n_clean):
        n_used = int(rng.integers(2, 5))
        base = _random_clean(rng, n_used, 8)
        suite.append(("clean", base))
        r = n_used                                  # restricted qubit index
        for cls in classes:
            mut = base.copy()
            if cls == "forbidden_gate":
                op = str(rng.choice(["t", "tdg", "s", "y"]))
                mut = _insert(base, int(rng.integers(0, len(base.data) + 1)),
                              op, [int(rng.integers(0, n_used))])
            elif cls == "restricted_gate":
                mut = _insert(base, int(rng.integers(0, len(base.data) + 1)),
                              "cx", [int(rng.integers(0, n_used)), r])
            elif cls == "restricted_measure":
                c = ClassicalRegister(1, "c")
                mut.add_register(c)
                mut.measure(r, c[0])
            elif cls == "depth_cap":
                while mut.depth() <= cap_depth:
                    mut.h(0)
            elif cls == "width_cap":
                mut.x(r + 1)                         # the idle, unrestricted spare
            suite.append((cls, mut))

    expect_text = {"forbidden_gate": "not in allow-list", "restricted_gate": "touches restricted",
                   "restricted_measure": "'measure' touches restricted",
                   "depth_cap": "depth", "width_cap": "width"}
    def of_class(cls, v):
        if cls == "restricted_gate":
            return "touches restricted" in v and not v.startswith("'measure'")
        if cls in ("depth_cap", "width_cap"):
            return v.startswith(expect_text[cls])
        return expect_text[cls] in v

    tally = {c: [0, 0] for c in ["clean"] + classes}
    single = {c: 0 for c in classes}
    agree = 0
    for cls, circ in suite:
        n_used = circ.num_qubits - 2
        # the width cap is n_used + 1 except for the width class, so that the
        # restricted-qubit mutants plant exactly one violation each
        cap_width = n_used if cls in ("width_cap", "clean") else n_used + 1
        cfg = ConcernConfig(allowed_gates=allowed, restricted_qubits=frozenset({n_used}),
                            max_qubits=cap_width, max_depth=cap_depth)
        wc = WeaveContext()
        Weaver([SecurityPolicyAspect(allowed_gates=allowed, restricted_qubits={n_used},
                                     max_qubits=cap_width, max_depth=cap_depth)]).weave(circ, wc)
        ctx: dict = {}
        check_policy(circ, cfg, ctx)
        agree += sorted(ctx["violations"]) == sorted(wc.violations)
        tally[cls][1] += 1
        if cls == "clean":
            tally[cls][0] += len(wc.violations) == 0
        else:
            tally[cls][0] += any(of_class(cls, v) for v in wc.violations)
            single[cls] += bool(wc.violations) and all(of_class(cls, v) for v in wc.violations)

    # physical layer: qubits with readout error above 5% are restricted
    dev = FakeTorino()
    ro_err = [dev.target["measure"][(q,)].error for q in range(dev.num_qubits)]
    bad = {q for q, e in enumerate(ro_err) if e > 0.05}
    phys = {}
    n_phys = 10 if args.quick else 60
    for level in (1, 2, 3):
        tp = fp = fn = tn = hits = 0
        for k in range(n_phys):
            w = int(rng.integers(3, 7))
            qc = hardware_efficient_ansatz(w, int(rng.integers(2, 6)), seed=SEED + 3000 + k)
            pm = generate_preset_pass_manager(optimization_level=level, backend=dev,
                                              seed_transpiler=SEED + k)
            attach(pm, Weaver([SecurityPolicyAspect(restricted_qubits=bad)]),
                   stage="post_optimization")
            routed = pm.run(qc)
            truth = len(pm.property_set["aspectq_context"].violations) > 0
            used = {routed.find_bit(q).index for i in routed.data
                    if not getattr(i.operation, "_directive", False) for q in i.qubits}
            assert truth == bool(used & bad)                 # independent oracle
            hits += truth
            wc = WeaveContext()             # same policy applied to logical indices
            Weaver([SecurityPolicyAspect(restricted_qubits=bad)]).weave(qc, wc)
            guess = len(wc.violations) > 0
            tp += truth and guess
            fp += guess and not truth
            fn += truth and not guess
            tn += not truth and not guess
        phys[f"level{level}"] = {"flagged_physical": hits, "n": n_phys,
                                 "logical_check": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}
    return {"suite_size": len(suite), "detection": tally, "single_class": single,
            "agreement_with_procedural": [int(agree), len(suite)],
            "restricted_physical_qubits": len(bad), "physical": phys}


def rq4_audit(args) -> dict:
    log("RQ4d tamper-evident audit")
    rng = np.random.default_rng(SEED)
    key = b"aspectq-evaluation-key"
    audit = ComplianceAuditAspect(hmac_key=key)
    w = Weaver([audit])
    n_records = 50
    for k in range(n_records):
        w.weave(random_circuit(3, 4, max_operands=2, seed=SEED + 4000 + k))
    chain = audit.chain
    anchor = audit.head()

    def rechain(c, start):
        prev = c[start - 1]["hash"] if start else "GENESIS"
        for e in c[start:]:
            e["prev"] = prev
            e["hash"] = ComplianceAuditAspect._hash(prev, e["payload"])
            prev = e["hash"]
        return c

    def attack(kind):
        c = copy.deepcopy(chain)
        i = int(rng.integers(1, n_records - 1))
        if kind == "modify":
            c[i]["payload"]["circuit"] = "tampered"
        elif kind == "delete":
            del c[i]
        elif kind == "insert":
            forged = copy.deepcopy(c[i])
            forged["payload"]["circuit"] = "forged"
            forged["prev"] = c[i - 1]["hash"]
            forged["hash"] = ComplianceAuditAspect._hash(forged["prev"], forged["payload"])
            c.insert(i, forged)
        elif kind == "reorder":
            c[i], c[i + 1] = c[i + 1], c[i]
        elif kind == "truncate":
            c = c[: n_records - int(rng.integers(1, 5))]
        elif kind == "rewrite":
            c[i]["payload"]["circuit"] = "tampered"
            c = rechain(c, i)
        return c

    trials = 20 if args.quick else 100
    matrix = {}
    for kind in ("modify", "delete", "insert", "reorder", "truncate", "rewrite"):
        det = {"hash_chain": 0, "anchored_head": 0, "hmac": 0}
        for _ in range(trials):
            c = attack(kind)
            det["hash_chain"] += not ComplianceAuditAspect.verify(c)[0]
            det["anchored_head"] += not ComplianceAuditAspect.verify(c, anchor=anchor)[0]
            det["hmac"] += not ComplianceAuditAspect.verify(c, hmac_key=key)[0]
        matrix[kind] = {k: v / trials for k, v in det.items()}
    return {"records": n_records, "trials": trials,
            "clean_verifies": ComplianceAuditAspect.verify(chain, anchor=anchor, hmac_key=key)[0],
            "detection": matrix}


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced configuration")
    ap.add_argument("--stamp", default="", help="timestamp recorded in results.json")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from results/results.json")
    args = ap.parse_args()
    if args.figures_only:
        with open(os.path.join(RES, "results.json")) as f:
            stored = json.load(f)
        fig_modularity(stored["rq1"])
        fig_overhead(stored["rq3"])
        fig_zne(stored["rq4_mitigation"])
        print("figures redrawn from results/results.json")
        return
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    t0 = time.perf_counter()
    import qiskit_ibm_runtime
    results = {"seed": SEED, "quick": args.quick, "stamp": args.stamp,
               "environment": {"python": platform.python_version(),
                               "qiskit": qiskit.__version__,
                               "qiskit_aer": qiskit_aer.__version__,
                               "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
                               "numpy": np.__version__,
                               "machine": f"{platform.system()} {platform.machine()}",
                               "processor": platform.processor()}}
    dev = FakeTorino()
    cz = [v.error for v in dev.target["cz"].values() if v is not None and v.error is not None]
    ro = [dev.target["measure"][(q,)].error for q in range(dev.num_qubits)]
    results["environment"]["machine_description"] = machine_description()
    results["device"] = {"name": dev.name, "num_qubits": dev.num_qubits,
                         "median_cz_error": float(np.median(cz)),
                         "median_readout_error": float(np.median(ro)),
                         "calibration_date": str(dev.properties().last_update_date)}
    results["protocol"] = {"zne_shots": 16384, "scale_factors": [1, 3, 5],
                           "weave_repetitions": 7, "transpile_repetitions": 7}
    timings = {}
    for key, fn in (("rq1", rq1), ("rq2", rq2), ("rq3", rq3)):
        t = time.perf_counter()
        results[key] = fn(args)
        timings[key] = time.perf_counter() - t
    t = time.perf_counter()
    raw = rq4_mitigation(args)
    results["rq4_mitigation"] = summarise_mitigation(raw)
    results["rq4_mitigation_raw"] = raw
    timings["rq4_mitigation"] = time.perf_counter() - t
    for key, fn in (("rq4_policy", rq4_policy), ("rq4_audit", rq4_audit)):
        t = time.perf_counter()
        results[key] = fn(args)
        timings[key] = time.perf_counter() - t
    results["timings_s"] = timings
    results["wall_time_s"] = time.perf_counter() - t0

    fig_modularity(results["rq1"])
    fig_overhead(results["rq3"])
    fig_zne(results["rq4_mitigation"])
    name = "results_quick.json" if args.quick else "results.json"
    with open(os.path.join(RES, name), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print_summary(results)


def machine_description() -> str:
    """CPU, memory and OS of the evaluation machine."""
    import subprocess
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip() or cpu
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True).stdout.strip()) // 2 ** 30
        return f"an {cpu} with {mem}~GB of memory under macOS~{platform.mac_ver()[0]}"
    except Exception:
        return f"{cpu} ({platform.system()} {platform.release()})"


def print_summary(r: dict) -> None:
    m = r["rq1"]["metrics"]
    line = "=" * 72
    print("\n" + line)
    print(f" AspectQ evaluation summary   (seed {r['seed']}, quick={r['quick']})")
    e = r["environment"]
    print(f" Python {e['python']} | Qiskit {e['qiskit']} | Aer {e['qiskit_aer']} | "
          f"NumPy {e['numpy']}")
    print(line)
    print(" RQ1  design       CL  FRAG  CDLOC  LOCC  driver-concern-LOC  tangling")
    for d in ("tangled", "procedural", "aspectq"):
        t, dr = m[d]["totals"], m[d]["driver"]
        print(f"      {d:<11} {t['CL']:>3} {t['FRAG']:>5} {t['CDLOC']:>6} {t['LOCC']:>5} "
              f"{dr['concern_lines']:>19} {dr['tangling']:>9.2f}")
    eq = r["rq1"]["equivalence_exact"]
    print(f"      exact equivalence on {len(eq)} programs: max|dE| = "
          f"{max(x['max_abs_diff_mitigated'] for x in eq):.1e}; artifacts identical: "
          f"{all(x['identical_violations'] and x['identical_observability'] and x['identical_audit'] and x['identical_quota'] for x in eq)}")
    r2 = r["rq2"]
    print(f" RQ2  unitary {r2['unitary_checks'][0]}/{r2['unitary_checks'][1]}  "
          f"DAG {r2['dag_checks'][0]}/{r2['dag_checks'][1]}  "
          f"folding {r2['folding_checks'][0]}/{r2['folding_checks'][1]}  "
          f"depth-neutral probes {r2['probe_depth_neutral'][0]}/{r2['probe_depth_neutral'][1]}")
    print(f"      naive probe placement depth +{100 * r2['naive_probe_depth_increase'][0]:.1f}% "
          f"(max +{100 * r2['naive_probe_depth_increase_max']:.0f}%);  stages identical: "
          f"{sum(s['identical'] for s in r2['transpiler_stages'])}/{len(r2['transpiler_stages'])}")
    c = r2["composition"]
    print(f"      quota inside ZNE meters {c['zne>quota']['metered_executions']} executions, "
          f"outside meters {c['quota>zne']['metered_executions']}; readout MAE "
          f"zne={c['readout_order']['zne'][0]:.4f} zne>ro={c['readout_order']['zne>ro'][0]:.4f} "
          f"ro>zne={c['readout_order']['ro>zne'][0]:.4f}")
    r3 = r["rq3"]
    print(f" RQ3  weave cost {r3['fit']['us_per_op']:.2f} us/op (R^2={r3['fit']['r2']:.3f}); "
          f"weave/transpile median {r3['weave_over_transpile']['median']:.2f}; "
          f"AspectQPass in level-1 pipeline x{r3['pass_in_pipeline']['median']:.2f} "
          f"(range {r3['pass_in_pipeline']['min']:.2f}-{r3['pass_in_pipeline']['max']:.2f}); "
          f"runtime chain {r3['runtime_chain']['us_per_aspect']:.1f} us/aspect")
    lf = r["rq4_mitigation"]["LOCALFOLD"]
    print(f" RQ4a local 2q folding MAE {lf['pooled']['local2q_mitigated'][0]:.4f} vs global "
          f"{lf['pooled']['global_mitigated'][0]:.4f} (unmitigated "
          f"{lf['pooled']['global_unmitigated'][0]:.4f}) at {100 * lf['executed_ops_ratio']:.0f}% "
          f"of the executed operations, n={lf['n']}")
    for name, s in r["rq4_mitigation"].items():
        if name in ("SWEEP", "LOCALFOLD"):
            continue
        p = s["pooled"]
        best = "zne_readout" if name.startswith("DEV") else "exponential"
        extra = (f" (ZNE alone {s['reduction_pct_zne_only']:.1f}%, policy flagged "
                 f"{100 * s['policy_flagged_fraction']:.0f}%)" if name.startswith("DEV") else "")
        print(f" RQ4a {name:<7} MAE {p['unmitigated'][0]:.4f} -> {p[best][0]:.4f} "
              f"({s['reduction_pct']:.1f}% lower [{s['reduction_pct_ci'][0]:.1f}, "
              f"{s['reduction_pct_ci'][1]:.1f}], {100 * s['fraction_improved']:.0f}% improved, "
              f"A12={s['a12']:.2f}, p={s['wilcoxon_p']:.1e}, n={s['n']}){extra}")
    pol = r["rq4_policy"]
    det = ", ".join(f"{k}:{v[0]}/{v[1]}" for k, v in pol["detection"].items())
    print(f" RQ4b {det}; single-class mutants "
          f"{sum(pol['single_class'].values())}/{sum(v[1] for k, v in pol['detection'].items() if k != 'clean')}")
    print("      compiled jobs touching a restricted qubit: " + ", ".join(
        f"level {k[-1]} {v['flagged_physical']}/{v['n']}" for k, v in pol["physical"].items()))
    au = r["rq4_audit"]["detection"]
    print(" RQ4d " + ", ".join(f"{k}:{v['hash_chain']:.0%}/{v['anchored_head']:.0%}/{v['hmac']:.0%}"
                              for k, v in au.items()) + "  (chain/anchor/HMAC)")
    print(f" wall time {r['wall_time_s']:.1f} s")
    print(line)


if __name__ == "__main__":
    main()
