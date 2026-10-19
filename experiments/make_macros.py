"""
Write the numbers and tables used by the paper.

Reads results/results.json and results/crosscheck_mitiq.json and writes
paper/generated/results_macros.tex plus one LaTeX table per file, so that the
manuscript never contains a hand-copied number.

    python experiments/make_macros.py [--paper ../paper]
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import subprocess
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sci(x: float, digits: int = 1) -> str:
    """Format x as LaTeX scientific notation."""
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def sci_up(x: float, digits: int = 1) -> str:
    """Scientific notation rounded *up*, for "to within" bounds."""
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = math.ceil(x / 10 ** e * 10 ** digits) / 10 ** digits
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def pval(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else sci(p).strip("$")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


def sloc(path: str) -> int:
    """Code lines without comments, docstrings or blank lines."""
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            b = getattr(node, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                doc.update(range(b[0].lineno, b[0].end_lineno + 1))
    lines = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in skip:
            lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines - doc)


def machine_description() -> str:
    """CPU and OS of this machine (the evaluation machine)."""
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip() or cpu
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True).stdout.strip()) // 2 ** 30
        osv = platform.mac_ver()[0]
        return f"an {cpu} with {mem}~GB of memory under macOS~{osv}"
    except Exception:
        return f"{cpu} ({platform.system()} {platform.release()})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=os.path.join(os.path.dirname(ROOT), "paper"))
    args = ap.parse_args()
    out_dir = os.path.join(args.paper, "generated")
    os.makedirs(out_dir, exist_ok=True)
    r = json.load(open(os.path.join(ROOT, "results", "results.json")))
    mx = json.load(open(os.path.join(ROOT, "results", "crosscheck_mitiq.json")))
    M: dict[str, str] = {}

    env = r["environment"]
    M.update(envPython=env["python"], envQiskit=env["qiskit"], envAer=env["qiskit_aer"],
             envNumpy=env["numpy"], envRuntime=env["qiskit_ibm_runtime"],
             envSeed=str(r["seed"]), envWallMinutes=f"{r['wall_time_s'] / 60:.0f}",
             envMachine=env.get("machine_description") or machine_description())

    # ---------------- RQ1
    m = r["rq1"]["metrics"]
    for tag, d in (("Tang", "tangled"), ("Proc", "procedural"), ("Aq", "aspectq")):
        t, dr = m[d]["totals"], m[d]["driver"]
        for k in ("CL", "FRAG", "CDLOC", "LOCC", "CDC", "CDO"):
            M[f"{tag}{k}"] = str(t[k])
        M[f"{tag}DrvLOC"] = str(dr["concern_lines"])
        M[f"{tag}DrvLines"] = str(dr["code_lines"])
        M[f"{tag}Core"] = str(dr["core_lines"])
        M[f"{tag}Tangling"] = pct(100 * dr["tangling"], 0)
    t_drv = m["tangled"]["driver"]["concern_lines"]
    a_drv = m["aspectq"]["driver"]["concern_lines"]
    M["drvReduction"] = pct(100 * (1 - a_drv / t_drv), 0)
    # sensitivity: count the weaver construction and the weave call as concern code
    infra = 3
    M["AqDrvLOCAlt"] = str(a_drv + infra)
    M["AqTanglingAlt"] = pct(100 * (a_drv + infra) / m["aspectq"]["driver"]["code_lines"], 0)
    M["drvReductionAlt"] = pct(100 * (1 - (a_drv + infra) / t_drv), 0)
    eq = r["rq1"]["equivalence_exact"]
    M["eqMaxDiff"] = sci(max(max(x["max_abs_diff_mitigated"], x["max_abs_diff_expectation"])
                             for x in eq))
    M["eqAllIdentical"] = "all five programs" if all(
        x["identical_violations"] and x["identical_observability"] and x["identical_audit"]
        and x["identical_quota"] for x in eq) else "NOT all programs"

    # ---------------- RQ2
    q = r["rq2"]
    M.update(presCircuits=str(q["n_circuits"]),
             presUnitaryPass=str(q["unitary_checks"][0]), presUnitaryTotal=str(q["unitary_checks"][1]),
             presDagPass=str(q["dag_checks"][0]), presDagTotal=str(q["dag_checks"][1]),
             presFoldPass=str(q["folding_checks"][0]), presFoldTotal=str(q["folding_checks"][1]),
             presProbePass=str(q["probe_depth_neutral"][0]), presProbeTotal=str(q["probe_depth_neutral"][1]),
             naiveMean=pct(100 * q["naive_probe_depth_increase"][0]),
             naiveMax=pct(100 * q["naive_probe_depth_increase_max"], 0),
             heaBase=str(q["hea6x5_depth"]["base"]), heaNaive=str(q["hea6x5_depth"]["program_order_probes"]),
             heaLayered=str(q["hea6x5_depth"]["layered_probes"]),
             obsPass=str(q["observability_exact"][0]), obsTotal=str(q["observability_exact"][1]),
             presFoldCircuits=str(q["folding_checks"][1] // 3),
             presHalf=str(q["n_circuits"] // 2))
    st = q["transpiler_stages"]
    M["stagePass"] = str(sum(s["identical"] for s in st))
    M["stageTotal"] = str(len(st))
    total_checks = (q["unitary_checks"][1] + q["dag_checks"][1] + q["folding_checks"][1] + len(st))
    passed = (q["unitary_checks"][0] + q["dag_checks"][0] + q["folding_checks"][0]
              + sum(s["identical"] for s in st))
    M["presTotalChecks"] = f"{total_checks:,}".replace(",", "{,}")
    M["presAllPassed"] = "yes" if passed == total_checks else "NO"
    c = q["composition"]
    M.update(compInner=str(c["zne>quota"]["metered_executions"]),
             compOuter=str(c["quota>zne"]["metered_executions"]),
             roZne=f"{c['readout_order']['zne'][0]:.4f}", roInner=f"{c['readout_order']['zne>ro'][0]:.4f}",
             roOuter=f"{c['readout_order']['ro>zne'][0]:.4f}")

    # ---------------- RQ3
    o = r["rq3"]
    M.update(weaveUsPerOp=f"{o['fit']['us_per_op']:.1f}", weaveRsq=f"{o['fit']['r2']:.3f}",
             weaveRatioMed=f"{o['weave_over_transpile']['median']:.2f}",
             weaveRatioMin=f"{o['weave_over_transpile']['min']:.2f}",
             weaveRatioMax=f"{o['weave_over_transpile']['max']:.2f}")
    big = max(o["rows"], key=lambda x: x["ops"])
    M["weaveMaxOps"] = f"{big['ops']:,}".replace(",", "{,}")
    tr_rows = [x for x in o["rows"] if "transpile_ms" in x]
    M["transpileRows"] = str(len(tr_rows))
    M["transpileMaxOps"] = f"{max(x['ops'] for x in tr_rows):,}".replace(",", "{,}")
    pp = o["pass_in_pipeline"]
    M.update(passPipeMed=f"{pp['median']:.1f}", passPipeMin=f"{pp['min']:.1f}",
             passPipeMax=f"{pp['max']:.1f}")
    M["chainUsPerAspect"] = f"{o['runtime_chain']['us_per_aspect']:.1f}"
    M["weaveMaxMs"] = f"{big['weave_full_ms']:.0f}"
    empty = sorted(o["rows"], key=lambda x: x["ops"])
    import numpy as np
    ops = np.array([x["ops"] for x in empty], float)
    M["weaveEmptyUsPerOp"] = f"{np.polyfit(ops, [x['weave_empty_ms'] for x in empty], 1)[0] * 1e3:.1f}"

    pr = o["probe_after_transpile"]
    inc = [100 * (x["probed_depth"] / x["plain_depth"] - 1) for x in pr]
    M["probeDepthIncMin"] = f"{min(inc):.0f}"
    M["probeDepthIncMax"] = f"{max(inc):.0f}"
    M["probeTwoQSame"] = "unchanged" if all(x["plain_2q"] == x["probed_2q"] for x in pr) else "changed"
    dur = [100 * (x["probed_duration_us"] / x["plain_duration_us"] - 1) for x in pr]
    M["probeDurIncMin"] = f"{min(dur):.0f}"
    M["probeDurIncMax"] = f"{max(dur):.0f}"

    # ---------------- RQ4 mitigation
    mit = r["rq4_mitigation"]
    M["rtOverheadMs"] = f"{mit['D1']['weaver_overhead_ms_median']:.1f}"
    M["rtOverheadDevMs"] = f"{mit['DEV_L2']['weaver_overhead_ms_median']:.1f}"
    for tag, name in (("DOne", "D1"), ("DTwo", "D2"), ("DevLTwo", "DEV_L2"), ("DevLOne", "DEV_L1")):
        s = mit[name]
        best = "zne_readout" if name.startswith("DEV") else "exponential"
        M[f"{tag}MAEun"] = f"{s['pooled']['unmitigated'][0]:.4f}"
        M[f"{tag}MAEbest"] = f"{s['pooled'][best][0]:.4f}"
        M[f"{tag}Red"] = pct(s["reduction_pct"])
        M[f"{tag}RedLo"] = pct(s["reduction_pct_ci"][0])
        M[f"{tag}RedHi"] = pct(s["reduction_pct_ci"][1])
        M[f"{tag}Impr"] = pct(100 * s["fraction_improved"], 0)
        M[f"{tag}Atwelve"] = f"{s['a12']:.2f}"
        M[f"{tag}P"] = pval(s["wilcoxon_p"])
        M[f"{tag}N"] = str(s["n"])
        if name.startswith("DEV"):
            M[f"{tag}RedZne"] = pct(s["reduction_pct_zne_only"])
            M[f"{tag}Flag"] = pct(100 * s["policy_flagged_fraction"], 0)
    raw = r["rq4_mitigation_raw"]
    d1 = raw["D1"]["records"]

    def sign_change(rec):
        e = rec["expvals"]
        return not (all(v > 0 for v in e) or all(v < 0 for v in e))
    for tag, name in (("DOne", "D1"), ("DTwo", "D2")):
        recs = raw[name]["records"]
        M[f"{tag}Fallback"] = str(sum(sign_change(x) for x in recs))
        small = [x for x in recs if abs(x["ideal"]) < 0.1]
        M[f"{tag}SmallN"] = str(len(small))
        M[f"{tag}SmallUn"] = f"{np.mean([abs(x['unmitigated'] - x['ideal']) for x in small]):.4f}"
        M[f"{tag}SmallMit"] = f"{np.mean([abs(x['exponential'] - x['ideal']) for x in small]):.4f}"
    lf = mit["LOCALFOLD"]
    M.update(LFUnmit=f"{lf['pooled']['global_unmitigated'][0]:.4f}",
             LFGlobal=f"{lf['pooled']['global_mitigated'][0]:.4f}",
             LFGlobalLo=f"{lf['pooled']['global_mitigated'][1]:.4f}",
             LFGlobalHi=f"{lf['pooled']['global_mitigated'][2]:.4f}",
             LFLocal=f"{lf['pooled']['local2q_mitigated'][0]:.4f}",
             LFLocalLo=f"{lf['pooled']['local2q_mitigated'][1]:.4f}",
             LFLocalHi=f"{lf['pooled']['local2q_mitigated'][2]:.4f}",
             LFOps=pct(100 * lf["executed_ops_ratio"], 0), LFN=str(lf["n"]),
             LFP=pval(lf["wilcoxon_two_sided_p"]),
             LFInstances=str(max(x["instance"] for x in raw["LOCALFOLD"]["records"]) + 1))
    M["sweepInstances"] = str(max(x["instance"] for x in raw["SWEEP"]["records"]) + 1)
    sw = mit["SWEEP"]["per_p2"]
    ps = sorted(sw, key=float)
    M.update(sweepUnLo=f"{sw[ps[0]]['unmitigated'][0]:.4f}", sweepUnHi=f"{sw[ps[-1]]['unmitigated'][0]:.4f}",
             sweepMitLo=f"{sw[ps[0]]['exponential'][0]:.4f}", sweepMitHi=f"{sw[ps[-1]]['exponential'][0]:.4f}")
    M["DOneExactMAE"] = f"{np.mean([abs(x['exact_mitigated'] - x['ideal']) for x in d1]):.4f}"
    M["zneShots"] = f"{r['protocol']['zne_shots']:,}".replace(",", "{,}")
    M["zneInstances"] = str(max(x["instance"] for x in d1) + 1)
    M["devInstances"] = str(max(x["instance"] for x in raw["DEV_L2"]["records"]) + 1)
    M["devLOneInstances"] = str(max(x["instance"] for x in raw["DEV_L1"]["records"]) + 1)
    M["devLOneInstancesTotal"] = str(len(raw["DEV_L1"]["records"]))
    M["devLOneRO"] = f"{100 * np.median([x['readout_error_q0'] for x in raw['DEV_L1']['records']]):.1f}\\%"
    M["devLTwoRO"] = f"{100 * np.median([x['readout_error_q0'] for x in raw['DEV_L2']['records']]):.1f}\\%"
    sw = raw["SWEEP"]["records"]
    M["sweepMin"] = f"{min(x['p2'] for x in sw):g}"
    M["sweepMax"] = f"{max(x['p2'] for x in sw):g}"
    dv = r["device"]
    M["torinoCZ"] = f"{100 * dv['median_cz_error']:.2f}\\%"
    M["torinoRO"] = f"{100 * dv['median_readout_error']:.1f}\\%"
    import datetime as _dt
    M["torinoDate"] = _dt.datetime.fromisoformat(dv["calibration_date"]).strftime("%B %-d, %Y")

    # Mitiq
    M.update(mitiqVer=mx["mitiq"], mitiqQiskit=mx["qiskit"], mitiqPython=mx["python"],
             mitiqN=str(mx["instances"]),
             mitiqRich=sci_up(max(mx["max_abs_diff"]["richardson"], mx["max_abs_diff"]["linear"])),
             mitiqExpSame=sci_up(mx["exponential_same_model"]["max_abs_diff"]),
             mitiqExpMatch=str(mx["exponential_same_model"]["instances"]),
             mitiqExpDiff=sci(mx["max_abs_diff"]["exponential"]))
    M["mitiqSameModelWorst"] = sci_up(max(mx["max_abs_diff"]["richardson"], mx["max_abs_diff"]["linear"],
                                          mx["exponential_same_model"]["max_abs_diff"]))
    M["mitiqComparisons"] = str(3 * mx["instances"])
    M["mitiqAgree"] = str(2 * mx["instances"] + mx["exponential_same_model"]["instances"])

    # policy
    pol = r["rq4_policy"]
    det = pol["detection"]
    M["polSuite"] = str(pol["suite_size"])
    M["polClean"] = str(det["clean"][1])
    M["polMutants"] = str(pol["suite_size"] - det["clean"][1])
    M["polAgree"] = f"{pol['agreement_with_procedural'][0]} of {pol['agreement_with_procedural'][1]}"
    M["polAllDetected"] = "yes" if all(v[0] == v[1] for v in det.values()) else "NO"
    M["polSingle"] = f"{sum(pol['single_class'].values())} of {pol['suite_size'] - det['clean'][1]}"
    M["physRestricted"] = str(pol["restricted_physical_qubits"])
    ph = pol["physical"]
    M["physN"] = str(ph["level1"]["n"])
    for lvl, tag in (("level1", "One"), ("level2", "Two"), ("level3", "Three")):
        M[f"phys{tag}"] = str(ph[lvl]["flagged_physical"])
        M[f"phys{tag}Pct"] = pct(100 * ph[lvl]["flagged_physical"] / ph[lvl]["n"], 0)

    # audit
    au = r["rq4_audit"]
    M["auditRecords"] = str(au["records"])
    M["auditTrials"] = str(au["trials"])

    # implementation size and test count
    lib = ["core.py", "aspects.py", "passes.py", "qutils.py", "execution.py", "__init__.py"]
    M["aqCoreLOC"] = str(sum(sloc(os.path.join(ROOT, "aspectq", f)) for f in lib))
    study = ["metrics.py", "benchmarks.py", "simulation.py",
             "designs/__init__.py", "designs/tangled.py", "designs/procedural.py",
             "designs/woven.py"]
    M["aqStudyLOC"] = str(sum(sloc(os.path.join(ROOT, "aspectq", f)) for f in study))
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                             cwd=ROOT, capture_output=True, text=True).stdout
        n_tests = [ln for ln in out.splitlines() if "tests collected" in ln or "test collected" in ln]
        M["nTests"] = n_tests[-1].split()[0] if n_tests else "?"
    except Exception:
        M["nTests"] = "?"

    with open(os.path.join(out_dir, "results_macros.tex"), "w") as f:
        f.write("% generated by experiments/make_macros.py -- do not edit\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    write_tables(r, out_dir)
    print(f"wrote {len(M)} macros and tables to {out_dir}")


def write_tables(r: dict, out_dir: str) -> None:
    m = r["rq1"]["metrics"]
    concerns = ["ErrorMitigation", "Observability", "SecurityPolicy", "ComplianceAudit",
                "ResourceQuota"]
    names = {"ErrorMitigation": "Error mitigation", "Observability": "Observability",
             "SecurityPolicy": "Security policy", "ComplianceAudit": "Compliance audit",
             "ResourceQuota": "Resource quota"}
    rows = []
    for cname in concerns:
        cells = []
        for d in ("tangled", "procedural", "aspectq"):
            v = m[d]["per_concern"][cname]
            cells += [v["CL"], v["FRAG"], v["CDLOC"], v["LOCC"]]
        rows.append(f"{names[cname]} & " + " & ".join(map(str, cells)) + r" \\")
    tot = []
    for d in ("tangled", "procedural", "aspectq"):
        t = m[d]["totals"]
        tot += [t["CL"], t["FRAG"], t["CDLOC"], t["LOCC"]]
    extra = []
    for label, fn in (("CDC / CDO", lambda d: f"{m[d]['totals']['CDC']} / {m[d]['totals']['CDO']}"),
                      ("Driver: concern / total lines",
                       lambda d: f"{m[d]['driver']['concern_lines']} / {m[d]['driver']['code_lines']}"),
                      ("Degree of tangling (driver)", lambda d: f"{100 * m[d]['driver']['tangling']:.0f}\\%")):
        extra.append(f"{label} & " + " & ".join(
            fr"\multicolumn{{4}}{{c}}{{{fn(d)}}}" for d in ("tangled", "procedural", "aspectq")) + r" \\")
    tex = r"""\begin{table*}[t]
\centering
\caption{Separation-of-concerns metrics for the three designs of the same five concerns. CL: change locality (edit sites in the driver); FRAG: concern fragments; CDLOC: transition points; LOCC: lines of concern code. Lower is better for all metrics.}
\label{tab:modularity}
\footnotesize
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{@{}l rrrr rrrr rrrr@{}}
\toprule
 & \multicolumn{4}{c}{\textbf{Tangled}} & \multicolumn{4}{c}{\textbf{Procedural}} & \multicolumn{4}{c}{\textbf{AspectQ}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}
\textbf{Concern} & CL & FRAG & CDLOC & LOCC & CL & FRAG & CDLOC & LOCC & CL & FRAG & CDLOC & LOCC \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\textbf{Total} & """ + " & ".join(fr"\textbf{{{x}}}" for x in tot) + r""" \\
\midrule
""" + "\n".join(extra) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(out_dir, "tab_modularity.tex"), "w").write(tex)

    q = r["rq2"]
    st = q["transpiler_stages"]
    tex = r"""\begin{table}[t]
\centering
\caption{Preservation checks (RQ2). Rows 1, 2, 4 and 6 compare woven and base circuits; row 3 checks global folding on further random circuits, and row 5 compares the observability metrics with Qiskit's own counts.}
\label{tab:preservation}
\footnotesize
\begin{tabular}{@{}p{5.2cm}r@{}}
\toprule
\textbf{Check} & \textbf{Passed} \\
\midrule
""" + f"""Operator equal, incl.\\ global phase (unitary circuits $\\times$ 8 aspect subsets) & {q['unitary_checks'][0]}/{q['unitary_checks'][1]} \\\\
DAG isomorphic after removing barriers (all circuits $\\times$ 8 aspect subsets) & {q['dag_checks'][0]}/{q['dag_checks'][1]} \\\\
Global folding: same operator and $\\lambda\\times$ gates ({q['folding_checks'][1] // 3} circuits $\\times$ $\\lambda \\in \\{{3,5,7\\}}$) & {q['folding_checks'][0]}/{q['folding_checks'][1]} \\\\
Layer-ordered probes leave the depth unchanged & {q['probe_depth_neutral'][0]}/{q['probe_depth_neutral'][1]} \\\\
Observability metrics equal Qiskit's depth and counts & {q['observability_exact'][0]}/{q['observability_exact'][1]} \\\\
Weaving at a transpiler stage leaves the compiled DAG unchanged & {sum(s['identical'] for s in st)}/{len(st)} \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(out_dir, "tab_preservation.tex"), "w").write(tex)

    pr = r["rq3"]["probe_after_transpile"]
    rows = "\n".join(f"{x['qubits']} & {x['plain_depth']} & {x['probed_depth']} & "
                     f"{x['plain_duration_us']:.1f} & {x['probed_duration_us']:.1f} & "
                     f"{x['plain_2q']} & {x['probed_2q']} \\\\" for x in pr)
    tex = r"""\begin{table}[t]
\centering
\caption{Effect of probe barriers after level-3 transpilation with ALAP scheduling for the Torino device (hardware-efficient ansatz, five layers). Duration is the scheduled duration estimated from the device target.}
\label{tab:probes}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{Depth}} & \multicolumn{2}{c}{\textbf{Duration ($\mu$s)}} & \multicolumn{2}{c}{\textbf{CZ gates}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
\textbf{Qubits} & plain & probed & plain & probed & plain & probed \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(out_dir, "tab_probes.tex"), "w").write(tex)

    mit = r["rq4_mitigation"]
    raw = r["rq4_mitigation_raw"]

    def sign_change(rec):
        e = rec["expvals"]
        return not (all(v > 0 for v in e) or all(v < 0 for v in e))
    fb1 = sum(sign_change(x) for x in raw["D1"]["records"])
    fb2 = sum(sign_change(x) for x in raw["D2"]["records"])
    n1, n2 = len(raw["D1"]["records"]), len(raw["D2"]["records"])

    def cell(ci):
        return f"{ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]"
    rows = []
    for name, label in (("D1", "D1 depolarizing"), ("D2", "D2 depolarizing (stress)")):
        p = mit[name]["pooled"]
        rows.append(f"{label} & {mit[name]['n']} & {cell(p['unmitigated'])} & "
                    f"{cell(p['exponential'])} & {cell(p['richardson'])} & {cell(p['linear'])} \\\\")
    dev_rows = []
    for name, label in (("DEV_L2", "Torino, level 2"), ("DEV_L1", "Torino, level 1")):
        p = mit[name]["pooled"]
        dev_rows.append(f"{label} & {mit[name]['n']} & {cell(p['unmitigated'])} & {cell(p['zne'])} & "
                        f"{cell(p['readout'])} & {cell(p['zne_readout'])} \\\\")
    tex = r"""\begin{table*}[t]
\centering
\caption{Mean absolute error of $\langle Z_0\rangle$ over all layer counts, with 95\% bootstrap confidence intervals (RQ4). Upper part: extrapolation models under depolarizing noise; the exponential column falls back to Richardson extrapolation when the scaled values change sign (""" + f"{fb1} of {n1} instances under D1, {fb2} of {n2} under D2" + r"""). Lower part: composition of ZNE (exponential) and readout mitigation on the Torino noise model.}
\label{tab:zne}
\footnotesize
\begin{tabular}{@{}lrllll@{}}
\toprule
\textbf{Noise model} & $n$ & \textbf{Unmitigated} & \textbf{ZNE, exponential} & \textbf{ZNE, Richardson} & \textbf{ZNE, linear} \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
 & & \textbf{Unmitigated} & \textbf{ZNE} & \textbf{Readout} & \textbf{ZNE $\supset$ readout} \\
\midrule
""" + "\n".join(dev_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(out_dir, "tab_zne.tex"), "w").write(tex)

    pol = r["rq4_policy"]
    ph = pol["physical"]
    rows = "\n".join(f"{lvl[-1]} & {v['flagged_physical']}/{v['n']} & "
                     f"{100 * v['flagged_physical'] / v['n']:.0f}\\% \\\\"
                     for lvl, v in ph.items())
    tex = r"""\begin{table}[t]
\centering
\caption{Physical-qubit policy on the Torino device: the qubits with readout error above 5\% are restricted, and the policy aspect is woven after optimization. The same """ + str(ph["level1"]["n"]) + r""" random ans\"atze were compiled at each optimization level.}
\label{tab:policy}
\footnotesize
\begin{tabular}{@{}crr@{}}
\toprule
\textbf{Optimization level} & \textbf{Jobs using a restricted qubit} & \textbf{Share} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(out_dir, "tab_policy.tex"), "w").write(tex)

    au = r["rq4_audit"]["detection"]
    labels = {"modify": "Modify a record", "delete": "Delete a record", "insert": "Insert a forged record",
              "reorder": "Swap two records", "truncate": "Truncate the tail",
              "rewrite": "Modify and re-chain"}
    rows = "\n".join(f"{labels[k]} & {100 * v['hash_chain']:.0f}\\% & {100 * v['anchored_head']:.0f}\\% & "
                     f"{100 * v['hmac']:.0f}\\% \\\\" for k, v in au.items())
    tex = r"""\begin{table}[t]
\centering
\caption{Detection rate of tampering with the audit chain, by verifier.}
\label{tab:audit}
\footnotesize
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Attack} & \textbf{Hash chain} & \textbf{+ anchored head} & \textbf{+ HMAC} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(out_dir, "tab_audit.tex"), "w").write(tex)


if __name__ == "__main__":
    main()
