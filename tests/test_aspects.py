"""Tests of the concrete aspects, the transpiler pass and the three designs."""
import copy
import json

import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers

from aspectq import (ComplianceAuditAspect, ObservabilityAspect,
                     PolicyViolation, SecurityPolicyAspect, Weaver,
                     WeaveContext)
from aspectq.benchmarks import workload_suite
from aspectq.designs import ConcernConfig
from aspectq.designs.procedural import run_program_procedural
from aspectq.designs.tangled import run_program_tangled
from aspectq.designs.woven import run_program_woven
from aspectq.metrics import compare
from aspectq.passes import attach
from aspectq.simulation import ExactBackend, depolarizing_noise

ALLOWED = {"ry", "rx", "rz", "h", "x", "cx", "rzz", "p", "u"}


def _report_woven(wc):
    return {"violations": sorted(wc.violations),
            "observability": wc.metrics["observability"],
            "zne": {k: wc.metrics["zne"][k] for k in ("expvals", "mitigated",
                                                      "method")},
            "audit": wc.audit[-1], "quota": wc.metrics["quota"]}


def _normalise(ctx):
    out = dict(ctx)
    out["violations"] = sorted(ctx["violations"])
    return out


@pytest.mark.parametrize("idx", range(5))
def test_three_designs_exactly_equivalent(idx):
    circ, obs = workload_suite(11)[idx]
    backend = ExactBackend(depolarizing_noise(1e-3, 1e-2))
    reports, values = [], []
    for fn in (run_program_tangled, run_program_procedural, run_program_woven):
        cfg = ConcernConfig(allowed_gates=ALLOWED, restricted_qubits=frozenset({3}),
                            max_qubits=4, max_depth=30, obs_qubits=obs,
                            max_shots=10 ** 6)
        if fn is run_program_woven:
            wc = WeaveContext()
            values.append(fn(circ, backend, 4096, cfg, wc)["expectation"])
            reports.append(_report_woven(wc))
        else:
            ctx = {}
            values.append(fn(circ, backend, 4096, cfg, ctx)["expectation"])
            reports.append(_normalise(ctx))
    dumped = [json.dumps(r, sort_keys=True) for r in reports]
    assert dumped[0] == dumped[1]
    # woven executes a re-linearised (equivalent) circuit: equal to rounding
    assert abs(values[0] - values[2]) < 1e-9
    assert abs(reports[0]["zne"]["mitigated"] - reports[2]["zne"]["mitigated"]) < 1e-9
    for key in ("violations", "observability", "audit", "quota"):
        assert reports[0][key] == reports[2][key]


def test_policy_detects_each_violation_class():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.t(1)                        # forbidden gate
    qc.cx(0, 3)                    # restricted qubit 3
    qc.measure_all()               # measurement on restricted qubit 3
    wc = WeaveContext()
    Weaver([SecurityPolicyAspect(allowed_gates={"h", "cx"},
                                 restricted_qubits={3}, max_qubits=4,
                                 max_depth=1)]).weave(qc, wc)
    text = " | ".join(wc.violations)
    assert "width 5 > max 4" in text
    assert "depth" in text
    assert "gate 't'" in text
    assert "'cx' touches restricted" in text
    assert "'measure' touches restricted" in text


def test_policy_strict_mode_raises():
    qc = QuantumCircuit(1)
    qc.t(0)
    with pytest.raises(PolicyViolation):
        Weaver([SecurityPolicyAspect(allowed_gates={"h"}, strict=True)]).weave(qc)


def _audited_chain(n=6, key=None):
    audit = ComplianceAuditAspect(hmac_key=key)
    w = Weaver([audit])
    for k in range(n):
        qc = QuantumCircuit(2, name=f"c{k}")
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1 * k, 1)
        w.weave(qc)
    return audit


def test_audit_detects_modification_deletion_reordering():
    audit = _audited_chain()
    assert ComplianceAuditAspect.verify(audit.chain) == (True, -1)
    mod = copy.deepcopy(audit.chain)
    mod[2]["payload"]["circuit"] = "swapped"
    assert ComplianceAuditAspect.verify(mod) == (False, 2)
    dele = copy.deepcopy(audit.chain)
    del dele[3]
    assert ComplianceAuditAspect.verify(dele)[0] is False
    swap = copy.deepcopy(audit.chain)
    swap[1], swap[2] = swap[2], swap[1]
    assert ComplianceAuditAspect.verify(swap)[0] is False


def test_audit_truncation_needs_anchor_and_rewrite_needs_hmac():
    audit = _audited_chain()
    head = audit.head()
    truncated = copy.deepcopy(audit.chain[:-1])
    assert ComplianceAuditAspect.verify(truncated) == (True, -1)       # undetected
    assert ComplianceAuditAspect.verify(truncated, anchor=head)[0] is False

    key = b"secret"
    keyed = _audited_chain(key=key)
    forged = copy.deepcopy(keyed.chain)
    forged[1]["payload"]["circuit"] = "forged"
    prev = forged[0]["hash"]
    for e in forged[1:]:                                  # attacker re-chains
        e["prev"] = prev
        e["hash"] = ComplianceAuditAspect._hash(prev, e["payload"])
        prev = e["hash"]
    assert ComplianceAuditAspect.verify(forged) == (True, -1)          # undetected
    assert ComplianceAuditAspect.verify(forged, hmac_key=key)[0] is False


def test_pass_manager_integration_physical_policy():
    backend = GenericBackendV2(7, coupling_map=CouplingMap.from_line(7), seed=5)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                      seed_transpiler=1)
    plain = pm.run(qc)
    physical = {plain.layout.initial_index_layout()[q] for q in range(3)}
    target = min(physical)
    pm2 = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                       seed_transpiler=1)
    attach(pm2, Weaver([ObservabilityAspect(insert_probes=True),
                        SecurityPolicyAspect(restricted_qubits={target})]),
           stage="post_optimization")
    woven = pm2.run(qc)
    wc = pm2.property_set["aspectq_context"]
    assert any(f"[{target}]" in v for v in wc.violations)
    assert circuit_to_dag(RemoveBarriers()(woven)) == \
        circuit_to_dag(RemoveBarriers()(plain))


def test_metrics_lint_clean_and_tangling_ordering():
    res = compare()
    for design in res.values():
        assert design["lint"] == []
    t, p, a = (res[d]["totals"] for d in ("tangled", "procedural", "aspectq"))
    assert a["CL"] < t["CL"]
    assert res["aspectq"]["driver"]["concern_lines"] < \
        res["tangled"]["driver"]["concern_lines"]
