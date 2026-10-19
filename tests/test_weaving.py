"""Behaviour-preservation and weaving-semantics tests for AspectQ."""
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import RemoveBarriers

from aspectq import (ComplianceAuditAspect, ErrorMitigationAspect,
                     ObservabilityAspect, ResourceQuotaAspect,
                     SecurityPolicyAspect, Weaver, WeaveContext, at_layer,
                     before_measure, fold_global, layer, on_qubits,
                     two_qubit_gates)
from aspectq.benchmarks import workload_suite
from aspectq.core import Aspect, JoinPointKind, Phase
from aspectq.simulation import ExactBackend, depolarizing_noise


def structural_weaver(probes=True):
    return Weaver([ObservabilityAspect(insert_probes=probes),
                   SecurityPolicyAspect(allowed_gates={"h", "cx"},
                                        restricted_qubits={1}, max_depth=3),
                   ComplianceAuditAspect()])


@pytest.mark.parametrize("seed", range(40))
def test_unitary_preserved(seed):
    circ = random_circuit(4, 8, max_operands=3, seed=seed)
    woven, _ = structural_weaver().weave(circ)
    assert Operator(woven) == Operator(circ)          # exact, incl. global phase


@pytest.mark.parametrize("seed", range(40))
def test_dag_preserved_with_measure_and_reset(seed):
    circ = random_circuit(4, 8, max_operands=3, measure=True, reset=True,
                          seed=1000 + seed)
    woven, _ = structural_weaver().weave(circ)
    assert circuit_to_dag(RemoveBarriers()(woven)) == \
        circuit_to_dag(RemoveBarriers()(circ))


@pytest.mark.parametrize("seed", range(20))
def test_layers_match_qiskit_depth_and_probes_add_no_depth(seed):
    circ = random_circuit(5, 10, max_operands=3, measure=True, seed=2000 + seed)
    woven, wc = structural_weaver().weave(circ)
    obs = wc.metrics["observability"]
    assert obs["num_layers"] == circ.depth()
    assert woven.depth() == circ.depth()
    assert obs["probes_inserted"] == circ.depth()
    assert sum(obs["per_layer"].values()) == sum(
        1 for i in circ.data if i.operation.name != "barrier")


@pytest.mark.parametrize("lam", [1, 3, 5, 7])
def test_global_folding_preserves_unitary(lam):
    circ = random_circuit(3, 6, max_operands=2, seed=lam)
    folded = fold_global(circ, lam)
    assert Operator(folded) == Operator(circ)
    assert folded.size() == lam * circ.size()


def test_folding_rejects_even_scale():
    with pytest.raises(ValueError):
        fold_global(QuantumCircuit(1), 2)


class Recorder(Aspect):
    """Records the order in which advice runs."""
    phase = Phase.STRUCTURAL

    def __init__(self, tag, log, pointcut):
        super().__init__(pointcut)
        self.name, self.log = tag, log

    def on_structural_before(self, jp, out, wc):
        self.log.append(("before", self.name, jp.kind.value))

    def on_structural_after(self, jp, out, wc):
        self.log.append(("after", self.name, jp.kind.value))


def test_precedence_before_and_after_order():
    log = []
    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    w = Weaver([Recorder("A", log, two_qubit_gates()),
                Recorder("B", log, two_qubit_gates())])
    w.weave(circ)
    assert log == [("before", "A", "two_qubit_gate"),
                   ("before", "B", "two_qubit_gate"),
                   ("after", "B", "two_qubit_gate"),
                   ("after", "A", "two_qubit_gate")]


def test_pointcut_algebra_selects_expected_join_points():
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])
    hits = []

    class Probe(Aspect):
        def on_structural_analysis(self, jp, original, wc):
            hits.append((jp.kind.value, jp.name, jp.qubits))

    Weaver([Probe(two_qubit_gates() & on_qubits({2}))]).weave(circ)
    assert hits == [("two_qubit_gate", "cx", (1, 2))]
    hits.clear()
    Weaver([Probe(before_measure() & ~on_qubits({0}))]).weave(circ)
    assert [h[2] for h in hits] == [(1,), (2,)]
    hits.clear()
    Weaver([Probe(layer() & at_layer(1))]).weave(circ)
    assert hits == [("layer", "layer1", (0, 1, 2))]


def test_runtime_around_chain_precedence():
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.measure_all()
    backend = ExactBackend(depolarizing_noise(1e-3, 1e-2))
    inner = WeaveContext()
    Weaver([ErrorMitigationAspect(obs_qubits=[0, 1]),
            ResourceQuotaAspect()]).run(circ, backend, shots=1000, wc=inner)
    outer = WeaveContext()
    Weaver([ResourceQuotaAspect(),
            ErrorMitigationAspect(obs_qubits=[0, 1])]).run(circ, backend,
                                                           shots=1000, wc=outer)
    assert inner.metrics["quota"]["executions"] == 3
    assert inner.metrics["quota"]["shots"] == 3000
    assert outer.metrics["quota"]["executions"] == 1
    assert inner.metrics["runtime"]["executions"] == 3


def test_quota_denies_over_budget():
    from aspectq.aspects import QuotaExceeded
    circ = QuantumCircuit(1)
    circ.measure_all()
    backend = ExactBackend()
    with pytest.raises(QuotaExceeded):
        Weaver([ErrorMitigationAspect(), ResourceQuotaAspect(max_shots=2000)]).run(
            circ, backend, shots=1000)


def test_zne_exact_in_noiseless_limit():
    """With no noise every folded circuit gives the ideal value."""
    for circ, obs in workload_suite(3):
        measured = circ.copy()
        measured.measure_all()
        wc = WeaveContext()
        res = Weaver([ErrorMitigationAspect(obs_qubits=list(obs))]).run(
            measured, ExactBackend(), shots=1, wc=wc)
        vals = wc.metrics["zne"]["expvals"]
        assert max(vals) - min(vals) < 1e-9
        assert abs(res["expectation_mitigated"] - vals[0]) < 1e-9


def test_split_final_measurements_keeps_mid_circuit_measure():
    from aspectq.qutils import split_final_measurements
    qc = QuantumCircuit(2, 3)
    qc.h(0)
    qc.measure(0, 2)              # mid-circuit: qubit 0 is used again
    qc.x(0)
    qc.cx(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    body, meas = split_final_measurements(qc)
    assert sorted(meas) == [(0, 0), (1, 1)]
    assert [i.operation.name for i in body.data] == ["h", "measure", "x", "cx"]
    assert body.num_clbits == 3


def test_readout_mitigation_removes_assignment_bias():
    from aspectq import ReadoutMitigationAspect
    from aspectq.simulation import depolarizing_backend
    qc = QuantumCircuit(2)
    qc.ry(0.7, 0)
    qc.cx(0, 1)
    ideal = float(__import__("numpy").cos(0.7))
    qc.measure_all()
    be = depolarizing_backend(0.0, 0.0, readout=0.08, seed=7)
    wc = WeaveContext()
    raw = Weaver([]).run(qc, be, shots=200_000)
    from aspectq import parity_expectation
    biased = parity_expectation(raw["counts"], [0])
    fixed = Weaver([ReadoutMitigationAspect()]).run(qc, be, shots=200_000, wc=wc)
    corrected = parity_expectation(fixed["counts"], [0])
    assert abs(biased - ideal) > 0.1                  # ~16% shrinkage
    assert abs(corrected - ideal) < 0.01
    assert wc.metrics["readout"]["calibration_executions"] == 2


def test_zne_reads_program_register_layout():
    """Counts keys with an extra classical register must not confuse ZNE."""
    from qiskit import ClassicalRegister, QuantumRegister
    from aspectq.simulation import depolarizing_backend
    qr, cr = QuantumRegister(2, "q"), ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    qc.x(0)                       # <Z_0> = -1, <Z_1> = +1
    qc.measure_all()              # adds register 'meas' after 'c'
    wc = WeaveContext()
    Weaver([ErrorMitigationAspect(obs_qubits=[0])]).run(
        qc, depolarizing_backend(0, 0, seed=1), shots=2000, wc=wc)
    assert wc.metrics["zne"]["expvals"] == [-1.0, -1.0, -1.0]


def test_zne_rejects_mid_circuit_measurement():
    qc = QuantumCircuit(1, 2)
    qc.h(0)
    qc.measure(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    with pytest.raises(ValueError):
        Weaver([ErrorMitigationAspect()]).run(qc, ExactBackend(), shots=10)


def test_two_qubit_join_points_exclude_directives_and_resets():
    qc = QuantumCircuit(2)
    qc.barrier(0, 1)
    qc.reset(0)
    qc.cx(0, 1)
    seen = []

    class Probe(Aspect):
        def on_structural_analysis(self, jp, original, wc):
            seen.append(jp.name)

    Weaver([Probe(two_qubit_gates())]).weave(qc)
    assert seen == ["cx"]


def test_device_folding_survives_translation():
    """Folded ISA circuits keep lambda x the two-qubit gates after translation."""
    from qiskit import transpile
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from aspectq.benchmarks import hardware_efficient_ansatz
    dev = FakeTorino()
    phys = generate_preset_pass_manager(optimization_level=2, backend=dev,
                                        seed_transpiler=3).run(
        hardware_efficient_ansatz(4, 3, seed=3))
    base_cz = phys.count_ops().get("cz", 0)
    for lam in (3, 5):
        folded = transpile(fold_global(phys, lam),
                           basis_gates=list(dev.operation_names), optimization_level=0)
        assert folded.count_ops().get("cz", 0) == lam * base_cz


def test_weaver_rejects_classical_variables():
    from qiskit.circuit.classical import expr, types
    qc = QuantumCircuit(1)
    v = qc.add_var("flag", expr.lift(False))
    qc.h(0)
    qc.store(v, expr.lift(True))
    with pytest.raises(NotImplementedError):
        Weaver([ObservabilityAspect()]).weave(qc)


def test_digest_independent_of_same_layer_order():
    from aspectq.qutils import circuit_digest
    a = QuantumCircuit(2)
    a.h(0)
    a.x(1)
    b = QuantumCircuit(2)
    b.x(1)
    b.h(0)
    assert circuit_digest(a) == circuit_digest(b)
    c = QuantumCircuit(2)
    c.x(0)
    c.h(0)
    d = QuantumCircuit(2)
    d.h(0)
    d.x(0)
    assert circuit_digest(c) != circuit_digest(d)


@pytest.mark.parametrize("lam", [3, 5])
def test_local_folding_preserves_unitary_and_scales_selected_gates(lam):
    from aspectq import GateFoldingAspect
    circ = random_circuit(4, 6, max_operands=2, seed=lam)
    folded, _ = Weaver([GateFoldingAspect(two_qubit_gates(), lam)]).weave(circ)
    assert Operator(folded) == Operator(circ)
    n2 = sum(1 for i in circ.data if len(i.qubits) == 2)
    assert folded.size() == circ.size() + (lam - 1) * n2


def test_zne_with_local_folding_is_exact_without_noise():
    for circ, obs in workload_suite(5):
        meas = circ.copy()
        meas.measure_all()
        wc = WeaveContext()
        Weaver([ErrorMitigationAspect(obs_qubits=list(obs),
                                      fold_pointcut=two_qubit_gates())]).run(
            meas, ExactBackend(), shots=1, wc=wc)
        vals = wc.metrics["zne"]["expvals"]
        assert max(vals) - min(vals) < 1e-9


def test_runtime_before_after_nest_like_aspectj():
    log = []

    class Tracer(Aspect):
        phase = Phase.RUNTIME

        def __init__(self, tag):
            super().__init__(run_scope_pc())
            self.name = tag

        def before_run(self, circuit, wc, ctx):
            log.append(("before", self.name))

        def after_run(self, circuit, wc, ctx, result):
            log.append(("after", self.name))

    from aspectq import run_scope as run_scope_pc
    circ = QuantumCircuit(1)
    circ.measure_all()
    Weaver([Tracer("A"), ErrorMitigationAspect(scale_factors=(1, 3)),
            Tracer("B")]).run(circ, ExactBackend(), shots=10)
    # A encloses everything; B runs once per folded execution, inside ZNE
    assert log[0] == ("before", "A") and log[-1] == ("after", "A")
    assert log.count(("before", "B")) == 2


def test_user_runtime_aspect_needs_no_flag():
    class Counter(Aspect):
        phase = Phase.RUNTIME

        def __init__(self):
            from aspectq import run_scope
            super().__init__(run_scope())
            self.calls = 0

        def around_run(self, circuit, backend, shots, wc, proceed):
            self.calls += 1
            return proceed(circuit)

    c = Counter()
    circ = QuantumCircuit(1)
    circ.measure_all()
    Weaver([c]).run(circ, ExactBackend(), shots=10)
    assert c.calls == 1


def test_run_context_designators_and_tags():
    from aspectq import backend_matches, shots_at_least, tagged
    from aspectq.aspects import ResourceQuotaAspect
    circ = QuantumCircuit(1)
    circ.measure_all()
    circ.metadata = {"tags": ["production"]}
    wc = WeaveContext()
    Weaver([ResourceQuotaAspect(pointcut=shots_at_least(100) & tagged("production")
                                & backend_matches("Exact"))]).run(
        circ, ExactBackend(), shots=500, wc=wc)
    assert wc.metrics["quota"]["shots"] == 500
    wc2 = WeaveContext()
    Weaver([ResourceQuotaAspect(pointcut=shots_at_least(1000))]).run(
        circ, ExactBackend(), shots=500, wc=wc2)
    assert "quota" not in wc2.metrics


def test_sampler_backend_adapter_matches_backend_run():
    from qiskit import ClassicalRegister, QuantumRegister
    from qiskit.primitives import BackendSamplerV2
    from qiskit_aer import AerSimulator
    from aspectq import SamplerBackend
    qr, cr = QuantumRegister(2, "q"), ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    qc.x(0)
    qc.measure_all()
    wc = WeaveContext()
    be = SamplerBackend(BackendSamplerV2(backend=AerSimulator()))
    Weaver([ErrorMitigationAspect(obs_qubits=[0])]).run(qc, be, shots=200, wc=wc)
    assert wc.metrics["zne"]["expvals"] == [-1.0, -1.0, -1.0]


def test_inserted_operations_trigger_analysis_warning():
    from aspectq import GateFoldingAspect
    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    wc = WeaveContext()
    Weaver([SecurityPolicyAspect(allowed_gates={"cx"}),
            GateFoldingAspect(two_qubit_gates(), 3)]).weave(circ, wc)
    assert wc.metrics["inserted_operations"] == 2
    assert any("WARNING" in line for line in wc.log)
