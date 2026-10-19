"""
Transpiler integration: structural weaving as a Qiskit transformation pass.

``AspectQPass`` runs the structural phase of a :class:`~aspectq.core.Weaver`
inside a Qiskit ``PassManager``.  The weave context of the last circuit is
published in the property set; a summary of every weave (violations, audit
records, log) is also stored in ``circuit.metadata["aspectq"]`` so that it
survives parallel compilation of circuit lists.  Aspect objects that keep
state across circuits (such as the audit chain) are copied into worker
processes when Qiskit compiles in parallel, so chain the per-circuit audit
records after compilation in that case.  Attached to a stage of a preset
``StagedPassManager`` (e.g. ``post_layout`` or ``post_optimization``), the
aspects see the circuit as it exists at that stage, so a policy aspect woven
after layout reasons about *physical* qubits rather than logical ones.
"""
from __future__ import annotations

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.basepasses import TransformationPass

from .core import WeaveContext, Weaver


class AspectQPass(TransformationPass):
    """Weave the structural aspects of ``weaver`` into the DAG being compiled.

    The resulting :class:`~aspectq.core.WeaveContext` is published in the
    pass manager's property set under ``"aspectq_context"``.
    """

    def __init__(self, weaver: Weaver):
        super().__init__()
        self.weaver = weaver

    def run(self, dag):
        circuit = dag_to_circuit(dag, copy_operations=False)
        woven, wc = self.weaver.weave(circuit, WeaveContext())
        self.property_set["aspectq_context"] = wc
        # the property set keeps only the last circuit's context and is not
        # returned from parallel workers, so a summary travels with the circuit
        meta = dict(woven.metadata or {})
        meta["aspectq"] = list(meta.get("aspectq", [])) + [{
            "aspects": list(wc.applied_aspects),
            "violations": list(wc.violations),
            "audit": [dict(e) for e in wc.audit],
            "log": list(wc.log)}]
        woven.metadata = meta
        return circuit_to_dag(woven, copy_operations=False)


def attach(pm: StagedPassManager, weaver: Weaver,
           stage: str = "post_optimization") -> StagedPassManager:
    """Attach structural weaving to ``stage`` of a staged pass manager.

    Any existing passes at that stage are kept and run first.
    """
    if stage not in pm.expanded_stages:
        raise ValueError(f"unknown stage {stage!r}; expected one of "
                         f"{pm.expanded_stages}")
    existing = getattr(pm, stage, None)
    combined = PassManager()
    if existing is not None:
        combined.append(existing.to_flow_controller())
    combined.append(AspectQPass(weaver))
    setattr(pm, stage, combined)
    return pm
