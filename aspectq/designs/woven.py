"""
WOVEN design: the base driver plus AspectQ aspects.

The driver only prepares, executes and estimates.  Concerns are attached by
registering aspects with the weaver (one registration per concern, marked
``# @C:<tag>``); their outputs land in the weave context supplied by the
caller.  The list order is the precedence order: the quota aspect sits
*inside* the mitigation aspect so that it meters every folded execution.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from ..aspects import (ComplianceAuditAspect, ErrorMitigationAspect,
                       ObservabilityAspect, ResourceQuotaAspect,
                       SecurityPolicyAspect)
from ..core import WeaveContext, Weaver
from ..qutils import parity_expectation
from . import ConcernConfig


def run_program_woven(circuit: QuantumCircuit, backend, shots: int,
                      cfg: ConcernConfig, wc: WeaveContext) -> dict:
    """Estimate <Z...> on ``backend``; concerns woven by AspectQ.

    Concern outputs are recorded in ``wc``; the return value is the core result.
    """
    weaver = Weaver([
        ObservabilityAspect(),                                          # @C:obs
        SecurityPolicyAspect(allowed_gates=cfg.allowed_gates,           # @C:sec
                             restricted_qubits=cfg.restricted_qubits,   # @C:sec
                             max_qubits=cfg.max_qubits,                 # @C:sec
                             max_depth=cfg.max_depth),                  # @C:sec
        ComplianceAuditAspect(chain=cfg.audit_chain),                   # @C:aud
        ErrorMitigationAspect(scale_factors=cfg.scale_factors,          # @C:mit
                              obs_qubits=list(cfg.obs_qubits),          # @C:mit
                              extrapolation=cfg.extrapolation),         # @C:mit
        ResourceQuotaAspect(max_shots=cfg.max_shots),                   # @C:quo
    ])
    woven, _ = weaver.weave(circuit, wc)
    executable = woven.copy()
    executable.measure_all()
    run = weaver.run(executable, backend, shots=shots, wc=wc)
    return {"expectation": parity_expectation(run["counts"], list(cfg.obs_qubits))}
