"""AspectQ: aspect-oriented weaving of crosscutting concerns into quantum programs."""
from .core import (Aspect, JoinPoint, JoinPointKind, Phase, Pointcut,
                   WeaveContext, Weaver, any_join_point, at_layer, before_measure,
                   backend_matches, circuit_scope, gate, kind_is, layer,
                   on_qubits, restricted_qubits, run_scope, shots_at_least,
                   tagged, two_qubit_gates)
from .aspects import (ComplianceAuditAspect, ErrorMitigationAspect,
                      GateFoldingAspect, ObservabilityAspect, PolicyViolation, QuotaExceeded,
                      ReadoutMitigationAspect, ResourceQuotaAspect,
                      SecurityPolicyAspect)
from .execution import SamplerBackend
from .passes import AspectQPass, attach
from .qutils import fold_global, parity_expectation

__all__ = [
    "Aspect", "JoinPoint", "JoinPointKind", "Phase", "Pointcut",
    "WeaveContext", "Weaver", "any_join_point", "at_layer", "before_measure",
    "circuit_scope", "gate", "kind_is", "layer", "on_qubits",
    "restricted_qubits", "run_scope", "tagged", "two_qubit_gates",
    "shots_at_least", "backend_matches", "GateFoldingAspect", "SamplerBackend",
    "ComplianceAuditAspect", "ErrorMitigationAspect", "ObservabilityAspect",
    "PolicyViolation", "QuotaExceeded", "ReadoutMitigationAspect",
    "ResourceQuotaAspect",
    "SecurityPolicyAspect", "AspectQPass", "attach", "fold_global",
    "parity_expectation",
]
__version__ = "0.2.0"
