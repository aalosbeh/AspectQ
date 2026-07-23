"""AspectQ: aspect-oriented weaving of crosscutting concerns into quantum programs."""
from .core import (Aspect, AdviceType, JoinPoint, JoinPointKind, Phase, Pointcut,
                   WeaveContext, Weaver, any_join_point, at_layer, before_measure,
                   circuit_scope, gate, kind_is, on_qubits, restricted_qubits,
                   run_scope, tagged, two_qubit_gates)
from .aspects import (ComplianceAuditAspect, ErrorMitigationAspect,
                      ObservabilityAspect, PolicyViolation, SecurityPolicyAspect,
                      fold_global, parity_expectation)

__all__ = [
    "Aspect", "AdviceType", "JoinPoint", "JoinPointKind", "Phase", "Pointcut",
    "WeaveContext", "Weaver", "any_join_point", "at_layer", "before_measure",
    "circuit_scope", "gate", "kind_is", "on_qubits", "restricted_qubits",
    "run_scope", "tagged", "two_qubit_gates", "ComplianceAuditAspect",
    "ErrorMitigationAspect", "ObservabilityAspect", "PolicyViolation",
    "SecurityPolicyAspect", "fold_global", "parity_expectation",
]
__version__ = "0.1.0"
