"""
Three functionally equivalent implementations of the same quantum driver,
used by the modularity study (RQ1):

  * tangled     -- all five concerns hand-inlined into one driver function
  * procedural  -- each concern factored into a helper function, invoked from
                   explicit call sites in the driver (the strongest non-AOP
                   design)
  * woven       -- the base driver plus AspectQ aspects

Every physical line that belongs to a crosscutting concern carries a marker
comment ``# @C:<tag>`` (tag in mit, obs, sec, aud, quo; ``+`` joins tags).
Whole helper functions and aspect classes are attributed to their concern by
name in :mod:`aspectq.metrics`.
"""
from dataclasses import dataclass, field
from typing import Optional

# Names recorded in audit payloads; identical across the three designs.
CONCERN_NAMES = ["Observability", "SecurityPolicy", "ComplianceAudit",
                 "ErrorMitigation(ZNE)", "ResourceQuota"]


@dataclass
class ConcernConfig:
    """Settings shared by the three designs."""
    allowed_gates: Optional[set] = None
    restricted_qubits: frozenset = frozenset()
    max_qubits: Optional[int] = None
    max_depth: Optional[int] = None
    scale_factors: tuple = (1, 3, 5)
    obs_qubits: tuple = (0,)
    extrapolation: str = "exponential"
    max_shots: Optional[int] = None
    audit_chain: list = field(default_factory=list)


class QuotaExceeded(Exception):
    pass
