"""
AspectQ reusable aspects: four crosscutting concerns of quantum software,
each modularised as a single, self-contained Aspect.

  * ErrorMitigationAspect  -- runtime AROUND advice: zero-noise extrapolation
  * ObservabilityAspect    -- structural advice: per-layer instrumentation/metrics
  * SecurityPolicyAspect   -- structural analysis advice: compile-time policy checks
  * ComplianceAuditAspect  -- circuit-scope AROUND advice: tamper-evident provenance

Every concern below would otherwise be *scattered* across a monolithic program
and *tangled* with its core logic (see aspectq.baseline).  Here each lives in
exactly one place and is (un)pluggable without touching the base program.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Optional

import numpy as np
from qiskit import QuantumCircuit

from .core import (Aspect, AdviceType, JoinPoint, JoinPointKind, Phase,
                   Pointcut, WeaveContext, any_join_point)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parity_expectation(counts: dict[str, int], obs_qubits: list[int]) -> float:
    """<Z_{q0} Z_{q1} ...> estimated from measurement counts.

    Qiskit count keys are little-endian bitstrings (qubit 0 is the rightmost
    character), so qubit q corresponds to bitstring[-1 - q].
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    acc = 0.0
    for bitstring, n in counts.items():
        bs = bitstring.replace(" ", "")
        ones = sum(int(bs[-1 - q]) for q in obs_qubits)
        acc += ((-1) ** ones) * n
    return acc / total


def fold_global(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Unitary global folding U -> U (U^dag U)^n, giving noise scale 2n+1."""
    if scale_factor == 1:
        return circuit.copy()
    n = (scale_factor - 1) // 2
    inv = circuit.inverse()
    folded = circuit.copy()
    for _ in range(n):
        folded = folded.compose(inv).compose(circuit)
    return folded


def canonical_ops(circuit: QuantumCircuit) -> str:
    """Deterministic textual encoding of a circuit's operations (for hashing)."""
    rows = []
    for instr in circuit.data:
        q = [circuit.find_bit(b).index for b in instr.qubits]
        try:
            p = [round(float(x), 10) for x in instr.operation.params]
        except (TypeError, ValueError):
            p = []
        rows.append(f"{instr.operation.name}:{q}:{p}")
    return ";".join(rows)


# --------------------------------------------------------------------------- #
# 1. Error mitigation (runtime, AROUND)
# --------------------------------------------------------------------------- #
class ErrorMitigationAspect(Aspect):
    """Zero-noise extrapolation woven *around* execution.

    The base program is unaware of mitigation; this aspect intercepts the run,
    generates noise-scaled copies via unitary folding, executes each, estimates
    the observable at every scale, and extrapolates to the zero-noise limit.
    """

    name = "ErrorMitigation(ZNE)"
    phase = Phase.RUNTIME
    advice_type = AdviceType.AROUND
    provides_around_run = True

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 scale_factors: tuple[int, ...] = (1, 3, 5),
                 obs_qubits: Optional[list[int]] = None,
                 extrapolation: str = "exponential"):
        super().__init__(pointcut or any_join_point())
        self.scale_factors = scale_factors
        self.obs_qubits = obs_qubits or [0]
        self.extrapolation = extrapolation

    @staticmethod
    def _extrapolate(lam_arr, exp_arr, method: str) -> float:
        """Extrapolate the observable to the zero-noise limit (lambda -> 0)."""
        if method == "exponential" and (np.all(exp_arr > 0) or np.all(exp_arr < 0)):
            # depolarising noise decays the observable multiplicatively:
            # E(l) ~ s * exp(b0 + k*l)  =>  E(0) = s * exp(b0)
            s = np.sign(exp_arr[0])
            k, b0 = np.polyfit(lam_arr, np.log(np.abs(exp_arr)), 1)
            return float(s * np.exp(b0))
        # Richardson / linear polynomial fallback
        deg = min(len(lam_arr) - 1, 2)
        coeffs = np.polyfit(lam_arr, exp_arr, deg)
        return float(np.polyval(coeffs, 0.0))

    def around_run(self, circuit, backend, shots, wc: WeaveContext,
                   proceed: Callable) -> dict:
        base = circuit.remove_final_measurements(inplace=False)
        expvals, exec_count = [], 0
        for lam in self.scale_factors:
            folded = fold_global(base, lam)
            folded.measure_all()
            out = proceed(folded)
            exec_count += 1
            expvals.append(parity_expectation(out["counts"], self.obs_qubits))
        lam_arr = np.array(self.scale_factors, dtype=float)
        exp_arr = np.array(expvals, dtype=float)
        mitigated = self._extrapolate(lam_arr, exp_arr, self.extrapolation)
        wc.metrics["zne"] = {
            "scale_factors": list(self.scale_factors),
            "expvals": [float(x) for x in expvals],
            "unmitigated": float(expvals[0]),
            "mitigated": mitigated,
            "executions": exec_count,
        }
        wc.emit(f"[{self.name}] unmitigated={expvals[0]:.4f} "
                f"mitigated={mitigated:.4f} ({exec_count} circuit executions)")
        return {"counts": None, "shots": shots, "expectation_unmitigated":
                float(expvals[0]), "expectation_mitigated": mitigated,
                "scaled_expvals": [float(x) for x in expvals]}


# --------------------------------------------------------------------------- #
# 2. Observability (structural instrumentation + metrics)
# --------------------------------------------------------------------------- #
class ObservabilityAspect(Aspect):
    """Collects per-layer structural metrics; optionally inserts barrier probes.

    By default it is analysis-only (zero added depth): it records circuit width,
    depth, two-qubit-gate count, and a per-layer gate histogram.  With
    ``insert_probes=True`` it weaves a barrier at each new layer boundary as an
    observable checkpoint.
    """

    name = "Observability"
    phase = Phase.STRUCTURAL
    advice_type = AdviceType.BEFORE

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 insert_probes: bool = False):
        super().__init__(pointcut or any_join_point())
        self.insert_probes = insert_probes

    def on_structural_analysis(self, jp: JoinPoint, original, wc: WeaveContext):
        obs = wc.metrics.setdefault("observability", {
            "num_qubits": original.num_qubits, "num_layers": 0,
            "two_qubit_gates": 0, "gate_histogram": {}, "per_layer": {},
            "probes_inserted": 0})
        if jp.kind == JoinPointKind.CIRCUIT_ENTRY:
            return
        obs["gate_histogram"][jp.name] = obs["gate_histogram"].get(jp.name, 0) + 1
        if jp.num_qubits == 2:
            obs["two_qubit_gates"] += 1
        layer_key = str(jp.layer_index)
        obs["per_layer"][layer_key] = obs["per_layer"].get(layer_key, 0) + 1
        obs["num_layers"] = max(obs["num_layers"], jp.layer_index + 1)

    def on_structural_before(self, jp: JoinPoint, out: QuantumCircuit,
                             wc: WeaveContext):
        if not self.insert_probes or jp.kind == JoinPointKind.CIRCUIT_ENTRY:
            return
        last = wc.metrics.get("_obs_last_layer", -1)
        if jp.layer_index > last:
            out.barrier()
            wc.metrics["_obs_last_layer"] = jp.layer_index
            obs = wc.metrics.setdefault("observability", {"probes_inserted": 0})
            obs["probes_inserted"] = obs.get("probes_inserted", 0) + 1


# --------------------------------------------------------------------------- #
# 3. Security policy (structural, compile-time analysis)
# --------------------------------------------------------------------------- #
class PolicyViolation(Exception):
    pass


class SecurityPolicyAspect(Aspect):
    """Enforces a compile-time security/resource policy over the circuit.

    Checks: (a) only allow-listed gates, (b) no operations on restricted
    (e.g. reserved / calibration) qubits, (c) circuit width and depth caps.
    Violations are recorded in the weave context; with ``strict=True`` the
    first violation raises ``PolicyViolation``.
    """

    name = "SecurityPolicy"
    phase = Phase.STRUCTURAL
    advice_type = AdviceType.AFTER

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 allowed_gates: Optional[set[str]] = None,
                 restricted_qubits: Optional[set[int]] = None,
                 max_qubits: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 strict: bool = False):
        super().__init__(pointcut or any_join_point())
        self.allowed_gates = allowed_gates
        self.restricted_qubits = set(restricted_qubits or set())
        self.max_qubits = max_qubits
        self.max_depth = max_depth
        self.strict = strict

    def _flag(self, wc: WeaveContext, msg: str):
        wc.violations.append(msg)
        wc.emit(f"[{self.name}] VIOLATION: {msg}")
        if self.strict:
            raise PolicyViolation(msg)

    def on_structural_analysis(self, jp: JoinPoint, original, wc: WeaveContext):
        if jp.kind == JoinPointKind.CIRCUIT_ENTRY:
            if self.max_qubits is not None and original.num_qubits > self.max_qubits:
                self._flag(wc, f"width {original.num_qubits} > max {self.max_qubits}")
            if self.max_depth is not None and original.depth() > self.max_depth:
                self._flag(wc, f"depth {original.depth()} > max {self.max_depth}")
            return
        if jp.kind == JoinPointKind.MEASURE:
            return
        if self.allowed_gates is not None and jp.name not in self.allowed_gates \
                and jp.name != "barrier":
            self._flag(wc, f"gate '{jp.name}' not in allow-list at op {jp.op_index}")
        illegal = set(jp.qubits) & self.restricted_qubits
        if illegal:
            self._flag(wc, f"gate '{jp.name}' touches restricted qubit(s) "
                           f"{sorted(illegal)} at op {jp.op_index}")


# --------------------------------------------------------------------------- #
# 4. Compliance / audit (circuit-scope AROUND, tamper-evident)
# --------------------------------------------------------------------------- #
class ComplianceAuditAspect(Aspect):
    """Emits a hash-chained provenance record for every woven circuit.

    Each record links to the previous via SHA-256(prev_hash || payload), giving
    a tamper-evident audit trail of *what ran, with which aspects, when* --
    supporting reproducibility and regulatory governance of quantum workloads.
    """

    name = "ComplianceAudit"
    phase = Phase.STRUCTURAL
    advice_type = AdviceType.AROUND

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 applied_aspects: Optional[list[str]] = None,
                 chain: Optional[list[dict]] = None):
        super().__init__(pointcut or any_join_point())
        self.applied_aspects = applied_aspects or []
        self.chain = chain if chain is not None else []

    @staticmethod
    def _hash(prev: str, payload: dict) -> str:
        blob = prev + json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def around_structural(self, out: QuantumCircuit, original: QuantumCircuit,
                          wc: WeaveContext) -> QuantumCircuit:
        prev = self.chain[-1]["hash"] if self.chain else "GENESIS"
        payload = {
            "circuit": original.name,
            "circuit_hash": hashlib.sha256(
                canonical_ops(original).encode()).hexdigest(),
            "woven_depth": out.depth(),
            "aspects": sorted(self.applied_aspects),
            "seq": len(self.chain),
        }
        entry = {"payload": payload, "prev": prev,
                 "hash": self._hash(prev, payload)}
        self.chain.append(entry)
        wc.audit.append(entry)
        wc.emit(f"[{self.name}] audit#{payload['seq']} hash={entry['hash'][:12]}...")
        return out

    @staticmethod
    def verify(chain: list[dict]) -> tuple[bool, int]:
        """Recompute the chain; return (ok, first_broken_index or -1)."""
        prev = "GENESIS"
        for i, entry in enumerate(chain):
            expect = ComplianceAuditAspect._hash(prev, entry["payload"])
            if expect != entry["hash"] or entry["prev"] != prev:
                return False, i
            prev = entry["hash"]
        return True, -1
