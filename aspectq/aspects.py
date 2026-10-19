"""
AspectQ reusable aspects: crosscutting concerns of quantum software, each
modularised as a single, self-contained Aspect class.

  * ErrorMitigationAspect  -- runtime AROUND advice: zero-noise extrapolation
  * ObservabilityAspect    -- structural analysis (+ optional LAYER probes)
  * SecurityPolicyAspect   -- structural analysis: compile-time policy checks
  * ComplianceAuditAspect  -- circuit-scope AROUND advice: tamper-evident provenance
  * ResourceQuotaAspect    -- runtime AROUND advice: execution/shot metering
  * ReadoutMitigationAspect -- runtime AROUND advice: tensored readout correction

Every concern below would otherwise be scattered across a driver program and
tangled with its core logic (see aspectq.baseline).  Here each one lives in
exactly one class and is (un)pluggable without touching the base program.
Domain helpers (folding, estimators, hashing) live in aspectq.qutils and are
shared with the baselines.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Callable, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit

from .core import (Aspect, JoinPoint, JoinPointKind, Phase, Pointcut,
                   WeaveContext, Weaver, any_join_point, circuit_scope, layer,
                   run_scope)
from .qutils import (circuit_digest, extrapolate,
                     extrapolation_stderr, fold_global, parity_expectation,
                     split_final_measurements)


# --------------------------------------------------------------------------- #
# 1. Error mitigation (runtime, AROUND)
# --------------------------------------------------------------------------- #
class ErrorMitigationAspect(Aspect):
    """Zero-noise extrapolation woven *around* execution.

    The base program is unaware of mitigation.  The aspect intercepts the RUN
    join point, builds noise-scaled copies by global unitary folding, executes
    each through ``proceed`` (so inner aspects still apply), estimates the
    observable at every scale and extrapolates to the zero-noise limit.
    ``obs_qubits`` name the qubits of the Z-string; their values are read
    from the clbits that the program's own final measurements write.
    ``fold_transform`` optionally re-translates each folded circuit (e.g. to a
    device basis with optimisation disabled, so U^dagger U is not cancelled).
    With ``fold_pointcut`` the aspect folds only the operations that the
    pointcut selects (local folding, G -> G (G^dagger G)^n, woven by a
    GateFoldingAspect) instead of the whole circuit.
    """

    name = "ErrorMitigation(ZNE)"
    phase = Phase.RUNTIME

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 scale_factors: tuple[int, ...] = (1, 3, 5),
                 obs_qubits: Optional[list[int]] = None,
                 extrapolation: str = "exponential",
                 fold_transform: Optional[Callable[[QuantumCircuit],
                                                   QuantumCircuit]] = None,
                 fold_pointcut: Optional[Pointcut] = None):
        super().__init__(pointcut or run_scope())
        for lam in scale_factors:
            if lam < 1 or lam % 2 != 1:
                raise ValueError(f"scale factors must be odd >= 1, got {lam}")
        self.scale_factors = tuple(scale_factors)
        self.obs_qubits = list(obs_qubits or [0])
        self.extrapolation = extrapolation
        self.fold_transform = fold_transform
        self.fold_pointcut = fold_pointcut

    def _fold(self, body: QuantumCircuit, lam: int) -> QuantumCircuit:
        if self.fold_pointcut is None:
            return fold_global(body, lam)
        folded, _ = Weaver([GateFoldingAspect(self.fold_pointcut, lam)]).weave(body)
        return folded

    def around_run(self, circuit, backend, shots, wc: WeaveContext,
                   proceed: Callable) -> dict:
        # fold the unitary body and re-apply the program's own final
        # measurements, so counts keep the format the base program expects
        body, meas = split_final_measurements(circuit)
        if not meas:
            raise ValueError("ErrorMitigationAspect needs final measurements")
        if any(i.operation.name in ("measure", "reset") or i.is_control_flow()
               for i in body.data):
            raise ValueError("global folding needs a unitary body; mid-circuit "
                             "measurements, resets and control flow are not supported")
        clbit_of = dict(meas)
        obs_bits = [clbit_of[q] for q in self.obs_qubits]
        expvals, first_counts = [], None
        for lam in self.scale_factors:
            folded = self._fold(body, lam)
            for q, c in meas:
                folded.measure(q, c)
            if self.fold_transform is not None:
                folded = self.fold_transform(folded)
            out = proceed(folded)
            first_counts = first_counts or out["counts"]
            expvals.append(parity_expectation(out["counts"], obs_bits))
        mitigated, used = extrapolate(self.scale_factors, expvals,
                                      self.extrapolation)
        stderr = extrapolation_stderr(self.scale_factors, expvals, shots, used
                                      if used != "richardson-fallback"
                                      else "richardson")
        wc.metrics["zne"] = {
            "scale_factors": list(self.scale_factors),
            "expvals": [float(x) for x in expvals],
            "unmitigated": float(expvals[0]),
            "mitigated": mitigated,
            "mitigated_stderr": stderr,
            "method": used,
            "executions": len(self.scale_factors),
            "folding": "global" if self.fold_pointcut is None
                       else f"local[{self.fold_pointcut.desc}]",
        }
        wc.emit(f"[{self.name}] unmitigated={expvals[0]:+.4f} "
                f"mitigated={mitigated:+.4f} +/- {stderr:.4f} ({used}, "
                f"{len(self.scale_factors)} executions)")
        # counts of the unscaled run keep the base program's result contract
        return {"counts": first_counts, "shots": shots,
                "expectation_unmitigated": float(expvals[0]),
                "expectation_mitigated": mitigated,
                "scaled_expvals": [float(x) for x in expvals]}


class GateFoldingAspect(Aspect):
    """Local unitary folding of the operations a pointcut selects.

    After every selected operation G the aspect appends (G^dagger G)^n, so the
    selected operations are executed 2n + 1 times while the circuit's unitary
    is unchanged (G (G^dagger G)^n = G).  ErrorMitigationAspect uses it for
    local folding; with a pointcut such as ``two_qubit_gates()`` it amplifies
    only the noise of the selected gates.
    """

    name = "GateFolding"
    phase = Phase.STRUCTURAL

    def __init__(self, pointcut: Pointcut, scale_factor: int):
        super().__init__(pointcut)
        if scale_factor < 1 or scale_factor % 2 != 1:
            raise ValueError(f"scale factor must be odd >= 1, got {scale_factor}")
        self.repeats = (scale_factor - 1) // 2

    def on_structural_after(self, jp: JoinPoint, out: QuantumCircuit,
                            wc: WeaveContext):
        if jp.operation is None or not self.repeats:
            return
        qargs = [out.qubits[i] for i in jp.qubits]
        inverse = jp.operation.inverse()
        for _ in range(self.repeats):
            out.append(inverse, qargs)
            out.append(jp.operation, qargs)


# --------------------------------------------------------------------------- #
# 2. Observability (structural analysis + optional probes)
# --------------------------------------------------------------------------- #
class ObservabilityAspect(Aspect):
    """Collects structural metrics; optionally inserts barrier probes.

    By default the aspect is analysis-only (nothing is added to the circuit):
    it records active width, depth, two-qubit-gate count, a gate histogram and
    a per-layer operation count.  With ``insert_probes=True`` it weaves a
    full-width barrier before every join point selected by ``probe_pointcut``
    (default: every LAYER), giving the transpiler and debuggers a stable
    checkpoint between layers.
    """

    name = "Observability"
    phase = Phase.STRUCTURAL

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 insert_probes: bool = False,
                 probe_pointcut: Optional[Pointcut] = None):
        super().__init__(pointcut or any_join_point())
        self.insert_probes = insert_probes
        self.probe_pointcut = probe_pointcut or layer()

    @staticmethod
    def _state(wc: WeaveContext) -> dict:
        return wc.metrics.setdefault("observability", {
            "num_qubits": 0, "active_qubits": 0, "num_layers": 0,
            "two_qubit_gates": 0, "measurements": 0, "gate_histogram": {},
            "per_layer": {}, "probes_inserted": 0})

    def on_structural_analysis(self, jp: JoinPoint, original, wc: WeaveContext):
        obs = self._state(wc)
        if jp.kind == JoinPointKind.CIRCUIT_ENTRY:
            obs["num_qubits"] = original.num_qubits
            obs["active_qubits"] = len(jp.context["active_qubits"])
            return
        if jp.kind in (JoinPointKind.CIRCUIT_EXIT, JoinPointKind.LAYER):
            return
        obs["gate_histogram"][jp.name] = obs["gate_histogram"].get(jp.name, 0) + 1
        if jp.kind == JoinPointKind.TWO_QUBIT_GATE:
            obs["two_qubit_gates"] += 1
        if jp.kind == JoinPointKind.MEASURE:
            obs["measurements"] += 1
        key = str(jp.layer_index)
        obs["per_layer"][key] = obs["per_layer"].get(key, 0) + 1
        obs["num_layers"] = max(obs["num_layers"], jp.layer_index + 1)

    def on_structural_before(self, jp: JoinPoint, out: QuantumCircuit,
                             wc: WeaveContext):
        if self.insert_probes and self.probe_pointcut.matches(jp):
            out.barrier()
            self._state(wc)["probes_inserted"] += 1


# --------------------------------------------------------------------------- #
# 3. Security policy (structural, compile-time analysis)
# --------------------------------------------------------------------------- #
class PolicyViolation(Exception):
    pass


class SecurityPolicyAspect(Aspect):
    """Enforces a compile-time security/resource policy over the circuit.

    Checks: (a) only allow-listed gates (measurements are always allowed),
    (b) no operation of any kind -- including measurement and reset -- on
    restricted qubits (e.g. reserved, calibration or faulty physical qubits),
    (c) caps on active width and depth.  Violations are recorded in the weave
    context; with ``strict=True`` the first violation raises PolicyViolation.
    Woven after layout (see aspectq.passes), the qubit indices are physical.
    """

    name = "SecurityPolicy"
    phase = Phase.STRUCTURAL

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 allowed_gates: Optional[set[str]] = None,
                 restricted_qubits: Optional[set[int]] = None,
                 max_qubits: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 strict: bool = False):
        super().__init__(pointcut or any_join_point())
        self.allowed_gates = set(allowed_gates) if allowed_gates is not None else None
        self.restricted_qubits = frozenset(restricted_qubits or ())
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
            width = len(jp.context["active_qubits"])
            if self.max_qubits is not None and width > self.max_qubits:
                self._flag(wc, f"width {width} > max {self.max_qubits}")
            depth = jp.context["depth"]
            if self.max_depth is not None and depth > self.max_depth:
                self._flag(wc, f"depth {depth} > max {self.max_depth}")
            return
        if jp.kind not in (JoinPointKind.GATE, JoinPointKind.TWO_QUBIT_GATE,
                           JoinPointKind.MEASURE):
            return
        if (self.allowed_gates is not None and jp.kind != JoinPointKind.MEASURE
                and jp.name not in self.allowed_gates):
            self._flag(wc, f"gate '{jp.name}' not in allow-list at op {jp.op_index}")
        illegal = self.restricted_qubits.intersection(jp.qubits)
        if illegal:
            self._flag(wc, f"'{jp.name}' touches restricted qubit(s) "
                           f"{sorted(illegal)} at op {jp.op_index}")


# --------------------------------------------------------------------------- #
# 4. Compliance / audit (circuit-scope AROUND, tamper-evident)
# --------------------------------------------------------------------------- #
class ComplianceAuditAspect(Aspect):
    """Emits a hash-chained provenance record for every woven circuit.

    Record i stores SHA-256(prev_hash || payload_i), where the payload names
    the circuit, digests of the input and the woven circuit, the woven depth,
    the aspects that were applied, the SDK version and a sequence number.  The
    chain detects in-place modification, deletion, insertion and reordering.
    Truncation of the tail is detected only against an externally kept head
    (``anchor``); a full rewrite of the chain is detected only when records
    are also authenticated with a secret key (``hmac_key``).
    """

    name = "ComplianceAudit"
    phase = Phase.STRUCTURAL

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 applied_aspects: Optional[list[str]] = None,
                 chain: Optional[list[dict]] = None,
                 hmac_key: Optional[bytes] = None):
        super().__init__(pointcut or circuit_scope())
        self.applied_aspects = applied_aspects
        self.chain = chain if chain is not None else []
        self.hmac_key = hmac_key

    @staticmethod
    def _hash(prev: str, payload: dict) -> str:
        blob = prev + json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _mac(key: bytes, digest: str) -> str:
        return hmac.new(key, digest.encode(), hashlib.sha256).hexdigest()

    def head(self) -> str:
        """Hash of the latest record (to be anchored outside the log)."""
        return self.chain[-1]["hash"] if self.chain else "GENESIS"

    def around_structural(self, out: QuantumCircuit, original: QuantumCircuit,
                          wc: WeaveContext) -> QuantumCircuit:
        prev = self.head()
        aspects = self.applied_aspects if self.applied_aspects is not None \
            else wc.applied_aspects
        payload = {
            "circuit": original.name,
            "circuit_hash": circuit_digest(original),
            "woven_hash": circuit_digest(out),
            "woven_depth": out.depth(),
            "aspects": sorted(aspects),
            "sdk": f"qiskit-{qiskit.__version__}",
            "seq": len(self.chain),
        }
        entry = {"payload": payload, "prev": prev,
                 "hash": self._hash(prev, payload)}
        if self.hmac_key is not None:
            entry["mac"] = self._mac(self.hmac_key, entry["hash"])
        self.chain.append(entry)
        wc.audit.append(entry)
        wc.emit(f"[{self.name}] audit#{payload['seq']} hash={entry['hash'][:12]}")
        return out

    @staticmethod
    def verify(chain: list[dict], anchor: Optional[str] = None,
               hmac_key: Optional[bytes] = None) -> tuple[bool, int]:
        """Recompute the chain; return (ok, first_broken_index or -1).

        ``anchor`` is a trusted copy of the head hash; a mismatch is reported
        at index ``len(chain)``.
        """
        prev = "GENESIS"
        for i, entry in enumerate(chain):
            if entry.get("prev") != prev or entry.get("payload", {}).get("seq") != i:
                return False, i
            expect = ComplianceAuditAspect._hash(prev, entry["payload"])
            if expect != entry["hash"]:
                return False, i
            if hmac_key is not None and not hmac.compare_digest(
                    entry.get("mac", ""),
                    ComplianceAuditAspect._mac(hmac_key, entry["hash"])):
                return False, i
            prev = entry["hash"]
        if anchor is not None and prev != anchor:
            return False, len(chain)
        return True, -1


# --------------------------------------------------------------------------- #
# 5. Resource quota / metering (runtime, AROUND)
# --------------------------------------------------------------------------- #
class QuotaExceeded(Exception):
    pass


class ResourceQuotaAspect(Aspect):
    """Meters circuit executions and shots and enforces an optional budget.

    The aspect counts what passes through its own ``proceed``.  Its position
    in the around chain therefore matters: placed *inside* the mitigation
    aspect (lower precedence) it sees every folded execution that reaches the
    backend; placed *outside* it sees one logical run.
    """

    name = "ResourceQuota"
    phase = Phase.RUNTIME

    def __init__(self, pointcut: Optional[Pointcut] = None,
                 max_shots: Optional[int] = None,
                 max_executions: Optional[int] = None):
        super().__init__(pointcut or run_scope())
        self.max_shots = max_shots
        self.max_executions = max_executions

    def around_run(self, circuit, backend, shots, wc: WeaveContext,
                   proceed: Callable) -> dict:
        q = wc.metrics.setdefault("quota", {"executions": 0, "shots": 0,
                                            "denied": 0})
        if ((self.max_shots is not None and q["shots"] + shots > self.max_shots)
                or (self.max_executions is not None
                    and q["executions"] + 1 > self.max_executions)):
            q["denied"] += 1
            wc.emit(f"[{self.name}] DENIED: budget exhausted")
            raise QuotaExceeded(f"budget exhausted after {q['executions']} "
                                f"executions / {q['shots']} shots")
        q["executions"] += 1
        q["shots"] += shots
        return proceed(circuit)


# --------------------------------------------------------------------------- #
# 6. Readout-error mitigation (runtime, AROUND)
# --------------------------------------------------------------------------- #
class ReadoutMitigationAspect(Aspect):
    """Tensored readout-error mitigation woven around every execution.

    For the qubits that a circuit measures, the aspect runs two calibration
    circuits (all qubits prepared in |0>, and all in |1>) through its own
    ``proceed`` and estimates, per measured bit k, the assignment matrix
    A_k = [[P(0|0), P(0|1)], [P(1|0), P(1|1)]].  It then applies the inverse of
    the tensor product of the A_k to the counts of every execution that passes
    through it (quasi-counts may be negative).  Calibrations are cached per
    backend and measurement map.  Global folding does not amplify readout
    error, so this aspect belongs *inside* ErrorMitigationAspect (lower
    precedence): each noise-scaled estimate is corrected before extrapolation.
    """

    name = "ReadoutMitigation"
    phase = Phase.RUNTIME

    def __init__(self, pointcut: Optional[Pointcut] = None):
        super().__init__(pointcut or run_scope())
        self._cache: dict = {}

    def _calibrate(self, circuit, meas, proceed) -> list:
        mats = []
        runs = []
        for prepare_one in (False, True):
            cal = circuit.copy_empty_like()
            for q, c in meas:
                if prepare_one:
                    cal.x(q)
                cal.measure(q, c)
            runs.append(proceed(cal)["counts"])
        for _, c in meas:
            p00 = _bit_fraction(runs[0], c, 0)
            p11 = _bit_fraction(runs[1], c, 1)
            mats.append(np.array([[p00, 1 - p11], [1 - p00, p11]]))
        return mats

    def around_run(self, circuit, backend, shots, wc: WeaveContext,
                   proceed: Callable) -> dict:
        _, meas = split_final_measurements(circuit)
        meas = sorted(meas, key=lambda m: m[1])
        if [c for _, c in meas] != list(range(circuit.num_clbits)):
            raise ValueError("ReadoutMitigationAspect needs every clbit to be "
                             "set by a final measurement")
        key = (str(backend), tuple(meas), shots)
        state = wc.metrics.setdefault("readout", {"calibration_executions": 0,
                                                  "corrected_runs": 0})
        if key not in self._cache:
            self._cache[key] = self._calibrate(circuit, meas, proceed)
            state["calibration_executions"] += 2
        mats = self._cache[key]
        state["assignment_fidelity"] = [float((m[0, 0] + m[1, 1]) / 2)
                                        for m in mats]
        out = proceed(circuit)
        corrected = dict(out)
        corrected["counts"] = _apply_inverse(out["counts"], mats)
        corrected["raw_counts"] = out["counts"]
        state["corrected_runs"] += 1
        return corrected


def _bit_fraction(counts: dict, bit: int, value: int) -> float:
    total = sum(counts.values())
    hit = sum(n for k, n in counts.items()
              if int(k.replace(" ", "")[-1 - bit]) == value)
    return hit / total


def _apply_inverse(counts: dict, mats: list) -> dict:
    """Apply (A_{m-1} x ... x A_0)^{-1} to a counts dictionary."""
    m = len(mats)
    total = sum(counts.values())
    arr = np.zeros((2,) * m)
    for key, n in counts.items():
        bits = key.replace(" ", "")
        arr[tuple(int(bits[-1 - k]) for k in range(m))] += n / total
    for k, a in enumerate(mats):
        arr = np.moveaxis(np.tensordot(np.linalg.inv(a), arr, axes=([1], [k])), 0, k)
    out = {}
    for idx in np.ndindex(*arr.shape):
        key = "".join(str(idx[k]) for k in range(m - 1, -1, -1))
        out[key] = float(arr[idx] * total)
    return out
