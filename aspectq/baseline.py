"""
TANGLED baseline: the same four crosscutting concerns (error mitigation,
observability, security policy, compliance audit) implemented *without* AspectQ,
hand-inlined into a monolithic driver.

Lines that implement a crosscutting concern are annotated with a marker comment
``#@C:<tag>`` where tag in {mit, obs, sec, aud}.  aspectq.metrics scans these
markers to quantify scattering and tangling versus the aspect-oriented version.
This module is fully functional and produces the same results as the woven
program; it exists to make the modularity comparison empirical rather than
rhetorical.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from qiskit import QuantumCircuit

from .aspects import parity_expectation, fold_global, canonical_ops


def run_program_tangled(circuit: QuantumCircuit, backend, shots: int,
                        allowed_gates, restricted_qubits, max_depth,
                        audit_chain, scale_factors=(1, 3, 5), obs_qubits=(0,)):
    """One monolith doing core work + 4 concerns, all interleaved."""
    log = []                                                          # @C:obs
    violations = []                                                   # @C:sec
    observability = {"gate_histogram": {}, "two_qubit_gates": 0,      # @C:obs
                     "per_layer": {}, "num_layers": 0}                # @C:obs

    # ---- security: circuit-level policy checks (scattered fragment 1) ----
    if max_depth is not None and circuit.depth() > max_depth:         # @C:sec
        violations.append(f"depth {circuit.depth()} > max {max_depth}")  # @C:sec

    # ---- walk the circuit once, tangling obs + sec into the core loop ----
    qubit_time = [0] * circuit.num_qubits                            # @C:obs
    for op_index, instr in enumerate(circuit.data):
        name = instr.operation.name
        qidx = tuple(circuit.find_bit(q).index for q in instr.qubits)
        # observability bookkeeping (scattered fragment 2)
        layer = max((qubit_time[q] for q in qidx), default=0)        # @C:obs
        for q in qidx:                                               # @C:obs
            qubit_time[q] = layer + 1                                # @C:obs
        observability["gate_histogram"][name] = \
            observability["gate_histogram"].get(name, 0) + 1         # @C:obs
        if len(qidx) == 2:                                           # @C:obs
            observability["two_qubit_gates"] += 1                    # @C:obs
        observability["per_layer"][str(layer)] = \
            observability["per_layer"].get(str(layer), 0) + 1        # @C:obs
        observability["num_layers"] = max(
            observability["num_layers"], layer + 1)                  # @C:obs
        # security per-gate checks (scattered fragment 3)
        if name not in ("measure", "barrier") and \
                allowed_gates is not None and name not in allowed_gates:  # @C:sec
            violations.append(f"gate '{name}' not allowed at {op_index}")  # @C:sec
        if set(qidx) & set(restricted_qubits):                       # @C:sec
            violations.append(f"gate '{name}' hits restricted qubit")  # @C:sec

    log.append(f"observed {observability['num_layers']} layers")     # @C:obs

    # ---- error mitigation: ZNE inlined into the run path (fragment 4) ----
    base = circuit.remove_final_measurements(inplace=False)          # @C:mit
    expvals = []                                                     # @C:mit
    for lam in scale_factors:                                        # @C:mit
        folded = fold_global(base, lam)                              # @C:mit
        folded.measure_all()                                         # @C:mit
        counts = backend.run(folded, shots=shots).result().get_counts()  # @C:mit
        expvals.append(parity_expectation(counts, list(obs_qubits)))     # @C:mit
    _lam = np.array(scale_factors, float)                           # @C:mit
    _e = np.array(expvals, float)                                   # @C:mit
    if np.all(_e > 0) or np.all(_e < 0):                            # @C:mit
        _s = np.sign(_e[0])                                         # @C:mit
        _k, _b0 = np.polyfit(_lam, np.log(np.abs(_e)), 1)          # @C:mit
        mitigated = float(_s * np.exp(_b0))                        # @C:mit
    else:                                                          # @C:mit
        mitigated = float(np.polyval(np.polyfit(_lam, _e, 2), 0.0))  # @C:mit
    log.append(f"zne mitigated={mitigated:.4f}")                     # @C:obs+mit

    # ---- compliance audit: hash-chain record (scattered fragment 5) ----
    prev = audit_chain[-1]["hash"] if audit_chain else "GENESIS"     # @C:aud
    payload = {"circuit": circuit.name,                              # @C:aud
               "circuit_hash": hashlib.sha256(                        # @C:aud
                   canonical_ops(circuit).encode()).hexdigest(),      # @C:aud
               "seq": len(audit_chain)}                               # @C:aud
    h = hashlib.sha256(                                              # @C:aud
        (prev + json.dumps(payload, sort_keys=True)).encode()).hexdigest()  # @C:aud
    audit_chain.append({"payload": payload, "prev": prev, "hash": h})  # @C:aud

    return {"expectation_unmitigated": float(expvals[0]),           # @C:mit
            "expectation_mitigated": mitigated,                      # @C:mit
            "violations": violations, "observability": observability,
            "log": log, "audit_chain": audit_chain}
