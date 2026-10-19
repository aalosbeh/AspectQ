"""
TANGLED design: the five crosscutting concerns (error mitigation,
observability, security policy, compliance audit, resource quota) written
directly into one driver function, the way they typically appear in
research and production scripts.

Lines that implement a concern carry ``# @C:<tag>``; unmarked lines are the
core program (prepare, execute, estimate).  aspectq.metrics scans the markers.
"""
from __future__ import annotations

import hashlib
import json

import qiskit
from qiskit import QuantumCircuit

from ..qutils import (active_qubits, asap_layers, circuit_digest, extrapolate,
                      fold_global, is_directive, is_two_qubit_gate,
                      parity_expectation)
from . import CONCERN_NAMES, ConcernConfig, QuotaExceeded


def run_program_tangled(circuit: QuantumCircuit, backend, shots: int,
                        cfg: ConcernConfig, ctx: dict) -> dict:
    """Estimate <Z...> on ``backend`` with all concerns inlined.

    Concern outputs are written to ``ctx``; the return value is the core result.
    """
    violations = ctx.setdefault("violations", [])                       # @C:sec
    observability = ctx.setdefault("observability", {                   # @C:obs
        "num_qubits": circuit.num_qubits,                               # @C:obs
        "active_qubits": len(active_qubits(circuit)),                   # @C:obs
        "num_layers": 0, "two_qubit_gates": 0, "measurements": 0,       # @C:obs
        "gate_histogram": {}, "per_layer": {}, "probes_inserted": 0})   # @C:obs
    quota = ctx.setdefault("quota", {"executions": 0, "shots": 0,       # @C:quo
                                     "denied": 0})                      # @C:quo

    width = len(active_qubits(circuit))                                 # @C:sec
    if cfg.max_qubits is not None and width > cfg.max_qubits:           # @C:sec
        violations.append(f"width {width} > max {cfg.max_qubits}")      # @C:sec
    depth = circuit.depth()                                             # @C:sec
    if cfg.max_depth is not None and depth > cfg.max_depth:             # @C:sec
        violations.append(f"depth {depth} > max {cfg.max_depth}")       # @C:sec

    layers = asap_layers(circuit)                                       # @C:obs
    for op_index, instr in enumerate(circuit.data):                     # @C:obs+sec
        name = instr.operation.name                                     # @C:obs+sec
        if is_directive(instr.operation):                               # @C:obs+sec
            continue                                                    # @C:obs+sec
        qidx = [circuit.find_bit(q).index for q in instr.qubits]        # @C:obs+sec
        hist = observability["gate_histogram"]                          # @C:obs
        hist[name] = hist.get(name, 0) + 1                              # @C:obs
        if name == "measure":                                           # @C:obs
            observability["measurements"] += 1                          # @C:obs
        elif is_two_qubit_gate(instr):                                  # @C:obs
            observability["two_qubit_gates"] += 1                       # @C:obs
        lay = int(layers[op_index])                                     # @C:obs
        per_layer = observability["per_layer"]                          # @C:obs
        per_layer[str(lay)] = per_layer.get(str(lay), 0) + 1            # @C:obs
        observability["num_layers"] = max(observability["num_layers"],  # @C:obs
                                          lay + 1)                      # @C:obs
        if (cfg.allowed_gates is not None and name != "measure"         # @C:sec
                and name not in cfg.allowed_gates):                     # @C:sec
            violations.append(                                          # @C:sec
                f"gate '{name}' not in allow-list at op {op_index}")    # @C:sec
        illegal = cfg.restricted_qubits.intersection(qidx)              # @C:sec
        if illegal:                                                     # @C:sec
            violations.append(f"'{name}' touches restricted qubit(s) "  # @C:sec
                              f"{sorted(illegal)} at op {op_index}")    # @C:sec

    expvals, first_counts = [], None                                    # @C:mit
    for lam in cfg.scale_factors:                                       # @C:mit
        executable = fold_global(circuit, lam)                          # @C:mit
        executable.measure_all()
        if (cfg.max_shots is not None                                   # @C:quo
                and quota["shots"] + shots > cfg.max_shots):            # @C:quo
            quota["denied"] += 1                                        # @C:quo
            raise QuotaExceeded("budget exhausted")                     # @C:quo
        quota["executions"] += 1                                        # @C:quo
        quota["shots"] += shots                                         # @C:quo
        counts = backend.run(executable, shots=shots).result().get_counts()
        first_counts = first_counts or counts                           # @C:mit
        expvals.append(parity_expectation(counts, list(cfg.obs_qubits)))  # @C:mit
    mitigated, method = extrapolate(cfg.scale_factors, expvals,         # @C:mit
                                    cfg.extrapolation)                  # @C:mit
    ctx["zne"] = {"expvals": expvals, "mitigated": mitigated,           # @C:mit
                  "method": method}                                     # @C:mit

    prev = cfg.audit_chain[-1]["hash"] if cfg.audit_chain else "GENESIS"  # @C:aud
    payload = {"circuit": circuit.name,                                 # @C:aud
               "circuit_hash": circuit_digest(circuit),                 # @C:aud
               "woven_hash": circuit_digest(circuit),                   # @C:aud
               "woven_depth": circuit.depth(),                          # @C:aud
               "aspects": sorted(CONCERN_NAMES),                        # @C:aud
               "sdk": f"qiskit-{qiskit.__version__}",                   # @C:aud
               "seq": len(cfg.audit_chain)}                             # @C:aud
    digest = hashlib.sha256(                                            # @C:aud
        (prev + json.dumps(payload, sort_keys=True)).encode()).hexdigest()  # @C:aud
    entry = {"payload": payload, "prev": prev, "hash": digest}          # @C:aud
    cfg.audit_chain.append(entry)                                       # @C:aud
    ctx["audit"] = entry                                                # @C:aud

    return {"expectation": parity_expectation(first_counts, list(cfg.obs_qubits))}
