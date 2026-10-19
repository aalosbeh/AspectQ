"""
PROCEDURAL design: each crosscutting concern is factored into a helper
function, and the driver calls the helpers explicitly.  This is how a careful
developer would modularise the concerns without AOP, and it is the strongest
non-AOP baseline in the study.

The helper functions are attributed to their concern by name in
aspectq.metrics; call sites inside the driver carry ``# @C:<tag>`` markers.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import qiskit
from qiskit import QuantumCircuit

from ..qutils import (active_qubits, asap_layers, circuit_digest, extrapolate,
                      fold_global, is_directive, is_two_qubit_gate,
                      parity_expectation)
from . import CONCERN_NAMES, ConcernConfig, QuotaExceeded


def observe(circuit: QuantumCircuit, ctx: dict) -> None:
    """Structural metrics of ``circuit`` (observability concern)."""
    obs = ctx["observability"] = {"num_qubits": circuit.num_qubits,
           "active_qubits": len(active_qubits(circuit)),
           "num_layers": 0, "two_qubit_gates": 0, "measurements": 0,
           "gate_histogram": {}, "per_layer": {}, "probes_inserted": 0}
    layers = asap_layers(circuit)
    for op_index, instr in enumerate(circuit.data):
        if is_directive(instr.operation):
            continue
        name = instr.operation.name
        obs["gate_histogram"][name] = obs["gate_histogram"].get(name, 0) + 1
        if name == "measure":
            obs["measurements"] += 1
        elif is_two_qubit_gate(instr):
            obs["two_qubit_gates"] += 1
        lay = int(layers[op_index])
        obs["per_layer"][str(lay)] = obs["per_layer"].get(str(lay), 0) + 1
        obs["num_layers"] = max(obs["num_layers"], lay + 1)


def check_policy(circuit: QuantumCircuit, cfg: ConcernConfig, ctx: dict) -> None:
    """Security/resource policy violations of ``circuit`` (policy concern)."""
    violations = ctx.setdefault("violations", [])
    width = len(active_qubits(circuit))
    if cfg.max_qubits is not None and width > cfg.max_qubits:
        violations.append(f"width {width} > max {cfg.max_qubits}")
    depth = circuit.depth()
    if cfg.max_depth is not None and depth > cfg.max_depth:
        violations.append(f"depth {depth} > max {cfg.max_depth}")
    for op_index, instr in enumerate(circuit.data):
        if is_directive(instr.operation):
            continue
        name = instr.operation.name
        qidx = [circuit.find_bit(q).index for q in instr.qubits]
        if (cfg.allowed_gates is not None and name != "measure"
                and name not in cfg.allowed_gates):
            violations.append(f"gate '{name}' not in allow-list at op {op_index}")
        illegal = cfg.restricted_qubits.intersection(qidx)
        if illegal:
            violations.append(f"'{name}' touches restricted qubit(s) "
                              f"{sorted(illegal)} at op {op_index}")


def zne_estimate(circuit: QuantumCircuit, cfg: ConcernConfig,
                 execute: Callable[[QuantumCircuit], dict], ctx: dict) -> dict:
    """Zero-noise extrapolation by global folding (mitigation concern)."""
    expvals, first_counts = [], None
    for lam in cfg.scale_factors:
        executable = fold_global(circuit, lam)
        executable.measure_all()
        counts = execute(executable)
        first_counts = first_counts or counts
        expvals.append(parity_expectation(counts, list(cfg.obs_qubits)))
    mitigated, method = extrapolate(cfg.scale_factors, expvals, cfg.extrapolation)
    ctx["zne"] = {"expvals": expvals, "mitigated": mitigated, "method": method}
    return first_counts


def metered(execute: Callable, shots: int, max_shots, ctx: dict) -> Callable:
    """Wrap ``execute`` with shot metering and a budget (quota concern)."""
    quota = ctx.setdefault("quota", {"executions": 0, "shots": 0, "denied": 0})

    def wrapped(executable):
        if max_shots is not None and quota["shots"] + shots > max_shots:
            quota["denied"] += 1
            raise QuotaExceeded("budget exhausted")
        quota["executions"] += 1
        quota["shots"] += shots
        return execute(executable)
    return wrapped


def audit_record(circuit: QuantumCircuit, chain: list, concerns: list,
                 ctx: dict) -> None:
    """Append a hash-chained provenance record (audit concern)."""
    prev = chain[-1]["hash"] if chain else "GENESIS"
    payload = {"circuit": circuit.name,
               "circuit_hash": circuit_digest(circuit),
               "woven_hash": circuit_digest(circuit),
               "woven_depth": circuit.depth(),
               "aspects": sorted(concerns),
               "sdk": f"qiskit-{qiskit.__version__}",
               "seq": len(chain)}
    digest = hashlib.sha256(
        (prev + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
    entry = {"payload": payload, "prev": prev, "hash": digest}
    chain.append(entry)
    ctx["audit"] = entry


def run_program_procedural(circuit: QuantumCircuit, backend, shots: int,
                           cfg: ConcernConfig, ctx: dict) -> dict:
    """Estimate <Z...> on ``backend``; concerns invoked from call sites.

    Concern outputs are written to ``ctx``; the return value is the core result.
    """
    observe(circuit, ctx)                                               # @C:obs
    check_policy(circuit, cfg, ctx)                                     # @C:sec

    def execute(executable):
        return backend.run(executable, shots=shots).result().get_counts()

    execute = metered(execute, shots, cfg.max_shots, ctx)               # @C:quo
    counts = zne_estimate(circuit, cfg, execute, ctx)                   # @C:mit
    audit_record(circuit, cfg.audit_chain, CONCERN_NAMES, ctx)          # @C:aud
    return {"expectation": parity_expectation(counts, list(cfg.obs_qubits))}
