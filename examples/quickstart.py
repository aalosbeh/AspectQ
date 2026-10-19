"""
AspectQ quick start.

Weaves five concerns into a hardware-efficient ansatz, compiles it for the IBM
Torino calibration snapshot with a policy and a user-defined aspect attached
after optimisation, and runs it with zero-noise extrapolation composed with
readout mitigation and a shot quota.

    python examples/quickstart.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import ClassicalRegister, transpile  # noqa: E402
from qiskit.quantum_info import Pauli, Statevector  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeTorino  # noqa: E402

from aspectq import (Aspect, ComplianceAuditAspect, ErrorMitigationAspect,  # noqa: E402
                     ObservabilityAspect, Phase, ReadoutMitigationAspect,
                     ResourceQuotaAspect, SecurityPolicyAspect, Weaver,
                     WeaveContext, attach, two_qubit_gates)
from aspectq.benchmarks import hardware_efficient_ansatz  # noqa: E402


class TwoQubitBudget(Aspect):
    """Listing 3 of the paper: a two-qubit-gate budget in nine lines."""
    name, phase = "TwoQubitBudget", Phase.STRUCTURAL

    def __init__(self, limit):
        super().__init__(two_qubit_gates())          # pointcut
        self.limit = limit

    def on_structural_analysis(self, jp, original, wc):
        n = wc.metrics["2q"] = wc.metrics.get("2q", 0) + 1
        if n > self.limit:
            wc.violations.append(f"2q budget: op {jp.op_index}")


def main():
    circuit = hardware_efficient_ansatz(4, 3, seed=7)
    ideal = Statevector(circuit).expectation_value(Pauli("IIIZ")).real
    device = FakeTorino()
    ro_err = [device.target["measure"][(q,)].error for q in range(device.num_qubits)]
    bad_qubits = {q for q, e in enumerate(ro_err) if e > 0.05}

    # structural phase, logical level
    logical = Weaver([ObservabilityAspect(), ComplianceAuditAspect()])
    woven, wc = logical.weave(circuit)
    obs = wc.metrics["observability"]
    print(f"[logical]  {obs['active_qubits']} qubits, {obs['num_layers']} layers, "
          f"{obs['two_qubit_gates']} two-qubit gates, audit "
          f"{wc.audit[-1]['hash'][:16]}...")

    # structural phase, physical level: aspects attached after optimisation
    for level in (1, 2):
        pm = generate_preset_pass_manager(optimization_level=level, backend=device,
                                          seed_transpiler=7)
        attach(pm, Weaver([SecurityPolicyAspect(restricted_qubits=bad_qubits),
                           TwoQubitBudget(limit=12)]), stage="post_optimization")
        physical = pm.run(woven)
        pwc = pm.property_set["aspectq_context"]
        q0 = physical.layout.final_index_layout()[0]
        print(f"[level {level}]  logical q0 -> physical q{q0} (readout error "
              f"{ro_err[q0]:.3f}); 2q gates {pwc.metrics['2q']}; "
              f"{len(pwc.violations)} policy violation(s)")
        for v in pwc.violations[:3]:
            print(f"           - {v}")

    # runtime phase on the level-2 circuit
    physical.add_register(ClassicalRegister(1, "m"))
    physical.measure(q0, physical.clbits[-1])
    sim = AerSimulator.from_backend(device)
    sim.set_options(seed_simulator=7)
    translate = lambda c: transpile(c, basis_gates=list(device.operation_names),
                                    optimization_level=0)
    for label, aspects in (
            ("ZNE", [ErrorMitigationAspect(obs_qubits=[q0], fold_transform=translate)]),
            ("ZNE > readout > quota",
             [ErrorMitigationAspect(obs_qubits=[q0], fold_transform=translate),
              ReadoutMitigationAspect(), ResourceQuotaAspect(max_shots=200_000)])):
        rwc = WeaveContext()
        Weaver(aspects).run(physical, sim, shots=16384, wc=rwc)
        z = rwc.metrics["zne"]
        extra = ""
        if "quota" in rwc.metrics:
            q = rwc.metrics["quota"]
            extra = f"; quota metered {q['executions']} executions / {q['shots']} shots"
        print(f"[run] {label:<22} unmitigated {z['unmitigated']:+.4f}  mitigated "
              f"{z['mitigated']:+.4f} +/- {z['mitigated_stderr']:.4f}  (ideal {ideal:+.4f}){extra}")


if __name__ == "__main__":
    main()
