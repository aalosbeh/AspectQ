"""
Benchmark quantum programs used in the evaluation.

All builders return unitary circuits without measurements; drivers add the
measurements they need.  Angles are drawn from a seeded generator so that every
run of the evaluation sees the same circuits.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full


def hardware_efficient_ansatz(n_qubits: int, n_layers: int, seed: int = 0,
                              name: str | None = None) -> QuantumCircuit:
    """R_y rotation layers interleaved with a linear CNOT chain.

    This is the hardware-efficient ansatz family of Kandala et al. (2017),
    restricted to R_y rotations so that <Z> observables are non-trivial.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=name or f"hea_{n_qubits}q_{n_layers}L")
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(rng.uniform(0, np.pi)), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def qaoa_maxcut_ring(n_qubits: int, p: int = 2, seed: int = 0) -> QuantumCircuit:
    """QAOA for MaxCut on an n-node ring with fixed (seeded) angles."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=f"qaoa_ring{n_qubits}_p{p}")
    qc.h(range(n_qubits))
    for _ in range(p):
        gamma, beta = rng.uniform(0, np.pi, size=2)
        for q in range(n_qubits):
            qc.rzz(2 * gamma, q, (q + 1) % n_qubits)
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    return qc


def ghz(n_qubits: int) -> QuantumCircuit:
    """GHZ state preparation with a CNOT chain."""
    qc = QuantumCircuit(n_qubits, name=f"ghz{n_qubits}")
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def qft_roundtrip(n_qubits: int, value: int = 5) -> QuantumCircuit:
    """Prepare |value>, apply the QFT and then the inverse QFT.

    The ideal output is |value>, so <Z_q> = (-1)^bit_q; the circuit is a
    standard mirror benchmark with a non-trivial gate count.
    """
    qc = QuantumCircuit(n_qubits, name=f"qft{n_qubits}_roundtrip")
    for q in range(n_qubits):
        if (value >> q) & 1:
            qc.x(q)
    qft = synth_qft_full(n_qubits)
    qc.compose(qft, inplace=True)
    qc.compose(qft.inverse(), inplace=True)
    return qc.decompose()


def grover(n_qubits: int = 3, marked: int = 5) -> QuantumCircuit:
    """One-solution Grover search with the optimal number of iterations."""
    qc = QuantumCircuit(n_qubits, name=f"grover{n_qubits}")
    qc.h(range(n_qubits))
    iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** n_qubits)))
    for _ in range(iterations):
        _phase_flip(qc, marked)
        qc.h(range(n_qubits))
        _phase_flip(qc, 0)
        qc.h(range(n_qubits))
    return qc.decompose()


def _phase_flip(qc: QuantumCircuit, basis_state: int) -> None:
    """Flip the phase of one computational basis state."""
    n = qc.num_qubits
    zeros = [q for q in range(n) if not (basis_state >> q) & 1]
    if zeros:
        qc.x(zeros)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    if zeros:
        qc.x(zeros)


def workload_suite(seed: int = 0) -> list[tuple[QuantumCircuit, tuple[int, ...]]]:
    """The five programs of the study, each with the Z-string it estimates."""
    return [(hardware_efficient_ansatz(4, 3, seed=seed, name="vqe_hea4"), (0,)),
            (qaoa_maxcut_ring(4, p=2, seed=seed), (0, 1)),
            (ghz(5), (0, 1)),
            (qft_roundtrip(4), (0,)),
            (grover(3), (0,))]
