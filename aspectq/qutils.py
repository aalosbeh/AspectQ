"""
Shared quantum utilities used by the aspects *and* by the non-AOP baselines.

Keeping these domain functions in a neutral module makes the modularity
comparison fair: every design reuses the same folding, estimation and
extrapolation code, so the metrics only measure how each design *places* its
concern code, not how much library code it re-implements.
"""
from __future__ import annotations

import hashlib

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


def is_directive(op) -> bool:
    """True for compiler directives such as barriers."""
    return bool(getattr(op, "_directive", False))


def is_two_qubit_gate(instr) -> bool:
    """True for a unitary gate acting on exactly two qubits.

    Directives, measurements, resets and control-flow blocks are excluded even
    when they span two qubits.
    """
    return len(instr.qubits) == 2 and isinstance(instr.operation, Gate)


def asap_layers(circuit: QuantumCircuit, data=None) -> list[float]:
    """Return an ASAP layer key for every instruction of ``circuit``.

    Operations occupy integer layers over all their wires (qubits and clbits),
    which reproduces ``QuantumCircuit.depth()``.  Directives (barriers) occupy
    no layer: they synchronise their wires and receive the key ``t - 0.5`` so
    that they sit on the boundary in front of layer ``t``.  Sorting the
    instructions by (key, original index) is therefore a topological order of
    the circuit DAG.  ``data`` may pass an already materialised
    ``list(circuit.data)``.
    """
    wire_time: dict = {}
    keys: list[float] = []
    for instr in (circuit.data if data is None else data):
        wires = instr.qubits + instr.clbits
        t = max((wire_time.get(w, 0) for w in wires), default=0)
        if instr.is_directive():
            for w in wires:
                wire_time[w] = t
            keys.append(t - 0.5)
        else:
            for w in wires:
                wire_time[w] = t + 1
            keys.append(float(t))
    return keys


def layered_order(circuit: QuantumCircuit, data=None) -> list[int]:
    """Instruction indices sorted by (ASAP key, original index)."""
    keys = asap_layers(circuit, data)
    return sorted(range(len(keys)), key=lambda i: (keys[i], i))


def parity_expectation(counts: dict[str, int], obs_qubits: list[int]) -> float:
    """Estimate <Z_{q0} Z_{q1} ...> from measurement counts.

    Qiskit count keys are little-endian bitstrings (qubit 0 is the rightmost
    character), so qubit q corresponds to ``bitstring[-1 - q]``.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    acc = 0
    for bitstring, n in counts.items():
        bs = bitstring.replace(" ", "")
        ones = sum(int(bs[-1 - q]) for q in obs_qubits)
        acc += (-1) ** ones * n
    return acc / total


def parity_stderr(expval: float, shots: int) -> float:
    """Shot-noise standard error of a +/-1 parity estimate."""
    return float(np.sqrt(max(1.0 - expval ** 2, 0.0) / shots))


def split_final_measurements(circuit: QuantumCircuit
                             ) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
    """Separate a circuit into its unitary part and its final measurements.

    Returns ``(body, meas)`` where ``body`` keeps every register and bit of
    ``circuit`` but drops final measurements and the barriers that precede
    them, and ``meas`` lists the dropped measurements as (qubit, clbit)
    indices.  A measurement is final if no later non-directive operation
    touches its qubit or its clbit.
    """
    data = list(circuit.data)
    closed: set = set()          # wires with a later non-final operation
    final = [False] * len(data)
    for i in range(len(data) - 1, -1, -1):
        instr = data[i]
        wires = set(instr.qubits) | set(instr.clbits)
        name = instr.operation.name
        if name in ("measure", "barrier") and not (wires & closed):
            final[i] = True
        else:
            closed |= wires
    body = circuit.copy_empty_like()
    meas = []
    for instr, is_final in zip(data, final):
        if not is_final:
            body.append(instr.operation, instr.qubits, instr.clbits, copy=False)
        elif instr.operation.name == "measure":
            meas.append((circuit.find_bit(instr.qubits[0]).index,
                         circuit.find_bit(instr.clbits[0]).index))
    return body, meas


def fold_global(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Global unitary folding U -> U (U^dagger U)^n, i.e. noise scale 2n + 1.

    In the noiseless limit the folded circuit implements the same unitary as
    ``circuit`` because U (U^dagger U)^n = U.  ``circuit`` must be unitary
    (remove final measurements first).
    """
    if scale_factor < 1 or scale_factor % 2 != 1:
        raise ValueError("global folding needs an odd scale factor >= 1, "
                         f"got {scale_factor}")
    n = (scale_factor - 1) // 2
    folded = circuit.copy()
    if n:
        inv = circuit.inverse()
        for _ in range(n):
            folded.compose(inv, inplace=True)
            folded.compose(circuit, inplace=True)
    return folded


def richardson_coefficients(scales) -> np.ndarray:
    """Coefficients g_i with E(0) ~ sum_i g_i E(lambda_i) (Lagrange at 0)."""
    lam = np.asarray(scales, dtype=float)
    g = np.ones_like(lam)
    for i in range(len(lam)):
        for j in range(len(lam)):
            if i != j:
                g[i] *= lam[j] / (lam[j] - lam[i])
    return g


def extrapolate(scales, values, method: str = "exponential") -> tuple[float, str]:
    """Extrapolate expectation values to the zero-noise limit.

    Returns ``(value, method_used)``.  ``exponential`` fits
    log|E(lambda)| linearly (E(lambda) = s * exp(a + b lambda), asymptote 0),
    weighting point i by sqrt|E_i| to offset the error propagation of the
    logarithm (the convention of Mitiq's ExpFactory), and falls back to
    Richardson extrapolation when the values change sign or vanish.
    """
    lam = np.asarray(scales, dtype=float)
    e = np.asarray(values, dtype=float)
    if method == "linear":
        b, a = np.polyfit(lam, e, 1)
        return float(a), "linear"
    if method == "richardson":
        return float(richardson_coefficients(lam) @ e), "richardson"
    if method == "exponential":
        if np.all(e > 0) or np.all(e < 0):
            s = np.sign(e[0])
            b, a = np.polyfit(lam, np.log(np.abs(e)), 1, w=np.sqrt(np.abs(e)))
            return float(s * np.exp(a)), "exponential"
        return float(richardson_coefficients(lam) @ e), "richardson-fallback"
    raise ValueError(f"unknown extrapolation method {method!r}")


def extrapolation_stderr(scales, values, shots: int, method: str,
                         n_boot: int = 200, seed: int = 0) -> float:
    """Parametric-bootstrap standard error of the extrapolated value.

    Each scaled estimate is resampled from N(E_i, sigma_i^2) with the shot-noise
    sigma_i of a parity measurement, and the extrapolation is repeated.
    """
    rng = np.random.default_rng(seed)
    e = np.asarray(values, dtype=float)
    sig = np.array([parity_stderr(v, shots) for v in e])
    draws = []
    for _ in range(n_boot):
        sample = np.clip(e + rng.normal(0.0, sig), -1.0, 1.0)
        draws.append(extrapolate(scales, sample, method)[0])
    return float(np.std(draws, ddof=1))


def canonical_ops(circuit: QuantumCircuit) -> str:
    """Deterministic encoding of a circuit's operations (for hashing).

    Operations are listed by ASAP layer and, within a layer, by their encoded
    text, so every linearisation of the same circuit DAG receives the same
    encoding.
    """
    data = list(circuit.data)
    keys = asap_layers(circuit, data)
    qidx = {b: i for i, b in enumerate(circuit.qubits)}
    cidx = {b: i for i, b in enumerate(circuit.clbits)}
    rows = []
    for key, instr in zip(keys, data):
        try:
            p = [round(float(x), 10) for x in instr.params]
        except (TypeError, ValueError):
            p = []
        rows.append((key, f"{instr.name}:{[qidx[b] for b in instr.qubits]}:"
                          f"{[cidx[b] for b in instr.clbits]}:{p}"))
    # within a layer, operations act on disjoint wires; sorting by the
    # encoded row makes the encoding independent of their order
    return ";".join(row for _, row in sorted(rows))


def circuit_digest(circuit: QuantumCircuit) -> str:
    """SHA-256 digest of :func:`canonical_ops`."""
    return hashlib.sha256(canonical_ops(circuit).encode()).hexdigest()


def active_qubits(circuit: QuantumCircuit) -> set[int]:
    """Indices of qubits touched by at least one non-directive operation."""
    qidx = {b: i for i, b in enumerate(circuit.qubits)}
    used: set[int] = set()
    for instr in circuit.data:
        if not instr.is_directive():
            used.update(qidx[q] for q in instr.qubits)
    return used
