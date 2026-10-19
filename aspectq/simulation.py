"""
Simulation backends for the evaluation.

* :func:`depolarizing_backend` -- Aer shot simulator with uniform depolarizing
  gate noise (optionally with readout error).
* :class:`ExactBackend` -- returns *expected* counts (probability x shots)
  computed with Aer's density-matrix method, i.e. the exact output
  distribution of the noisy channel with no shot noise.  It is used to show
  that different designs compute exactly the same noisy result.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .qutils import split_final_measurements

ONE_Q = ["ry", "rx", "rz", "h", "x", "sx", "sxdg", "s", "sdg", "t", "tdg",
         "p", "u", "u1", "u2", "u3"]
TWO_Q = ["cx", "cz", "rzz", "cp", "swap", "ecr"]


def depolarizing_noise(p1: float, p2: float, readout: float = 0.0) -> NoiseModel:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ONE_Q)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_Q)
    if readout > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def depolarizing_backend(p1: float = 1e-3, p2: float = 1e-2, readout: float = 0.0,
                         seed: int | None = None) -> AerSimulator:
    be = AerSimulator(noise_model=depolarizing_noise(p1, p2, readout))
    if seed is not None:
        be.set_options(seed_simulator=seed)
    return be


class _Job:
    def __init__(self, counts):
        self._counts = counts

    def result(self):
        return self

    def get_counts(self):
        return self._counts


class ExactBackend:
    """Expected counts of a measured circuit under a (gate) noise model.

    Only final measurements are supported and every clbit must be written by
    one of them.  Readout errors are not modelled.
    """

    def __init__(self, noise_model: NoiseModel | None = None):
        self.noise_model = noise_model
        self._sim = AerSimulator(method="density_matrix", noise_model=noise_model)

    def __str__(self) -> str:
        return "ExactBackend(density_matrix)"

    def run(self, circuit: QuantumCircuit, shots: int = 1024):
        body, meas = split_final_measurements(circuit)
        if sorted(c for _, c in meas) != list(range(circuit.num_clbits)):
            raise ValueError("every clbit must be set by a final measurement")
        qubits = [q for q, _ in sorted(meas, key=lambda m: m[1])]
        body.save_probabilities_dict(qubits=qubits)
        probs = self._sim.run(body).result().data()["probabilities"]
        counts = {}
        for key, p in probs.items():
            index = int(key, 16) if isinstance(key, str) else int(key)
            counts[format(index, f"0{len(qubits)}b")] = p * shots
        return _Job(counts)
