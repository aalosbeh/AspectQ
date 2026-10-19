"""
Execution adapters for the runtime weaver.

``Weaver.run`` executes circuits through ``backend.run(circuit, shots=...)``,
which Aer simulators and fake backends provide.  IBM hardware backends accept
jobs only through the Sampler and Estimator primitives; ``SamplerBackend``
wraps any SamplerV2 (``qiskit.primitives.BackendSamplerV2`` or
``qiskit_ibm_runtime.SamplerV2``) so that the same runtime aspects run on it.
"""
from __future__ import annotations

from qiskit import QuantumCircuit


class _Result:
    def __init__(self, counts: dict):
        self._counts = counts

    def result(self):
        return self

    def get_counts(self):
        return self._counts


class SamplerBackend:
    """Expose a SamplerV2 through the ``run(...).result().get_counts()`` API.

    Counts join all classical registers into one bit string, in the same
    global clbit order as ``BackendV2.run`` (register separators removed).
    """

    def __init__(self, sampler, name: str = ""):
        self.sampler = sampler
        self.name = name or type(sampler).__name__

    def __str__(self) -> str:
        return f"SamplerBackend({self.name})"

    def run(self, circuit: QuantumCircuit, shots: int = 1024):
        pub_result = self.sampler.run([circuit], shots=shots).result()[0]
        return _Result(pub_result.join_data().get_counts())
