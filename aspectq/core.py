"""
AspectQ core: a Quantum Join-Point Model (QJPM), a pointcut language, an advice
model, and a two-phase weaver for aspect-oriented quantum programming.

The design deliberately mirrors classical AOP (AspectJ-style join points,
pointcuts, advice, weaving) but adapts it to two peculiarities of quantum
software:

  1. Quantum programs have a *structural* phase (a circuit / DAG that is
     compiled and transpiled) and a *runtime* phase (repeated shot-based
     execution whose measurements collapse the state). AspectQ therefore
     supports STRUCTURAL advice (woven at compile time over the circuit DAG)
     and RUNTIME advice (woven around execution as classical post-processing).

  2. Reading the quantum state destroys superposition, so observability and
     verification concerns cannot simply "print a variable" mid-execution.
     AspectQ models these as first-class crosscutting aspects that are woven
     either structurally (instrumentation / policy) or at runtime (mitigation /
     audit over aggregate statistics).

Author: (research prototype accompanying the AspectQ paper)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from qiskit import QuantumCircuit


# --------------------------------------------------------------------------- #
# Quantum Join-Point Model (QJPM)
# --------------------------------------------------------------------------- #
class JoinPointKind(Enum):
    """Kinds of well-defined points in a quantum program's structure/execution."""
    CIRCUIT_ENTRY = "circuit_entry"     # around the whole program
    CIRCUIT_EXIT = "circuit_exit"
    GATE = "gate"                       # application of a gate operation
    TWO_QUBIT_GATE = "two_qubit_gate"   # convenience specialization of GATE
    LAYER = "layer"                     # a parallel layer (DAG time-slice)
    MEASURE = "measure"                 # a measurement operation
    RUN = "run"                         # a runtime execution (shot batch)


@dataclass
class JoinPoint:
    """A concrete point in the program exposed to aspects, plus its context."""
    kind: JoinPointKind
    name: str = ""                      # e.g. gate name ("cx", "ry")
    qubits: tuple[int, ...] = ()        # physical/logical qubit indices
    clbits: tuple[int, ...] = ()
    params: tuple[float, ...] = ()
    layer_index: int = -1
    op_index: int = -1
    tags: frozenset[str] = frozenset()
    # runtime context (populated only for RUN join points)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)


# --------------------------------------------------------------------------- #
# Pointcut language: composable predicates over join points
# --------------------------------------------------------------------------- #
class Pointcut:
    """A predicate over join points with boolean combinators (& | ~)."""

    def __init__(self, pred: Callable[[JoinPoint], bool], desc: str = ""):
        self._pred = pred
        self.desc = desc or "pointcut"

    def matches(self, jp: JoinPoint) -> bool:
        return bool(self._pred(jp))

    def __and__(self, other: "Pointcut") -> "Pointcut":
        return Pointcut(lambda jp: self.matches(jp) and other.matches(jp),
                        f"({self.desc} && {other.desc})")

    def __or__(self, other: "Pointcut") -> "Pointcut":
        return Pointcut(lambda jp: self.matches(jp) or other.matches(jp),
                        f"({self.desc} || {other.desc})")

    def __invert__(self) -> "Pointcut":
        return Pointcut(lambda jp: not self.matches(jp), f"!({self.desc})")

    def __repr__(self) -> str:
        return f"Pointcut[{self.desc}]"


# --- pointcut factory (designator) functions ------------------------------- #
def any_join_point() -> Pointcut:
    return Pointcut(lambda jp: True, "any")


def kind_is(k: JoinPointKind) -> Pointcut:
    return Pointcut(lambda jp: jp.kind == k, f"kind=={k.value}")


def gate(name: str) -> Pointcut:
    return Pointcut(lambda jp: jp.kind in (JoinPointKind.GATE,
                                           JoinPointKind.TWO_QUBIT_GATE)
                    and jp.name == name, f"gate({name})")


def two_qubit_gates() -> Pointcut:
    return Pointcut(lambda jp: jp.num_qubits == 2
                    and jp.kind in (JoinPointKind.GATE,
                                    JoinPointKind.TWO_QUBIT_GATE),
                    "twoQubitGates")


def on_qubits(qset: set[int]) -> Pointcut:
    qs = set(qset)
    return Pointcut(lambda jp: bool(set(jp.qubits) & qs),
                    f"onQubits({sorted(qs)})")


def restricted_qubits(qset: set[int]) -> Pointcut:
    return on_qubits(qset)


def at_layer(i: int) -> Pointcut:
    return Pointcut(lambda jp: jp.layer_index == i, f"atLayer({i})")


def before_measure() -> Pointcut:
    return Pointcut(lambda jp: jp.kind == JoinPointKind.MEASURE, "beforeMeasure")


def circuit_scope() -> Pointcut:
    return Pointcut(lambda jp: jp.kind in (JoinPointKind.CIRCUIT_ENTRY,
                                           JoinPointKind.CIRCUIT_EXIT),
                    "circuitScope")


def run_scope() -> Pointcut:
    return Pointcut(lambda jp: jp.kind == JoinPointKind.RUN, "runScope")


def tagged(tag: str) -> Pointcut:
    return Pointcut(lambda jp: tag in jp.tags, f"tagged({tag})")


# --------------------------------------------------------------------------- #
# Advice model
# --------------------------------------------------------------------------- #
class AdviceType(Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"


class Phase(Enum):
    STRUCTURAL = "structural"   # woven over the circuit at compile time
    RUNTIME = "runtime"         # woven around execution (post-processing)


@dataclass
class WeaveContext:
    """Mutable state threaded through a weave so aspects can record results."""
    log: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, msg: str) -> None:
        self.log.append(msg)


# --------------------------------------------------------------------------- #
# Aspect base class
# --------------------------------------------------------------------------- #
class Aspect:
    """Base class for a crosscutting concern.

    An aspect binds a *pointcut* (where) to *advice* (what) at a given *phase*
    (structural = compile time, runtime = execution).  Concrete aspects
    override only the hooks they need; the weaver dispatches to whatever is
    provided.  Capability flags let the weaver know which runtime hooks to call
    without inspecting the MRO.
    """

    name: str = "Aspect"
    phase: Phase = Phase.STRUCTURAL
    advice_type: AdviceType = AdviceType.AFTER
    provides_around_run: bool = False

    def __init__(self, pointcut: Pointcut):
        self.pointcut = pointcut

    # -- structural advice ------------------------------------------------- #
    def on_structural_before(self, jp: JoinPoint, out: QuantumCircuit,
                             wc: WeaveContext) -> None:
        """Insert operations *before* the join-point op (out is being built)."""

    def on_structural_after(self, jp: JoinPoint, out: QuantumCircuit,
                            wc: WeaveContext) -> None:
        """Insert operations *after* the join-point op."""

    def on_structural_analysis(self, jp: JoinPoint, original: QuantumCircuit,
                               wc: WeaveContext) -> None:
        """Observe a join point without modifying the circuit."""

    def around_structural(self, out: QuantumCircuit, original: QuantumCircuit,
                          wc: WeaveContext) -> QuantumCircuit:
        """Wrap the whole woven circuit (circuit-scope around advice)."""
        return out

    # -- runtime advice ---------------------------------------------------- #
    def before_run(self, circuit: QuantumCircuit, wc: WeaveContext,
                   run_ctx: dict) -> None:
        pass

    def after_run(self, circuit: QuantumCircuit, wc: WeaveContext,
                  run_ctx: dict, result: dict) -> None:
        pass

    def around_run(self, circuit: QuantumCircuit, backend, shots: int,
                   wc: WeaveContext, proceed: Callable) -> dict:
        """Wrap execution; must call proceed(circuit) to run the base program."""
        return proceed(circuit)


# --------------------------------------------------------------------------- #
# The two-phase weaver
# --------------------------------------------------------------------------- #
class Weaver:
    """Composes aspects and weaves them into a program in two phases."""

    def __init__(self, aspects: Optional[list[Aspect]] = None):
        self.aspects: list[Aspect] = list(aspects or [])

    def register(self, aspect: Aspect) -> "Weaver":
        self.aspects.append(aspect)
        return self

    # ----- structural weaving (compile time) ------------------------------ #
    def weave(self, circuit: QuantumCircuit,
              wc: Optional[WeaveContext] = None) -> tuple[QuantumCircuit, WeaveContext]:
        wc = wc or WeaveContext()
        t0 = time.perf_counter()

        structural = [a for a in self.aspects if a.phase == Phase.STRUCTURAL]

        # circuit-entry join point (for before / analysis at circuit scope)
        entry = JoinPoint(JoinPointKind.CIRCUIT_ENTRY, name=circuit.name,
                          tags=frozenset((circuit.metadata or {}).get("tags", [])))
        for a in structural:
            if a.pointcut.matches(entry):
                a.on_structural_analysis(entry, circuit, wc)

        out = QuantumCircuit(circuit.num_qubits, circuit.num_clbits,
                             name=circuit.name)
        out.metadata = dict(circuit.metadata or {})

        # ASAP layering to give each op a layer index
        qubit_time = [0] * circuit.num_qubits
        tags = frozenset((circuit.metadata or {}).get("tags", []))

        for op_index, instr in enumerate(circuit.data):
            op = instr.operation
            qidx = tuple(circuit.find_bit(q).index for q in instr.qubits)
            cidx = tuple(circuit.find_bit(c).index for c in instr.clbits)
            try:
                params = tuple(float(p) for p in op.params)
            except (TypeError, ValueError):
                params = ()
            layer = max((qubit_time[q] for q in qidx), default=0)
            for q in qidx:
                qubit_time[q] = layer + 1

            if op.name == "measure":
                kind = JoinPointKind.MEASURE
            elif len(qidx) == 2:
                kind = JoinPointKind.TWO_QUBIT_GATE
            else:
                kind = JoinPointKind.GATE
            jp = JoinPoint(kind, op.name, qidx, cidx, params, layer, op_index, tags)

            for a in structural:
                if a.pointcut.matches(jp):
                    a.on_structural_before(jp, out, wc)
            out.append(op, [out.qubits[i] for i in qidx],
                       [out.clbits[i] for i in cidx])
            for a in structural:
                if a.pointcut.matches(jp):
                    a.on_structural_analysis(jp, circuit, wc)
                    a.on_structural_after(jp, out, wc)

        # circuit-exit + around (circuit scope) advice
        exit_jp = JoinPoint(JoinPointKind.CIRCUIT_EXIT, name=circuit.name, tags=tags)
        for a in structural:
            if a.pointcut.matches(exit_jp) or a.pointcut.matches(entry):
                out = a.around_structural(out, circuit, wc)

        wc.metrics["weave_time_ms"] = (time.perf_counter() - t0) * 1e3
        return out, wc

    # ----- runtime weaving (execution) ------------------------------------ #
    def run(self, circuit: QuantumCircuit, backend, shots: int = 4096,
            wc: Optional[WeaveContext] = None) -> dict:
        wc = wc or WeaveContext()
        run_ctx: dict = {"shots": shots}
        run_jp = JoinPoint(JoinPointKind.RUN, name=circuit.name)
        runtime = [a for a in self.aspects if a.phase == Phase.RUNTIME]

        for a in runtime:
            if a.pointcut.matches(run_jp):
                a.before_run(circuit, wc, run_ctx)

        def proceed(circ: QuantumCircuit) -> dict:
            job = backend.run(circ, shots=shots)
            counts = job.result().get_counts()
            return {"counts": counts, "shots": shots}

        result: dict = None
        for a in runtime:
            if a.provides_around_run and a.pointcut.matches(run_jp):
                result = a.around_run(circuit, backend, shots, wc, proceed)
                break
        if result is None:
            result = proceed(circuit)

        for a in runtime:
            if a.pointcut.matches(run_jp):
                a.after_run(circuit, wc, run_ctx, result)
        return result
