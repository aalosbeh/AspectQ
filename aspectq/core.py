"""
AspectQ core: a Quantum Join-Point Model (QJPM), a pointcut language, an advice
model, and a two-phase weaver for aspect-oriented quantum programming.

The design mirrors classical AOP (AspectJ-style join points, pointcuts, advice,
weaving) and adapts it to two properties of quantum software:

  1. A quantum program has a *structural* phase (a circuit / DAG that is
     compiled and transpiled) and a *runtime* phase (repeated shot-based
     execution whose measurements collapse the state).  AspectQ therefore
     supports STRUCTURAL advice (woven over the circuit at compile time) and
     RUNTIME advice (woven around execution as classical pre/post-processing).

  2. Reading a qubit disturbs its state, so observability and verification
     concerns cannot simply "print a variable" in the middle of a run.  AspectQ
     models them as aspects that are woven either structurally (analysis,
     directives, policy) or at runtime (mitigation, metering, auditing of
     aggregate statistics).

Weaving order follows AspectJ precedence rules: aspects earlier in the
``Weaver`` list have higher precedence; *before* advice runs in precedence
order, *after* advice in reverse precedence order, and *around* advice of
higher precedence encloses *around* advice of lower precedence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Store

from .qutils import asap_layers, is_two_qubit_gate


# --------------------------------------------------------------------------- #
# Quantum Join-Point Model (QJPM)
# --------------------------------------------------------------------------- #
class JoinPointKind(Enum):
    """Kinds of well-defined points in a quantum program's structure/execution."""
    CIRCUIT_ENTRY = "circuit_entry"     # start of the program (circuit scope)
    CIRCUIT_EXIT = "circuit_exit"       # end of the program (circuit scope)
    GATE = "gate"                       # any other operation (gate, reset, block, ...)
    TWO_QUBIT_GATE = "two_qubit_gate"   # a unitary gate on exactly two qubits
    LAYER = "layer"                     # a parallel layer (ASAP time-slice of the DAG)
    MEASURE = "measure"                 # a measurement operation
    RUN = "run"                         # one logical run (a Weaver.run call)


@dataclass(slots=True)
class JoinPoint:
    """A concrete point in the program exposed to aspects, plus its context."""
    kind: JoinPointKind
    name: str = ""                      # gate name ("cx", "ry"), circuit name, ...
    qubits: tuple[int, ...] = ()        # qubit indices (logical or physical)
    clbits: tuple[int, ...] = ()
    params: tuple[float, ...] = ()
    layer_index: int = -1
    op_index: int = -1                  # position of the op in the input circuit
    tags: frozenset[str] = frozenset()  # circuit tags (circuit.metadata["tags"])
    # extra context (circuit facts for entry/exit, shots/backend for RUN, ...)
    context: dict[str, Any] = field(default_factory=dict)
    operation: Any = None               # the Qiskit operation (operation join points)

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


_OP_KINDS = (JoinPointKind.GATE, JoinPointKind.TWO_QUBIT_GATE)


# --- pointcut factory (designator) functions ------------------------------- #
def any_join_point() -> Pointcut:
    return Pointcut(lambda jp: True, "any")


def kind_is(k: JoinPointKind) -> Pointcut:
    return Pointcut(lambda jp: jp.kind == k, f"kind=={k.value}")


def gate(name: str) -> Pointcut:
    return Pointcut(lambda jp: jp.kind in _OP_KINDS and jp.name == name,
                    f"gate({name})")


def two_qubit_gates() -> Pointcut:
    return Pointcut(lambda jp: jp.kind == JoinPointKind.TWO_QUBIT_GATE,
                    "twoQubitGates")


def on_qubits(qset: set[int]) -> Pointcut:
    qs = frozenset(qset)
    return Pointcut(lambda jp: bool(qs.intersection(jp.qubits)),
                    f"onQubits({sorted(qs)})")


def restricted_qubits(qset: set[int]) -> Pointcut:
    """Alias of :func:`on_qubits`, named for policy aspects."""
    return on_qubits(qset)


def layer() -> Pointcut:
    return Pointcut(lambda jp: jp.kind == JoinPointKind.LAYER, "layer")


def at_layer(i: int) -> Pointcut:
    return Pointcut(lambda jp: jp.layer_index == i, f"atLayer({i})")


def before_measure() -> Pointcut:
    """Selects measurement join points (use with *before* advice)."""
    return Pointcut(lambda jp: jp.kind == JoinPointKind.MEASURE, "measure")


def circuit_scope() -> Pointcut:
    return Pointcut(lambda jp: jp.kind in (JoinPointKind.CIRCUIT_ENTRY,
                                           JoinPointKind.CIRCUIT_EXIT),
                    "circuitScope")


def run_scope() -> Pointcut:
    return Pointcut(lambda jp: jp.kind == JoinPointKind.RUN, "runScope")


def tagged(tag: str) -> Pointcut:
    return Pointcut(lambda jp: tag in jp.tags, f"tagged({tag})")


def shots_at_least(n: int) -> Pointcut:
    """RUN join points whose shot count is at least ``n``."""
    return Pointcut(lambda jp: jp.kind == JoinPointKind.RUN
                    and jp.context.get("shots", 0) >= n, f"shotsAtLeast({n})")


def backend_matches(text: str) -> Pointcut:
    """RUN join points whose backend description contains ``text``."""
    return Pointcut(lambda jp: jp.kind == JoinPointKind.RUN
                    and text in jp.context.get("backend", ""), f"backend({text})")


# --------------------------------------------------------------------------- #
# Advice model
# --------------------------------------------------------------------------- #
class Phase(Enum):
    STRUCTURAL = "structural"   # woven over the circuit at compile time
    RUNTIME = "runtime"         # woven around execution


@dataclass
class WeaveContext:
    """Mutable state threaded through a weave so aspects can record results."""
    log: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    applied_aspects: list[str] = field(default_factory=list)

    def emit(self, msg: str) -> None:
        self.log.append(msg)


# --------------------------------------------------------------------------- #
# Aspect base class
# --------------------------------------------------------------------------- #
class Aspect:
    """Base class for a crosscutting concern.

    An aspect binds a *pointcut* (where) to *advice* (what) in a *phase*
    (structural = compile time, runtime = execution).  Concrete aspects
    override only the hooks they need; the weaver dispatches to whatever is
    provided (the default hooks do nothing, and the default around advice
    simply proceeds).
    """

    name: str = "Aspect"
    phase: Phase = Phase.STRUCTURAL

    def __init__(self, pointcut: Pointcut):
        self.pointcut = pointcut

    # -- structural advice ------------------------------------------------- #
    def on_structural_before(self, jp: JoinPoint, out: QuantumCircuit,
                             wc: WeaveContext) -> None:
        """Insert operations *before* the join point (``out`` is being built)."""

    def on_structural_after(self, jp: JoinPoint, out: QuantumCircuit,
                            wc: WeaveContext) -> None:
        """Insert operations *after* the join point."""

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
        """Wrap execution; must call ``proceed(circuit)`` to run the program."""
        return proceed(circuit)


def _overrides(aspect: Aspect, hook: str) -> bool:
    return getattr(type(aspect), hook) is not getattr(Aspect, hook)


# --------------------------------------------------------------------------- #
# The two-phase weaver
# --------------------------------------------------------------------------- #
class Weaver:
    """Composes aspects and weaves them into a program in two phases.

    The list order defines precedence (index 0 = highest precedence).
    """

    def __init__(self, aspects: Optional[list[Aspect]] = None):
        self.aspects: list[Aspect] = list(aspects or [])

    def register(self, aspect: Aspect) -> "Weaver":
        self.aspects.append(aspect)
        return self

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.aspects]

    # ----- structural weaving (compile time) ------------------------------ #
    def weave(self, circuit: QuantumCircuit,
              wc: Optional[WeaveContext] = None) -> tuple[QuantumCircuit, WeaveContext]:
        """Weave all structural aspects into ``circuit``.

        Operations are visited layer by layer (a stable sort of the ASAP layer
        keys), which is a topological order of the circuit DAG, so copying the
        operations in this order preserves the circuit's semantics.  The
        result keeps the input's registers, bits, global phase and metadata.
        """
        wc = wc or WeaveContext()
        wc.applied_aspects = self.names
        t0 = time.perf_counter()

        structural = [a for a in self.aspects if a.phase == Phase.STRUCTURAL]
        reverse = structural[::-1]
        # only aspects that override a hook take part in its dispatch; the
        # base hooks do nothing, so skipping them does not change the result
        analysis_aspects = [a for a in structural
                            if _overrides(a, "on_structural_analysis")]
        before_aspects = [a for a in structural if _overrides(a, "on_structural_before")]
        after_aspects = [a for a in reverse if _overrides(a, "on_structural_after")]
        around_aspects = [a for a in reverse if _overrides(a, "around_structural")]
        tags = frozenset((circuit.metadata or {}).get("tags", []))
        data = list(circuit.data)                 # materialise once
        if circuit.num_vars or any(isinstance(i.operation, Store) for i in data):
            raise NotImplementedError(
                "AspectQ weaves circuits whose dependencies are carried by "
                "qubit and clbit wires; classical variables and Store "
                "instructions are not supported")
        out = circuit.copy_empty_like()
        qidx = {b: i for i, b in enumerate(circuit.qubits)}
        cidx = {b: i for i, b in enumerate(circuit.clbits)}
        keys = asap_layers(circuit, data)
        order = sorted(range(len(keys)), key=lambda i: (keys[i], i))
        all_qubits = tuple(range(circuit.num_qubits))

        # circuit-level facts, computed once and exposed on CIRCUIT_ENTRY
        active: set[int] = set()
        depth = 0
        for instr, key in zip(data, keys):
            if not instr.is_directive():
                active.update(qidx[q] for q in instr.qubits)
                depth = max(depth, int(key) + 1)
        facts = {"depth": depth, "active_qubits": frozenset(active),
                 "size": sum(1 for i in data if not i.is_directive())}

        def dispatch(jp: JoinPoint, before: bool, after: bool, analysis: bool):
            if analysis:
                for a in analysis_aspects:
                    if a.pointcut.matches(jp):
                        a.on_structural_analysis(jp, circuit, wc)
            if before:
                for a in before_aspects:
                    if a.pointcut.matches(jp):
                        a.on_structural_before(jp, out, wc)
            if after:
                for a in after_aspects:
                    if a.pointcut.matches(jp):
                        a.on_structural_after(jp, out, wc)

        entry = JoinPoint(JoinPointKind.CIRCUIT_ENTRY, name=circuit.name,
                          qubits=all_qubits, tags=tags, context=facts)
        dispatch(entry, before=True, after=False, analysis=True)

        current: Optional[JoinPoint] = None
        for op_index in order:
            instr = data[op_index]
            if instr.is_directive():           # barriers are copied, not advised
                out._append(instr)
                continue
            lay = int(keys[op_index])
            if current is None or lay != current.layer_index:
                if current is not None:
                    dispatch(current, before=False, after=True, analysis=False)
                current = JoinPoint(JoinPointKind.LAYER, name=f"layer{lay}",
                                    qubits=all_qubits, layer_index=lay, tags=tags)
                dispatch(current, before=True, after=False, analysis=True)

            q = tuple(qidx[b] for b in instr.qubits)
            c = tuple(cidx[b] for b in instr.clbits)
            try:
                params = tuple(float(p) for p in instr.params)
            except (TypeError, ValueError):
                params = ()
            name = instr.name
            if name == "measure":
                kind = JoinPointKind.MEASURE
            elif is_two_qubit_gate(instr):
                kind = JoinPointKind.TWO_QUBIT_GATE
            else:
                kind = JoinPointKind.GATE
            jp = JoinPoint(kind, name, q, c, params, lay, op_index, tags,
                           operation=instr.operation)

            dispatch(jp, before=True, after=False, analysis=False)
            out._append(instr)
            dispatch(jp, before=False, after=True, analysis=True)

        if current is not None:
            dispatch(current, before=False, after=True, analysis=False)

        exit_jp = JoinPoint(JoinPointKind.CIRCUIT_EXIT, name=circuit.name,
                            qubits=all_qubits, tags=tags, context=facts)
        dispatch(exit_jp, before=False, after=True, analysis=True)

        # circuit-scope around advice: lowest precedence innermost
        for a in around_aspects:
            if a.pointcut.matches(exit_jp) or a.pointcut.matches(entry):
                out = a.around_structural(out, circuit, wc)

        # operations inserted by advice are not join points of this weave;
        # warn when analysis advice may therefore have missed them
        inserted = (sum(1 for i in out.data if not i.is_directive())
                    - facts["size"])
        wc.metrics["inserted_operations"] = inserted
        if inserted and analysis_aspects:
            wc.emit(f"[Weaver] WARNING: advice inserted {inserted} operation(s) "
                    "that analysis advice did not see; weave the analysis "
                    "aspects again at a later stage to cover them")

        wc.metrics["weave_time_ms"] = (time.perf_counter() - t0) * 1e3
        return out, wc

    # ----- runtime weaving (execution) ------------------------------------ #
    def run(self, circuit: QuantumCircuit, backend, shots: int = 4096,
            wc: Optional[WeaveContext] = None) -> dict:
        """Execute ``circuit`` on ``backend`` with all runtime aspects woven.

        The runtime aspects whose pointcuts match the RUN join point form a
        chain ordered by precedence.  Entering an aspect runs its before
        advice, then its around advice with a ``proceed`` that enters the next
        aspect, then its after advice, so advice nests as in AspectJ and
        inner aspects run once per ``proceed`` call.  The innermost
        ``proceed`` calls ``backend.run``.
        """
        wc = wc or WeaveContext()
        if not wc.applied_aspects:
            wc.applied_aspects = self.names
        t0 = time.perf_counter()
        backend_s = 0.0
        run_ctx: dict = {"shots": shots, "executions": 0}
        run_jp = JoinPoint(JoinPointKind.RUN, name=circuit.name,
                           qubits=tuple(range(circuit.num_qubits)),
                           tags=frozenset((circuit.metadata or {}).get("tags", [])),
                           context={"shots": shots, "backend": str(backend)})
        chain = [a for a in self.aspects
                 if a.phase == Phase.RUNTIME and a.pointcut.matches(run_jp)]

        def execute(circ: QuantumCircuit) -> dict:
            nonlocal backend_s
            tb = time.perf_counter()
            counts = backend.run(circ, shots=shots).result().get_counts()
            backend_s += time.perf_counter() - tb
            run_ctx["executions"] += 1
            return {"counts": counts, "shots": shots}

        def proceed_at(i: int) -> Callable[[QuantumCircuit], dict]:
            if i == len(chain):
                return execute
            aspect = chain[i]

            def advised(circ: QuantumCircuit) -> dict:
                aspect.before_run(circ, wc, run_ctx)
                res = aspect.around_run(circ, backend, shots, wc, proceed_at(i + 1))
                aspect.after_run(circ, wc, run_ctx, res)
                return res
            return advised

        result = proceed_at(0)(circuit)

        total = time.perf_counter() - t0
        wc.metrics["runtime"] = {
            "executions": run_ctx["executions"],
            "backend_ms": backend_s * 1e3,
            # time outside the backend: weaving plus the advice's own work
            "weaver_overhead_ms": (total - backend_s) * 1e3,
        }
        return result
