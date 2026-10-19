"""
Separation-of-concerns metrics for the modularity study (RQ1).

The metrics follow Sant'Anna et al. (SBES 2003) and are computed directly from
source code.  A *code line* is a non-blank physical line that is not a comment,
a docstring or an import.  A code line belongs to a concern if it carries a
``# @C:<tag>`` marker or lies inside a unit (function or class) that is
dedicated to the concern (``units`` below).

Per concern and design we report

  * CDC   -- Concern Diffusion over Components: source modules with concern code
  * CDO   -- Concern Diffusion over Operations: classes, functions and methods
             with concern code (class-body lines count for their class)
  * CDLOC -- Concern Diffusion over LOC: transition points, i.e. the number of
             places where consecutive code lines switch into or out of the
             concern (Sant'Anna et al.'s definition)
  * FRAG  -- concern fragments: maximal runs of consecutive concern lines
  * LOCC  -- lines of concern code
  * CL    -- change locality: concern fragments inside the *driver* (the base
             program), i.e. the edit sites needed to remove or add the concern

and the degree of tangling of the driver (share of its code lines that belong
to some concern).
"""
from __future__ import annotations

import ast
import io
import os
import re
import tokenize
from collections import defaultdict

CONCERNS = {"mit": "ErrorMitigation", "obs": "Observability",
            "sec": "SecurityPolicy", "aud": "ComplianceAudit",
            "quo": "ResourceQuota"}

_MARKER = re.compile(r"#\s*@C:([a-z+]+)")

PKG = os.path.dirname(os.path.abspath(__file__))

DESIGNS = {
    "tangled": {
        "files": ["designs/tangled.py"],
        "driver": ("designs/tangled.py", "run_program_tangled"),
        "units": {},
    },
    "procedural": {
        "files": ["designs/procedural.py"],
        "driver": ("designs/procedural.py", "run_program_procedural"),
        "units": {"observe": "obs", "check_policy": "sec",
                  "zne_estimate": "mit", "metered": "quo",
                  "audit_record": "aud"},
    },
    "aspectq": {
        "files": ["aspects.py", "designs/woven.py"],
        "driver": ("designs/woven.py", "run_program_woven"),
        "units": {"ErrorMitigationAspect": "mit", "ObservabilityAspect": "obs",
                  "SecurityPolicyAspect": "sec", "PolicyViolation": "sec",
                  "ComplianceAuditAspect": "aud", "ResourceQuotaAspect": "quo",
                  "QuotaExceeded": "quo"},
    },
}


# --------------------------------------------------------------------------- #
# line classification
# --------------------------------------------------------------------------- #
def _excluded_lines(tree: ast.AST) -> set[int]:
    """Docstring and import lines."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(range(node.lineno, node.end_lineno + 1))
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.update(range(body[0].lineno, body[0].end_lineno + 1))
    return out


def _code_lines(src: str, tree: ast.AST) -> list[int]:
    """Physical lines that contain at least one code token."""
    has_code: set[int] = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in skip:
            has_code.update(range(tok.start[0], tok.end[0] + 1))
    return sorted(has_code - _excluded_lines(tree))


def _operations(tree: ast.AST) -> list[tuple[int, int, str, int]]:
    """(start, end, qualified name, nesting) for every def/class."""
    spans = []

    def visit(node, prefix, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                q = f"{prefix}{child.name}"
                spans.append((child.lineno, child.end_lineno, q, depth))
                visit(child, q + ".", depth + 1)
            else:
                visit(child, prefix, depth)
    visit(tree, "", 0)
    return spans


def _innermost(spans, line: int) -> str:
    best, depth = "<module>", -1
    for s, e, q, d in spans:
        if s <= line <= e and d > depth:
            best, depth = q, d
    return best


def scan_file(path: str, units: dict[str, str]) -> dict:
    """Map each code line of ``path`` to its concern tags."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines()
    code = _code_lines(src, tree)
    spans = _operations(tree)
    top_units = [(s, e, units[q]) for s, e, q, d in spans if d == 0 and q in units]
    tags: dict[int, set[str]] = {}
    for ln in code:
        t: set[str] = set()
        m = _MARKER.search(lines[ln - 1])
        if m:
            t.update(x for x in m.group(1).split("+") if x in CONCERNS)
        for s, e, tag in top_units:
            if s <= ln <= e:
                t.add(tag)
        tags[ln] = t
    return {"path": path, "code": code, "tags": tags, "spans": spans,
            "tree": tree, "lines": lines}


def lint_markers(scan: dict) -> list[str]:
    """Report multi-line statements that mix marked and unmarked lines.

    An unmarked continuation line splits one concern fragment into two and
    inflates FRAG/CDLOC; this check guards against that counting error.  Lines
    that only open or close a bracket (``Weaver([`` ... ``])``) are exempt so
    that one concern can be registered per line of a list display.
    """
    warnings = []
    code = set(scan["code"])
    for node in ast.walk(scan["tree"]):
        if not isinstance(node, ast.stmt) or isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        end = node.end_lineno
        if hasattr(node, "body") and node.body:     # compound: header only
            end = node.body[0].lineno - 1
        rows = [ln for ln in range(node.lineno, end + 1) if ln in code]
        if len(rows) < 2:
            continue
        marked = [ln for ln in rows if scan["tags"][ln]]
        unmarked = [ln for ln in rows if not scan["tags"][ln]
                    and not _bracket_only(scan["lines"][ln - 1])]
        if marked and unmarked:
            warnings.append(f"{os.path.basename(scan['path'])}:{node.lineno}"
                            f" unmarked line(s) {unmarked} in a marked statement")
    return warnings


def _bracket_only(text: str) -> bool:
    """True if a line just opens or closes a bracket (``x = f([`` / ``])``)."""
    code = text.split("#", 1)[0].strip()
    return code.endswith(("(", "[", "{")) or set(code) <= set(")]}, ")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _runs(code: list[int], member) -> list[list[int]]:
    runs, cur = [], []
    for ln in code:
        if member(ln):
            cur.append(ln)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def measure_design(name: str, root: str = PKG) -> dict:
    spec = DESIGNS[name]
    scans = {f: scan_file(os.path.join(root, f), spec["units"])
             for f in spec["files"]}
    drv_file, drv_func = spec["driver"]
    drv_scan = scans[drv_file]
    drv_span = next((s, e) for s, e, q, d in drv_scan["spans"] if q == drv_func)
    drv_code = [ln for ln in drv_scan["code"] if drv_span[0] <= ln <= drv_span[1]]

    per = {}
    for tag, cname in CONCERNS.items():
        files, ops, frag, cdloc, locc, cl = set(), set(), 0, 0, 0, 0
        for f, sc in scans.items():
            member = (lambda ln, sc=sc: tag in sc["tags"][ln])
            runs = _runs(sc["code"], member)
            if not runs:
                continue
            files.add(f)
            frag += len(runs)
            locc += sum(len(r) for r in runs)
            code = sc["code"]
            cdloc += sum(1 for a, b in zip(code, code[1:])
                         if member(a) != member(b))
            for r in runs:
                ops.update(f"{f}:{_innermost(sc['spans'], ln)}" for ln in r)
        drv_member = (lambda ln: tag in drv_scan["tags"][ln])
        cl = len(_runs(drv_code, drv_member))
        per[cname] = {"CDC": len(files), "CDO": len(ops), "CDLOC": cdloc,
                      "FRAG": frag, "LOCC": locc, "CL": cl}

    totals = {k: sum(v[k] for v in per.values())
              for k in ("CDC", "CDO", "CDLOC", "FRAG", "LOCC", "CL")}
    tangled_lines = sum(1 for ln in drv_code if drv_scan["tags"][ln])
    driver = {"code_lines": len(drv_code), "concern_lines": tangled_lines,
              "core_lines": len(drv_code) - tangled_lines,
              "tangling": tangled_lines / len(drv_code) if drv_code else 0.0,
              "concerns_in_driver": sum(1 for t in CONCERNS
                                        if any(t in drv_scan["tags"][ln]
                                               for ln in drv_code))}
    lint = [w for sc in scans.values() for w in lint_markers(sc)]
    return {"per_concern": per, "totals": totals, "driver": driver,
            "lint": lint}


def compare(root: str = PKG) -> dict:
    return {name: measure_design(name, root) for name in DESIGNS}
