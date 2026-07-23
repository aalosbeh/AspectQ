"""
Modularity metrics for the AspectQ evaluation (RQ1).

We adapt classic aspect-oriented separation-of-concerns metrics
(Sant'Anna et al., 2003) to compare the tangled baseline against the
aspect-oriented implementation of the same four concerns:

  * CDC  -- Concern Diffusion over Components  (files touched by a concern)
  * CDO  -- Concern Diffusion over Operations  (functions/classes touched)
  * CDLOC-- Concern Diffusion over LOC         (concern code fragments)
  * LOCC -- Lines Of Concern Code
  * CL   -- Change Locality (edit sites to add/remove the concern)

Numbers are derived directly from the source files (marker comments in the
baseline, AST class spans for the aspects), so the comparison is reproducible.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict

CONCERNS = {"mit": "ErrorMitigation", "obs": "Observability",
            "sec": "SecurityPolicy", "aud": "ComplianceAudit"}

ASPECT_CLASS = {"ErrorMitigationAspect": "mit", "ObservabilityAspect": "obs",
                "SecurityPolicyAspect": "sec", "ComplianceAuditAspect": "aud"}

_MARKER = re.compile(r"#\s*@C:([a-z+]+)")


def _enclosing_functions(source: str) -> list[tuple[int, int, str]]:
    tree = ast.parse(source)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    return spans


def scan_baseline(path: str) -> dict:
    src = open(path).read()
    lines = src.splitlines()
    funcs = _enclosing_functions(src)

    tagged: dict[str, list[int]] = defaultdict(list)   # concern -> line numbers
    for i, line in enumerate(lines, start=1):
        m = _MARKER.search(line)
        if not m:
            continue
        for tag in m.group(1).split("+"):
            if tag in CONCERNS:
                tagged[tag].append(i)

    result = {}
    for tag, lns in tagged.items():
        lns_sorted = sorted(lns)
        fragments = 1
        for a, b in zip(lns_sorted, lns_sorted[1:]):
            if b - a > 1:
                fragments += 1
        fns = set()
        for ln in lns_sorted:
            for s, e, name in funcs:
                if s <= ln <= e:
                    fns.add(name)
        result[tag] = {"concern": CONCERNS[tag], "files": 1,
                       "operations": len(fns) or 1, "fragments": fragments,
                       "locc": len(lns_sorted), "change_locality": fragments}
    return result


def scan_aspects(path: str) -> dict:
    src = open(path).read()
    tree = ast.parse(src)
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ASPECT_CLASS:
            tag = ASPECT_CLASS[node.name]
            locc = node.end_lineno - node.lineno + 1
            result[tag] = {"concern": CONCERNS[tag], "files": 1, "operations": 1,
                           "fragments": 1, "locc": locc, "change_locality": 1}
    return result


def compare(baseline_path: str, aspects_path: str) -> dict:
    base = scan_baseline(baseline_path)
    asp = scan_aspects(aspects_path)
    rows = {}
    for tag, cname in CONCERNS.items():
        b = base.get(tag, {"files": 0, "operations": 0, "fragments": 0,
                           "locc": 0, "change_locality": 0})
        a = asp.get(tag, {"files": 0, "operations": 0, "fragments": 0,
                          "locc": 0, "change_locality": 0})
        rows[cname] = {"baseline": b, "aspectq": a}
    totals = {}
    for side, data in (("baseline", base), ("aspectq", asp)):
        totals[side] = {
            "CDC": sum(v["files"] for v in data.values()),
            "CDO": sum(v["operations"] for v in data.values()),
            "CDLOC": sum(v["fragments"] for v in data.values()),
            "LOCC": sum(v["locc"] for v in data.values()),
            "CL": sum(v["change_locality"] for v in data.values()),
        }
    return {"per_concern": rows, "totals": totals}
