#!/usr/bin/env python3
"""
design-governance parser

Reads a design.html (source of truth) + schema.json (variable tag/attribute
vocabulary for this project) and:
  1. Checks the complexity budget INVARIANT (soft 7/12, hard 9/15) — these
     numbers are hardcoded here on purpose. They are never read from
     schema.json and never overridable per layer.
  2. Generates README.md and MANUAL.md derived views, stamped with a
     generation timestamp and the design.html's git short hash (if the file
     is tracked in a git repo; otherwise "untracked").

Usage:
    python parser.py <path-to-design.html> [--schema <path-to-schema.json>]

If --schema is omitted, looks for schema.json next to design.html.

Recurses into subdirectories automatically: if design.html sits in
design/, and design/auth/design.html also exists, both get processed as
independent layers (each with its own module/behaviour count against the
same fixed invariant).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

# ---- fixed invariants: never move these into schema.json, never override ----
SOFT_MODULE_LIMIT = 7
HARD_MODULE_LIMIT = 9
SOFT_BEHAVIOUR_LIMIT = 12
HARD_BEHAVIOUR_LIMIT = 15

DEFAULT_SCHEMA = {
    "module_tag": "module",
    "behaviour_tag": "behaviour",
    "behaviour_scope_attr": "data-modules",
}


class HardLimitExceeded(Exception):
    pass


def load_schema(schema_path: Path) -> dict:
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            user_schema = json.load(f)
        schema = {**DEFAULT_SCHEMA, **user_schema}
    else:
        schema = dict(DEFAULT_SCHEMA)
    return schema


def git_short_hash(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(path)],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        h = out.stdout.strip()
        return h if h else "untracked"
    except Exception:
        return "untracked"


def parse_layer(html_path: Path, schema: dict) -> dict:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    module_tag = schema["module_tag"]
    behaviour_tag = schema["behaviour_tag"]
    scope_attr = schema["behaviour_scope_attr"]

    modules = []
    for el in soup.find_all(attrs={"data-role": module_tag}):
        modules.append({
            "name": el.get("data-name", "(unnamed)"),
            "body": el.get_text(strip=True),
        })

    behaviours = []
    for el in soup.find_all(attrs={"data-role": behaviour_tag}):
        scope_raw = el.get(scope_attr, "")
        scoped_modules = [m.strip() for m in scope_raw.split(",") if m.strip()]
        behaviours.append({
            "name": el.get("data-name", "(unnamed)"),
            "modules": scoped_modules,
            "body": el.get_text(strip=True),
        })

    return {"modules": modules, "behaviours": behaviours}


def check_budget(layer: dict, html_path: Path) -> list:
    warnings = []
    n_mod = len(layer["modules"])
    n_beh = len(layer["behaviours"])

    if n_mod > HARD_MODULE_LIMIT:
        raise HardLimitExceeded(
            f"{html_path}: {n_mod} modules > hard limit {HARD_MODULE_LIMIT}. "
            f"Split this layer — sink a cluster of tightly-related modules "
            f"into a sub-module. Do not raise the limit."
        )
    if n_beh > HARD_BEHAVIOUR_LIMIT:
        raise HardLimitExceeded(
            f"{html_path}: {n_beh} behaviours > hard limit {HARD_BEHAVIOUR_LIMIT}. "
            f"Split this layer — sink the modules a cluster of behaviours "
            f"spans into a sub-layer, and move those behaviours with them. "
            f"Do not raise the limit."
        )
    if n_mod > SOFT_MODULE_LIMIT:
        warnings.append(
            f"WARNING {html_path}: {n_mod} modules > soft limit {SOFT_MODULE_LIMIT}. "
            f"Consider splitting this layer soon."
        )
    if n_beh > SOFT_BEHAVIOUR_LIMIT:
        warnings.append(
            f"WARNING {html_path}: {n_beh} behaviours > soft limit {SOFT_BEHAVIOUR_LIMIT}. "
            f"Consider splitting this layer soon."
        )
    return warnings


def render_readme(layer: dict, html_path: Path, schema: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    ghash = git_short_hash(html_path)
    lines = [
        f"<!-- generated: {ts} from {html_path.name}@{ghash} -->",
        "",
        f"# {html_path.parent.name or 'design'} — module map",
        "",
        "## Modules",
        "",
    ]
    for m in layer["modules"]:
        lines.append(f"- **{m['name']}** — {m['body'] or '(no description)'}")
    lines += ["", "## Behaviours (cross-module paths)", ""]
    for b in layer["behaviours"]:
        path = " → ".join(b["modules"])
        lines.append(f"- **{b['name']}**: {path}")
        if b["body"]:
            lines.append(f"  {b['body']}")
    lines += [
        "",
        "## Budget",
        "",
        f"- modules: {len(layer['modules'])} "
        f"(soft {SOFT_MODULE_LIMIT} / hard {HARD_MODULE_LIMIT})",
        f"- behaviours: {len(layer['behaviours'])} "
        f"(soft {SOFT_BEHAVIOUR_LIMIT} / hard {HARD_BEHAVIOUR_LIMIT})",
        "",
    ]
    return "\n".join(lines)


def render_manual(layer: dict, html_path: Path) -> str | None:
    if not layer["behaviours"]:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    ghash = git_short_hash(html_path)
    lines = [
        f"<!-- generated: {ts} from {html_path.name}@{ghash} -->",
        "",
        f"# {html_path.parent.name or 'design'} — behaviour manual",
        "",
    ]
    for b in layer["behaviours"]:
        lines.append(f"## {b['name']}")
        lines.append(f"Spans: {', '.join(b['modules'])}")
        if b["body"]:
            lines.append("")
            lines.append(b["body"])
        lines.append("")
    return "\n".join(lines)


def process(html_path: Path, schema_arg: Path | None):
    schema_path = schema_arg or (html_path.parent / "schema.json")
    schema = load_schema(schema_path)

    layer = parse_layer(html_path, schema)
    warnings = check_budget(layer, html_path)
    for w in warnings:
        print(w, file=sys.stderr)

    readme = render_readme(layer, html_path, schema)
    (html_path.parent / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {html_path.parent / 'README.md'}")

    manual = render_manual(layer, html_path)
    if manual:
        (html_path.parent / "MANUAL.md").write_text(manual, encoding="utf-8")
        print(f"wrote {html_path.parent / 'MANUAL.md'}")

    # recurse into sub-layers
    for child_dir in sorted(p for p in html_path.parent.iterdir() if p.is_dir()):
        child_html = child_dir / "design.html"
        if child_html.exists():
            process(child_html, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_html", type=Path)
    ap.add_argument("--schema", type=Path, default=None)
    args = ap.parse_args()

    try:
        process(args.design_html, args.schema)
    except HardLimitExceeded as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
