#!/usr/bin/env python3
"""
Summarize capability benchmark results into a table.

Usage:
    python -m ideogen.capabilities.summarize results/capabilities/

    # Or specific files
    python -m ideogen.capabilities.summarize results/capabilities/capabilities_openai_gpt-4o_*.json
"""

import argparse
import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Capability Benchmarks</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 2em; background: #f9f9f9; }}
  h1 {{ color: #333; }}
  table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 10px 18px; text-align: left; }}
  td {{ padding: 9px 18px; border-bottom: 1px solid #eee; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4f8; }}
  .skip {{ color: #aaa; }}
  .err {{ color: #e74c3c; }}
  p.meta {{ color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Capability Benchmarks</h1>
<p class="meta">Generated {timestamp}</p>
<table>
<tr>{headers}</tr>
{rows}
</table>
</body>
</html>"""


def load_results(paths: list[str]) -> list[dict]:
    """Load result JSONs from paths (files or directories)."""
    results = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.glob("capabilities_*.json")):
                results.append(json.loads(f.read_text()))
        elif path.suffix == ".json":
            results.append(json.loads(path.read_text()))
    return results


def format_score(benchmark_data: dict | None) -> str:
    """Format a benchmark score for display."""
    if benchmark_data is None:
        return "-"
    if benchmark_data.get("skipped"):
        return "skip"
    if benchmark_data.get("error"):
        return "ERR"
    acc = benchmark_data.get("accuracy")
    if acc is None:
        return "?"
    return f"{acc:.1%}"


def make_table(results: list[dict]) -> str:
    """Build a text table from results."""
    benchmarks = ["gsm8k", "mmlu", "humaneval"]
    headers = ["Model", "GSM8K", "MMLU", "HumanEval"]

    rows = []
    for r in results:
        model = r["model"]
        scores = [format_score(r["benchmarks"].get(b)) for b in benchmarks]
        if r.get("limit"):
            model += f" (n={r['limit']})"
        rows.append([model] + scores)

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build table
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)

    return "\n".join(lines)


def score_css_class(score: str) -> str:
    if score == "skip":
        return "skip"
    if score == "ERR":
        return "err"
    return ""


def make_html(results: list[dict], out_path: Path) -> None:
    from datetime import datetime

    benchmarks = ["gsm8k", "mmlu", "humaneval"]
    headers = ["Model", "GSM8K", "MMLU", "HumanEval"]

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for r in results:
        model = r["model"]
        if r.get("limit"):
            model += f" (n={r['limit']})"
        cells = f"<td>{model}</td>"
        for b in benchmarks:
            score = format_score(r["benchmarks"].get(b))
            css = score_css_class(score)
            cells += f'<td class="{css}">{score}</td>'
        rows_html += f"<tr>{cells}</tr>\n"

    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        headers=header_html,
        rows=rows_html,
    )
    out_path.write_text(html)
    print(f"Saved HTML to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize capability benchmark results")
    parser.add_argument("paths", nargs="+", help="Result JSON files or directories")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    parser.add_argument("--html", metavar="FILE", help="Save HTML report to this path")
    args = parser.parse_args()

    results = load_results(args.paths)
    if not results:
        print("No results found.")
        return

    if args.json:
        summary = []
        for r in results:
            entry = {"model": r["model"]}
            for bench in ["gsm8k", "mmlu", "humaneval"]:
                bd = r["benchmarks"].get(bench, {})
                entry[bench] = bd.get("accuracy") if not bd.get("error") and not bd.get("skipped") else None
            summary.append(entry)
        print(json.dumps(summary, indent=2))
    else:
        print(make_table(results))

    # Auto-save HTML alongside the first input path, or use --html
    if args.html:
        make_html(results, Path(args.html))
    elif len(args.paths) == 1:
        p = Path(args.paths[0])
        html_path = (p if p.is_dir() else p.parent) / "summary.html"
        make_html(results, html_path)


if __name__ == "__main__":
    main()
