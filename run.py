#!/usr/bin/env python3
"""Run the book's independent chapter laboratories and save reproducible logs."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--list", action="store_true", help="List the 26 chapters")
    select.add_argument("--chapter", type=int, nargs="+", help="Chapter number(s), 1 to 26")
    select.add_argument("--all", action="store_true", help="Run all chapters in order")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per chapter")
    parser.add_argument("--quiet", action="store_true", help="Show progress, keep detailed output in logs")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    catalog = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["chapters"]
    if args.list:
        for ch in catalog:
            print(f'{ch["number"]:02d}  {ch["slug"]}  ({len(ch["examples"])} examples)')
        return 0
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required; Python 3.12 is recommended")
    requested = set(args.chapter or range(1, 27))
    if not requested.issubset(range(1, 27)):
        parser.error("Chapter numbers must be between 1 and 26")
    versions = {}
    for name in ("numpy", "qiskit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            parser.exit(2, "Missing dependencies. Run: python -m pip install -r requirements.txt\n")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    results = []
    for ch in catalog:
        if ch["number"] not in requested:
            continue
        number = ch["number"]
        log_path = output / f"ch{number:02d}.log"
        print(f"[START] Chapter {number:02d}: {ch['slug']}", flush=True)
        started = time.perf_counter()
        status = "passed"
        try:
            process = subprocess.run(
                [sys.executable, "-u", str(ROOT / ch["path"])],
                cwd=ROOT, env=environment, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=args.timeout,
            )
            stdout, stderr = process.stdout, process.stderr
            code = process.returncode
            if code:
                status = "failed"
        except subprocess.TimeoutExpired as error:
            def decode(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
            stdout, stderr = decode(error.stdout), decode(error.stderr)
            stderr += f"\nChapter exceeded {args.timeout} seconds. Increase --timeout if needed.\n"
            status, code = "timeout", 124
        duration = round(time.perf_counter() - started, 3)
        log_path.write_text(stdout + ("\n--- stderr ---\n" + stderr if stderr else ""), encoding="utf-8")
        if not args.quiet:
            print(stdout, end="")
            if stderr:
                print(stderr, file=sys.stderr, end="")
        print(f"[{status.upper()}] Chapter {number:02d} | {duration:.3f}s | {log_path}", flush=True)
        results.append({"chapter": number, "slug": ch["slug"], "status": status,
                        "returncode": code, "seconds": duration,
                        "examples": len(ch["examples"]), "log": log_path.name})
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "dependencies": versions, "results": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    passed = sum(row["status"] == "passed" for row in results)
    print(f"Completed: {passed}/{len(results)} chapters passed. Report: {output / 'summary.json'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
