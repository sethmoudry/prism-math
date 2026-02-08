#!/usr/bin/env python3
"""Comprehensive audit of all registered Prism methods.

Iterates every method in the MethodRegistry, runs generate_parameters -> compute
trials with timeout enforcement, tests inversions, checks known values, and writes
incremental JSONL results.

Usage:
    python -m prism.tests.audit_methods
    python -m prism.tests.audit_methods --trials 5 --timeout 10
    python -m prism.tests.audit_methods --method fibonacci --quick
    python -m prism.tests.audit_methods --resume --output audit_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# sys.path setup -- allow running from repo root or tests/ directory
# ---------------------------------------------------------------------------
_PRISM_SRC = str(Path(__file__).parent.parent / "src")
_PRISM_ROOT = str(Path(__file__).parent.parent)
if _PRISM_SRC not in sys.path:
    sys.path.insert(0, _PRISM_SRC)
if _PRISM_ROOT not in sys.path:
    sys.path.insert(0, _PRISM_ROOT)

from prism.techniques import MethodRegistry, MethodResult  # noqa: E402

# audit_input_factory for per-type test inputs
from tests.audit_input_factory import get_test_input  # noqa: E402

# Known ground-truth values
try:
    from tests.known_values import KNOWN_VALUES  # noqa: E402
except ImportError:
    KNOWN_VALUES: Dict[str, List[dict]] = {}


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
PASS = "PASS"
FAIL_GENERATE = "FAIL_GENERATE"
FAIL_COMPUTE = "FAIL_COMPUTE"
FAIL_TIMEOUT = "FAIL_TIMEOUT"
FAIL_RANGE = "FAIL_RANGE"
FAIL_KNOWN_VALUE = "FAIL_KNOWN_VALUE"
FAIL_IMPORT = "FAIL_IMPORT"
WARN_SLOW = "WARN_SLOW"

ALL_FAIL_STATUSES = {
    FAIL_GENERATE, FAIL_COMPUTE, FAIL_TIMEOUT,
    FAIL_RANGE, FAIL_KNOWN_VALUE, FAIL_IMPORT,
}


# ---------------------------------------------------------------------------
# Dataclass for per-method audit result
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    method_name: str
    input_type: str
    output_type: str
    status: str
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration_ms: float = 0.0
    result_value: Any = None
    result_type: Optional[str] = None
    can_invert: bool = False
    invert_tested: bool = False
    invert_passed: bool = False
    known_value_tested: bool = False
    known_value_passed: bool = False
    tags: List[str] = field(default_factory=list)
    difficulty: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_serialize(value: Any, max_len: int = 200) -> Any:
    """Convert *value* to something JSON-serializable, truncating if needed."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, (list, tuple)):
        items = [_safe_serialize(v, max_len=50) for v in value[:20]]
        if len(value) > 20:
            items.append(f"... ({len(value)} total)")
        return items
    if isinstance(value, dict):
        out = {}
        for k, v in list(value.items())[:20]:
            out[str(k)] = _safe_serialize(v, max_len=50)
        return out
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"...[truncated, len={len(s)}]"
    return s


def _values_close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Check whether two numeric values are approximately equal."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) or math.isnan(fb):
        return False
    if fa == fb:
        return True
    denom = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) / denom <= tol


def _infer_domain(tags: List[str]) -> str:
    """Infer a high-level domain from a method's tags."""
    tag_set = {t.lower() for t in tags}
    for domain in ("geometry", "number_theory", "algebra", "combinatorics",
                   "analysis", "graph", "probability", "topology"):
        if domain in tag_set:
            return domain
    # Fallback heuristics
    for t in tag_set:
        if "geom" in t or "triangle" in t or "circle" in t:
            return "geometry"
        if "prime" in t or "divis" in t or "modular" in t or "totient" in t:
            return "number_theory"
        if "poly" in t or "vieta" in t or "inequal" in t or "equation" in t:
            return "algebra"
        if "count" in t or "permut" in t or "combinator" in t or "graph" in t:
            return "combinatorics"
    return "other"


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def _run_with_timeout(fn, timeout_sec: float):
    """Execute *fn()* in a worker thread, raising FuturesTimeoutError on expiry."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        return future.result(timeout=timeout_sec)


def _audit_single_trial(method, input_value, timeout_sec: float) -> AuditResult:
    """Run one generate -> compute trial for *method*.  Returns AuditResult."""
    result = AuditResult(
        method_name=method.name,
        input_type=getattr(method, "input_type", "unknown"),
        output_type=getattr(method, "output_type", "unknown"),
        status=PASS,
        tags=list(getattr(method, "tags", [])),
        difficulty=getattr(method, "difficulty", 1),
        can_invert=False,
    )

    t0 = time.monotonic()

    # --- generate_parameters ---
    try:
        params = _run_with_timeout(
            lambda: method.generate_parameters(input_value), timeout_sec
        )
    except FuturesTimeoutError:
        result.status = FAIL_TIMEOUT
        result.error = "generate_parameters timed out"
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result
    except Exception as exc:
        result.status = FAIL_GENERATE
        result.error = f"{type(exc).__name__}: {exc}"
        result.error_type = type(exc).__name__
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    if params is None or not isinstance(params, dict):
        result.status = FAIL_GENERATE
        result.error = f"generate_parameters returned {type(params).__name__} (expected dict)"
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    # --- compute ---
    try:
        compute_result = _run_with_timeout(
            lambda: method.compute(input_value, params), timeout_sec
        )
    except FuturesTimeoutError:
        result.status = FAIL_TIMEOUT
        result.error = "compute timed out"
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result
    except Exception as exc:
        result.status = FAIL_COMPUTE
        result.error = f"{type(exc).__name__}: {exc}"
        result.error_type = type(exc).__name__
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    elapsed_ms = (time.monotonic() - t0) * 1000
    result.duration_ms = elapsed_ms

    if compute_result is None:
        result.status = FAIL_COMPUTE
        result.error = "compute returned None"
        return result

    if not isinstance(compute_result, MethodResult):
        result.status = FAIL_COMPUTE
        result.error = f"compute returned {type(compute_result).__name__}, expected MethodResult"
        return result

    result.result_value = _safe_serialize(compute_result.value)
    result.result_type = type(compute_result.value).__name__

    # --- can_invert ---
    try:
        result.can_invert = bool(method.can_invert())
    except Exception:
        result.can_invert = False

    # --- slow warning ---
    if elapsed_ms > 2000 and result.status == PASS:
        result.status = WARN_SLOW

    return result


def _test_inversion(method, input_value, params, compute_result,
                    timeout_sec: float) -> tuple[bool, bool]:
    """Test method inversion.  Returns (tested, passed)."""
    try:
        can_inv = method.can_invert()
    except Exception:
        return False, False

    if not can_inv:
        return False, False

    try:
        inverted = _run_with_timeout(
            lambda: method.invert(compute_result.value, params), timeout_sec
        )
    except (FuturesTimeoutError, Exception):
        return True, False

    if inverted is None:
        return True, False

    return True, _values_close(inverted, input_value)


def _test_known_values(method_name: str, method, timeout_sec: float,
                       ) -> tuple[bool, bool]:
    """Check results against known ground-truth values.  Returns (tested, passed)."""
    if method_name not in KNOWN_VALUES:
        return False, False

    cases = KNOWN_VALUES[method_name]
    if not cases:
        return False, False

    for case in cases:
        inp = case.get("input")
        params = case.get("params", {})
        expected = case.get("expected")
        if expected is None:
            continue
        try:
            res = _run_with_timeout(
                lambda: method.compute(inp, params), timeout_sec
            )
            if res is None or not isinstance(res, MethodResult):
                return True, False
            if not _values_close(res.value, expected):
                return True, False
        except Exception:
            return True, False

    return True, True


def audit_method(method, trials: int, timeout_sec: float,
                 skip_known: bool = False,
                 skip_inversion: bool = False) -> AuditResult:
    """Run *trials* audit rounds on *method* and return the worst AuditResult."""
    input_value = get_test_input(getattr(method, "input_type", "any"))

    worst: Optional[AuditResult] = None

    for _ in range(trials):
        res = _audit_single_trial(method, input_value, timeout_sec)

        if res.status in ALL_FAIL_STATUSES:
            return res  # Fail fast on hard failures

        if worst is None or (res.status != PASS and worst.status == PASS):
            worst = res

    assert worst is not None

    # --- Inversion test (once, not per trial) ---
    if not skip_inversion and worst.status in (PASS, WARN_SLOW):
        try:
            params = _run_with_timeout(
                lambda: method.generate_parameters(input_value), timeout_sec
            )
            if params and isinstance(params, dict):
                compute_res = _run_with_timeout(
                    lambda: method.compute(input_value, params), timeout_sec
                )
                if compute_res and isinstance(compute_res, MethodResult):
                    tested, passed = _test_inversion(
                        method, input_value, params, compute_res, timeout_sec
                    )
                    worst.invert_tested = tested
                    worst.invert_passed = passed
        except Exception:
            pass  # inversion test is best-effort

    # --- Known-value test (once) ---
    if not skip_known and worst.status in (PASS, WARN_SLOW):
        tested, passed = _test_known_values(method.name, method, timeout_sec)
        worst.known_value_tested = tested
        worst.known_value_passed = passed
        if tested and not passed:
            worst.status = FAIL_KNOWN_VALUE
            worst.error = "Known-value check failed"

    return worst


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _load_completed_methods(path: str) -> Set[str]:
    """Load method names already audited from an existing JSONL file."""
    completed = set()
    if not os.path.exists(path):
        return completed
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    name = obj.get("method_name")
                    if name:
                        completed.add(name)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return completed


def _write_result(fp, result: AuditResult) -> None:
    """Append one AuditResult as a JSON line and flush."""
    d = asdict(result)
    fp.write(json.dumps(d, default=str) + "\n")
    fp.flush()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: List[AuditResult]) -> None:
    """Print a human-readable summary table."""
    if not results:
        print("\nNo methods audited.")
        return

    # --- By status ---
    status_counts: Dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    total = len(results)
    pass_count = status_counts.get(PASS, 0)
    warn_count = status_counts.get(WARN_SLOW, 0)
    fail_count = total - pass_count - warn_count

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"{'Total methods':<30} {total}")
    print(f"{'PASS':<30} {pass_count}")
    print(f"{'WARN_SLOW':<30} {warn_count}")
    print(f"{'FAIL (all types)':<30} {fail_count}")
    print("-" * 60)

    for status in sorted(status_counts.keys()):
        if status not in (PASS, WARN_SLOW):
            print(f"  {status:<28} {status_counts[status]}")

    # --- By domain ---
    domain_counts: Dict[str, Dict[str, int]] = {}
    for r in results:
        domain = _infer_domain(r.tags)
        if domain not in domain_counts:
            domain_counts[domain] = {"total": 0, "pass": 0, "fail": 0, "warn": 0}
        domain_counts[domain]["total"] += 1
        if r.status == PASS:
            domain_counts[domain]["pass"] += 1
        elif r.status == WARN_SLOW:
            domain_counts[domain]["warn"] += 1
        else:
            domain_counts[domain]["fail"] += 1

    print("\n" + "-" * 60)
    print(f"{'Domain':<20} {'Total':>6} {'Pass':>6} {'Warn':>6} {'Fail':>6}")
    print("-" * 60)
    for domain in sorted(domain_counts.keys()):
        c = domain_counts[domain]
        print(f"{domain:<20} {c['total']:>6} {c['pass']:>6} {c['warn']:>6} {c['fail']:>6}")

    # --- Inversion summary ---
    inv_tested = sum(1 for r in results if r.invert_tested)
    inv_passed = sum(1 for r in results if r.invert_passed)
    if inv_tested:
        print(f"\nInversion: {inv_passed}/{inv_tested} passed "
              f"({inv_passed / inv_tested * 100:.0f}%)")

    # --- Known-value summary ---
    kv_tested = sum(1 for r in results if r.known_value_tested)
    kv_passed = sum(1 for r in results if r.known_value_passed)
    if kv_tested:
        print(f"Known values: {kv_passed}/{kv_tested} passed "
              f"({kv_passed / kv_tested * 100:.0f}%)")

    print("=" * 60)

    # Pass rate
    effective_pass = pass_count + warn_count
    pct = effective_pass / total * 100 if total else 0
    print(f"\nEffective pass rate: {effective_pass}/{total} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit all registered Prism methods."
    )
    parser.add_argument("--output", default="audit_results.jsonl",
                        help="Path for JSONL output (default: audit_results.jsonl)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout in seconds per operation (default: 5)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per method (default: 3)")
    parser.add_argument("--method", default=None,
                        help="Substring filter on method name")
    parser.add_argument("--resume", action="store_true",
                        help="Skip methods already in output file")
    parser.add_argument("--quick", action="store_true",
                        help="1 trial, skip known values and inversions (CI mode)")
    args = parser.parse_args()

    if args.quick:
        args.trials = 1

    # Collect all methods sorted by name
    all_methods = sorted(MethodRegistry.get_all(), key=lambda m: m.name)

    # Filter by name substring
    if args.method:
        filt = args.method.lower()
        all_methods = [m for m in all_methods if filt in m.name.lower()]

    if not all_methods:
        print("No methods matched the filter. Exiting.")
        sys.exit(1)

    # Resume support
    completed: Set[str] = set()
    if args.resume:
        completed = _load_completed_methods(args.output)
        if completed:
            print(f"Resuming: {len(completed)} methods already audited, skipping.")

    methods_to_audit = [m for m in all_methods if m.name not in completed]
    total = len(methods_to_audit)

    if total == 0:
        print("All methods already audited. Nothing to do.")
        sys.exit(0)

    print(f"Auditing {total} methods | trials={args.trials} | "
          f"timeout={args.timeout}s | output={args.output}")
    if args.quick:
        print("Quick mode: 1 trial, no known-value checks, no inversion tests.")
    print()

    results: List[AuditResult] = []
    mode = "a" if args.resume else "w"

    with open(args.output, mode) as fp:
        for idx, method in enumerate(methods_to_audit, 1):
            result = audit_method(
                method,
                trials=args.trials,
                timeout_sec=args.timeout,
                skip_known=args.quick,
                skip_inversion=args.quick,
            )
            results.append(result)
            _write_result(fp, result)

            # Print failures immediately
            if result.status in ALL_FAIL_STATUSES:
                print(f"  [{idx}/{total}] FAIL {result.method_name}: "
                      f"{result.status} -- {result.error}")
            elif result.status == WARN_SLOW:
                print(f"  [{idx}/{total}] WARN {result.method_name}: "
                      f"{result.duration_ms:.0f}ms")

            # Periodic progress every 50 methods
            if idx % 50 == 0 or idx == total:
                pass_so_far = sum(1 for r in results
                                  if r.status in (PASS, WARN_SLOW))
                print(f"  Progress: {idx}/{total} audited, "
                      f"{pass_so_far} passing")

    _print_summary(results)

    # Exit code: non-zero if any hard failure
    fail_count = sum(1 for r in results if r.status in ALL_FAIL_STATUSES)
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
