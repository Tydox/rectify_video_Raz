"""
Small, dependency-free debug/utility helpers shared across scripts in this
pipeline. Nothing in here should be required for the pipeline to run --
it's all optional instrumentation.
"""

import functools
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Global switch: when False, @timeit / @profile_resources become
# zero-overhead no-ops instead of timing anything. Flip this off for
# production/full-pipeline runs.
TIMING_ENABLED = True

# Global switch: when False, per-call results are still timed and printed,
# but are NOT accumulated into _TIMING_STATS (so print_timing_summary() /
# get_timing_stats() will have nothing to show). Turn this off if you just
# want live per-call prints without the memory overhead of keeping history.
STATS_ENABLED = True

# Separate global switch just for @profile_resources -- independent of
# TIMING_ENABLED, so you can time everything with @timeit while keeping the
# heavier RAM/GPU queries off (or vice versa). When False, @profile_resources
# calls the function directly with no psutil/torch.cuda overhead.
PROFILE_RESOURCES_ENABLED = True

# Global switch for dprint(). - DEBUG Print- . When False, dprint() is a no-op -- lets you
# silence all of your debug prints from one place instead of commenting
# them out / guarding each one manually.
DPRINT_ENABLED = True


def dprint(*args, **kwargs):
    """
    Drop-in replacement for print() that can be globally turned off via
    DPRINT_ENABLED, instead of manually removing/guarding every print()
    call. Either prints exactly like print() would, or does nothing.

    Usage:
        from general_useful_functions import dprint
        dprint("some debug info:", value)

    To silence everywhere:
        import general_useful_functions as guf
        guf.DPRINT_ENABLED = False
    """
    if not DPRINT_ENABLED:
        return
    print(*args, **kwargs)


# Optional dependencies -- only used by profile_resources(). Everything else
# in this file works without them.
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

# ANSI colors for console output.
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

# Per-function call stats, keyed by label. Only populated while
# TIMING_ENABLED and STATS_ENABLED are both True. Read via
# print_timing_summary() / get_timing_stats().
_TIMING_STATS = defaultdict(lambda: {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0})


def _record_stat(name, elapsed):
    if not STATS_ENABLED:
        return
    stats = _TIMING_STATS[name]
    stats["count"] += 1
    stats["total"] += elapsed
    stats["min"] = min(stats["min"], elapsed)
    stats["max"] = max(stats["max"], elapsed)


def timeit(label=None, enabled=None):
    """
    Decorator that prints how long a function call took, in seconds, and
    (if STATS_ENABLED) records it for print_timing_summary().

    Works on any function/method regardless of its signature, since it just
    forwards *args/**kwargs through. Uses time.perf_counter() (monotonic,
    high resolution) rather than time.time().

    Usage:
        @timeit()
        def my_func(...): ...

        @timeit(label="rectify one video")
        def my_func(...): ...

    Set TIMING_ENABLED = False (module-level, see below) to disable timing
    everywhere without removing the decorators -- the wrapper then just
    calls the function directly, with no perf_counter overhead.

    :param enabled: per-function override. None (default) follows the
        module-level TIMING_ENABLED flag. Pass True/False to force this
        one function on/off regardless of the global setting.
    """
    def decorator(func):
        name = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_enabled = TIMING_ENABLED if enabled is None else enabled
            if not is_enabled:
                return func(*args, **kwargs)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                elapsed = time.perf_counter() - start
                print(f"{_BLUE}{name}{_RESET} {_RED}FAILED{_RESET} after {_YELLOW}{elapsed:.4f} s{_RESET}")
                raise

            elapsed = time.perf_counter() - start
            _record_stat(name, elapsed)

            stats = _TIMING_STATS[name] if STATS_ENABLED else None
            if stats:
                print(f"{_BLUE}{name}{_RESET}: {_YELLOW}{elapsed:.4f} s{_RESET} "
                      f"(call #{stats['count']}, avg {stats['total'] / stats['count']:.4f} s)")
            else:
                print(f"{_BLUE}{name}{_RESET}: {_YELLOW}{elapsed:.4f} s{_RESET}")
            return result

        return wrapper

    return decorator


def get_timing_stats():
    """Return the raw {label: {count, total, min, max}} stats dict."""
    return dict(_TIMING_STATS)


def save_timing_stats(path, fmt=None, add_timestamp=True):
    """
    Dump get_timing_stats() to a file, so timings can be tracked/compared
    across runs instead of only being visible in the console.

    :param path: output file path (str or Path).
    :param fmt: "csv" or "json". If None (default), inferred from path's
        extension (.csv / .json); defaults to "csv" for anything else.
    :param add_timestamp: if True (default), inserts the current
        year-month-day-time into the filename (before the extension), so
        each run's file is distinguishable, e.g.
        "timing_report.csv" -> "timing_report_2026-08-01-18-42-07.csv".

    CSV columns: name, count, total, avg, min, max.
    JSON: {name: {count, total, avg, min, max}, ...}.
    """
    path = Path(path)
    if fmt is None:
        fmt = "json" if path.suffix.lower() == ".json" else "csv"

    suffix = path.suffix or f".{fmt}"
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = path.with_name(f"{path.stem}_{timestamp}{suffix}")
    else:
        path = path.with_suffix(suffix)

    stats = get_timing_stats()
    rows = {
        name: {
            "count": s["count"],
            "total": s["total"],
            "avg": s["total"] / s["count"] if s["count"] else 0.0,
            "min": s["min"],
            "max": s["max"],
        }
        for name, s in stats.items()
    }

    if fmt == "json":
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    else:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "count", "total", "avg", "min", "max"])
            for name, s in rows.items():
                writer.writerow([name, s["count"], s["total"], s["avg"], s["min"], s["max"]])

    print(f"Saved timing stats to: {path}\n---------------------------\n")


def print_timing_summary():
    """
    Print an aggregated summary (call count, total/avg/min/max seconds) for
    every @timeit/@profile_resources-decorated function that has been called
    so far, sorted by total time descending. Useful to call once at the end
    of a pipeline run to see where the time actually went, instead of
    scrolling through every individual per-call line.

    Requires STATS_ENABLED to have been True while those calls ran.
    """
    if not _TIMING_STATS:
        print("No timing data recorded.")
        return

    print(f"\n\n{_BLUE}--- Timing summary ---{_RESET}")
    rows = sorted(_TIMING_STATS.items(), key=lambda kv: kv[1]["total"], reverse=True)
    for name, stats in rows:
        avg = stats["total"] / stats["count"]
        print(
            f"{_BLUE}{name}{_RESET}: "
            f"calls={stats['count']}, "
            f"total={_YELLOW}{stats['total']:.4f} s{_RESET}, "
            f"avg={_YELLOW}{avg:.4f} s{_RESET}, "
            f"min={stats['min']:.4f} s, max={stats['max']:.4f} s"
        )
    print(f"------------------\n\n")


def _process_rss_mb():
    """Current process RAM (resident set size), in MB. None if psutil isn't installed."""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 ** 2)


def _gpu_allocated_mb():
    """Currently allocated CUDA memory, in MB. None if torch/CUDA isn't available."""
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 ** 2)


def _gpu_peak_mb():
    """Peak allocated CUDA memory since the last reset, in MB. None if torch/CUDA isn't available."""
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def profile_resources(label=None, enabled=None):
    """
    Heavier sibling of @timeit, for GPU- and/or memory-intensive functions
    (e.g. model inference, big batch frame processing). In addition to
    elapsed time, it reports:
      - process RAM before/after (via psutil, if installed)
      - CUDA memory allocated before/after, and the peak during the call
        (via torch.cuda, if torch + a CUDA device are available)

    Missing optional dependencies degrade gracefully -- whatever can't be
    measured is simply omitted from the printout, nothing raises.

    This is deliberately a separate decorator from @timeit rather than
    extra logic bolted onto it: querying process RSS and CUDA memory stats
    has real (if small) overhead per call, and most functions in this
    pipeline (file I/O, folder loops, simple OpenCV calls) don't need it.
    Reserve @profile_resources for the handful of functions that actually
    touch the GPU or allocate large arrays/tensors.

    Usage:
        @profile_resources()
        def run_model(...): ...

    :param enabled: per-function override. None (default) follows the
        module-level PROFILE_RESOURCES_ENABLED flag (independent of
        TIMING_ENABLED). Pass True/False to force this one function on/off
        regardless of the global setting.
    """
    def decorator(func):
        name = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_enabled = PROFILE_RESOURCES_ENABLED if enabled is None else enabled
            if not is_enabled:
                return func(*args, **kwargs)

            ram_before = _process_rss_mb()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            gpu_before = _gpu_allocated_mb()

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                elapsed = time.perf_counter() - start
                print(f"{_BLUE}{name}{_RESET} {_RED}FAILED{_RESET} after {_YELLOW}{elapsed:.4f} s{_RESET}")
                raise
            elapsed = time.perf_counter() - start

            ram_after = _process_rss_mb()
            gpu_after = _gpu_allocated_mb()
            gpu_peak = _gpu_peak_mb()

            _record_stat(name, elapsed)

            parts = [f"{_BLUE}{name}{_RESET}: {_YELLOW}{elapsed:.4f} s{_RESET}"]
            if ram_before is not None and ram_after is not None:
                parts.append(f"RAM {ram_before:.1f}->{ram_after:.1f} MB (delta {ram_after - ram_before:+.1f})")
            if gpu_before is not None and gpu_after is not None:
                parts.append(f"GPU {gpu_before:.1f}->{gpu_after:.1f} MB (peak {gpu_peak:.1f})")
            print(", ".join(parts))

            return result

        return wrapper

    return decorator
