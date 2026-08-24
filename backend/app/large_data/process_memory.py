"""Portable "how much memory is this OS process actually using" reading, for the
benchmark script (``app/large_data/benchmark.py``) to report real peak-RSS
numbers rather than an estimate.

This is deliberately separate from ``memory_guard.py``: the guard enforces a
ceiling using exact, cheap, OS-independent object-size accounting
(`DataFrame.memory_usage`), which is what makes it fast and unit-testable. This
module instead answers "what did the OS actually allocate to this process," which
is only needed for *reporting* real benchmark numbers, and is inherently
platform-specific.

No new third-party dependency: ``psutil`` isn't in requirements.txt (checked —
see report), so this uses stdlib only — ``ctypes`` + the Win32
``GetProcessMemoryInfo`` API on Windows (this project's dev sandbox), and the
POSIX ``resource`` module elsewhere. If ``psutil`` is ever added to
requirements.txt (recommended for production — see report), prefer it there;
it's used here automatically if already importable.
"""

from __future__ import annotations

import sys


def get_peak_rss_bytes() -> int:
    """Peak resident-set-size (peak working-set on Windows) of the current
    process since it started, in bytes. Monotonically non-decreasing — callers
    benchmarking a single operation should read this once *before* and once
    *after* the operation in an otherwise-fresh process (see benchmark.py, which
    runs naive vs. chunked as separate subprocesses for exactly this reason) and
    report the "after" value as that operation's peak.
    """
    try:
        import psutil  # type: ignore

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except ImportError:
        pass

    if sys.platform == "win32":
        return _win32_peak_working_set()
    return _posix_peak_rss()


def _win32_peak_working_set() -> int:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(f"GetProcessMemoryInfo failed (WinError {err}: {ctypes.FormatError(err)}).")
    return int(counters.PeakWorkingSetSize)


def _posix_peak_rss() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS.
    return usage * 1024 if sys.platform != "darwin" else usage
