"""Run a script under a hard RSS cap and wall-clock timeout.

A watchdog for long, memory-hungry local recomputations (the potential-size
re-solves in this directory). It launches the target script in its own process
group, samples the group's total RSS every 2 s, and SIGTERM/SIGKILLs the whole
group if it exceeds ``cap_mb`` or runs past ``timeout_s``. This is the safety
net that lets us parallelise the PS solve on a laptop without risking an
out-of-memory hard-freeze (dask's memory controls are not trustworthy here, so
we cap externally and use joblib inside the jobs).

Usage:
    python rerun/mem_guard.py <script.py> <cap_mb> <timeout_s> [-- arg ...]

The child runs with cwd = the repo root (this file's grandparent) and that same
path on PYTHONPATH, so ``import w22`` / ``import tcpips`` resolve without an
install. Extra args after ``--`` are forwarded to the target script.

Exit code 0 only if the child completed on its own with rc 0; 1 otherwise
(cap/timeout kill, or non-zero child exit).
"""

import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    script, cap_mb, timeout_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    child_args = sys.argv[4:]
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]

    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    child = subprocess.Popen(
        [sys.executable, script, *child_args],
        start_new_session=True,  # own process group, so we can kill workers too
        cwd=REPO_ROOT,
        env=env,
    )
    pgid = os.getpgid(child.pid)
    peak = 0
    t0 = time.time()
    verdict = "completed"

    def kill(reason: str) -> None:
        nonlocal verdict
        verdict = reason
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    while child.poll() is None:
        time.sleep(2.0)
        try:
            out = subprocess.run(
                ["ps", "-ax", "-o", "pgid=,rss="], capture_output=True, text=True
            ).stdout
            rss_kb = sum(
                int(p[1])
                for line in out.splitlines()
                if (p := line.split()) and p[0] == str(pgid)
            )
        except Exception:
            continue
        peak = max(peak, rss_kb)
        if rss_kb > cap_mb * 1024:
            kill(f"KILLED: RSS {rss_kb // 1024}MB > cap {cap_mb}MB")
            break
        if time.time() - t0 > timeout_s:
            kill(f"KILLED: wall > {timeout_s}s")
            break

    rc = child.wait()
    print(
        f"\n[watchdog] {verdict}; exit={rc}; peak RSS {peak // 1024}MB; "
        f"wall {time.time() - t0:.0f}s",
        flush=True,
    )
    return 0 if verdict == "completed" and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
