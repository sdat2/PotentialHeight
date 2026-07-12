import os, signal, subprocess, sys, time
script, cap_mb, timeout_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = dict(os.environ, PYTHONPATH="/Users/simon/worstsurge")
child = subprocess.Popen([sys.executable, script], start_new_session=True, cwd="/Users/simon/worstsurge", env=env)
pgid = os.getpgid(child.pid); peak=0; t0=time.time(); verdict="completed"
while child.poll() is None:
    time.sleep(2.0)
    try:
        out = subprocess.run(["ps","-ax","-o","pgid=,rss="], capture_output=True, text=True).stdout
        rss_kb = sum(int(p[1]) for line in out.splitlines() if (p:=line.split()) and p[0]==str(pgid))
    except Exception: continue
    peak=max(peak,rss_kb)
    if rss_kb > cap_mb*1024:
        verdict=f"KILLED: RSS {rss_kb//1024}MB > cap {cap_mb}MB"
        os.killpg(pgid,signal.SIGTERM); time.sleep(3)
        try: os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError: pass
        break
    if time.time()-t0 > timeout_s:
        verdict=f"KILLED: wall > {timeout_s}s"
        os.killpg(pgid,signal.SIGTERM); time.sleep(3)
        try: os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError: pass
        break
rc=child.wait()
print(f"\n[watchdog] {verdict}; exit={rc}; peak RSS {peak//1024}MB; wall {time.time()-t0:.0f}s")
sys.exit(0 if verdict=="completed" and rc==0 else 1)
