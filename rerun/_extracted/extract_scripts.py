import json, re, os
T="/Users/simon/.claude/projects/-Users-simon-thesis/3ecac1c7-3c1c-48be-9b89-cf847cc43e0d.jsonl"
OUT="/Users/simon/worstsurge/rerun/_extracted"
# regex: capture `cat > <path> <<'?DELIM'? \n body \n DELIM`
pat = re.compile(r"cat > (\S+) <<'?(\w+)'?\n(.*?)\n\2(?:\n|$)", re.DOTALL)
scripts = {}   # basename -> (body, order_index)
order = 0
def harvest(cmd):
    global order
    for m in pat.finditer(cmd):
        path, delim, body = m.group(1), m.group(2), m.group(3)
        base = os.path.basename(path)
        order += 1
        scripts[base] = (body, order)   # keep LAST occurrence (latest version)

writes = {}  # file_path basename -> content (from Write tool)
with open(T) as f:
    for line in f:
        try: obj = json.loads(line)
        except Exception: continue
        msg = obj.get("message", {})
        content = msg.get("content")
        if not isinstance(content, list): continue
        for blk in content:
            if not isinstance(blk, dict): continue
            if blk.get("type")=="tool_use":
                inp = blk.get("input", {}) or {}
                cmd = inp.get("command")
                if isinstance(cmd, str) and "cat > " in cmd:
                    harvest(cmd)
                # Write tool
                fp = inp.get("file_path"); ct = inp.get("content")
                if isinstance(fp,str) and fp.endswith(".py") and isinstance(ct,str):
                    writes[os.path.basename(fp)] = ct

for base,(body,_) in sorted(scripts.items()):
    with open(os.path.join(OUT, base), "w") as g: g.write(body+"\n")
for base,ct in writes.items():
    if base not in scripts:
        with open(os.path.join(OUT, "write_"+base), "w") as g: g.write(ct)

print(f"extracted {len(scripts)} heredoc scripts + {len([b for b in writes if b not in scripts])} Write-only .py")
for b in sorted(scripts): print("  ", b, f"({len(scripts[b][0])} bytes)")
