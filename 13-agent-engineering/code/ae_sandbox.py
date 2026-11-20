"""Sandboxing basics for agent tools: path, command and network policy, plus a real
process limit. Stdlib only. This is a policy layer and a timeout, NOT an OS sandbox:
real isolation needs containers, VMs or seccomp (see the lesson).

Run:  python ae_sandbox.py
"""
import os
import posixpath
import shlex
import subprocess
import sys
from urllib.parse import urlparse

WORKSPACE = "/work/project"
ALLOWED_COMMANDS = {"ls", "cat", "grep", "python3", "pytest"}
ALLOWED_HOSTS = {"api.example.com", "pypi.org"}


def resolve_path(user_path, root=WORKSPACE):
    """Join to the workspace and normalise. Returns the absolute path."""
    joined = user_path if user_path.startswith("/") else posixpath.join(root, user_path)
    return posixpath.normpath(joined)


def path_allowed(user_path, root=WORKSPACE):
    p = resolve_path(user_path, root)
    return p == root or p.startswith(root + "/")


def command_allowed(command_line, allowed=ALLOWED_COMMANDS):
    """Allow only a known program, with no shell operators. Returns (ok, reason)."""
    try:
        argv = shlex.split(command_line)
    except ValueError:
        return False, "unparseable"
    if not argv:
        return False, "empty"
    # Scan the RAW string: shlex.split keeps "a.txt;" as ONE token, so checking tokens misses "cat a.txt; rm -rf /".
    if any(ch in command_line for ch in (";", "&", "|", ">", "<", "\n")):
        return False, "shell operator"
    if any(ch in command_line for ch in ("`", "$")):
        return False, "substitution"
    if argv[0] not in allowed:
        return False, "program not allowed"
    return True, "ok"


def url_allowed(url, hosts=ALLOWED_HOSTS):
    u = urlparse(url)
    if u.scheme != "https":
        return False
    return (u.hostname or "") in hosts


def run_limited(code, timeout=1.0, max_output=2000):
    """Run Python code in a child process with an empty environment and a hard timeout.
    Returns dict(status, output). status is ok | error | timeout."""
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                           timeout=timeout, env={}, cwd="/")
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": ""}
    out = (r.stdout + r.stderr)[:max_output]
    return {"status": "ok" if r.returncode == 0 else "error", "output": out}


if __name__ == "__main__":
    for p in ("src/app.py", "../../etc/passwd", "/etc/passwd", "src/../../secret", "src/./x/../y.py"):
        print(f"path  {p:<20} -> {resolve_path(p):<26} allowed={path_allowed(p)}")
    for c in ("ls -la", "cat notes.txt", "cat a.txt; rm -rf /", "rm -rf /", "grep x $(whoami)", "python3 -c 'print(1)'"):
        print(f"cmd   {c:<26} -> {command_allowed(c)}")
    for u in ("https://api.example.com/v1", "http://api.example.com/v1", "https://api.example.com.evil.io/x", "https://evil.io/api.example.com"):
        print(f"url   {u:<38} -> {url_allowed(u)}")
    print("run   fine     ->", run_limited("print(2+2)"))
    print("run   loop     ->", run_limited("while True: pass", timeout=0.5))
    print("run   env leak ->", run_limited("import os; print(len(os.environ))"))
