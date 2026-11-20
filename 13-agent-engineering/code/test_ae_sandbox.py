import ae_sandbox as m


def test_traversal_is_blocked():
    assert m.path_allowed("src/app.py")
    assert not m.path_allowed("../../etc/passwd")
    assert not m.path_allowed("/etc/passwd")
    assert not m.path_allowed("src/../../secret")


def test_dot_segments_inside_workspace_ok():
    assert m.path_allowed("src/./x/../y.py")
    assert m.resolve_path("src/./x/../y.py") == "/work/project/src/y.py"


def test_prefix_trick_is_blocked():
    assert not m.path_allowed("/work/project-evil/x")


def test_command_policy():
    assert m.command_allowed("ls -la")[0]
    assert m.command_allowed("cat a.txt; rm -rf /") == (False, "shell operator")
    assert m.command_allowed("cat a.txt;rm -rf /") == (False, "shell operator")
    assert m.command_allowed("ls && id") == (False, "shell operator")
    assert m.command_allowed("cat a | sh") == (False, "shell operator")
    assert m.command_allowed("rm -rf /") == (False, "program not allowed")
    assert m.command_allowed("grep x $(whoami)") == (False, "substitution")
    assert m.command_allowed("") == (False, "empty")


def test_url_policy_blocks_lookalikes():
    assert m.url_allowed("https://api.example.com/v1")
    assert not m.url_allowed("http://api.example.com/v1")
    assert not m.url_allowed("https://api.example.com.evil.io/x")
    assert not m.url_allowed("https://evil.io/api.example.com")


def test_run_limited_ok_error_timeout():
    assert m.run_limited("print(2+2)") == {"status": "ok", "output": "4\n"}
    assert m.run_limited("raise SystemExit(3)")["status"] == "error"
    assert m.run_limited("while True: pass", timeout=0.4)["status"] == "timeout"


def test_child_has_empty_environment():
    assert m.run_limited("import os; print(len(os.environ))")["output"].strip() in {"0", "1", "2"}
