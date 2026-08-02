"""Check cli._parse_kill_args: the bulk selector and explicit job ids are mutually
exclusive, and neither is ever inferred silently -- dkill deletes Batch jobs
irreversibly, so `dkill all JOB_ID` must error rather than pick one meaning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dsub_submit.cli import _parse_kill_args


def expect_exit(args):
    try:
        _parse_kill_args(args)
    except SystemExit:
        return
    raise AssertionError(f"{args} should have exited")


def test_parse_kill_args():
    # explicit ids -> no selector, ids verbatim (order preserved)
    assert _parse_kill_args(["job-a", "job-b"]) == (None, ["job-a", "job-b"])
    # bulk selectors -> "" for all, substring for name=
    assert _parse_kill_args(["all"]) == ("", [])
    assert _parse_kill_args(["-a"]) == ("", [])
    assert _parse_kill_args(["name=train-delp"]) == ("train-delp", [])
    # a name= substring containing '=' survives (only the first = is the separator)
    assert _parse_kill_args(["name=lr=1e-3"]) == ("lr=1e-3", [])

    expect_exit([])  # no args: usage, never "delete everything"
    expect_exit(["all", "job-a"])  # mixing would delete a set you didn't name
    expect_exit(["name=x", "job-a"])
    expect_exit(["name=x", "all"])
    expect_exit(["name=x", "name=y"])


if __name__ == "__main__":
    test_parse_kill_args()
    print("OK")
