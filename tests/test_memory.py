"""Check resolve_machine_type's memory= handling: GPU jobs reject it, CPU jobs
get the smallest n1-standard-* that fits, and machine_type= never mixes with it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dsub_submit.cli import DsubConfig, resolve_machine_type


def expect_exit(**kw):
    try:
        resolve_machine_type(DsubConfig(**kw))
    except SystemExit:
        return
    raise AssertionError(f"{kw} should have exited")


def test_memory():
    cpu = dict(gpu=False)
    # smallest fit, boundaries inclusive (n1-standard-N = N * 3.75 GB)
    assert resolve_machine_type(DsubConfig(**cpu, memory=1)) == "n1-standard-1"
    assert resolve_machine_type(DsubConfig(**cpu, memory=3.75)) == "n1-standard-1"
    assert resolve_machine_type(DsubConfig(**cpu, memory=3.76)) == "n1-standard-2"
    assert resolve_machine_type(DsubConfig(**cpu, memory=64)) == "n1-standard-32"
    assert resolve_machine_type(DsubConfig(**cpu, memory=360)) == "n1-standard-96"

    expect_exit(gpu=False, memory=361)  # past the ladder -> machine_type= yourself
    expect_exit(gpu=True, gpu_num=1, memory=16)  # GPU jobs: not yet supported
    expect_exit(gpu=False, memory=16, machine_type="n1-highmem-8")  # pick one

    # memory=None leaves the existing paths untouched
    assert resolve_machine_type(DsubConfig(gpu=False)) == "n1-standard-8"
    assert resolve_machine_type(DsubConfig(gpu=True, gpu_num=1)) == "n1-standard-8"


if __name__ == "__main__":
    test_memory()
    print("OK")
