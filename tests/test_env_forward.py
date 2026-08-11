"""Check cli.build_env_pairs: every .env-declared key is forwarded at its
effective os.environ value, and the launcher stays payload-agnostic — it
overwrites only the keys it owns (resolved project, generic plumbing) and
never emits a duplicate --env."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dsub_submit.cli import DsubConfig, build_env_pairs


def test_forward_all_env_keys():
    cfg = DsubConfig()  # wandb_mode="offline"
    os.environ["WORKSPACE_CDR"] = "cdr-x"
    os.environ["DELPHI_DATASET"] = "aou"
    os.environ["DELPHI_DATA_DIR"] = "gs://env-data"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "env-proj"
    env_keys = [
        "GOOGLE_CLOUD_PROJECT",
        "DELPHI_DATA_DIR",
        "DELPHI_DATASET",
        "WORKSPACE_CDR",
    ]
    pairs = build_env_pairs(cfg, "proj", env_keys)
    keys = [k for k, _ in pairs]
    d = dict(pairs)

    # no duplicate --env keys
    assert len(keys) == len(set(keys)), keys
    # the launcher-resolved project wins over the raw .env value
    assert d["GOOGLE_CLOUD_PROJECT"] == "proj"
    # payload vars are forwarded verbatim — the launcher knows nothing about them
    assert d["DELPHI_DATA_DIR"] == "gs://env-data"
    assert d["DELPHI_DATASET"] == "aou"
    assert d["WORKSPACE_CDR"] == "cdr-x"
    # generic plumbing still present
    assert d["PYTHONUNBUFFERED"] == "1"
    assert d["WANDB_MODE"] == "offline"


if __name__ == "__main__":
    test_forward_all_env_keys()
    print("OK")
