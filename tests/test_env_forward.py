"""Check cli.build_env_pairs: dedup against the resolved set + forward the
remaining .env-declared keys (so a raw .env value never duplicates / clobbers a
resolved one like DELPHI_DATA_DIR)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dsub_submit.cli import DsubConfig, build_env_pairs


def test_dedup_and_forward():
    cfg = DsubConfig()  # dataset="aou", wandb_mode="offline"
    os.environ["WORKSPACE_CDR"] = "cdr-x"
    os.environ["DELPHI_LOG_BACKEND"] = "tensorboard"
    # .env declares keys that overlap the resolved set + some new ones
    env_keys = [
        "GOOGLE_CLOUD_PROJECT",
        "DELPHI_DATA_DIR",
        "DELPHI_CKPT_DIR",
        "WORKSPACE_CDR",
        "DELPHI_LOG_BACKEND",
    ]
    pairs = build_env_pairs(cfg, "proj", "gs://data", "gs://ckpt", env_keys)
    keys = [k for k, _ in pairs]
    d = dict(pairs)

    # no duplicate --env keys
    assert len(keys) == len(set(keys)), keys
    # resolved values win for the overlapping keys (not the raw .env value)
    assert d["GOOGLE_CLOUD_PROJECT"] == "proj"
    assert d["DELPHI_DATA_DIR"] == "gs://data"
    assert d["DELPHI_CKPT_DIR"] == "gs://ckpt"
    # the remaining .env vars are forwarded (effective value)
    assert d["WORKSPACE_CDR"] == "cdr-x"
    assert d["DELPHI_LOG_BACKEND"] == "tensorboard"
    # constants still present
    assert d["PYTHONUNBUFFERED"] == "1"
    assert d["WANDB_MODE"] == "offline"


if __name__ == "__main__":
    test_dedup_and_forward()
    print("OK")
