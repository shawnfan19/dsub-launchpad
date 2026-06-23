# dsub_launchpad

Submit **Delphi** jobs to the **All of Us** Researcher Workbench (Google Batch via
`dsub`) with the **same CLI as the SLURM [`launchpad`](https://github.com/) `submit`
tool**, so the same muscle memory works on both platforms.

Sibling to `slurm_submit` — *not* a refactor of it. The SLURM tool is untouched.

```
dsubmit <script + args...> -- <launcher cfg...>
```

Everything before `--` is the script and its args (run as `python <that> $OVERRIDE`
inside the container); everything after `--` is OmegaConf dotlist config merged over
`DsubConfig` (with `config=file.yaml` for a config file).

## Why `dsub050` and the rest of the AoU plumbing

This wraps the AoU dsub workflow documented in the Delphi repo at
`installation/containers/dsub/README.md`. In particular it always uses the pinned
**`dsub050`** (0.5.0 — the Workbench default 0.5.2 fails every GPU job with exit 125
and no log), routes images through the `$ARTIFACT_REGISTRY_DOCKER_REPO` proxy, and
bakes the mandatory `--project / --user-project / --service-account / --network /
--subnetwork / --use-private-address` flags from your environment.

## Install

**On the Workbench** (the terminal can reach github):
```bash
pip install git+https://github.com/shawnfan19/dsub-launchpad
echo 'alias submit=dsubmit' >> ~/.bashrc   # so you type `submit`, same as SLURM
```
`dsubmit` is the console script; the alias makes the *command* identical to the cluster.

**On the SLURM cluster** (for `--dry` debugging only; coexists with the SLURM `submit`):
```bash
pip install -e /hps/software/users/birney/sfan/dsub_launchpad
```

## Same syntax across platforms

```bash
# cluster (SLURM)
submit apps/train-delphi-m4.py batch_size=128 -- time=5 memory=64  gpu_num=1 sweep.lr=lrs.yaml
# Workbench (dsub)   ← after `source .env`
submit apps/train-delphi-m4.py batch_size=128 -- time=5 machine_type=n1-standard-8 gpu_num=1 sweep.lr=lrs.yaml
```
The only divergence is `memory=` → `machine_type=` (GCP predefined machine types couple
RAM:vCPU, and T4 GPUs require the `n1` family). `time`, `gpu`/`gpu_num`/`gpu_type`,
`sweep.*`, `overrides=`, `dry`, `job_name` are identical.

## Verify GCS access first (once per workspace)

Before trusting the launcher, prove the container can actually read/write `gs://` as your
pet service account. See `examples/verify-gcs.sh` for the one-shot job (submit it directly,
not via `dsubmit`). `VERIFY OK` ⇒ you're good. A READ/WRITE failure is an IAM grant on the
data/checkpoint bucket, not a launcher problem.

## How it works

- **Fan-out:** `sweep.<flag>=values.yaml|[inline,list]` and `overrides=file.yaml` (a list,
  or a dict of lists) are combined by `itertools.product` into **N independent `dsub050`
  jobs**. dsub has no positional passthrough, so each combo is injected as
  `--env OVERRIDE="k=v ..."` and the job runs `python <script> $OVERRIDE`.
- **I/O is fully GCS-native** (no `--input`/`--output` staging): `DELPHI_DATA_DIR`,
  `DELPHI_CKPT_DIR`, `GOOGLE_CLOUD_PROJECT`, `DELPHI_DATASET`, `WANDB_MODE` are forwarded
  into the container via `--env` (an allowlist — no secrets). Reads come straight from
  `gs://` (AoU readers are cloudpathlib-native); checkpoints **and** training logs
  (wandb/tb) write straight to `gs://` from inside Delphi (`delphi/log.py`), so there is
  nothing to delocalize after the run.
- **`.env` defaults:** values are read from `.env` (default `./.env` then
  `~/repos/delphi/.env`, override with `env_file=`) using the same `setdefault` semantics
  as `delphi.env.load_env_file`, so it doesn't depend on your shell having sourced it.
- **Image / hash:** `image_repo:tag` (default `shawnclarkefan/delphi:latest`) is resolved
  to an immutable `@sha256` digest via the Docker Hub API at submit time
  (`resolve_digest=true`) — always-newest *and* reproducible / proxy-cache-safe. `image=`
  bypasses with an explicit ref. An empty tag is rejected (the bug that produces exit 125).
- **wandb / tensorboard:** Batch VMs have no internet, so `WANDB_MODE=offline` (+
  `WANDB_DIR=/tmp/wandb`) is injected. Delphi's logging backend pushes its own run dir to
  `gs://$DELPHI_CKPT_DIR/<ckpt_dir>/<run>/{wandb,tb}/` at the checkpoint cadence —
  mid-run and crash-safe. The launcher no longer uploads after the run (it used to push
  the wandb dir on clean exit); the mechanism now lives in `delphi/log.py`
  `Logger.flush_to_gcs`.
- **Logs:** `--logging` defaults to `gs://misc-$GOOGLE_CLOUD_PROJECT/logs` (override with `logging=`).
- **After submit:** fire-and-forget (no `--wait`); prints each job-id and a ready-to-paste
  monitor command. Check status with **`dqueue [JOB_ID ...]`** — a thin `dstat050` wrapper
  with the invariant AoU flags baked in (no args lists all your jobs; ids show `--full`
  detail). Or `wait=true` for a single blocking debug job.

## Dry run

`dry=true` prints the generated job script and the exact `dsub050` command(s) without
submitting (and tolerates a machine without the AoU env, for debugging on the cluster):

```bash
dsubmit apps/train-delphi-m4.py -- dry=true gpu_num=2 machine_type=n1-standard-8 sweep.lr='[1e-3,1e-4]'
```

## Config reference (`DsubConfig`)

| Field | Default | Meaning |
|---|---|---|
| `time` | `0` (uncapped) | walltime hours → `--timeout`; **default omits it** (run to completion, dsub ~7-day cap). Set e.g. `time=24` to cap. |
| `gpu` / `gpu_num` / `gpu_type` | `true` / `1` / `t4` | `--accelerator-*`; `gpu_num>1` → `torchrun` |
| `machine_type` | `n1-standard-8` | `--machine-type` (T4 needs `n1`) |
| `boot_disk_size` | `80` | GB; must fit the CUDA image + cloudpathlib cache |
| `image_repo` / `tag` / `resolve_digest` / `image` | `shawnclarkefan/delphi` / `latest` / `true` / – | image + digest pinning |
| `job_name` | script stem | `--name` |
| `overrides` / `sweep.*` | – | fan-out axes |
| `project` / `service_account` / `logging` / `data_dir` / `ckpt_dir` | from env/`.env` | identity + GCS paths |
| `dataset` | `aou` | `--env DELPHI_DATASET` |
| `wandb_mode` | `offline` | `--env WANDB_MODE` |
| `wait` | `false` | block on the job (debug) |
| `dry` | `false` | print, don't submit |
| `env_file` | `./.env`, `~/repos/delphi/.env` | where to read defaults |
