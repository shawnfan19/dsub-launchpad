"""dsub_launchpad — submit Delphi jobs to AoU Google Batch via dsub.

Mirrors the `slurm_submit` (`submit`) CLI so the same muscle memory works on both
platforms:

    dsubmit <script + args...> -- <launcher cfg...>

Everything before ``--`` is the script and its args (run as ``python <that> $OVERRIDE``
inside the container); everything after ``--`` is OmegaConf dotlist config merged over
:class:`DsubConfig` (with ``config=file.yaml`` for a config file).

Sweeps fan out into N independent ``dsub050`` jobs, one per ``itertools.product`` combo;
each combo is injected as ``--env OVERRIDE="k=v ..."`` (dsub has no positional passthrough).
"""

import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

# gpu_type -> dsub --accelerator-type value. AoU's default-quota GPU is t4.
ACCELERATOR_TYPES = {
    "t4": "nvidia-tesla-t4",
    "a100": "nvidia-tesla-a100",
    "v100": "nvidia-tesla-v100",
    "p100": "nvidia-tesla-p100",
    "l4": "nvidia-l4",
}


@dataclass
class DsubConfig:
    # --- shared with slurm_submit RunConfig (same names => same syntax) ---
    dry: bool = False
    script_with_args: str = "apps/train-delphi-m4.py"
    overrides: Optional[str] = None
    time: float = 3.0  # walltime in hours -> --timeout
    job_name: Optional[str] = None  # -> --name (default: script stem)
    gpu: bool = True
    gpu_num: int = 1  # -> --accelerator-count (and torchrun --nproc-per-node if >1)
    gpu_type: str = "t4"  # -> --accelerator-type

    # --- dsub / AoU specific ---
    machine_type: str = "n1-standard-8"  # GCP-native sizing (T4 requires n1)
    boot_disk_size: int = 80  # GB; must fit the CUDA image + cloudpathlib cache
    regions: str = "us-central1"
    wait: bool = False  # block until the (single) job finishes; for debugging

    # image + tag->digest resolution
    image: Optional[str] = (
        None  # explicit ref (repo:tag or repo@sha256); bypasses below
    )
    image_repo: str = "shawnclarkefan/delphi"
    tag: str = "latest"
    resolve_digest: bool = (
        True  # resolve repo:tag -> repo@sha256 (reproducible, cache-safe)
    )

    # GCS paths / identity (default from environment / .env, all overridable)
    project: Optional[str] = None  # default $GOOGLE_CLOUD_PROJECT
    service_account: Optional[str] = None  # default $GOOGLE_SERVICE_ACCOUNT_EMAIL
    logging: Optional[str] = None  # default gs://$CKPT_BUCKET/dsub/logs
    data_dir: Optional[str] = (
        None  # -> --env DELPHI_DATA_DIR (default $DELPHI_DATA_DIR)
    )
    ckpt_dir: Optional[str] = (
        None  # -> --env DELPHI_CKPT_DIR (default $DELPHI_CKPT_DIR)
    )
    dataset: str = (
        "aou"  # -> --env DELPHI_DATASET (explicit; auto-detect over gs:// is flaky)
    )
    wandb_mode: str = "offline"  # -> --env WANDB_MODE (Batch VMs have no internet)

    # AoU plumbing (constant per workspace; rarely overridden)
    network: str = "global/networks/network"
    subnetwork: str = "regions/us-central1/subnetworks/subnetwork"
    dsub_bin: str = "dsub050"  # the pinned 0.5.0; 0.5.2 breaks GPU jobs
    env_file: Optional[str] = None  # .env to read for defaults (acquisition mode B)


# --------------------------------------------------------------------------- #
# .env loading (acquisition mode B: don't depend on the submitter's shell)
# --------------------------------------------------------------------------- #
def load_env_file(env_file: Optional[str]) -> Optional[Path]:
    """Populate os.environ from a .env, mirroring delphi.env.load_env_file semantics.

    os.environ wins on conflict (setdefault). Returns the file used, or None.
    """
    if env_file:
        candidates = [Path(env_file).expanduser()]
    else:
        candidates = [Path.cwd() / ".env", Path.home() / "repos" / "delphi" / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return path
    return None


# --------------------------------------------------------------------------- #
# image / digest resolution
# --------------------------------------------------------------------------- #
def dockerhub_digest(repo: str, tag: str) -> str:
    """Resolve a Docker Hub tag to its immutable @sha256 digest (public repos)."""
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.load(resp)
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable message
        raise RuntimeError(
            f"could not query Docker Hub for {repo}:{tag} ({e}). "
            "Repo private, tag missing, or no network?"
        )
    digest = data.get("digest")
    if not digest:
        raise RuntimeError(f"no digest for {repo}:{tag} (private repo or missing tag?)")
    return digest


def resolve_image(cfg: DsubConfig, dry: bool) -> str:
    """Build the fully-qualified --image ref, routed through the AoU proxy.

    With resolve_digest, the tag is pinned to an immutable @sha256 at submit time
    (always-newest + reproducible + proxy-cache-safe). We resolve in dry mode too
    so the printed command shows the real digest; if Docker Hub is unreachable, a
    real submit hard-fails (never silently ships a mutable tag) while --dry falls
    back to :tag with a note.
    """
    proxy = os.environ.get("ARTIFACT_REGISTRY_DOCKER_REPO")
    if not proxy:
        if not dry:
            raise SystemExit(
                "ARTIFACT_REGISTRY_DOCKER_REPO is not set — run inside the AoU Workbench."
            )
        proxy = (
            "${ARTIFACT_REGISTRY_DOCKER_REPO}"  # placeholder so --dry stays readable
        )

    if cfg.image:
        ref = cfg.image
    elif not cfg.tag:
        raise SystemExit(
            "empty image tag — set tag= or image= (this is the empty-tag bug)"
        )
    elif cfg.resolve_digest:
        try:
            ref = f"{cfg.image_repo}@{dockerhub_digest(cfg.image_repo, cfg.tag)}"
        except RuntimeError as e:
            if not dry:
                raise SystemExit(str(e))
            print(f"[dry] {e}")
            print(f"[dry] showing :{cfg.tag} (a real submit resolves it to @sha256)")
            ref = f"{cfg.image_repo}:{cfg.tag}"
    else:
        ref = f"{cfg.image_repo}:{cfg.tag}"

    return ref if ref.startswith(proxy) else f"{proxy}/{ref}"


# --------------------------------------------------------------------------- #
# sweep / override axes (copied semantics from slurm_submit)
# --------------------------------------------------------------------------- #
def flatten_sweep(node: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in node.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_sweep(v, key))
        else:
            flat[key] = v
    return flat


def parse_sweep(flag: str, source) -> list[str]:
    if isinstance(source, str):
        with open(source, "r") as f:
            values = yaml.safe_load(f)
    elif isinstance(source, list):
        values = source
    else:
        raise ValueError(
            f"sweep.{flag} must be a YAML file path or an inline list, got: {source!r}"
        )
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"sweep values for '{flag}' must be a non-empty list")
    return [f"{flag}={'null' if v is None else v}" for v in values]


def build_axes(cfg: DsubConfig, sweeps: dict) -> list[str]:
    """Return the list of combined override strings (one per product combo)."""
    overrides = []
    if cfg.overrides is not None:
        with open(cfg.overrides, "r") as f:
            overrides = yaml.safe_load(f) or []

    axes = []
    if overrides:
        if isinstance(overrides, list):
            axes.append(overrides)
        elif isinstance(overrides, dict):
            for k, vs in overrides.items():
                if not isinstance(vs, list) or len(vs) == 0:
                    raise ValueError(
                        f"override values for '{k}' must be a non-empty list"
                    )
                axes.append([f"{k}={'null' if v is None else v}" for v in vs])
        else:
            raise ValueError(
                f"overrides file must be a list or dict, got {type(overrides)}"
            )

    for flag, source in sweeps.items():
        axes.append(parse_sweep(flag, source))

    if not axes:
        return [""]
    return [" ".join(str(t) for t in combo) for combo in itertools.product(*axes)]


# --------------------------------------------------------------------------- #
# command + script construction
# --------------------------------------------------------------------------- #
def sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-") or "job"
    if not name[0].isalpha():
        name = "j-" + name
    return name[:50]


def parse_timeout(time_in_hours: float) -> str:
    return f"{int(round(time_in_hours * 3600))}s"


def build_job_script(cfg: DsubConfig, script_tokens: list[str]) -> str:
    """One script reused across all combos; per-combo args arrive via $OVERRIDE.

    Script-side tokens are shell-quoted so spaces/$VAR/$(...) etc. reach python
    literally; $OVERRIDE is intentionally left UNQUOTED so a multi-token sweep
    override word-splits into separate argv (the dsub analog of `sbatch "$@"`).
    """
    quoted = " ".join(shlex.quote(t) for t in script_tokens)
    launcher = (
        f"torchrun --standalone --nproc-per-node={cfg.gpu_num}"
        if cfg.gpu and cfg.gpu_num > 1
        else "python"
    )
    lines = [
        "#!/bin/bash",
        "set -e",
        "source /entrypoint.sh",  # cd /workspace/Delphi + GPU detect (no .env in container)
        f"{launcher} {quoted} ${{OVERRIDE:-}}",  # OVERRIDE unset (no sweep) -> empty
    ]
    if cfg.wandb_mode == "offline":
        # Offline wandb writes to $WANDB_DIR/wandb/ (ephemeral on the VM). Push it
        # to the gs:// checkpoint dir via cloudpathlib — gsutil is NOT in the image,
        # but cloudpathlib[gs] is. `|| echo` keeps a push hiccup from failing the run;
        # we do NOT swallow stderr, so failures show in the --logging stream.
        lines += [
            "python - <<'PY' || echo '[wandb] offline-dir upload failed (non-fatal)'",
            "import os",
            "from cloudpathlib import AnyPath",
            "src = os.path.join(os.environ.get('WANDB_DIR', '/tmp/wandb'), 'wandb')",
            "root = os.environ.get('DELPHI_CKPT_DIR')",
            "if os.path.isdir(src) and root:",
            "    (AnyPath(root) / 'wandb').upload_from(src)",
            "    print('[wandb] uploaded offline runs from', src)",
            "else:",
            "    print('[wandb] nothing to upload at', src)",
            "PY",
        ]
    return "\n".join(lines) + "\n"


def build_env_pairs(
    cfg: DsubConfig, project, data_dir, ckpt_dir
) -> list[tuple[str, str]]:
    pairs = [
        ("GOOGLE_CLOUD_PROJECT", project),
        ("DELPHI_DATA_DIR", data_dir),
        ("DELPHI_CKPT_DIR", ckpt_dir),
        ("DELPHI_DATASET", cfg.dataset),
    ]
    if cfg.wandb_mode:
        pairs.append(("WANDB_MODE", cfg.wandb_mode))
        if cfg.wandb_mode == "offline":
            pairs.append(("WANDB_DIR", "/tmp/wandb"))
    return [(k, v) for k, v in pairs if v]


def build_dsub_command(
    cfg: DsubConfig,
    image: str,
    logging_path: str,
    project: str,
    service_account: str,
    env_pairs: list[tuple[str, str]],
    script_path: str,
) -> list[str]:
    cmd = [
        cfg.dsub_bin,
        "--provider",
        "google-batch",
        "--project",
        project,
        "--user-project",
        project,
        "--regions",
        cfg.regions,
        "--service-account",
        service_account,
        "--network",
        cfg.network,
        "--subnetwork",
        cfg.subnetwork,
        "--use-private-address",
        "--machine-type",
        cfg.machine_type,
        "--boot-disk-size",
        str(cfg.boot_disk_size),
        "--image",
        image,
        "--logging",
        logging_path,
    ]
    # time<=0 omits --timeout -> the job runs to completion (dsub default cap
    # ~7 days, plus AoU Batch quota). Set a generous time= for long training.
    if cfg.time and cfg.time > 0:
        cmd += ["--timeout", parse_timeout(cfg.time)]
    if cfg.job_name:
        cmd += ["--name", sanitize_name(cfg.job_name)]
    if cfg.gpu and cfg.gpu_num > 0:
        if cfg.gpu_type not in ACCELERATOR_TYPES:
            raise SystemExit(
                f"unknown gpu_type={cfg.gpu_type!r}; known: {sorted(ACCELERATOR_TYPES)}"
            )
        cmd += [
            "--accelerator-type",
            ACCELERATOR_TYPES[cfg.gpu_type],
            "--accelerator-count",
            str(cfg.gpu_num),
        ]
    for key, value in env_pairs:
        cmd += ["--env", f"{key}={value}"]
    if cfg.wait:
        cmd += ["--wait"]
    cmd += ["--script", script_path]
    return cmd


def resolve_service_account(cfg: DsubConfig, dry: bool) -> str:
    if cfg.service_account:
        return cfg.service_account
    sa = os.environ.get("GOOGLE_SERVICE_ACCOUNT_EMAIL")
    if sa:
        return sa
    try:
        sa = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        sa = ""
    if not sa and not dry:
        raise SystemExit(
            "no service account: set service_account=, $GOOGLE_SERVICE_ACCOUNT_EMAIL, "
            "or `gcloud config set account`."
        )
    return sa or "${GOOGLE_SERVICE_ACCOUNT_EMAIL}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        script_args, launcher_args = argv[:i], argv[i + 1 :]
    else:
        script_args, launcher_args = argv, []

    default_cfg = OmegaConf.structured(DsubConfig)
    cli_cfg = OmegaConf.from_cli(launcher_args)

    if hasattr(cli_cfg, "config"):
        file_cfg = OmegaConf.load(cli_cfg.config)
        del cli_cfg.config
    else:
        file_cfg = OmegaConf.create({})

    sweeps = {}
    if hasattr(cli_cfg, "sweep"):
        if not OmegaConf.is_dict(cli_cfg.sweep):
            raise ValueError(
                "sweep expects sweep.<flag>=<values.yaml|[inline,list]>, "
                "e.g. sweep.lr=lrs.yaml"
            )
        sweeps = flatten_sweep(OmegaConf.to_object(cli_cfg.sweep))  # type: ignore
        raw_keys = {
            a[len("sweep.") :].split("=", 1)[0]
            for a in launcher_args
            if a.startswith("sweep.") and "=" in a
        }
        lost = raw_keys - set(sweeps)
        if lost:
            raise ValueError(
                f"conflicting sweep flags {sorted(lost)}: a sweep key cannot also be a "
                "prefix of another sweep key"
            )
        del cli_cfg.sweep

    try:
        cfg: DsubConfig = OmegaConf.to_object(
            OmegaConf.merge(default_cfg, file_cfg, cli_cfg)  # type: ignore
        )
    except ConfigKeyError as e:
        raise SystemExit(
            f"unknown config key ({e}). Note: sweep.* and config= must be passed "
            "on the CLI, not inside a config=file.yaml."
        )

    # Script tokens: prefer the positional list (preserves arg boundaries for
    # shell-quoting); otherwise split the script_with_args config string.
    if script_args:
        script_tokens = list(script_args)
        cfg.script_with_args = " ".join(script_args)
    else:
        script_tokens = shlex.split(cfg.script_with_args)
    if not script_tokens:
        raise SystemExit(
            "no script to run — pass a script before `--`, or set script_with_args="
        )
    if not cfg.job_name:
        cfg.job_name = Path(script_tokens[0]).stem

    # Acquisition mode B: load .env ourselves so we don't depend on the shell.
    used = load_env_file(cfg.env_file)
    if used:
        print(f"loaded env defaults from {used}")

    project = cfg.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    data_dir = cfg.data_dir or os.environ.get("DELPHI_DATA_DIR")
    ckpt_dir = cfg.ckpt_dir or os.environ.get("DELPHI_CKPT_DIR")
    ckpt_bucket = os.environ.get("CKPT_BUCKET")

    logging_path = cfg.logging or (
        f"gs://{ckpt_bucket}/dsub/logs" if ckpt_bucket else None
    )

    if not cfg.dry:
        missing = [
            n
            for n, v in [
                ("project/GOOGLE_CLOUD_PROJECT", project),
                ("data_dir/DELPHI_DATA_DIR", data_dir),
                ("ckpt_dir/DELPHI_CKPT_DIR", ckpt_dir),
                ("logging/CKPT_BUCKET", logging_path),
            ]
            if not v
        ]
        if missing:
            raise SystemExit(
                "missing required values: "
                + ", ".join(missing)
                + ". `source .env` or pass env_file=/the matching flags."
            )
    else:  # keep --dry readable on a machine without the AoU env
        project = project or "${GOOGLE_CLOUD_PROJECT}"
        data_dir = data_dir or "${DELPHI_DATA_DIR}"
        ckpt_dir = ckpt_dir or "${DELPHI_CKPT_DIR}"
        logging_path = logging_path or "gs://${CKPT_BUCKET}/dsub/logs"

    # Invariant: both branches above guarantee these are set (dry -> placeholders,
    # non-dry -> raised on any missing). Narrow for the type checker.
    assert project and data_dir and ckpt_dir and logging_path

    image = resolve_image(cfg, cfg.dry)
    service_account = resolve_service_account(cfg, cfg.dry)
    env_base = build_env_pairs(cfg, project, data_dir, ckpt_dir)
    combos = build_axes(cfg, sweeps)

    script_text = build_job_script(cfg, script_tokens)
    tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else "."
    fd, script_path = tempfile.mkstemp(
        prefix="dsub_", suffix=".sh", dir=tmp_dir, text=True
    )
    with os.fdopen(fd, "w") as f:
        f.write(script_text)
    os.chmod(script_path, 0o755)
    print(f"job script written to {script_path}:")
    print("  " + script_text.replace("\n", "\n  ").rstrip())

    summary = (
        f"[image:{image.rsplit('/', 1)[-1]} machine:{cfg.machine_type} "
        f"gpu:{cfg.gpu_type if cfg.gpu else 'none'}x{cfg.gpu_num if cfg.gpu else 0} "
        f"time:{f'{cfg.time}h' if cfg.time and cfg.time > 0 else 'uncapped'}]"
    )
    print(f"{summary} {len(combos)} job(s)")

    job_ids = []
    try:
        for combo in combos:
            env_pairs = env_base + ([("OVERRIDE", combo)] if combo else [])
            cmd = build_dsub_command(
                cfg,
                image,
                logging_path,
                project,
                service_account,
                env_pairs,
                script_path,
            )
            label = f"  override: {combo}" if combo else "  (no sweep)"
            print(label)
            if cfg.dry:
                print("  " + " ".join(shlex.quote(c) for c in cmd))
                continue
            result = subprocess.run(cmd, capture_output=True, text=True)
            sys.stdout.write(result.stdout)
            if result.returncode != 0:
                sys.stderr.write(result.stderr)
                raise SystemExit(f"dsub failed (exit {result.returncode})")
            for line in result.stdout.splitlines():
                if "Launched job-id:" in line:
                    job_ids.append(line.split(":", 1)[1].strip())
    finally:
        if cfg.dry:
            print(f"(dry) job script kept at {script_path}")
        else:
            os.unlink(script_path)

    if job_ids:
        ids = " ".join(f"'{j}'" for j in job_ids)
        print(f"\nlaunched {len(job_ids)} job(s). Monitor:")
        print(
            f"  dstat050 --provider google-batch --project {project} "
            f"--location {cfg.regions} --jobs {ids} --status '*'"
        )


if __name__ == "__main__":
    main()
