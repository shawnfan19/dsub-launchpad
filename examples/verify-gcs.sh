#!/bin/bash
# Verify the CONTAINER side end-to-end before trusting the launcher:
#   - forwarded env vars reach delphi.env
#   - in-container gs:// READ (data bucket) under ADC (pet service account)
#   - in-container gs:// WRITE (checkpoint bucket) under ADC
#
# Submit it directly (NOT via dsubmit — it runs bash, not `python <script>`):
#
#   source .env
#   dsub050 --provider google-batch \
#     --project $GOOGLE_CLOUD_PROJECT --user-project $GOOGLE_CLOUD_PROJECT --regions us-central1 \
#     --service-account ${GOOGLE_SERVICE_ACCOUNT_EMAIL:-$(gcloud config get-value account)} \
#     --network global/networks/network --subnetwork regions/us-central1/subnetworks/subnetwork \
#     --use-private-address \
#     --machine-type n1-standard-1 --boot-disk-size 80 \
#     --image $ARTIFACT_REGISTRY_DOCKER_REPO/shawnclarkefan/delphi:latest \
#     --env DELPHI_DATA_DIR=$DELPHI_DATA_DIR \
#     --env DELPHI_CKPT_DIR=$DELPHI_CKPT_DIR \
#     --env GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT \
#     --env DELPHI_DATASET=aou \
#     --logging gs://$CKPT_BUCKET/dsub/logs \
#     --script examples/verify-gcs.sh --wait
#
# VERIFY OK  -> the I/O + auth design is proven; build/run real jobs with confidence.
# fail READ  -> pet SA lacks IAM on the data bucket (or metadata/ADC unreachable).
# fail WRITE -> pet SA lacks IAM on the checkpoint bucket.
set -e
source /entrypoint.sh
python - <<'PY'
import os
from cloudpathlib import AnyPath
from delphi import env

print("forwarded DATA =", os.environ.get("DELPHI_DATA_DIR"))
print("forwarded CKPT =", os.environ.get("DELPHI_CKPT_DIR"))
print("resolved  DATA =", env.DELPHI_DATA_DIR)
print("resolved  CKPT =", env.DELPHI_CKPT_DIR)

# READ test: list the data dir (proves list auth) then read the first object
# found (proves get auth) — no assumption about which files exist in the bucket.
data = AnyPath(env.DELPHI_DATA_DIR)
top = list(data.iterdir())
print("READ/list ok:", [p.name for p in top[:10]] or "(empty)")
target = next((p for p in top if p.is_file()), None)
if target is None:
    for d in (p for p in top if p.is_dir()):
        target = next((p for p in d.iterdir() if p.is_file()), None)
        if target:
            break
if target is not None:
    print("READ file ok:", target.name, f"({len(target.read_bytes())} bytes)")
else:
    print("READ file: list worked but found no object to read")

# WRITE test (the checkpoint bucket, exactly what the patched Checkpointer uses).
probe = AnyPath(env.DELPHI_CKPT_DIR) / "dsub_verify_probe.txt"
probe.write_text("ok")
assert probe.read_text() == "ok"
probe.unlink()
print("WRITE ok")

print("VERIFY OK")
PY
