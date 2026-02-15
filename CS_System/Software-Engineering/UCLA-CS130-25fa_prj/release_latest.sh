#!/usr/bin/env bash
# release_latest.sh — promote :latest → :release in Artifact Registry

set -euo pipefail  # Exit on error, unset var, or pipeline failure

PROJECT="semiconductor-savants"   # GCP project ID
REPO="semiconductor-savants"      # Artifact Registry repo name
IMAGE="semiconductor-savants"     # Image name inside repo
REGION="us"                       # Artifact Registry region (for gcr.io)
INSTANCE="web-server"             # GCE instance name to restart
ZONE="us-west1-b"                 # GCE zone for the instance
PULL_LATEST=true                   # Default: pull latest before tagging
RESTART_VM=true                    # Default: restart VM to pick up :release

# Parse optional flag to skip pulling
# Flags
for arg in "$@"; do
  case "$arg" in
    --no-pull) PULL_LATEST=false ;;
    --no-restart) RESTART_VM=false ;;
    --instance=*) INSTANCE="${arg#*=}" ;;
    --zone=*) ZONE="${arg#*=}" ;;
    --project=*) PROJECT="${arg#*=}" ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Define full image paths
REMOTE="gcr.io/${PROJECT}/${REPO}"
SRC="${REMOTE}:latest"
DST="${REMOTE}:release"

# Optionally pull the latest image
if $PULL_LATEST; then
  echo "Pulling latest image..."
  docker pull "$SRC"
else
  echo "Skipping image pull (--no-pull specified)"
fi

# Tag and push the release version
echo "Tagging $SRC → $DST"
docker tag "$SRC" "$DST"

echo "Pushing release tag..."
docker push "$DST"

echo "Successfully promoted :latest → :release"

# Restart VM to pick up the new :release image unless disabled
if $RESTART_VM; then
  echo "Restarting VM ${INSTANCE} in ${ZONE} to use ${DST}..."
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud not found in PATH; cannot restart VM" >&2
    exit 3
  fi
  gcloud compute instances update-container "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --container-image="${DST}"
  echo "VM restart command issued."
else
  echo "Skipping VM restart (--no-restart specified)"
fi
