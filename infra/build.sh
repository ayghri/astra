#!/usr/bin/env bash
# Build Docker image then convert to Apptainer SIF.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=bonsainet:latest
SIF="${REPO_ROOT}/infra/bonsainet.sif"

echo "==> Building Docker image ${IMAGE}"
docker build -t "${IMAGE}" -f "${REPO_ROOT}/infra/Dockerfile" "${REPO_ROOT}"

echo "==> Converting to Apptainer SIF: ${SIF}"
apptainer build --fakeroot "${SIF}" "${REPO_ROOT}/infra/bonsainet.def"

echo "==> Done: ${SIF}"
