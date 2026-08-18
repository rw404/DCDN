#!/usr/bin/env bash
set -euo pipefail
cd /workspace/codex-home/dcdn/deep_division_network
source scripts/env.sh
exec "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_worker.py --project-root /workspace/codex-home/dcdn/deep_division_network --registry-root runs/registry --run-id vk-cp-smoke-dcnv2-full-rank-l2-s42-1eb-v1 --gpu 1 --resolved configs/runs/vk-cp-smoke-dcnv2-full-rank-l2-s42-1eb-v1.yaml --poll-seconds 5
