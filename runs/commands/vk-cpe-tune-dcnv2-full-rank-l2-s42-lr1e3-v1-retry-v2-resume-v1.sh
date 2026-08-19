#!/usr/bin/env bash
set -euo pipefail
cd /workspace/codex-home/dcdn/deep_division_network
source scripts/env.sh
exec "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_worker.py --project-root /workspace/codex-home/dcdn/deep_division_network --registry-root runs/registry --run-id vk-cpe-tune-dcnv2-full-rank-l2-s42-lr1e3-v1-retry-v2-resume-v1 --gpu 1 --resolved configs/runs/vk-cpe-tune-dcnv2-full-rank-l2-s42-lr1e3-v1-retry-v2-resume-v1.yaml --poll-seconds 5
