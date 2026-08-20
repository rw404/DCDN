#!/usr/bin/env bash
set -euo pipefail
cd /workspace/codex-home/dcdn/deep_division_network
source scripts/env.sh
exec "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_worker.py --project-root /workspace/codex-home/dcdn/deep_division_network --registry-root runs/registry --run-id vk-cpe-final-dcnv2-full-rank-l2-s44-v1-receipt-v3 --gpu 1 --resolved configs/runs/vk-cpe-final-dcnv2-full-rank-l2-s44-v1-receipt-v3.yaml --poll-seconds 5
