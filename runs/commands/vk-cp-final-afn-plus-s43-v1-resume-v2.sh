#!/usr/bin/env bash
set -euo pipefail
cd /workspace/codex-home/dcdn/deep_division_network
source scripts/env.sh
exec "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_worker.py --project-root /workspace/codex-home/dcdn/deep_division_network --registry-root runs/registry --run-id vk-cp-final-afn-plus-s43-v1-resume-v2 --gpu 0 --resolved configs/runs/vk-cp-final-afn-plus-s43-v1-resume-v2.yaml --poll-seconds 5
