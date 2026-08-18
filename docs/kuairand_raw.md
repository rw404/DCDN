# KuaiRand-Pure raw-source checkpoint

The raw release was acquired from the only frozen route:

`https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`

The archive is regenerable public data and remains only under
`/home/codex/.cache/dcdn-public`. No raw bytes are stored in the persistent
repository.

## Reproduction

Run the local guards before acquisition:

```bash
python3 scripts/cost_guard.py --eta-hours 0.5 --phase compute \
  --work-class mandatory --json
python3 scripts/storage_guard.py --project-root "$PWD" --overlay-path / --json
```

Download or resume only the frozen URL into the ephemeral download directory:

```bash
install -d -m 700 /home/codex/.cache/dcdn-public/downloads/kuairand_pure
curl --disable --fail --silent --show-error --location \
  --proto '=https' --proto-redir '=https' \
  --retry 5 --retry-all-errors \
  --continue-at - \
  --output /home/codex/.cache/dcdn-public/downloads/kuairand_pure/KuaiRand-Pure.tar.gz \
  https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
```

Then run the fail-closed verifier/extractor in the isolated environment. The
target extraction directory must be absent or empty:

```bash
source scripts/env.sh
UV_OFFLINE=1 uv run --frozen python scripts/data/kuairand_pure_raw.py \
  --archive /home/codex/.cache/dcdn-public/downloads/kuairand_pure/KuaiRand-Pure.tar.gz \
  --extract-root /home/codex/.cache/dcdn-public/raw/kuairand_pure \
  --manifest-output data/manifests/kuairand_pure_source_v1.yaml \
  --extract
```

## Verified boundary

- archive bytes: `47432272`;
- archive MD5: `0820331067a3784d9691136f772b35a7`;
- archive SHA-256:
  `c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0`;
- no traversal, absolute path, symlink, hardlink, or special tar member;
- extracted raw files are byte-hashed, while CSV inspection is limited to
  each first header line;
- no row values, dates, label values, `is_rand` counts/distributions,
  README/results, validation metrics, test labels, or test metrics were read;
- `video_features_statistic_pure.csv` is explicitly excluded from the strict
  causal experiment;
- `test_unlocked=false` remains in force.
- the source manifest and exact-name SHA-256 sidecar are written through
  fsynced same-directory temporary files and atomically replaced; readers
  reject noncanonical sidecars, duplicate YAML keys, symlinks, and hardlinks.

The durable metadata is
`data/manifests/kuairand_pure_source_v1.yaml`, with SHA-256
`bc5b441fdd3188d033e0ecbf0bdfd73f8bcf07bbc744d474d260ee220ae4d6e9`.
