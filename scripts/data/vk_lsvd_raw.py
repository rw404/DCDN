#!/usr/bin/env python3
"""Acquire the frozen VK-LSVD raw allowlist without semantic test access.

Only the official ``deepvk/VK-LSVD`` Hugging Face dataset route is accepted.
The repository revision, object paths, sizes, LFS SHA-256 ETags, Git blob OIDs,
and Xet hashes are frozen in ``data/manifests/vk_lsvd_source_v1.yaml``.

This module deliberately has no Parquet/NumPy decoding path.  In particular,
week 26 may be transferred and mechanically hashed, but it is never opened as
a dataset; it is also excluded from ``unsealed_objects`` while
``test_unlocked`` is false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
    urlopen,
)

from dcdn.source_manifest import (
    SourceManifestError,
    load_source_manifest,
)

OFFICIAL_DATASET_ID = "deepvk/VK-LSVD"
OFFICIAL_DATASET_URL = "https://huggingface.co/datasets/deepvk/VK-LSVD"
OFFICIAL_API_ROOT = "https://huggingface.co/api/datasets/deepvk/VK-LSVD"
DEFAULT_CACHE_ROOT = Path("/home/codex/.cache/dcdn-public/raw/vk_lsvd")
MIN_OVERLAY_AVAILABLE_BYTES = 20 * 1024**3
MAX_OVERLAY_USAGE_FRACTION = 0.85
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
USER_AGENT = "dcdn-vk-lsvd-raw/1.0"
TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


class AcquisitionError(RuntimeError):
    """Raised when acquisition would violate the frozen source contract."""


@dataclass(frozen=True)
class SourceObject:
    """One immutable LFS object in the acquisition allowlist."""

    role: str
    path: str
    size_bytes: int
    sha256: str
    etag: str
    git_blob_oid: str
    xet_hash: str
    last_commit: str
    content_access: str
    week: int | None = None
    split: str | None = None

    @property
    def sealed(self) -> bool:
        return self.content_access == "sealed_hash_only"


@dataclass(frozen=True)
class SourceManifest:
    """Validated immutable VK-LSVD source manifest."""

    path: Path
    digest: str
    revision: str
    test_unlocked: bool
    week26_sealed: bool
    objects: tuple[SourceObject, ...]
    raw: Mapping[str, Any]

    @property
    def total_bytes(self) -> int:
        return sum(obj.size_bytes for obj in self.objects)

    def unsealed_objects(self) -> tuple[SourceObject, ...]:
        """Return non-test objects without opening or decoding their bytes."""

        return tuple(obj for obj in self.objects if not obj.sealed)


def _sha256_bytes(path: Path) -> str:
    """Hash bytes without interpreting their file format or row contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_etag(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:]
    return normalized.strip('"')


def _require_hash(value: Any, field: str, *, length: int = 64) -> str:
    pattern = SHA256_RE if length == 64 else REVISION_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise AcquisitionError(f"{field} must be {length} lowercase hex characters")
    return value


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AcquisitionError("object path must be a non-empty string")
    if "\\" in value:
        raise AcquisitionError(f"object path uses a backslash: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise AcquisitionError(f"unsafe object path: {value!r}")
    return value


def _parse_object(payload: Mapping[str, Any]) -> SourceObject:
    size = payload.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise AcquisitionError("object bytes must be a positive integer")
    week = payload.get("week")
    if week is not None and (
        not isinstance(week, int) or isinstance(week, bool) or not 0 <= week <= 26
    ):
        raise AcquisitionError("interaction week must be an integer in [0, 26]")
    split = payload.get("split")
    if split is not None and split not in {"train", "validation", "test"}:
        raise AcquisitionError(f"invalid split: {split!r}")
    content_access = payload.get("content_access")
    if content_access not in {"not_decoded", "sealed_hash_only"}:
        raise AcquisitionError(f"invalid content_access: {content_access!r}")
    sha256 = _require_hash(payload.get("sha256"), "object.sha256")
    etag = _normalize_etag(str(payload.get("etag", "")))
    _require_hash(etag, "object.etag")
    if etag != sha256:
        raise AcquisitionError("LFS ETag must equal the frozen content SHA-256")
    return SourceObject(
        role=str(payload.get("role", "")),
        path=_safe_relative_path(payload.get("path")),
        size_bytes=size,
        sha256=sha256,
        etag=etag,
        git_blob_oid=_require_hash(
            payload.get("git_blob_oid"), "object.git_blob_oid", length=40
        ),
        xet_hash=_require_hash(payload.get("xet_hash"), "object.xet_hash"),
        last_commit=_require_hash(
            payload.get("last_commit"), "object.last_commit", length=40
        ),
        content_access=content_access,
        week=week,
        split=split,
    )


def _validate_exact_object_set(objects: Sequence[SourceObject]) -> None:
    if len(objects) != 30:
        raise AcquisitionError("VK-LSVD allowlist must contain exactly 30 objects")
    paths = [obj.path for obj in objects]
    collision_keys = [unicodedata.normalize("NFC", path).casefold() for path in paths]
    if len(paths) != len(set(paths)) or len(paths) != len(set(collision_keys)):
        raise AcquisitionError(
            "VK-LSVD allowlist contains duplicate or case-colliding paths"
        )

    interactions = [obj for obj in objects if obj.week is not None]
    if len(interactions) != 27 or {obj.week for obj in interactions} != set(range(27)):
        raise AcquisitionError("interaction allowlist must contain weeks 0 through 26")
    for obj in interactions:
        assert obj.week is not None
        if obj.week <= 24:
            split = "train"
        elif obj.week == 25:
            split = "validation"
        else:
            split = "test"
        expected = f"subsamples/up0.01_ir0.01/{split}/week_{obj.week:02d}.parquet"
        if obj.path != expected or obj.split != split:
            raise AcquisitionError(f"unexpected week-{obj.week} path or split")
        if obj.week == 26:
            if obj.role != "sealed_test" or not obj.sealed:
                raise AcquisitionError("week 26 must be sealed_hash_only")
        elif obj.sealed:
            raise AcquisitionError("only week 26 may be sealed")

    expected_metadata = {
        "metadata/item_embeddings.npz": (
            2_362_338_243,
            "6a4024ad7623f5d8150e59c5faaaf42fb7d7f48e2beec44c6917e5e57debea7c",
        ),
        "metadata/items_metadata.parquet": (
            155_747_843,
            "607ffd09d3fb6627f7eec07ed53583d45c47263a2e6538e4deee66cd8a22e51e",
        ),
        "metadata/users_metadata.parquet": (
            72_526_148,
            "51fa9528f71bb0da947311d7baa4727d325285a3e748b54816477b82492aff20",
        ),
    }
    metadata = {obj.path: obj for obj in objects if obj.week is None}
    if set(metadata) != set(expected_metadata):
        raise AcquisitionError(
            "metadata allowlist differs from the three frozen objects"
        )
    for path, (size, digest) in expected_metadata.items():
        obj = metadata[path]
        if obj.role != "metadata" or obj.size_bytes != size or obj.sha256 != digest:
            raise AcquisitionError(f"metadata contract mismatch: {path}")


def load_manifest(path: str | Path) -> SourceManifest:
    """Load the source manifest only after its conventional sidecar verifies."""

    source = Path(path).absolute()
    try:
        payload = load_source_manifest(source)
    except SourceManifestError as exc:
        raise AcquisitionError(f"invalid source manifest: {exc}") from exc
    digest = _sha256_bytes(source)
    if payload.get("schema_version") != 1:
        raise AcquisitionError("manifest schema_version must equal 1")
    if payload.get("dataset_id") != OFFICIAL_DATASET_ID:
        raise AcquisitionError("manifest dataset_id is not the official VK-LSVD repo")
    if payload.get("dataset_url") != OFFICIAL_DATASET_URL:
        raise AcquisitionError("manifest dataset_url is not the official HTTPS route")
    if payload.get("subset") != "up0.01_ir0.01":
        raise AcquisitionError("manifest subset must be up0.01_ir0.01")
    if payload.get("license") != "apache-2.0":
        raise AcquisitionError("manifest license must be apache-2.0")
    if type(payload.get("test_unlocked")) is not bool:
        raise AcquisitionError("test_unlocked must be a boolean")
    if payload["test_unlocked"]:
        raise AcquisitionError("raw acquisition manifest must keep test_unlocked=false")
    if payload.get("week26_sealed") is not True:
        raise AcquisitionError("week26_sealed must be true")
    revision = _require_hash(
        payload.get("resolved_revision"), "resolved_revision", length=40
    )
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not all(
        isinstance(item, Mapping) for item in raw_objects
    ):
        raise AcquisitionError("manifest objects must be a list of mappings")
    objects = tuple(_parse_object(item) for item in raw_objects)
    _validate_exact_object_set(objects)

    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise AcquisitionError("manifest totals must be a mapping")
    interaction_bytes = sum(obj.size_bytes for obj in objects if obj.week is not None)
    metadata_bytes = sum(obj.size_bytes for obj in objects if obj.week is None)
    expected_totals = {
        "interaction_objects": 27,
        "interaction_bytes": interaction_bytes,
        "metadata_objects": 3,
        "metadata_bytes": metadata_bytes,
        "all_objects": 30,
        "all_bytes": interaction_bytes + metadata_bytes,
    }
    if dict(totals) != expected_totals:
        raise AcquisitionError("manifest totals do not match object records")
    return SourceManifest(
        path=source,
        digest=digest,
        revision=revision,
        test_unlocked=False,
        week26_sealed=True,
        objects=objects,
        raw=payload,
    )


def resolver_url(manifest: SourceManifest, obj: SourceObject) -> str:
    """Return a revision-pinned official resolver URL for one object."""

    encoded = quote(obj.path, safe="/")
    return f"{OFFICIAL_DATASET_URL}/resolve/{manifest.revision}/{encoded}"


def _api_json(url: str) -> Any:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise AcquisitionError(f"refusing non-official API URL: {url}")
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=120) as response:  # noqa: S310 - host checked
        return json.load(response)


def _tree_url(revision: str, subtree: str) -> str:
    encoded_subtree = quote(subtree, safe="/")
    query = urlencode({"recursive": "true", "expand": "true"})
    return f"{OFFICIAL_API_ROOT}/tree/{revision}/{encoded_subtree}?{query}"


def _validate_remote_entries(
    manifest: SourceManifest, entries: Iterable[Mapping[str, Any]]
) -> None:
    files = {
        str(entry.get("path")): entry
        for entry in entries
        if entry.get("type") == "file"
    }
    expected_paths = {obj.path for obj in manifest.objects}
    if set(files) != expected_paths:
        missing = sorted(expected_paths - set(files))
        extra = sorted(set(files) - expected_paths)
        raise AcquisitionError(
            f"remote tree path mismatch; missing={missing}, extra={extra}"
        )
    for obj in manifest.objects:
        remote = files[obj.path]
        lfs = remote.get("lfs")
        last_commit = remote.get("lastCommit")
        if not isinstance(lfs, Mapping) or not isinstance(last_commit, Mapping):
            raise AcquisitionError(f"missing expanded LFS/commit metadata: {obj.path}")
        comparisons = {
            "size": (remote.get("size"), obj.size_bytes),
            "lfs.size": (lfs.get("size"), obj.size_bytes),
            "lfs.oid": (lfs.get("oid"), obj.sha256),
            "oid": (remote.get("oid"), obj.git_blob_oid),
            "xetHash": (remote.get("xetHash"), obj.xet_hash),
            "lastCommit.id": (last_commit.get("id"), obj.last_commit),
        }
        for field, (actual, expected) in comparisons.items():
            if actual != expected:
                raise AcquisitionError(
                    f"remote {field} mismatch for {obj.path}: "
                    f"{actual!r} != {expected!r}"
                )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _resolver_linked_headers(url: str) -> Mapping[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise AcquisitionError(f"refusing non-official resolver URL: {url}")
    opener = build_opener(_NoRedirect())
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        response = opener.open(request, timeout=120)
    except HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise AcquisitionError(f"resolver HEAD failed for {url}: {exc}") from exc
        return exc.headers
    with response:
        return response.headers


def audit_remote(manifest: SourceManifest) -> None:
    """Validate revision, tree metadata, sizes, and linked ETags without data GETs."""

    repo = _api_json(f"{OFFICIAL_API_ROOT}/revision/{manifest.revision}")
    if not isinstance(repo, Mapping):
        raise AcquisitionError("official repository API returned a non-mapping")
    checks = {
        "sha": manifest.revision,
        "private": False,
        "gated": False,
        "disabled": False,
    }
    for field, expected in checks.items():
        if repo.get(field) != expected:
            raise AcquisitionError(
                f"official repository {field} mismatch: {repo.get(field)!r}"
            )
    card_data = repo.get("cardData")
    if not isinstance(card_data, Mapping) or card_data.get("license") != "apache-2.0":
        raise AcquisitionError("official repository license is not apache-2.0")

    entries: list[Mapping[str, Any]] = []
    for subtree in ("subsamples/up0.01_ir0.01", "metadata"):
        payload = _api_json(_tree_url(manifest.revision, subtree))
        if not isinstance(payload, list) or not all(
            isinstance(item, Mapping) for item in payload
        ):
            raise AcquisitionError(f"invalid official tree response for {subtree}")
        entries.extend(payload)
    _validate_remote_entries(manifest, entries)

    for obj in manifest.objects:
        headers = _resolver_linked_headers(resolver_url(manifest, obj))
        commit = headers.get("X-Repo-Commit")
        linked_size = headers.get("X-Linked-Size")
        linked_etag = _normalize_etag(headers.get("X-Linked-ETag", ""))
        if commit != manifest.revision:
            raise AcquisitionError(f"resolver revision mismatch: {obj.path}")
        if linked_size != str(obj.size_bytes):
            raise AcquisitionError(f"resolver linked size mismatch: {obj.path}")
        if linked_etag != obj.etag:
            raise AcquisitionError(f"resolver linked ETag mismatch: {obj.path}")


def destination(cache_root: Path, obj: SourceObject) -> Path:
    root = cache_root.resolve()
    try:
        safe_path = _safe_relative_path(obj.path)
    except AcquisitionError as exc:
        raise AcquisitionError(f"object escapes cache root: {obj.path}") from exc
    target = root.joinpath(*PurePosixPath(safe_path).parts)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AcquisitionError(f"object escapes cache root: {obj.path}") from exc
    return target


def verify_object(path: Path, obj: SourceObject) -> str:
    """Verify exact size/SHA-256 while performing no format decoding."""

    if path.is_symlink() or not path.is_file():
        raise AcquisitionError(f"object must be a regular non-symlink file: {path}")
    metadata = path.stat()
    if metadata.st_nlink != 1:
        raise AcquisitionError(f"object must not be hard-linked: {path}")
    size = metadata.st_size
    if size != obj.size_bytes:
        raise AcquisitionError(
            f"size mismatch for {obj.path}: expected {obj.size_bytes}, got {size}"
        )
    digest = _sha256_bytes(path)
    if digest != obj.sha256:
        raise AcquisitionError(
            f"SHA-256 mismatch for {obj.path}: expected {obj.sha256}, got {digest}"
        )
    return digest


def _remaining_bytes(manifest: SourceManifest, cache_root: Path) -> int:
    remaining = 0
    for obj in manifest.objects:
        target = destination(cache_root, obj)
        partial = target.with_name(f"{target.name}.part")
        for candidate in (target, partial):
            if candidate.is_symlink() or (
                candidate.exists() and not candidate.is_file()
            ):
                raise AcquisitionError(f"unsafe local object path: {candidate}")
        existing = 0
        if target.is_file():
            size = target.stat().st_size
            if size > obj.size_bytes:
                raise AcquisitionError(f"local object exceeds frozen size: {target}")
            existing = size
        elif partial.is_file():
            size = partial.stat().st_size
            if size > obj.size_bytes:
                raise AcquisitionError(f"partial exceeds frozen size: {partial}")
            existing = size
        remaining += obj.size_bytes - existing
    return remaining


def guard_overlay_space(manifest: SourceManifest, cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(cache_root)
    unit = stats.f_frsize or stats.f_bsize
    total = unit * stats.f_blocks
    available = unit * stats.f_bavail
    remaining = _remaining_bytes(manifest, cache_root)
    projected_available = available - remaining
    projected_usage = 1.0 - projected_available / total
    if projected_available < MIN_OVERLAY_AVAILABLE_BYTES:
        raise AcquisitionError(
            "download would leave less than 20 GiB free on the temporary overlay"
        )
    if projected_usage >= MAX_OVERLAY_USAGE_FRACTION:
        raise AcquisitionError("download would raise temporary overlay usage to 85%")


def _credential_free_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in TOKEN_ENV_VARS:
        env.pop(name, None)
    env.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_DATASETS_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
    )
    return env


def _download_one(
    manifest: SourceManifest, cache_root: Path, obj: SourceObject
) -> None:
    target = destination(cache_root, obj)
    expected_mode = stat.S_IRUSR if obj.sealed else stat.S_IRUSR | stat.S_IWUSR
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.resolve(strict=True) != target.parent.absolute():
        raise AcquisitionError(f"object parent contains a symlink: {target.parent}")
    if target.is_symlink():
        raise AcquisitionError(f"object target must not be a symlink: {target}")
    if target.exists():
        verify_object(target, obj)
        target.chmod(expected_mode)
        return

    partial = target.with_name(f"{target.name}.part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise AcquisitionError(f"partial must be a regular non-symlink file: {partial}")
    if partial.exists():
        metadata = partial.stat()
        if metadata.st_nlink != 1:
            raise AcquisitionError(f"partial must not be hard-linked: {partial}")
        if metadata.st_size > obj.size_bytes:
            raise AcquisitionError(f"partial exceeds frozen size: {partial}")
        if metadata.st_size == obj.size_bytes:
            verify_object(partial, obj)
            partial.chmod(expected_mode)
            os.replace(partial, target)
            return
        if metadata.st_size == 0:
            partial.unlink()
    command = [
        "curl",
        "--disable",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--continue-at",
        "-",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--output",
        str(partial),
        resolver_url(manifest, obj),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            env=_credential_free_environment(),
            timeout=3600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AcquisitionError(
            f"download interrupted for {obj.path}; "
            f"resumable partial retained at {partial}"
        ) from exc
    verify_object(partial, obj)
    partial.chmod(expected_mode)
    partial.replace(target)


def acquire(manifest: SourceManifest, cache_root: Path) -> None:
    """Acquire all allowlisted objects, putting sealed week 26 strictly last."""

    guard_overlay_space(manifest, cache_root)
    ordered = [obj for obj in manifest.objects if not obj.sealed]
    ordered.extend(obj for obj in manifest.objects if obj.sealed)
    for index, obj in enumerate(ordered, start=1):
        _download_one(manifest, cache_root, obj)
        guard_overlay_space(manifest, cache_root)
        access = "sealed-byte-hash" if obj.sealed else "raw"
        print(f"VERIFIED {index:02d}/{len(ordered):02d} {access} {obj.path}")


def verify_all(manifest: SourceManifest, cache_root: Path) -> None:
    """Mechanically verify every allowlisted object, including sealed bytes."""

    for obj in manifest.objects:
        verify_object(destination(cache_root, obj), obj)
    print(
        f"VERIFIED all {len(manifest.objects)} objects ({manifest.total_bytes} bytes)"
    )


def _default_manifest() -> Path:
    return Path(__file__).resolve().parents[2] / "data/manifests/vk_lsvd_source_v1.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_default_manifest())
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-manifest")
    subparsers.add_parser("audit-remote")
    subparsers.add_parser("download")
    subparsers.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate-manifest":
            print(
                f"VALID manifest sha256={manifest.digest} "
                "objects=30 test_unlocked=false"
            )
        elif args.command == "audit-remote":
            audit_remote(manifest)
            print(
                f"VALID remote revision={manifest.revision} objects=30 "
                "week26_sealed=true"
            )
        elif args.command == "download":
            audit_remote(manifest)
            acquire(manifest, args.cache_root)
            verify_all(manifest, args.cache_root)
        elif args.command == "verify":
            verify_all(manifest, args.cache_root)
        else:  # pragma: no cover - argparse enforces the command choices
            raise AcquisitionError(f"unknown command: {args.command}")
    except AcquisitionError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
