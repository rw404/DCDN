from scripts.storage_guard import (
    GIB,
    FilesystemStats,
    MountInfo,
    evaluate_storage,
)


def _fuse_mount() -> MountInfo:
    return MountInfo(
        mount_point="/workspace",
        fs_type="fuse.mfs",
        source="mfs#us-md-1.runpod.net",
    )


def _local_mount() -> MountInfo:
    return MountInfo(mount_point="/workspace", fs_type="ext4", source="/dev/vdb")


def _healthy_overlay() -> FilesystemStats:
    return FilesystemStats(total_bytes=100 * GIB, available_bytes=70 * GIB)


def test_shared_fuse_enforces_logical_50_gib_envelope() -> None:
    result = evaluate_storage(
        project_bytes=40 * GIB,
        persistent=FilesystemStats(
            total_bytes=400_000 * GIB,
            available_bytes=1,
        ),
        persistent_mount=_fuse_mount(),
        overlay=_healthy_overlay(),
    )
    assert result["ok"]
    assert result["shared_fuse_df_caveat"]
    assert result["logical_capacity_gib"] == 50
    assert result["logical_headroom_gib"] == 10
    assert result["persistent_df_enforcement"] == "informational_shared_backend"


def test_one_byte_over_durable_cap_is_blocked() -> None:
    result = evaluate_storage(
        project_bytes=40 * GIB + 1,
        persistent=FilesystemStats(400_000 * GIB, 100_000 * GIB),
        persistent_mount=_fuse_mount(),
        overlay=_healthy_overlay(),
    )
    assert not result["ok"]
    assert any("dynamic cap" in item for item in result["violations"])
    assert any("logical persistent headroom" in item for item in result["violations"])


def test_smaller_filesystem_keeps_twenty_percent_and_five_gib() -> None:
    result = evaluate_storage(
        project_bytes=24 * GIB,
        persistent=FilesystemStats(30 * GIB, 6 * GIB),
        persistent_mount=_local_mount(),
        overlay=_healthy_overlay(),
    )
    assert result["ok"]
    assert result["project_cap_gib"] == 24
    assert result["required_headroom_gib"] == 6
    assert result["capacity_basis"] == "smaller_locally_observed_filesystem"


def test_smaller_local_filesystem_enforces_observed_free_space() -> None:
    result = evaluate_storage(
        project_bytes=20 * GIB,
        persistent=FilesystemStats(30 * GIB, 5 * GIB),
        persistent_mount=_local_mount(),
        overlay=_healthy_overlay(),
    )
    assert not result["ok"]
    assert any("observed persistent free" in item for item in result["violations"])


def test_overlay_requires_twenty_gib_and_less_than_85_percent_usage() -> None:
    result = evaluate_storage(
        project_bytes=1 * GIB,
        persistent=FilesystemStats(50 * GIB, 49 * GIB),
        persistent_mount=_local_mount(),
        overlay=FilesystemStats(total_bytes=100 * GIB, available_bytes=15 * GIB),
    )
    assert not result["ok"]
    assert any("less than 20 GiB" in item for item in result["violations"])
    assert any("at or above 85%" in item for item in result["violations"])
