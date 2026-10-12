from __future__ import annotations

from unittest.mock import MagicMock, patch

from shawaf_vlm.retrieval.tier_a_download import (
    DownloadResult,
    ensure_groot,
    ensure_rstpreid,
    ensure_tier_a_datasets,
    ensure_tvpreid,
)


def test_ensure_tvpreid_skips_when_present() -> None:
    with patch("shawaf_vlm.data.tvpreid.discover_local_tvpreid", return_value="/data/TVPReid"):
        with patch("shawaf_vlm.data.tvpreid.download_tvpreid") as dl:
            result = ensure_tvpreid()
    assert result.action == "skipped"
    dl.assert_not_called()


def test_ensure_tvpreid_downloads_when_missing() -> None:
    with patch("shawaf_vlm.data.tvpreid.discover_local_tvpreid", return_value=None):
        with patch(
            "shawaf_vlm.data.tvpreid.download_tvpreid",
            return_value="/cache/TVPReid",
        ) as dl:
            result = ensure_tvpreid()
    assert result.action == "downloaded"
    dl.assert_called_once()


def test_ensure_tier_a_all_three() -> None:
    with patch(
        "shawaf_vlm.retrieval.tier_a_download.ensure_tvpreid",
        return_value=DownloadResult("TVPReid", "skipped", "/a"),
    ):
        with patch(
            "shawaf_vlm.retrieval.tier_a_download.ensure_groot",
            return_value=DownloadResult("GroOT", "skipped", "/b"),
        ):
            with patch(
                "shawaf_vlm.retrieval.tier_a_download.ensure_rstpreid",
                return_value=DownloadResult("RSTP", "skipped", "/c"),
            ):
                rows = ensure_tier_a_datasets()
    assert len(rows) == 3


def test_dataset_status_keys() -> None:
    from shawaf_vlm.retrieval.datasets import dataset_status

    st = dataset_status()
    assert set(st.keys()) == {"tvpreid", "groot", "rstpreid"}
