"""Tests for the dashboard payload functions."""

import json
from pathlib import Path

import pytest

from kairnz_train.dashboard import metrics_payload, status_payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """Returns a populated work directory with status.json and metrics.jsonl."""
    status = {"iter": 3, "phase": "train", "promoted": True}
    (tmp_path / "status.json").write_text(json.dumps(status))

    rows = [
        {"iter": 0, "loss": 1.5, "samples": 1024},
        {"iter": 1, "loss": 1.2, "samples": 2048},
        {"iter": 2, "loss": 0.9, "samples": 3072},
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    return tmp_path


@pytest.fixture()
def empty_dir(tmp_path: Path) -> Path:
    """Returns a work directory with no data files."""
    return tmp_path


# ---------------------------------------------------------------------------
# status_payload
# ---------------------------------------------------------------------------


def test_status_payload_returns_parsed_dict(work_dir: Path) -> None:
    """status_payload reads status.json and returns the correct dict."""
    result = status_payload(work_dir)
    assert result == {"iter": 3, "phase": "train", "promoted": True}


def test_status_payload_returns_empty_dict_when_missing(empty_dir: Path) -> None:
    """status_payload returns {} when status.json does not exist."""
    result = status_payload(empty_dir)
    assert result == {}


def test_status_payload_returns_empty_dict_on_corrupt_json(tmp_path: Path) -> None:
    """status_payload returns {} when status.json is not valid JSON."""
    (tmp_path / "status.json").write_text("not valid json{{{")
    result = status_payload(tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# metrics_payload
# ---------------------------------------------------------------------------


def test_metrics_payload_returns_list_of_dicts(work_dir: Path) -> None:
    """metrics_payload reads metrics.jsonl and returns the correct list."""
    result = metrics_payload(work_dir)
    assert len(result) == 3
    assert result[0] == {"iter": 0, "loss": 1.5, "samples": 1024}
    assert result[2]["loss"] == pytest.approx(0.9)


def test_metrics_payload_returns_empty_list_when_missing(empty_dir: Path) -> None:
    """metrics_payload returns [] when metrics.jsonl does not exist."""
    result = metrics_payload(empty_dir)
    assert result == []


def test_metrics_payload_skips_blank_lines(tmp_path: Path) -> None:
    """metrics_payload ignores blank lines in metrics.jsonl."""
    content = '\n{"iter": 0}\n\n{"iter": 1}\n'
    (tmp_path / "metrics.jsonl").write_text(content)
    result = metrics_payload(tmp_path)
    assert len(result) == 2
    assert result[1]["iter"] == 1
