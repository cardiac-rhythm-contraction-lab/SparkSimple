"""tests/unit/test_cli.py — Tests for the sparksimple CLI entry point."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from sparksimple.cli import main


def test_version_flag(capsys):
    with patch.object(sys, "argv", ["sparksimple", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "sparksimple" in out


def test_run_missing_dir(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    with patch.object(sys, "argv", ["sparksimple", "run", missing]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_run_help(capsys):
    with patch.object(sys, "argv", ["sparksimple", "run", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "experiment_dir" in out


def test_run_calls_pipeline(tmp_path):
    fake_meta = MagicMock()

    fake_result = MagicMock()
    fake_result.filtering.accepted = []
    fake_result.filtering.rejected = []
    fake_result.clustering.rois = []
    fake_result.phase10 = None

    with (
        patch("sparksimple.cli.load_metadata", return_value=fake_meta),
        patch("sparksimple.cli.load_image_stack", return_value=(MagicMock(), 0)),
        patch("sparksimple.cli.run_pipeline", return_value=fake_result) as mock_pipeline,
        patch.object(sys, "argv", ["sparksimple", "run", str(tmp_path)]),
    ):
        main()

    mock_pipeline.assert_called_once()
    _, kwargs = mock_pipeline.call_args
    assert kwargs.get("is_simulated") is False
