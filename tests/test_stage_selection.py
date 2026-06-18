from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from main import STAGE_ORDER, parse_arguments


def test_config_only_selects_all_stages() -> None:
    """A config path with no --stage argument should select all stages in order."""

    with patch.object(sys, "argv", ["main.py", "--config", "config.yaml"]):
        result = parse_arguments()
        assert result.config_path.name == "config.yaml"
        assert result.stages == list(STAGE_ORDER)


def test_config_argument_required() -> None:
    """The pipeline should require explicit run configuration."""

    with patch.object(sys, "argv", ["main.py"]):
        with pytest.raises(SystemExit):
            parse_arguments()


def test_single_stage_selection() -> None:
    """A single --stage value should return just that stage."""

    with patch.object(
        sys, "argv", ["main.py", "--config", "config.yaml", "--stage", "colocalization"]
    ):
        assert parse_arguments().stages == ["colocalization"]


def test_multiple_stages_sorted_to_pipeline_order() -> None:
    """Stages given out of order should be sorted into pipeline order."""

    with patch.object(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            "config.yaml",
            "--stage",
            "colocalization",
            "preprocess",
            "ingest",
        ],
    ):
        assert parse_arguments().stages == ["ingest", "preprocess", "colocalization"]


def test_duplicate_stages_deduplicated() -> None:
    """Duplicate stage names should not produce duplicate entries."""

    with patch.object(
        sys, "argv", ["main.py", "--config", "config.yaml", "--stage", "ingest", "ingest"]
    ):
        result = parse_arguments()
        assert result.stages == ["ingest"]


def test_invalid_stage_rejected() -> None:
    """An invalid stage name should cause argparse to exit."""

    with patch.object(
        sys, "argv", ["main.py", "--config", "config.yaml", "--stage", "nonexistent"]
    ):
        with pytest.raises(SystemExit):
            parse_arguments()
