from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from main import STAGE_ORDER, parse_arguments


def test_no_arguments_selects_all_stages() -> None:
    """No --stage argument should select all stages in order."""

    with patch.object(sys, "argv", ["main.py"]):
        assert parse_arguments() == list(STAGE_ORDER)


def test_single_stage_selection() -> None:
    """A single --stage value should return just that stage."""

    with patch.object(sys, "argv", ["main.py", "--stage", "colocalization"]):
        assert parse_arguments() == ["colocalization"]


def test_multiple_stages_sorted_to_pipeline_order() -> None:
    """Stages given out of order should be sorted into pipeline order."""

    with patch.object(
        sys, "argv", ["main.py", "--stage", "colocalization", "preprocess", "ingest"]
    ):
        assert parse_arguments() == ["ingest", "preprocess", "colocalization"]


def test_duplicate_stages_deduplicated() -> None:
    """Duplicate stage names should not produce duplicate entries."""

    with patch.object(sys, "argv", ["main.py", "--stage", "ingest", "ingest"]):
        result = parse_arguments()
        assert result == ["ingest"]


def test_invalid_stage_rejected() -> None:
    """An invalid stage name should cause argparse to exit."""

    with patch.object(sys, "argv", ["main.py", "--stage", "nonexistent"]):
        with pytest.raises(SystemExit):
            parse_arguments()
