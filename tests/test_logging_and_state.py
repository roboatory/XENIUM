from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src import logging as pipeline_logging
from src import state
from src.config import Configuration


def test_initialize_logging_creates_log_file_and_pointer(tmp_path: Path) -> None:
    """initialize_logging creates a timestamped .log file and an active pointer."""

    logs_directory = tmp_path / "logs"
    log_path = pipeline_logging.initialize_logging(logs_directory, reset=True)

    assert log_path.exists()
    assert log_path.parent == logs_directory
    pointer_path = logs_directory / ".active_log"
    assert pointer_path.exists()
    assert pointer_path.read_text().strip() == str(log_path)


def test_initialize_logging_reuses_existing_active_log(tmp_path: Path) -> None:
    """A second initialize_logging call in a fresh process reuses the pointer target."""

    logs_directory = tmp_path / "logs"
    first_log = pipeline_logging.initialize_logging(logs_directory, reset=True)

    pipeline_logging._LOG_PATH = None
    second_log = pipeline_logging.initialize_logging(logs_directory, reset=False)

    assert second_log == first_log


def test_initialize_logging_uses_stdout_under_slurm(tmp_path: Path) -> None:
    """SLURM runs rely on the scheduler output file instead of a second log file."""

    logs_directory = tmp_path / "logs"

    with patch.dict("os.environ", {"SLURM_JOB_ID": "123"}, clear=False):
        log_path = pipeline_logging.initialize_logging(logs_directory, reset=True)

    assert log_path == Path("/dev/stdout")
    assert not (logs_directory / ".active_log").exists()
    assert not list(logs_directory.glob("*.log"))


def test_clear_active_log_removes_pointer_file(tmp_path: Path) -> None:
    """clear_active_log deletes the pointer; the timestamped log file is left in place."""

    logs_directory = tmp_path / "logs"
    log_path = pipeline_logging.initialize_logging(logs_directory, reset=True)
    pipeline_logging.clear_active_log(logs_directory)

    assert not (logs_directory / ".active_log").exists()
    assert log_path.exists()


def test_get_logger_uses_pipeline_namespace() -> None:
    """get_logger returns loggers rooted under the 'pipeline' namespace."""

    root_logger = pipeline_logging.get_logger()
    assert root_logger.name == "pipeline"
    child_logger = pipeline_logging.get_logger("submodule")
    assert child_logger.name == "pipeline.submodule"


def test_build_state_path_lives_under_results_directory(
    configuration: Configuration,
) -> None:
    """state.build_state_path returns results/state.json."""

    assert state.build_state_path(configuration) == (
        configuration.results_directory / "state.json"
    )


def test_configuration_settings_snapshot_round_trips_key_values(
    configuration: Configuration,
) -> None:
    """configuration_settings_snapshot reproduces top-level and nested pipeline config values."""

    snapshot = state.configuration_settings_snapshot(configuration)

    assert snapshot["annotation_model"] == configuration.annotation_model
    assert snapshot["pipeline"]["pca_n_components"] == (
        configuration.pipeline.pca_n_components
    )
    assert snapshot["pipeline"]["colocalization_number_of_permutations"] == (
        configuration.pipeline.colocalization_number_of_permutations
    )
    assert snapshot["pipeline"]["mad_threshold"] == configuration.pipeline.mad_threshold
    assert snapshot["pipeline"]["leiden_resolution"] == (
        configuration.pipeline.leiden_resolution
    )


def test_configuration_settings_snapshot_includes_samples(
    configuration: Configuration,
) -> None:
    """The snapshot lists each sample's id and path."""

    snapshot = state.configuration_settings_snapshot(configuration)

    assert snapshot["samples"] == [
        {"id": sample.id, "path": str(sample.path)} for sample in configuration.samples
    ]
