from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Configuration
from .logging import get_logger

logger = get_logger(__name__)


def build_state_path(configuration: Configuration) -> Path:
    """Return the path for provenance state output."""

    path = configuration.results_directory / "state.json"
    logger.debug("resolved state path to %s", path)
    return path


def configuration_settings_snapshot(configuration: Configuration) -> dict[str, Any]:
    """Build a serializable snapshot of config settings used for this run."""

    pipeline = configuration.pipeline
    snapshot = {
        "samples": [
            {"id": sample.id, "path": str(sample.path)}
            for sample in configuration.samples
        ],
        "output_directory": str(configuration.output_directory),
        "logs_directory": str(configuration.logs_directory),
        "annotation_model": configuration.annotation_model,
        "pipeline": {
            "minimum_cells": pipeline.minimum_cells,
            "mad_threshold": pipeline.mad_threshold,
            "pca_n_components": pipeline.pca_n_components,
            "leiden_resolution": pipeline.leiden_resolution,
            "neighborhood_colocalization_radius": (
                pipeline.neighborhood_colocalization_radius
            ),
            "colocalization_number_of_permutations": (
                pipeline.colocalization_number_of_permutations
            ),
            "colocalization_minimum_cells": pipeline.colocalization_minimum_cells,
            "domain_n_clusters": pipeline.domain_n_clusters,
            "rank_top_n": pipeline.rank_top_n,
            "minimum_logarithm_fold_change": pipeline.minimum_logarithm_fold_change,
            "maximum_adjusted_p_value": pipeline.maximum_adjusted_p_value,
        },
    }
    logger.debug("built configuration snapshot for state persistence")
    return snapshot
