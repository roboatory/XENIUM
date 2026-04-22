from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import Configuration, PipelineConfiguration, PlotsConfiguration, Sample


def _base_config_dict(raw: Path, output: Path) -> dict:
    return {
        "samples": [{"id": "sample_a", "path": str(raw / "sample_a")}],
        "output_directory": str(output),
        "annotation_model": "llama3.1:8b",
        "pipeline": {
            "minimum_counts": 10,
            "maximum_counts_quantile": 0.99,
            "minimum_cells": 5,
            "pca_n_components": 20,
            "neighborhood_colocalization_radius": 50.0,
            "colocalization_number_of_permutations": 100,
            "colocalization_minimum_cells": 5,
            "domain_n_clusters": 4,
            "rank_top_n": 10,
            "minimum_logarithm_fold_change": 0.5,
            "maximum_adjusted_p_value": 0.05,
        },
        "plots": {"genes_to_plot": ["geneB", "geneA"]},
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_load_from_yaml_populates_all_fields(tmp_path: Path) -> None:
    """load_from_yaml fills directories, samples, and nested configs from YAML."""

    config_path = _write_config(
        tmp_path, _base_config_dict(tmp_path / "raw", tmp_path / "output")
    )
    configuration = Configuration()
    configuration.load_from_yaml(config_path)

    assert configuration.annotation_model == "llama3.1:8b"
    assert [sample.id for sample in configuration.samples] == ["sample_a"]
    assert configuration.samples[0].path == (tmp_path / "raw" / "sample_a").resolve()
    assert configuration.output_directory == (tmp_path / "output").resolve()
    assert (
        configuration.processed_data_directory
        == (tmp_path / "output").resolve() / "processed"
    )
    assert (
        configuration.results_directory == (tmp_path / "output").resolve() / "analysis"
    )
    assert (
        configuration.figures_directory == (tmp_path / "output").resolve() / "figures"
    )
    assert configuration.logs_directory == (tmp_path / "output").resolve() / "logs"
    assert isinstance(configuration.pipeline, PipelineConfiguration)
    assert configuration.pipeline.pca_n_components == 20
    assert isinstance(configuration.plots, PlotsConfiguration)


def test_plots_genes_are_sorted(tmp_path: Path) -> None:
    """PlotsConfiguration sorts genes_to_plot into deterministic order."""

    config_path = _write_config(
        tmp_path, _base_config_dict(tmp_path / "raw", tmp_path / "output")
    )
    configuration = Configuration()
    configuration.load_from_yaml(config_path)

    assert configuration.plots.genes_to_plot == ("geneA", "geneB")


def test_defaults_used_when_optional_keys_omitted(tmp_path: Path) -> None:
    """Missing optional keys fall back to Configuration defaults."""

    payload = _base_config_dict(tmp_path / "raw", tmp_path / "output")
    payload.pop("annotation_model")
    config_path = _write_config(tmp_path, payload)

    configuration = Configuration()
    configuration.load_from_yaml(config_path)
    assert configuration.annotation_model == "llama3.1:8b"


def test_create_directories_makes_all_outputs(tmp_path: Path) -> None:
    """create_directories creates processed/results/figures/logs under the output root."""

    config_path = _write_config(
        tmp_path, _base_config_dict(tmp_path / "raw", tmp_path / "output")
    )
    configuration = Configuration()
    configuration.load_from_yaml(config_path)
    configuration.create_directories()

    for directory in (
        configuration.processed_data_directory,
        configuration.results_directory,
        configuration.figures_directory,
        configuration.logs_directory,
    ):
        assert directory.is_dir()


def test_multiple_samples_parsed_in_order(tmp_path: Path) -> None:
    """Multiple sample records are loaded preserving the order given in YAML."""

    payload = _base_config_dict(tmp_path / "raw", tmp_path / "output")
    payload["samples"] = [
        {"id": "patient_001", "path": str(tmp_path / "raw" / "p001")},
        {"id": "patient_002", "path": str(tmp_path / "raw" / "p002")},
        {"id": "patient_003", "path": str(tmp_path / "raw" / "p003")},
    ]
    config_path = _write_config(tmp_path, payload)

    configuration = Configuration()
    configuration.load_from_yaml(config_path)

    assert [sample.id for sample in configuration.samples] == [
        "patient_001",
        "patient_002",
        "patient_003",
    ]


def test_sample_from_dictionary_parses_id_and_path(tmp_path: Path) -> None:
    """Sample.from_dictionary reads id and path fields."""

    sample = Sample.from_dictionary({"id": "a", "path": str(tmp_path / "a")})
    assert sample.id == "a"
    assert sample.path == (tmp_path / "a").resolve()


def test_pipeline_configuration_casts_numeric_types() -> None:
    """PipelineConfiguration.from_dictionary coerces numeric strings to int/float."""

    raw = {
        "minimum_counts": "100",
        "maximum_counts_quantile": "0.99",
        "minimum_cells": "5",
        "pca_n_components": "20",
        "neighborhood_colocalization_radius": "50.0",
        "colocalization_number_of_permutations": "200",
        "colocalization_minimum_cells": "5",
        "domain_n_clusters": "4",
        "rank_top_n": "10",
        "minimum_logarithm_fold_change": "0.5",
        "maximum_adjusted_p_value": "0.05",
    }
    pipeline = PipelineConfiguration.from_dictionary(raw)

    assert pipeline.minimum_counts == 100
    assert pipeline.maximum_counts_quantile == pytest.approx(0.99)
    assert pipeline.pca_n_components == 20
    assert pipeline.colocalization_number_of_permutations == 200
