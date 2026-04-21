from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from spatialdata import SpatialData

from src import io
from src.config import Configuration


def test_spatialdata_zarr_round_trip(
    configuration: Configuration,
    tiny_spatialdata: SpatialData,
) -> None:
    """write_spatialdata_zarr + read_spatialdata_zarr preserves the table."""

    io.write_spatialdata_zarr(configuration, tiny_spatialdata)
    round_tripped = io.read_spatialdata_zarr(configuration)

    zarr_path = configuration.processed_data_directory / "processed.zarr"
    assert zarr_path.exists()
    assert round_tripped["table"].n_obs == tiny_spatialdata["table"].n_obs
    assert round_tripped["table"].n_vars == tiny_spatialdata["table"].n_vars
    assert "cell_boundaries" in round_tripped.shapes


def test_write_spatialdata_zarr_overwrites_previous_zarr(
    configuration: Configuration,
    tiny_spatialdata: SpatialData,
) -> None:
    """Re-writing replaces the prior zarr without leaving a stale temp directory."""

    io.write_spatialdata_zarr(configuration, tiny_spatialdata)
    io.write_spatialdata_zarr(configuration, tiny_spatialdata)

    processed_dir = configuration.processed_data_directory
    tmp_entries = [
        p for p in processed_dir.iterdir() if p.name.startswith(".processed")
    ]
    assert tmp_entries == []
    assert (processed_dir / "processed.zarr").exists()


def test_enriched_genes_round_trip(configuration: Configuration) -> None:
    """write_enriched_genes then read_enriched_genes restores the mapping."""

    payload = {"0": ["g1", "g2"], "1": ["g3"]}
    io.write_enriched_genes(configuration, payload)
    restored = io.read_enriched_genes(configuration)

    assert restored == payload


def test_write_annotations_creates_expected_files(configuration: Configuration) -> None:
    """write_annotations writes cluster and domain JSON files at expected paths."""

    cluster_annotations = {"0": {"cell_type": "Luminal", "confidence": 0.9}}
    domain_annotations = {"0": {"cell_type": "Stromal niche", "confidence": 0.8}}

    io.write_annotations(configuration, cluster_annotations, "cluster")
    io.write_annotations(configuration, domain_annotations, "domain")

    cluster_path = configuration.results_directory / "cluster_annotations.json"
    domain_path = configuration.results_directory / "spatial_domain_annotations.json"

    assert json.loads(cluster_path.read_text()) == cluster_annotations
    assert json.loads(domain_path.read_text()) == domain_annotations


def test_write_labels_cluster_mode_writes_csv_with_cell_id(
    configuration: Configuration,
    tiny_adata,
) -> None:
    """write_labels produces leiden_clusters.csv with the cell_id and group columns."""

    tiny_adata.obs["leiden"] = ["0"] * (tiny_adata.n_obs // 2) + ["1"] * (
        tiny_adata.n_obs - tiny_adata.n_obs // 2
    )
    io.write_labels(configuration, tiny_adata, "cluster")

    output_path = configuration.results_directory / "leiden_clusters.csv"
    dataframe = pd.read_csv(output_path)
    assert list(dataframe.columns) == ["cell_id", "group"]
    assert len(dataframe) == tiny_adata.n_obs
    assert set(dataframe["group"].astype(str).unique()) == {"0", "1"}


def test_write_labels_domain_mode_writes_csv_at_expected_path(
    configuration: Configuration,
    tiny_adata,
) -> None:
    """write_labels in domain mode writes spatial_domain_labels.csv."""

    tiny_adata.obs["spatial_domain"] = pd.Categorical(["A"] * tiny_adata.n_obs)
    io.write_labels(configuration, tiny_adata, "domain")

    output_path = configuration.results_directory / "spatial_domain_labels.csv"
    dataframe = pd.read_csv(output_path)
    assert list(dataframe.columns) == ["cell_id", "group"]
    assert len(dataframe) == tiny_adata.n_obs


def test_save_state_writes_sorted_json(tmp_path: Path) -> None:
    """save_state writes a JSON file with sorted keys."""

    state_path = tmp_path / "state.json"
    io.save_state(state_path, {"b": 1, "a": 2})

    body = state_path.read_text()
    assert body.index('"a"') < body.index('"b"')
    assert json.loads(body) == {"a": 2, "b": 1}
