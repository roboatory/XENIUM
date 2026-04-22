from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from anndata import AnnData

from src import io
from src.config import Configuration


def test_processed_anndata_round_trip(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """write_processed_anndata + read_processed_anndata preserves the AnnData."""

    io.write_processed_anndata(configuration, tiny_adata)
    round_tripped = io.read_processed_anndata(configuration)

    h5ad_path = configuration.processed_data_directory / "processed.h5ad"
    assert h5ad_path.exists()
    assert round_tripped.n_obs == tiny_adata.n_obs
    assert round_tripped.n_vars == tiny_adata.n_vars
    assert list(round_tripped.obs.columns) == list(tiny_adata.obs.columns)


def test_write_processed_anndata_overwrites_previous_file(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """Re-writing replaces the prior h5ad without leaving a stale temp file."""

    io.write_processed_anndata(configuration, tiny_adata)
    io.write_processed_anndata(configuration, tiny_adata)

    processed_dir = configuration.processed_data_directory
    tmp_entries = [
        p for p in processed_dir.iterdir() if p.name.startswith(".processed")
    ]
    assert tmp_entries == []
    assert (processed_dir / "processed.h5ad").exists()


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


def test_write_labels_cluster_mode_writes_per_sample_csv(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """write_labels produces one leiden_clusters.csv per sample under results/<sample_id>/."""

    tiny_adata.obs["sample_id"] = ["sample_a"] * (tiny_adata.n_obs // 2) + [
        "sample_b"
    ] * (tiny_adata.n_obs - tiny_adata.n_obs // 2)
    tiny_adata.obs["leiden"] = ["0"] * (tiny_adata.n_obs // 2) + ["1"] * (
        tiny_adata.n_obs - tiny_adata.n_obs // 2
    )

    io.write_labels(configuration, tiny_adata, "cluster")

    for sample_id in ("sample_a", "sample_b"):
        output_path = (
            configuration.results_directory / sample_id / "leiden_clusters.csv"
        )
        dataframe = pd.read_csv(output_path)
        assert list(dataframe.columns) == ["cell_id", "group"]
        assert (
            dataframe["group"].astype(str) == ("0" if sample_id == "sample_a" else "1")
        ).all()


def test_write_labels_domain_mode_writes_per_sample_csv(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """write_labels in domain mode writes per-sample spatial_domain_labels.csv files."""

    tiny_adata.obs["sample_id"] = ["sample_a"] * tiny_adata.n_obs
    tiny_adata.obs["spatial_domain"] = pd.Categorical(["A"] * tiny_adata.n_obs)

    io.write_labels(configuration, tiny_adata, "domain")

    output_path = (
        configuration.results_directory / "sample_a" / "spatial_domain_labels.csv"
    )
    dataframe = pd.read_csv(output_path)
    assert list(dataframe.columns) == ["cell_id", "group"]
    assert len(dataframe) == tiny_adata.n_obs


def test_write_labels_preserves_original_cell_id_per_sample(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """Each sample's CSV contains only that sample's original cell_id values."""

    tiny_adata.obs["sample_id"] = ["sample_a"] * (tiny_adata.n_obs // 2) + [
        "sample_b"
    ] * (tiny_adata.n_obs - tiny_adata.n_obs // 2)
    tiny_adata.obs["leiden"] = ["0"] * tiny_adata.n_obs

    io.write_labels(configuration, tiny_adata, "cluster")

    a_csv = pd.read_csv(
        configuration.results_directory / "sample_a" / "leiden_clusters.csv"
    )
    b_csv = pd.read_csv(
        configuration.results_directory / "sample_b" / "leiden_clusters.csv"
    )
    assert len(a_csv) + len(b_csv) == tiny_adata.n_obs
    # Per-sample CSVs use original cell_ids (not uniqueified obs_names with _sample_a suffix).
    assert "_sample_a" not in "".join(a_csv["cell_id"].astype(str))
    assert "_sample_b" not in "".join(b_csv["cell_id"].astype(str))


def test_save_state_writes_sorted_json(tmp_path: Path) -> None:
    """save_state writes a JSON file with sorted keys."""

    state_path = tmp_path / "state.json"
    io.save_state(state_path, {"b": 1, "a": 2})

    body = state_path.read_text()
    assert body.index('"a"') < body.index('"b"')
    assert json.loads(body) == {"a": 2, "b": 1}
