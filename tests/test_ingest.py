from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from anndata import AnnData

from src import io
from src.config import Configuration, PipelineConfiguration, Sample
from src.ingest import assign_core_labels, run_ingest

from .conftest import build_synthetic_adata


@pytest.fixture
def multi_sample_configuration(tmp_path: Path) -> Configuration:
    """Configuration with two sample records pointing at placeholder paths."""

    configuration = Configuration(
        samples=[
            Sample(id="patient_001", path=tmp_path / "raw" / "patient_001"),
            Sample(id="patient_002", path=tmp_path / "raw" / "patient_002"),
        ],
        output_directory=tmp_path / "output",
        processed_data_directory=tmp_path / "output" / "processed",
        results_directory=tmp_path / "output" / "analysis",
        figures_directory=tmp_path / "output" / "figures",
        logs_directory=tmp_path / "output" / "logs",
        pipeline=PipelineConfiguration(
            minimum_cells=1,
            mad_threshold=5.0,
            pca_n_components=2,
            leiden_resolution=0.5,
            neighborhood_colocalization_radius=1.0,
            colocalization_number_of_permutations=1,
            colocalization_minimum_cells=1,
            domain_n_clusters=2,
            rank_top_n=1,
            minimum_logarithm_fold_change=0.0,
            maximum_adjusted_p_value=1.0,
        ),
    )
    configuration.create_directories()
    return configuration


def _mock_xenium_from_samples(
    samples_by_id: dict[str, AnnData],
):
    """Return a side_effect callable for patching xenium() that returns AnnData wrapped in a table-like dict."""

    def fake_xenium(path):
        sample_id = Path(path).name
        return {"table": samples_by_id[sample_id]}

    return fake_xenium


def test_run_ingest_merges_two_samples_into_one_anndata(
    multi_sample_configuration: Configuration,
) -> None:
    """run_ingest concatenates per-sample tables and writes a merged processed.h5ad."""

    per_sample = {
        "patient_001": build_synthetic_adata(seed=0),
        "patient_002": build_synthetic_adata(seed=1, coord_offset=(1000.0, 0.0)),
    }

    with patch("src.ingest.xenium", side_effect=_mock_xenium_from_samples(per_sample)):
        run_ingest(multi_sample_configuration)

    merged = io.read_processed_anndata(multi_sample_configuration)

    assert (
        merged.n_obs
        == per_sample["patient_001"].n_obs + per_sample["patient_002"].n_obs
    )
    assert "sample_id" in merged.obs.columns
    assert "core_id" in merged.obs.columns
    assert set(merged.obs["sample_id"].astype(str).unique()) == {
        "patient_001",
        "patient_002",
    }


def test_run_ingest_makes_obs_names_unique_across_samples(
    multi_sample_configuration: Configuration,
) -> None:
    """obs_names in the merged AnnData are globally unique even when per-sample cell_ids collide."""

    per_sample = {
        "patient_001": build_synthetic_adata(seed=0),
        "patient_002": build_synthetic_adata(seed=1),
    }

    with patch("src.ingest.xenium", side_effect=_mock_xenium_from_samples(per_sample)):
        run_ingest(multi_sample_configuration)

    merged = io.read_processed_anndata(multi_sample_configuration)

    assert len(set(merged.obs_names)) == merged.n_obs


def test_run_ingest_preserves_original_cell_id(
    multi_sample_configuration: Configuration,
) -> None:
    """obs['cell_id'] retains each sample's original per-sample cell id for Explorer export."""

    per_sample = {
        "patient_001": build_synthetic_adata(seed=0, n_per_type=5),
        "patient_002": build_synthetic_adata(seed=1, n_per_type=5),
    }

    with patch("src.ingest.xenium", side_effect=_mock_xenium_from_samples(per_sample)):
        run_ingest(multi_sample_configuration)

    merged = io.read_processed_anndata(multi_sample_configuration)
    original_ids = set(merged.obs["cell_id"].astype(str).unique())
    assert original_ids == {f"cell_{i:04d}" for i in range(15)}


def test_run_ingest_single_sample_writes_anndata(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """Single-sample run writes an AnnData with a sample_id column equal to that sample's id."""

    per_sample = {"sample_a": tiny_adata}

    with patch("src.ingest.xenium", side_effect=_mock_xenium_from_samples(per_sample)):
        run_ingest(configuration)

    merged = io.read_processed_anndata(configuration)
    assert merged.n_obs == tiny_adata.n_obs
    assert (merged.obs["sample_id"].astype(str) == "sample_a").all()
    assert merged.obs["core_id"].astype(str).str.startswith("sample_a_core_").all()


def test_assign_core_labels_orders_cores_from_left_to_right() -> None:
    """assign_core_labels labels spatially separated cores by x-position."""

    adata = build_synthetic_adata(seed=7, n_per_type=8)
    adata.obsm["spatial"][:8, 0] -= 500
    adata.obsm["spatial"][8:16, 0] += 500
    adata.obsm["spatial"][16:, 0] += 1500

    assign_core_labels(adata, "50702-2")

    labels = adata.obs["core_id"].astype(str).to_numpy()
    assert set(labels) == {
        "50702-2_core_1",
        "50702-2_core_2",
        "50702-2_core_3",
    }
    assert (labels[:8] == "50702-2_core_1").all()
    assert (labels[8:16] == "50702-2_core_2").all()
    assert (labels[16:] == "50702-2_core_3").all()


def test_run_ingest_raises_on_missing_sample_path(
    multi_sample_configuration: Configuration,
) -> None:
    """A non-existent sample path surfaces as an error from the underlying xenium loader."""

    with pytest.raises(Exception):
        run_ingest(multi_sample_configuration)
