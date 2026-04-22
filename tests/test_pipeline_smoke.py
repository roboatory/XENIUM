from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from anndata import AnnData

import main
from src.config import Configuration, PipelineConfiguration, PlotsConfiguration, Sample

from .conftest import build_synthetic_adata


@pytest.fixture
def two_sample_configuration(tmp_path: Path) -> Configuration:
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
            pca_n_components=5,
            neighborhood_colocalization_radius=30.0,
            colocalization_number_of_permutations=5,
            colocalization_minimum_cells=3,
            domain_n_clusters=3,
            rank_top_n=5,
            minimum_logarithm_fold_change=-1e9,
            maximum_adjusted_p_value=1.0,
        ),
        plots=PlotsConfiguration(genes_to_plot=()),
    )
    configuration.create_directories()
    return configuration


def _fake_xenium_for_paths(
    samples_by_path: dict[str, AnnData],
):
    """Return a side_effect callable that looks up AnnData by the stem of the requested path."""

    def fake_xenium(path):
        sample_id = Path(path).name
        return {"table": samples_by_path[sample_id]}

    return fake_xenium


def _fake_harmony(adata, key):
    """Mimic sce.pp.harmony_integrate: copy X_pca into X_pca_harmony without correction."""

    adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()


class _FakeOllamaResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _ollama_response_for_clusters(cluster_ids: list[str]) -> bytes:
    """Produce the body an Ollama /api/chat would return with one annotation per cluster."""

    annotations = [
        {
            "cluster_id": cluster_id,
            "cell_type": f"cell_type_{cluster_id}",
            "confidence": 0.9,
            "rationale": "fixture",
        }
        for cluster_id in cluster_ids
    ]
    return json.dumps(
        {"message": {"content": json.dumps({"annotations": annotations})}}
    ).encode("utf-8")


def _fake_urlopen(request, timeout):
    """Parse the Ollama payload to discover cluster ids and return matching annotations."""

    payload = json.loads(request.data.decode("utf-8"))
    user_message = next(
        message["content"]
        for message in payload["messages"]
        if message["role"] == "user"
    )
    cluster_ids = []
    for line in user_message.splitlines():
        if line.startswith("- "):
            remainder = line[2:]
            cluster_id = remainder.split(":", 1)[0].strip()
            cluster_ids.append(cluster_id)
    return _FakeOllamaResponse(_ollama_response_for_clusters(cluster_ids))


def test_pipeline_runs_end_to_end_on_two_samples(
    two_sample_configuration: Configuration,
) -> None:
    """ingest → preprocess → annotate → domains → colocalization runs and writes expected files."""

    samples_by_path = {
        "patient_001": build_synthetic_adata(seed=0),
        "patient_002": build_synthetic_adata(seed=1, coord_offset=(500.0, 500.0)),
    }
    for sample_data in samples_by_path.values():
        sample_data.obs["cell_id"] = sample_data.obs["cell_id"].astype(str)

    with (
        patch("src.ingest.xenium", side_effect=_fake_xenium_for_paths(samples_by_path)),
        patch("src.analysis.sce.pp.harmony_integrate", side_effect=_fake_harmony),
        patch("src.annotation.urlopen", side_effect=_fake_urlopen),
    ):
        main.run_ingest_stage(two_sample_configuration)
        main.run_preprocess_stage(two_sample_configuration)
        main.run_annotate_stage(two_sample_configuration)
        main.run_domains_stage(two_sample_configuration)
        main.run_colocalization_stage(two_sample_configuration)

    figures = two_sample_configuration.figures_directory
    results = two_sample_configuration.results_directory

    assert (figures / "xenium_qc_histograms.png").exists()
    assert (figures / "harmony_umap_before_after.png").exists()
    assert (figures / "umap_leiden.png").exists()
    assert (figures / "rank_genes_dotplot_top_5.png").exists()
    for filename in (
        "xenium_colocalization_contact_counts.png",
        "xenium_colocalization_contact_row_proportions.png",
        "xenium_colocalization_log2_fold_enrichment.png",
        "xenium_colocalization_log2_fold_enrichment_significant_only.png",
    ):
        assert (figures / filename).exists(), f"missing {filename}"

    for sample_id in ("patient_001", "patient_002"):
        assert (figures / sample_id / "xenium_cell_type_overlay.png").exists()
        assert (
            figures / sample_id / "xenium_spatial_domain_label_overlay.png"
        ).exists()
        assert (results / sample_id / "leiden_clusters.csv").exists()
        assert (results / sample_id / "spatial_domain_labels.csv").exists()

    assert (results / "cluster_enriched_genes.json").exists()
    assert (results / "cluster_annotations.json").exists()
    assert (results / "spatial_domain_annotations.json").exists()
