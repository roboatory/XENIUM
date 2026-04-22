from __future__ import annotations

from unittest.mock import patch

import anndata as ad
import numpy as np
import scanpy as sc
from anndata import AnnData

from src import analysis
from src.preprocessing import normalize_and_scale

from .conftest import build_synthetic_adata


def _prepare_for_clustering(adata: AnnData) -> None:
    """Run the usual QC + normalization scaffolding prior to clustering."""

    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(adata)


def test_run_clustering_single_sample_skips_harmony(tiny_adata: AnnData) -> None:
    """Single-sample run populates X_pca, X_umap, leiden but not X_pca_harmony."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5, leiden_resolution=0.5)

    assert "X_pca" in tiny_adata.obsm
    assert "X_pca_harmony" not in tiny_adata.obsm
    assert "X_umap_uncorrected" not in tiny_adata.obsm
    assert tiny_adata.obsm["X_umap"].shape == (tiny_adata.n_obs, 2)
    assert "leiden" in tiny_adata.obs.columns
    assert tiny_adata.obs["leiden"].nunique() >= 2


def test_run_clustering_multi_sample_runs_harmony_and_stores_uncorrected_umap() -> None:
    """Multi-sample run writes X_pca_harmony, X_umap_uncorrected, and canonical X_umap."""

    sample_a = build_synthetic_adata(seed=0)
    sample_b = build_synthetic_adata(seed=1)
    merged = ad.concat(
        [sample_a, sample_b],
        keys=["sample_a", "sample_b"],
        label="sample_id",
        index_unique="_",
    )
    merged.obs.index.name = None
    _prepare_for_clustering(merged)

    def fake_harmony(adata, key):
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()

    with patch(
        "src.analysis.sce.pp.harmony_integrate", side_effect=fake_harmony
    ) as mocked:
        analysis.run_clustering(merged, pca_n_components=5, leiden_resolution=0.5)

    mocked.assert_called_once()
    assert mocked.call_args.kwargs["key"] == "sample_id"
    assert "X_pca_harmony" in merged.obsm
    assert "X_umap_uncorrected" in merged.obsm
    assert "X_umap" in merged.obsm
    assert merged.obsm["X_umap"].shape == (merged.n_obs, 2)
    assert "leiden" in merged.obs.columns


def test_rank_genes_populates_uns(tiny_adata: AnnData) -> None:
    """rank_genes stores rank_genes_groups results in uns."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5, leiden_resolution=0.5)
    analysis.rank_genes(tiny_adata)

    assert "rank_genes_groups" in tiny_adata.uns
    assert "names" in tiny_adata.uns["rank_genes_groups"]


def test_compute_enriched_genes_respects_filters(tiny_adata: AnnData) -> None:
    """compute_enriched_genes returns a dict keyed by cluster id with filtered gene lists."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5, leiden_resolution=0.5)
    analysis.rank_genes(tiny_adata)

    permissive = analysis.compute_enriched_genes(
        tiny_adata,
        top_n=10,
        minimum_logarithm_fold_change=-np.inf,
        maximum_adjusted_p_value=1.0,
    )
    assert set(permissive.keys()) == set(tiny_adata.obs["leiden"].astype(str).unique())
    assert all(isinstance(genes, list) for genes in permissive.values())

    strict = analysis.compute_enriched_genes(
        tiny_adata,
        top_n=10,
        minimum_logarithm_fold_change=1e9,
        maximum_adjusted_p_value=0.0,
    )
    assert all(genes == [] for genes in strict.values())


def test_compute_enriched_genes_respects_top_n(tiny_adata: AnnData) -> None:
    """compute_enriched_genes caps each cluster's list at top_n entries."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5, leiden_resolution=0.5)
    analysis.rank_genes(tiny_adata)

    result = analysis.compute_enriched_genes(
        tiny_adata,
        top_n=2,
        minimum_logarithm_fold_change=-np.inf,
        maximum_adjusted_p_value=1.0,
    )
    for genes in result.values():
        assert len(genes) <= 2
