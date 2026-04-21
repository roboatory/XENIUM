from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData

from src import analysis
from src.preprocessing import normalize_and_scale


def _prepare_for_clustering(adata: AnnData) -> None:
    """Run the usual QC + normalization scaffolding prior to clustering."""

    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(adata)


def test_run_clustering_adds_pca_and_leiden(tiny_adata: AnnData) -> None:
    """run_clustering populates X_pca and a leiden obs column with multiple groups."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5)

    assert "X_pca" in tiny_adata.obsm
    assert tiny_adata.obsm["X_pca"].shape == (tiny_adata.n_obs, 5)
    assert "leiden" in tiny_adata.obs.columns
    assert tiny_adata.obs["leiden"].nunique() >= 2


def test_run_umap_adds_two_dimensional_embedding(tiny_adata: AnnData) -> None:
    """run_umap attaches a 2-D X_umap embedding to obsm."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5)
    analysis.run_umap(tiny_adata)

    assert tiny_adata.obsm["X_umap"].shape == (tiny_adata.n_obs, 2)


def test_rank_genes_populates_uns(tiny_adata: AnnData) -> None:
    """rank_genes stores rank_genes_groups results in uns."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5)
    analysis.rank_genes(tiny_adata)

    assert "rank_genes_groups" in tiny_adata.uns
    assert "names" in tiny_adata.uns["rank_genes_groups"]


def test_compute_enriched_genes_respects_filters(tiny_adata: AnnData) -> None:
    """compute_enriched_genes returns a dict keyed by cluster id with filtered gene lists."""

    _prepare_for_clustering(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5)
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
    analysis.run_clustering(tiny_adata, pca_n_components=5)
    analysis.rank_genes(tiny_adata)

    result = analysis.compute_enriched_genes(
        tiny_adata,
        top_n=2,
        minimum_logarithm_fold_change=-np.inf,
        maximum_adjusted_p_value=1.0,
    )
    for genes in result.values():
        assert len(genes) <= 2
