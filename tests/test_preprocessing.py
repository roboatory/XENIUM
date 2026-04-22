from __future__ import annotations

import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from src.preprocessing import (
    compute_qc_metrics,
    filter_cells_and_genes,
    normalize_and_scale,
)

from .conftest import build_synthetic_adata


def _concat_with_sample_ids(
    adatas: list[AnnData],
    sample_ids: list[str],
) -> AnnData:
    """Helper: concat per-sample AnnDatas with a sample_id column and unique obs_names."""

    import anndata as ad

    merged = ad.concat(
        adatas,
        keys=sample_ids,
        label="sample_id",
        index_unique="_",
    )
    merged.obs.index.name = None
    return merged


def test_filter_cells_and_genes_drops_extreme_outliers_single_sample(
    tiny_adata: AnnData,
) -> None:
    """A cell with wildly inflated counts is flagged as a MAD outlier and dropped."""

    original_obs = tiny_adata.obs.index[0]
    extreme_row = tiny_adata.X.toarray().copy()
    extreme_row[0] = extreme_row[0] * 1000
    tiny_adata.X = csr_matrix(extreme_row)

    compute_qc_metrics(tiny_adata)
    filter_cells_and_genes(tiny_adata, minimum_cells=1)

    assert original_obs not in tiny_adata.obs.index


def test_filter_cells_and_genes_applies_mad_per_sample(
    tiny_adata: AnnData,
) -> None:
    """Per-sample MAD filtering derives different cutoffs when samples have different scales."""

    dense = tiny_adata.X.toarray()
    sample_a = AnnData(
        X=csr_matrix(dense),
        obs=tiny_adata.obs.copy(),
        var=tiny_adata.var.copy(),
    )
    sample_a.obsm["spatial"] = tiny_adata.obsm["spatial"].copy()
    sample_a.layers["counts"] = sample_a.X.copy()

    sample_b = AnnData(
        X=csr_matrix(dense * 10),
        obs=tiny_adata.obs.copy(),
        var=tiny_adata.var.copy(),
    )
    sample_b.obsm["spatial"] = tiny_adata.obsm["spatial"].copy()
    sample_b.layers["counts"] = sample_b.X.copy()

    merged = _concat_with_sample_ids([sample_a, sample_b], ["sample_a", "sample_b"])

    compute_qc_metrics(merged)
    filter_cells_and_genes(merged, minimum_cells=1)

    sample_b_rows = merged[merged.obs["sample_id"].astype(str) == "sample_b"]
    assert sample_b_rows.n_obs > 0


def test_filter_cells_and_genes_raises_when_qc_metrics_missing(
    tiny_adata: AnnData,
) -> None:
    """Filter refuses to run on obs that doesn't carry the required QC metric columns."""

    import pytest

    assert "log1p_total_counts" not in tiny_adata.obs.columns
    with pytest.raises(ValueError, match="compute_qc_metrics first"):
        filter_cells_and_genes(tiny_adata, minimum_cells=1)


def test_filter_cells_and_genes_applies_global_gene_filter(tiny_adata: AnnData) -> None:
    """The global min_cells gene filter drops genes below the minimum cell count."""

    dense = tiny_adata.X.toarray()
    dense[:, -1] = 0
    dense[0, -1] = 1
    tiny_adata.X = csr_matrix(dense)
    rare_gene = tiny_adata.var_names[-1]

    compute_qc_metrics(tiny_adata)
    filter_cells_and_genes(tiny_adata, minimum_cells=10)

    assert rare_gene not in tiny_adata.var_names


def test_filter_cells_and_genes_skips_zero_spread_metric(tiny_adata: AnnData) -> None:
    """When a sample's MAD collapses to zero on a metric, no cells are flagged on that metric."""

    from src.preprocessing import _flag_per_sample_mad_outliers

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=[20], log1p=True)

    # Force one metric to be constant across every cell: MAD on that metric
    # collapses to zero, so the 5-MAD band has no width.
    tiny_adata.obs["pct_counts_in_top_20_genes"] = 50.0

    mask = _flag_per_sample_mad_outliers(tiny_adata)
    # Only the two remaining metrics drive flagging; the constant one must not
    # drag every cell into the outlier set.
    assert mask.sum() < tiny_adata.n_obs


def test_normalize_and_scale_single_sample_hvg(tiny_adata: AnnData) -> None:
    """Single-sample run does not add the highly_variable_nbatches column."""

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(tiny_adata)

    assert "highly_variable" in tiny_adata.var.columns
    assert "highly_variable_nbatches" not in tiny_adata.var.columns
    assert "log_normalized" in tiny_adata.layers


def test_normalize_and_scale_multi_sample_uses_batch_key() -> None:
    """Multi-sample HVG selection adds the highly_variable_nbatches column."""

    sample_a = build_synthetic_adata(seed=0)
    sample_b = build_synthetic_adata(seed=1)
    merged = _concat_with_sample_ids([sample_a, sample_b], ["sample_a", "sample_b"])

    sc.pp.calculate_qc_metrics(merged, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(merged)

    assert "highly_variable_nbatches" in merged.var.columns
