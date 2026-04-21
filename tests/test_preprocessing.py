from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from src.preprocessing import filter_cells_and_genes, normalize_and_scale


def test_filter_cells_removes_cells_below_min_counts(tiny_adata: AnnData) -> None:
    """filter_cells_and_genes drops cells whose total_counts falls below min_counts."""

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=None, log1p=False)
    target_threshold = int(tiny_adata.obs["total_counts"].quantile(0.5))
    n_before = tiny_adata.n_obs

    filter_cells_and_genes(
        tiny_adata,
        minimum_counts=target_threshold,
        maximum_counts_quantile=1.0,
        minimum_cells=1,
    )

    assert tiny_adata.n_obs < n_before
    assert (tiny_adata.obs["total_counts"] >= target_threshold).all()


def test_filter_genes_removes_genes_below_min_cells() -> None:
    """filter_cells_and_genes drops genes expressed in fewer cells than min_cells."""

    counts = np.zeros((20, 5), dtype=np.float32)
    counts[:, 0] = 10.0
    counts[:10, 1] = 5.0
    counts[:2, 2] = 1.0

    obs = pd.DataFrame(
        {
            "total_counts": counts.sum(axis=1),
            "cell_id": [f"c{i}" for i in range(20)],
        },
        index=[f"c{i}" for i in range(20)],
    )
    var = pd.DataFrame(index=pd.Index([f"g{i}" for i in range(5)], name="gene_id"))
    adata = AnnData(X=counts, obs=obs, var=var)
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)

    filter_cells_and_genes(
        adata,
        minimum_counts=1,
        maximum_counts_quantile=1.0,
        minimum_cells=5,
    )

    assert set(adata.var_names) == {"g0", "g1"}


def test_normalize_and_scale_sets_expected_anndata_state(tiny_adata: AnnData) -> None:
    """normalize_and_scale computes HVGs, writes log_normalized layer, and scales X."""

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(tiny_adata)

    assert "highly_variable" in tiny_adata.var.columns
    assert "log_normalized" in tiny_adata.layers
    assert tiny_adata.X.max() <= 10.0 + 1e-6
