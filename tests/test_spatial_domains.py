from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from src import spatial_domains


def test_compute_neighborhood_composition_rows_sum_to_one(tiny_adata: AnnData) -> None:
    """Each cell's neighborhood composition vector sums to 1 or to 0 when isolated."""

    spatial_domains.compute_neighborhood_composition(tiny_adata, radius=30.0)
    composition = np.asarray(tiny_adata.obsm["neighborhood_composition"])

    row_sums = composition.sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))
    n_categories = tiny_adata.obs["cell_type"].nunique()
    assert composition.shape == (tiny_adata.n_obs, n_categories)


def test_assign_spatial_domains_populates_categorical_labels(
    tiny_adata: AnnData,
) -> None:
    """assign_spatial_domains writes a categorical spatial_domain column with n_clusters levels."""

    spatial_domains.compute_neighborhood_composition(tiny_adata, radius=30.0)
    spatial_domains.assign_spatial_domains(tiny_adata, n_clusters=3)

    assert "spatial_domain" in tiny_adata.obs.columns
    assert isinstance(tiny_adata.obs["spatial_domain"].dtype, pd.CategoricalDtype)
    assert tiny_adata.obs["spatial_domain"].nunique() <= 3


def test_spatial_neighbors_is_block_diagonal_across_samples(
    tiny_adata: AnnData,
) -> None:
    """With multi-sample data, spatial_connectivities has zero cross-sample edges."""

    import anndata as ad

    a = tiny_adata.copy()
    b = tiny_adata.copy()
    b.obsm["spatial"] = b.obsm["spatial"] + np.array([200.0, 200.0]).astype(
        b.obsm["spatial"].dtype
    )
    merged = ad.concat(
        [a, b], keys=["sample_a", "sample_b"], label="sample_id", index_unique="_"
    )
    merged.obs.index.name = None

    spatial_domains.compute_neighborhood_composition(merged, radius=30.0)

    connectivities = merged.obsp["spatial_connectivities"].tocoo()
    sample_of_cell = merged.obs["sample_id"].astype(str).to_numpy()
    cross_sample_edges = (
        sample_of_cell[connectivities.row] != sample_of_cell[connectivities.col]
    )
    assert not cross_sample_edges.any()


def test_build_domain_signatures_is_sorted_descending(tiny_adata: AnnData) -> None:
    """Each domain signature is a list of (cell_type, proportion) sorted descending."""

    spatial_domains.compute_neighborhood_composition(tiny_adata, radius=30.0)
    spatial_domains.assign_spatial_domains(tiny_adata, n_clusters=3)
    signatures = spatial_domains.build_domain_signatures(tiny_adata)

    assert set(signatures.keys()) == set(
        tiny_adata.obs["spatial_domain"].astype(str).unique()
    )
    for components in signatures.values():
        proportions = [proportion for _, proportion in components]
        assert proportions == sorted(proportions, reverse=True)
