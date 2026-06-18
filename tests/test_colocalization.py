from __future__ import annotations

import numpy as np
import squidpy as sq
from anndata import AnnData

from src import colocalization


def _attach_spatial_graph(adata: AnnData, radius: float = 30.0) -> None:
    """Populate obsp['spatial_connectivities'] for a tiny AnnData."""

    sq.gr.spatial_neighbors(
        adata,
        radius=radius,
        coord_type="generic",
        delaunay=True,
    )


def test_compute_observed_contact_matrices_is_symmetric(tiny_adata: AnnData) -> None:
    """The observed contact matrix is symmetric with matching cell-type axes."""

    _attach_spatial_graph(tiny_adata)
    counts, proportions = colocalization.compute_observed_contact_matrices(tiny_adata)

    assert list(counts.index) == list(counts.columns)
    counts_array = counts.to_numpy()
    assert np.array_equal(counts_array, counts_array.T)
    row_sums = proportions.to_numpy().sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))


def test_compute_permutation_significance_outputs_have_expected_shape(
    tiny_adata: AnnData,
) -> None:
    """Permutation results have square frames keyed on cell types with a symmetric mask."""

    _attach_spatial_graph(tiny_adata)
    results = colocalization.compute_permutation_significance(
        tiny_adata,
        number_of_permutations=20,
        minimum_cells=3,
    )

    log2_enrichment = results["log2_fold_enrichment"]
    mask = results["significant_mask"]
    assert log2_enrichment.shape == mask.shape
    assert list(log2_enrichment.index) == list(log2_enrichment.columns)
    mask_array = mask.to_numpy()
    assert np.array_equal(mask_array, mask_array.T)


def test_compute_permutation_significance_skips_when_too_few_types(
    tiny_adata: AnnData,
) -> None:
    """If fewer than two cell types meet minimum_cells, results are empty frames."""

    _attach_spatial_graph(tiny_adata)
    results = colocalization.compute_permutation_significance(
        tiny_adata,
        number_of_permutations=10,
        minimum_cells=10_000,
    )

    assert results["log2_fold_enrichment"].shape == (0, 0)
    assert results["significant_mask"].shape == (0, 0)


def test_symmetric_counts_matches_manual_count() -> None:
    """_symmetric_counts produces the expected symmetric count matrix."""

    rows = np.array([0, 0, 1])
    cols = np.array([1, 2, 2])
    counts = colocalization._symmetric_counts(rows, cols, n_types=3)

    expected = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(counts, expected)


def test_benjamini_hochberg_matches_scipy() -> None:
    """_benjamini_hochberg matches scipy's FDR implementation on a flat vector."""

    from scipy.stats import false_discovery_control

    p_values = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
    result = colocalization._benjamini_hochberg(p_values)
    expected = np.asarray(false_discovery_control(p_values, method="bh"))
    assert np.allclose(result, expected)


def test_sample_partitions_cover_eligible_cells_without_overlap(
    tiny_adata: AnnData,
) -> None:
    """_sample_partitions falls back to sample_id when core_id is absent."""

    import anndata as ad

    merged = ad.concat(
        [tiny_adata.copy(), tiny_adata.copy()],
        keys=["sample_a", "sample_b"],
        label="sample_id",
        index_unique="_",
    )
    merged.obs.index.name = None
    eligible_index = np.arange(merged.n_obs)

    partitions = colocalization._sample_partitions(merged, eligible_index)
    assert len(partitions) == 2
    covered = np.concatenate(partitions)
    assert covered.size == merged.n_obs
    assert len(set(covered.tolist())) == merged.n_obs


def test_sample_partitions_prefer_core_id(tiny_adata: AnnData) -> None:
    """_sample_partitions keeps permutations within cores when core labels exist."""

    tiny_adata.obs["sample_id"] = "sample_a"
    tiny_adata.obs["core_id"] = ["core_1"] * 40 + ["core_2"] * 40 + ["core_3"] * 40
    eligible_index = np.arange(tiny_adata.n_obs)

    partitions = colocalization._sample_partitions(tiny_adata, eligible_index)

    assert len(partitions) == 3
    assert [partition.size for partition in partitions] == [40, 40, 40]


def test_permutation_significance_does_not_mutate_obs_cell_type(
    tiny_adata: AnnData,
) -> None:
    """compute_permutation_significance does not modify adata.obs (permutations are local)."""

    _attach_spatial_graph(tiny_adata, radius=30.0)
    before = tiny_adata.obs["cell_type"].astype(str).to_numpy().copy()

    colocalization.compute_permutation_significance(
        tiny_adata,
        number_of_permutations=5,
        minimum_cells=3,
    )
    after = tiny_adata.obs["cell_type"].astype(str).to_numpy()
    assert (before == after).all()


def test_benjamini_hochberg_handles_empty_input() -> None:
    """_benjamini_hochberg on an empty array returns an empty float array."""

    result = colocalization._benjamini_hochberg(np.array([]))
    assert result.size == 0
    assert result.dtype == np.float64
