from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData

from src import analysis, colocalization, plotting
from src.config import Configuration
from src.preprocessing import normalize_and_scale


def _assert_nonempty_file(path) -> None:
    """Helper: the given path exists and has content."""

    assert path.exists(), f"expected figure at {path}"
    assert path.stat().st_size > 0, f"figure at {path} is empty"


def test_plot_cluster_overlay_writes_per_sample_centroid_scatter(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_cluster_overlay emits a per-sample PNG using obsm['spatial']."""

    tiny_adata.obs["sample_id"] = ["sample_a"] * tiny_adata.n_obs

    plotting.plot_cluster_overlay(
        configuration,
        tiny_adata,
        cluster_key="cell_type",
        sample_id="sample_a",
    )
    _assert_nonempty_file(
        configuration.figures_directory / "cell_type_overlays" / "sample_a.png"
    )


def test_plot_cluster_overlay_filters_to_requested_sample(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """A sample_id with no matching cells produces no file (and no crash)."""

    tiny_adata.obs["sample_id"] = ["sample_a"] * tiny_adata.n_obs

    plotting.plot_cluster_overlay(
        configuration,
        tiny_adata,
        cluster_key="cell_type",
        sample_id="missing_sample",
    )
    assert not (
        configuration.figures_directory / "cell_type_overlays" / "missing_sample.png"
    ).exists()


def test_plot_cluster_overlay_raises_when_sample_id_column_missing(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """Requesting a sample_id when the column is absent raises rather than mislabeling output."""

    import pytest

    assert "sample_id" not in tiny_adata.obs.columns
    with pytest.raises(ValueError, match="no 'sample_id' column"):
        plotting.plot_cluster_overlay(
            configuration,
            tiny_adata,
            cluster_key="cell_type",
            sample_id="patient_001",
        )


def test_plot_cluster_overlay_uses_global_palette_size(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """Per-sample plots derive their palette from the full dataset's categories."""

    from unittest.mock import patch

    import anndata as ad

    # Build a merged adata where each sample has a strict subset of categories
    # to make per-sample category-set != full category-set.
    sample_a = tiny_adata[tiny_adata.obs["cell_type"].astype(str) != "type_2"].copy()
    sample_b = tiny_adata[tiny_adata.obs["cell_type"].astype(str) != "type_0"].copy()
    merged = ad.concat(
        [sample_a, sample_b],
        keys=["sample_a", "sample_b"],
        label="sample_id",
        index_unique="_",
    )
    merged.obs.index.name = None
    full_category_count = merged.obs["cell_type"].astype(str).nunique()

    palette_call_sizes: list[int] = []
    real_builder = plotting._build_categorical_palette

    def recording_builder(n_categories: int) -> list[str]:
        palette_call_sizes.append(n_categories)
        return real_builder(n_categories)

    with patch.object(
        plotting, "_build_categorical_palette", side_effect=recording_builder
    ):
        plotting.plot_cluster_overlay(
            configuration, merged, cluster_key="cell_type", sample_id="sample_a"
        )
        plotting.plot_cluster_overlay(
            configuration, merged, cluster_key="cell_type", sample_id="sample_b"
        )

    assert palette_call_sizes == [full_category_count, full_category_count]


def test_plot_umap_leiden_takes_anndata_directly(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_umap_leiden writes a UMAP PNG when given an AnnData with cell_type and X_umap."""

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5, leiden_resolution=0.5)
    tiny_adata.obs["cell_type"] = tiny_adata.obs["leiden"].astype(str)

    plotting.plot_umap_leiden(configuration, tiny_adata)
    _assert_nonempty_file(configuration.figures_directory / "umap_leiden.png")


def test_plot_qc_histogram_faceted_by_sample(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_qc_histogram writes a faceted PNG using per-sample MAD cutoffs."""

    import anndata as ad

    a = tiny_adata.copy()
    b = tiny_adata.copy()
    merged = ad.concat(
        [a, b], keys=["sample_a", "sample_b"], label="sample_id", index_unique="_"
    )
    merged.obs.index.name = None
    sc.pp.calculate_qc_metrics(merged, inplace=True, percent_top=[20], log1p=True)
    from src.preprocessing import compute_per_sample_mad_cutoffs

    cutoffs = compute_per_sample_mad_cutoffs(merged, mad_threshold=5.0)

    plotting.plot_qc_histogram(configuration, merged, cutoffs)
    _assert_nonempty_file(configuration.figures_directory / "xenium_qc_histograms.png")


def test_plot_harmony_diagnostic_writes_figure(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_harmony_diagnostic writes a side-by-side pre/post UMAP composite."""

    tiny_adata.obs["sample_id"] = ["sample_a"] * (tiny_adata.n_obs // 2) + [
        "sample_b"
    ] * (tiny_adata.n_obs - tiny_adata.n_obs // 2)
    tiny_adata.obsm["X_umap_uncorrected"] = np.random.default_rng(0).normal(
        size=(tiny_adata.n_obs, 2)
    )
    tiny_adata.obsm["X_umap"] = np.random.default_rng(1).normal(
        size=(tiny_adata.n_obs, 2)
    )

    plotting.plot_harmony_diagnostic(configuration, tiny_adata)
    _assert_nonempty_file(
        configuration.figures_directory / "harmony_umap_before_after.png"
    )


def test_plot_harmony_diagnostic_skips_when_no_uncorrected_umap(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """The diagnostic is a no-op when the uncorrected UMAP is absent (single-sample run)."""

    plotting.plot_harmony_diagnostic(configuration, tiny_adata)
    assert not (
        configuration.figures_directory / "harmony_umap_before_after.png"
    ).exists()


def test_plot_rank_genes_dotplot_writes_figure(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_rank_genes_dotplot writes the dotplot PNG after rank_genes."""

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5, leiden_resolution=0.5)
    analysis.rank_genes(tiny_adata)
    sc.tl.dendrogram(tiny_adata, groupby="leiden", use_rep="X_pca")

    plotting.plot_rank_genes_dotplot(configuration, tiny_adata)
    _assert_nonempty_file(
        configuration.figures_directory / "rank_genes_dotplot_top_5.png"
    )


def test_plot_colocalization_heatmaps_write_figures(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """All colocalization heatmap plots write non-empty PNGs."""

    import squidpy as sq

    sq.gr.spatial_neighbors(
        tiny_adata, radius=30.0, coord_type="generic", delaunay=True
    )
    counts, proportions = colocalization.compute_observed_contact_matrices(tiny_adata)
    results = colocalization.compute_permutation_significance(
        tiny_adata,
        number_of_permutations=10,
        minimum_cells=3,
    )

    plotting.plot_colocalization_contact_counts(configuration, counts)
    plotting.plot_colocalization_contact_row_proportions(configuration, proportions)
    plotting.plot_colocalization_log2_fold_enrichment(
        configuration, results["log2_fold_enrichment"]
    )
    plotting.plot_colocalization_log2_fold_enrichment_significant_only(
        configuration,
        results["log2_fold_enrichment"],
        results["significant_mask"],
    )

    for filename in (
        "contact_counts.png",
        "contact_row_proportions.png",
        "log2_fold_enrichment.png",
        "log2_fold_enrichment_significant_only.png",
    ):
        _assert_nonempty_file(
            configuration.figures_directory / "colocalizations" / filename
        )
