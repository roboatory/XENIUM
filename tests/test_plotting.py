from __future__ import annotations

import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData

from src import analysis, colocalization, plotting
from src.config import Configuration
from src.preprocessing import normalize_and_scale


def _assert_nonempty_file(path) -> None:
    """Helper: the given path exists and has content."""

    assert path.exists(), f"expected figure at {path}"
    assert path.stat().st_size > 0, f"figure at {path} is empty"


def test_plot_cell_and_nucleus_boundaries_missing_element_is_skipped(
    configuration: Configuration,
    tiny_spatialdata: SpatialData,
) -> None:
    """The fixture lacks nucleus_boundaries, so this smoke test only covers cell_boundaries path."""

    # nucleus_boundaries is optional in the fixture; we patch it in by aliasing cell_boundaries.
    tiny_spatialdata.shapes["nucleus_boundaries"] = tiny_spatialdata.shapes[
        "cell_boundaries"
    ]
    plotting.plot_cell_and_nucleus_boundaries(configuration, tiny_spatialdata)

    output_path = configuration.figures_directory / "xenium_cell_nucleus_boundaries.png"
    _assert_nonempty_file(output_path)


def test_plot_qc_histogram_writes_figure(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_qc_histogram writes the transcripts-per-cell histogram PNG."""

    plotting.plot_qc_histogram(
        configuration,
        tiny_adata,
        cutoffs=[1, 0.99],
        cutoff_colors=["crimson", "goldenrod"],
    )

    output_path = configuration.figures_directory / "xenium_transcripts_per_cell.png"
    _assert_nonempty_file(output_path)


def test_plot_umap_leiden_writes_figure(
    configuration: Configuration,
    tiny_spatialdata: SpatialData,
) -> None:
    """plot_umap_leiden writes the UMAP figure after clustering and UMAP."""

    adata = tiny_spatialdata["table"]
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(adata)
    analysis.run_clustering(adata, pca_n_components=5)
    analysis.run_umap(adata)
    adata.obs["cell_type"] = adata.obs["leiden"].astype(str)

    plotting.plot_umap_leiden(configuration, tiny_spatialdata)
    _assert_nonempty_file(configuration.figures_directory / "umap_leiden.png")


def test_plot_cluster_overlay_writes_figure(
    configuration: Configuration,
    tiny_spatialdata: SpatialData,
) -> None:
    """plot_cluster_overlay writes a PNG named after the cluster key."""

    plotting.plot_cluster_overlay(
        configuration, tiny_spatialdata, cluster_key="cell_type"
    )
    _assert_nonempty_file(
        configuration.figures_directory / "xenium_cell_type_overlay.png"
    )


def test_plot_rank_genes_dotplot_writes_figure(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """plot_rank_genes_dotplot writes the dotplot PNG after rank_genes."""

    sc.pp.calculate_qc_metrics(tiny_adata, inplace=True, percent_top=None, log1p=False)
    normalize_and_scale(tiny_adata)
    analysis.run_clustering(tiny_adata, pca_n_components=5)
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
        "xenium_colocalization_contact_counts.png",
        "xenium_colocalization_contact_row_proportions.png",
        "xenium_colocalization_log2_fold_enrichment.png",
        "xenium_colocalization_log2_fold_enrichment_significant_only.png",
    ):
        _assert_nonempty_file(configuration.figures_directory / filename)
