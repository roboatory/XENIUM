from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData
import spatialdata_plot  # noqa: F401

from .config import Configuration
from .logging import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 300


@contextmanager
def _suppress_show() -> Iterator[None]:
    """Temporarily suppress plt.show calls from third-party plotting code."""

    original_show = plt.show
    plt.show = lambda: None
    try:
        yield
    finally:
        plt.show = original_show


def plot_cell_and_nucleus_boundaries(
    configuration: Configuration,
    spatial_data: SpatialData,
) -> None:
    """Plot cell and nucleus boundaries."""

    out_path = configuration.figures_directory / "xenium_cell_nucleus_boundaries.png"
    logger.debug("rendering cell and nucleus boundary plot to %s", out_path)

    with _suppress_show():
        fig, ax = plt.subplots(figsize=(14, 14), dpi=FIGURE_DPI)

        (
            spatial_data.pl.render_shapes(
                element="cell_boundaries",
                fill_alpha=0.0,
                outline_color="#2d2d2d",
                outline_width=0.2,
                outline_alpha=1.0,
            )
            .pl.render_shapes(
                element="nucleus_boundaries",
                fill_alpha=0.0,
                outline_color="#e64b35",
                outline_width=0.35,
                outline_alpha=1.0,
            )
            .pl.show(ax=ax, title="", dpi=FIGURE_DPI)
        )

        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_transcripts(
    configuration: Configuration,
    spatial_data: SpatialData,
    genes: list[str],
    palette: list[str],
    max_points: int = 50_000,
) -> None:
    """Plot transcripts."""

    out_path = (
        configuration.figures_directory / f"xenium_transcripts_{'_'.join(genes)}.png"
    )
    logger.debug("rendering transcript plot for genes=%s to %s", genes, out_path)

    with _suppress_show():
        fig, ax = plt.subplots(figsize=(14, 14), dpi=FIGURE_DPI)

        (
            spatial_data.pl.render_points(
                element="transcripts",
                color="feature_name",
                groups=genes,
                size=2,
                max_points=max_points,
                palette=palette,
            ).pl.show(ax=ax, title="", dpi=FIGURE_DPI, colorbar=False)
        )

        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_qc_histogram(
    configuration: Configuration,
    annotated_data: AnnData,
    cutoffs: list[float],
    cutoff_colors: list[str],
) -> None:
    """Plot QC histogram."""

    out_path = configuration.figures_directory / "xenium_transcripts_per_cell.png"
    logger.debug("rendering QC histogram to %s", out_path)

    cutoffs = [
        int(cutoffs[0]),
        int(np.quantile(annotated_data.obs["total_counts"], cutoffs[1])),
    ]

    with _suppress_show():
        figure, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)
        ax.hist(annotated_data.obs["total_counts"], bins=100, color="black", alpha=0.8)
        ax.set_xlabel("transcripts per cell")
        ax.set_ylabel("number of cells")
        ax.set_title("QC: transcripts per cell (full range)")

        for index, (cutoff, color) in enumerate(zip(cutoffs, cutoff_colors)):
            ax.axvline(cutoff, color=color, linestyle="--", linewidth=2, alpha=0.8)
            ax.text(
                cutoff,
                ax.get_ylim()[1] * (0.9 - 0.05 * index),
                f"cutoff={cutoff}",
                color=color,
                fontsize=12,
                ha="left",
                va="top",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.2),
            )

        figure.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(figure)
        logger.debug("saved figure %s", out_path)


def plot_umap_leiden(
    configuration: Configuration,
    spatial_data: SpatialData,
) -> None:
    """Plot UMAP colored by Leiden clusters."""

    out_path = configuration.figures_directory / "umap_leiden.png"
    annotated_data = spatial_data["table"]
    logger.debug("rendering UMAP plot to %s", out_path)

    with _suppress_show():
        umap_figure = sc.pl.umap(
            annotated_data,
            color="cell_type",
            show=False,
            return_fig=True,
        )
        umap_figure.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(umap_figure)
        logger.debug("saved figure %s", out_path)


def plot_cluster_overlay(
    configuration: Configuration,
    spatial_data: SpatialData,
    cluster_key: str = "cell_type",
) -> None:
    """Plot cell shapes colored by cluster labels (e.g. Leiden)."""

    out_path = configuration.figures_directory / f"xenium_{cluster_key}_overlay.png"
    logger.debug("rendering overlay for %s to %s", cluster_key, out_path)
    table = spatial_data.tables["table"]
    gdf = spatial_data.shapes["cell_boundaries"].copy()

    cell_to_cluster = table.obs.set_index("cell_id")[cluster_key]
    gdf[cluster_key] = gdf.index.map(cell_to_cluster)
    gdf = gdf.dropna(subset=[cluster_key])

    with _suppress_show():
        fig, ax = plt.subplots(figsize=(14, 14), dpi=FIGURE_DPI)
        n_clusters = max(gdf[cluster_key].nunique(), 1)
        cmap = matplotlib.colormaps["Set3"].resampled(n_clusters)
        gdf.plot(
            column=cluster_key,
            ax=ax,
            categorical=True,
            legend=True,
            edgecolor="gray",
            linewidth=0.15,
            cmap=cmap,
            legend_kwds={
                "loc": "center left",
                "bbox_to_anchor": (1, 0.5),
                "fontsize": 9,
            },
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.invert_yaxis()
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_rank_genes_dotplot(
    configuration: Configuration,
    annotated_data: AnnData,
    n_genes: int = 5,
) -> None:
    """Plot rank genes groups dotplot."""

    out_path = configuration.figures_directory / f"rank_genes_dotplot_top_{n_genes}.png"
    logger.debug("rendering ranked-genes dotplot to %s", out_path)

    with _suppress_show():
        dotplot = sc.pl.rank_genes_groups_dotplot(
            annotated_data,
            n_genes=n_genes,
            show=False,
            return_fig=True,
        )
        dotplot.make_figure()
        fig = dotplot.fig
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_colocalization_contact_counts(
    configuration: Configuration,
    counts: pd.DataFrame,
) -> None:
    """Plot heatmap of observed cell-type contact counts."""

    out_path = (
        configuration.figures_directory / "xenium_colocalization_contact_counts.png"
    )
    logger.debug("rendering colocalization count heatmap to %s", out_path)

    with _suppress_show():
        fig, ax = plt.subplots(figsize=(12, 10), dpi=FIGURE_DPI)
        image = ax.imshow(np.log1p(counts.to_numpy()), cmap="magma")
        ax.set_xticks(np.arange(counts.shape[1]))
        ax.set_yticks(np.arange(counts.shape[0]))
        ax.set_xticklabels(counts.columns)
        ax.set_yticklabels(counts.index)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("cell type")
        ax.set_ylabel("cell type")
        ax.set_title("observed first degree contacts")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("log1p(contact count)")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_colocalization_contact_row_proportions(
    configuration: Configuration,
    proportions: pd.DataFrame,
) -> None:
    """Plot heatmap of row-normalized observed cell-type contact proportions."""

    out_path = (
        configuration.figures_directory
        / "xenium_colocalization_contact_row_proportions.png"
    )
    logger.debug("rendering row-normalized colocalization heatmap to %s", out_path)

    with _suppress_show():
        fig, ax = plt.subplots(figsize=(12, 10), dpi=FIGURE_DPI)
        image = ax.imshow(proportions.to_numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(proportions.shape[1]))
        ax.set_yticks(np.arange(proportions.shape[0]))
        ax.set_xticklabels(proportions.columns)
        ax.set_yticklabels(proportions.index)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("cell type")
        ax.set_ylabel("cell type")
        ax.set_title("observed first degree contacts (row-normalized)")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("row-normalized contact proportion")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_colocalization_log2_fold_enrichment(
    configuration: Configuration,
    log2_fold_enrichment: pd.DataFrame,
) -> None:
    """Plot heatmap of log2 fold enrichment from permutation testing."""

    out_path = (
        configuration.figures_directory
        / "xenium_colocalization_log2_fold_enrichment.png"
    )
    logger.debug("rendering log2 fold enrichment heatmap to %s", out_path)

    with _suppress_show():
        figure, axis = plt.subplots(figsize=(12, 10), dpi=FIGURE_DPI)
        _render_log2_enrichment_heatmap(
            axis, log2_fold_enrichment, title="first-degree contact enrichment"
        )
        figure.tight_layout()
        figure.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(figure)
        logger.debug("saved figure %s", out_path)


def plot_colocalization_log2_fold_enrichment_significant_only(
    configuration: Configuration,
    log2_fold_enrichment: pd.DataFrame,
    significant_mask: pd.DataFrame,
) -> None:
    """Plot heatmap of log2 fold enrichment for significant pairs only."""

    out_path = (
        configuration.figures_directory
        / "xenium_colocalization_log2_fold_enrichment_significant_only.png"
    )
    logger.debug("rendering significant-only enrichment heatmap to %s", out_path)
    significant_only = log2_fold_enrichment.where(significant_mask, np.nan)

    with _suppress_show():
        figure, axis = plt.subplots(figsize=(12, 10), dpi=FIGURE_DPI)
        _render_log2_enrichment_heatmap(
            axis,
            significant_only,
            title="significant first-degree contact enrichment",
        )
        figure.tight_layout()
        figure.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(figure)
        logger.debug("saved figure %s", out_path)


def _render_log2_enrichment_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
) -> None:
    """Render an enrichment heatmap or a no-data placeholder."""

    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]

    max_abs = float(np.nanmax(np.abs(finite_values)))
    if max_abs == 0.0:
        max_abs = 1.0

    image = axis.imshow(values, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_yticks(np.arange(matrix.shape[0]))
    axis.set_xticklabels(matrix.columns)
    axis.set_yticklabels(matrix.index)
    axis.tick_params(axis="x", labelrotation=90)
    axis.set_xlabel("cell type")
    axis.set_ylabel("cell type")
    axis.set_title(title)
    colorbar = axis.figure.colorbar(image, ax=axis)
    colorbar.set_label("log2(fold enrichment)")
