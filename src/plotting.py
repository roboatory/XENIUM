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

from .config import Configuration
from .logging import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 300
_TAB20 = [matplotlib.colors.rgb2hex(c) for c in matplotlib.colormaps["tab20"].colors]


def _build_categorical_palette(n_categories: int) -> list[str]:
    """Return a list of n hex colors, using tab20 and extending with HSL if needed."""

    if n_categories <= len(_TAB20):
        return _TAB20[:n_categories]
    extra = [
        matplotlib.colors.hsv_to_rgb((i / n_categories, 0.65, 0.85))
        for i in range(n_categories)
    ]
    return [matplotlib.colors.rgb2hex(c) for c in extra]


@contextmanager
def _suppress_show() -> Iterator[None]:
    """Temporarily suppress plt.show calls from third-party plotting code."""

    original_show = plt.show
    plt.show = lambda: None
    try:
        yield
    finally:
        plt.show = original_show


def plot_qc_histogram(
    configuration: Configuration,
    annotated_data: AnnData,
    cutoffs_by_sample: dict[str, dict[str, tuple[float, float]]],
) -> None:
    """Faceted per-sample QC histogram of log1p_total_counts with MAD cutoffs as vlines."""

    out_path = configuration.figures_directory / "xenium_qc_histograms.png"
    metric = "log1p_total_counts"
    if metric not in annotated_data.obs.columns:
        logger.debug("skipping QC histogram: %s not on obs", metric)
        return

    if "sample_id" in annotated_data.obs.columns:
        grouping = annotated_data.obs["sample_id"].astype(str)
    else:
        grouping = pd.Series("__single__", index=annotated_data.obs.index)
    sample_ids = sorted(grouping.unique().tolist())
    if not sample_ids:
        return

    logger.debug("rendering faceted QC histogram to %s", out_path)

    with _suppress_show():
        n_samples = len(sample_ids)
        fig, axes = plt.subplots(
            1,
            n_samples,
            figsize=(max(6, 5 * n_samples), 4),
            dpi=FIGURE_DPI,
            squeeze=False,
        )
        axes = axes[0]
        for axis, sample_id in zip(axes, sample_ids):
            sample_mask = (grouping == sample_id).to_numpy()
            values = annotated_data.obs.loc[sample_mask, metric].to_numpy()
            axis.hist(values, bins=80, color="black", alpha=0.8)
            bounds = cutoffs_by_sample.get(sample_id, {}).get(metric)
            if bounds is not None:
                lower, upper = bounds
                for cutoff in (lower, upper):
                    if np.isfinite(cutoff):
                        axis.axvline(
                            cutoff,
                            color="crimson",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.8,
                        )
            axis.set_xlabel(metric)
            axis.set_ylabel("cells")
            axis.set_title(sample_id)
        fig.suptitle("QC: 5-MAD cutoffs on log1p total counts per sample")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_umap_leiden(
    configuration: Configuration,
    annotated_data: AnnData,
) -> None:
    """Plot the composite UMAP colored by cell_type."""

    out_path = configuration.figures_directory / "umap_leiden.png"
    logger.debug("rendering UMAP plot to %s", out_path)

    with _suppress_show():
        n_types = annotated_data.obs["cell_type"].nunique()
        palette = _build_categorical_palette(n_types)
        umap_figure = sc.pl.umap(
            annotated_data,
            color="cell_type",
            palette=palette,
            show=False,
            return_fig=True,
        )
        umap_figure.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(umap_figure)
        logger.debug("saved figure %s", out_path)


def plot_cluster_overlay(
    configuration: Configuration,
    annotated_data: AnnData,
    cluster_key: str,
    sample_id: str | None = None,
) -> None:
    """Plot a per-sample centroid scatter colored by the given cluster label column."""

    if sample_id is not None:
        if "sample_id" not in annotated_data.obs.columns:
            raise ValueError(
                f"sample_id={sample_id!r} requested but adata has no 'sample_id' column"
            )
        sample_mask = (
            annotated_data.obs["sample_id"].astype(str) == sample_id
        ).to_numpy()
        if not sample_mask.any():
            logger.warning(
                "no cells found for sample_id=%r; skipping overlay", sample_id
            )
            return
        view = annotated_data[sample_mask]
    else:
        view = annotated_data

    # Build the cluster->color map from the full dataset so the same cluster keeps
    # the same color across every per-sample plot.
    all_categories = (
        annotated_data.obs[cluster_key].dropna().astype("category").cat.categories
    )
    palette = _build_categorical_palette(len(all_categories))
    color_map = dict(zip(all_categories, palette))

    valid_mask = view.obs[cluster_key].notna().to_numpy()
    coordinates = np.asarray(view.obsm["spatial"])[valid_mask]
    cluster_values = view.obs[cluster_key].dropna()
    if coordinates.size == 0:
        logger.debug(
            "skipping overlay for %s/%s: no labeled cells", cluster_key, sample_id
        )
        return

    if sample_id is not None:
        out_dir = configuration.figures_directory / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"xenium_{cluster_key}_overlay.png"
    else:
        out_path = configuration.figures_directory / f"xenium_{cluster_key}_overlay.png"
    logger.debug("rendering overlay for %s to %s", cluster_key, out_path)

    with _suppress_show():
        fig, ax = plt.subplots(figsize=(14, 14), dpi=FIGURE_DPI)
        for category in all_categories:
            in_category = (cluster_values == category).to_numpy()
            if not in_category.any():
                continue
            ax.scatter(
                coordinates[in_category, 0],
                coordinates[in_category, 1],
                s=4,
                color=color_map[category],
                label=str(category),
                alpha=0.85,
                linewidths=0,
            )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{cluster_key}{' — ' + sample_id if sample_id else ''}")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            markerscale=2,
            title=cluster_key,
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def plot_harmony_diagnostic(
    configuration: Configuration,
    annotated_data: AnnData,
) -> None:
    """Emit a side-by-side pre/post-Harmony UMAP colored by sample_id."""

    if "X_umap_uncorrected" not in annotated_data.obsm:
        logger.debug("skipping Harmony diagnostic: no uncorrected UMAP present")
        return
    if "sample_id" not in annotated_data.obs.columns:
        logger.debug("skipping Harmony diagnostic: no sample_id column")
        return

    out_path = configuration.figures_directory / "harmony_umap_before_after.png"
    logger.debug("rendering Harmony diagnostic to %s", out_path)

    sample_categories = annotated_data.obs["sample_id"].astype("category")
    palette = _build_categorical_palette(sample_categories.cat.categories.size)

    with _suppress_show():
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=FIGURE_DPI)

        _scatter_umap(
            axes[0],
            annotated_data.obsm["X_umap_uncorrected"],
            sample_categories,
            palette,
            title="before Harmony",
        )
        _scatter_umap(
            axes[1],
            annotated_data.obsm["X_umap"],
            sample_categories,
            palette,
            title="after Harmony",
        )
        axes[1].legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=9,
            markerscale=2,
            title="sample_id",
        )

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        logger.debug("saved figure %s", out_path)


def _scatter_umap(
    axis: plt.Axes,
    coordinates: np.ndarray,
    categories: pd.Series,
    palette: list[str],
    title: str,
) -> None:
    """Scatter the given UMAP coordinates colored by the given categorical series."""

    for color, category in zip(palette, categories.cat.categories):
        mask = (categories == category).to_numpy()
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=3,
            color=color,
            label=str(category),
            alpha=0.7,
            linewidths=0,
        )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title(title)
    axis.set_aspect("equal")


def plot_rank_genes_dotplot(
    configuration: Configuration,
    annotated_data: AnnData,
    n_genes: int = 5,
) -> None:
    """Plot rank genes groups dotplot."""

    out_path = configuration.figures_directory / f"rank_genes_dotplot_top_{n_genes}.png"
    logger.debug("rendering ranked-genes dotplot to %s", out_path)

    # Precompute the dendrogram on X_pca so scanpy does not fall back to a
    # groupby-mean over obs, which breaks on sparse X and object-dtype columns.
    sc.tl.dendrogram(annotated_data, groupby="leiden", use_rep="X_pca")

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

    if finite_values.size == 0:
        max_abs = 1.0
    else:
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
