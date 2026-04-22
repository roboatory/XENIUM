from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import median_abs_deviation

from .logging import get_logger

logger = get_logger(__name__)

QC_METRICS = (
    "log1p_total_counts",
    "log1p_n_genes_by_counts",
    "pct_counts_in_top_20_genes",
)
MAD_THRESHOLD = 5.0


def compute_qc_metrics(
    annotated_data: AnnData,
) -> None:
    """Populate log1p total counts, log1p n_genes_by_counts, and pct_counts_in_top_20_genes on obs."""

    sc.pp.calculate_qc_metrics(
        annotated_data,
        percent_top=[20],
        log1p=True,
        inplace=True,
    )


def compute_per_sample_mad_cutoffs(
    annotated_data: AnnData,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {sample_id: {metric: (lower, upper)}} 5-MAD bounds used by the filter and diagnostic."""

    obs = annotated_data.obs
    if "sample_id" in obs.columns:
        grouping = obs["sample_id"].astype(str)
    else:
        grouping = pd.Series("__single__", index=obs.index)

    cutoffs: dict[str, dict[str, tuple[float, float]]] = {}
    for sample_value, group in grouping.groupby(grouping):
        sample_cutoffs: dict[str, tuple[float, float]] = {}
        for metric in QC_METRICS:
            if metric not in obs.columns:
                continue
            sample_values = obs.loc[group.index, metric].to_numpy()
            center = float(np.median(sample_values))
            spread = float(median_abs_deviation(sample_values))
            if spread == 0:
                sample_cutoffs[metric] = (float("-inf"), float("inf"))
                continue
            sample_cutoffs[metric] = (
                center - MAD_THRESHOLD * spread,
                center + MAD_THRESHOLD * spread,
            )
        cutoffs[str(sample_value)] = sample_cutoffs
    return cutoffs


def filter_cells_and_genes(
    annotated_data: AnnData,
    minimum_cells: int,
) -> None:
    """Flag per-sample MAD outliers, drop them, then apply the global gene filter."""

    missing_metrics = [m for m in QC_METRICS if m not in annotated_data.obs.columns]
    if missing_metrics:
        raise ValueError(
            "filter_cells_and_genes requires QC metrics on obs; call "
            f"compute_qc_metrics first (missing: {missing_metrics})"
        )

    outlier_mask = _flag_per_sample_mad_outliers(annotated_data)
    n_flagged = int(outlier_mask.sum())
    logger.info(
        "MAD outlier filter flagged %s cells at %s MADs across %s",
        n_flagged,
        MAD_THRESHOLD,
        ", ".join(QC_METRICS),
    )
    annotated_data._inplace_subset_obs(~outlier_mask)

    sc.pp.filter_genes(annotated_data, min_cells=minimum_cells)
    logger.info(
        "QC retained %s cells and %s genes",
        annotated_data.n_obs,
        annotated_data.n_vars,
    )


def _flag_per_sample_mad_outliers(annotated_data: AnnData) -> np.ndarray:
    """Return a boolean mask of cells flagged as outliers by 5-MAD per-sample QC."""

    obs = annotated_data.obs
    has_sample_id = "sample_id" in obs.columns
    if has_sample_id:
        grouping = obs["sample_id"].astype(str)
    else:
        grouping = pd.Series("__single__", index=obs.index)

    outlier_mask = pd.Series(False, index=obs.index)
    for metric in QC_METRICS:
        values = obs[metric]
        grouped = values.groupby(grouping)
        center = grouped.transform("median")
        spread = grouped.transform(lambda group: median_abs_deviation(group.to_numpy()))
        # When a sample's MAD is zero the metric is degenerate (bimodal cluster
        # at the median, or a single cell). The 5-MAD band collapses to a point
        # and any deviation becomes an "outlier", which over-flags. Treat the
        # metric as uninformative for that sample instead.
        has_spread = spread > 0
        lower = center - MAD_THRESHOLD * spread
        upper = center + MAD_THRESHOLD * spread
        outlier_mask |= has_spread & ((values < lower) | (values > upper))

    return outlier_mask.to_numpy()


def normalize_and_scale(
    annotated_data: AnnData,
) -> None:
    """Normalize and scale the annotated data, picking HVGs with batch awareness when applicable."""

    logger.debug("normalizing and scaling %s cells", annotated_data.n_obs)
    hvg_kwargs = {"flavor": "seurat_v3", "n_top_genes": 2000}
    if (
        "sample_id" in annotated_data.obs.columns
        and annotated_data.obs["sample_id"].nunique() > 1
    ):
        hvg_kwargs["batch_key"] = "sample_id"
    sc.pp.highly_variable_genes(annotated_data, **hvg_kwargs)

    sc.pp.normalize_total(annotated_data)
    sc.pp.log1p(annotated_data)
    annotated_data.layers["log_normalized"] = annotated_data.X.copy()
    sc.pp.scale(annotated_data, zero_center=False, max_value=10)
    logger.debug("normalization and scaling complete")
