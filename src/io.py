from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import anndata as ad
from anndata import AnnData

from .config import Configuration
from .logging import get_logger

logger = get_logger(__name__)


def read_processed_anndata(
    configuration: Configuration,
) -> AnnData:
    """Read the merged processed AnnData from disk."""

    path = configuration.processed_data_directory / "processed.h5ad"
    logger.debug("reading processed anndata from %s", path)
    return ad.read_h5ad(path)


def write_processed_anndata(
    configuration: Configuration,
    annotated_data: AnnData,
) -> None:
    """Write the merged processed AnnData using atomic temp-then-rename."""

    target_path = configuration.processed_data_directory / "processed.h5ad"
    temporary_path = target_path.parent / f".{target_path.name}.tmp-{uuid4().hex}"

    annotated_data.write_h5ad(temporary_path)

    try:
        if target_path.exists():
            target_path.unlink()
        temporary_path.rename(target_path)
        logger.debug("wrote processed anndata to %s", target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def read_enriched_genes(
    configuration: Configuration,
) -> dict[str, list[str]]:
    """Load enriched genes per cluster from the analysis directory JSON."""

    enriched_genes_path = (
        configuration.results_directory / "cluster_enriched_genes.json"
    )
    with enriched_genes_path.open("r") as file_handle:
        data = json.load(file_handle)

    enriched_genes = {
        str(cluster_id): [str(gene) for gene in genes]
        for cluster_id, genes in data.items()
    }
    logger.debug("loaded enriched gene sets from %s", enriched_genes_path)
    return enriched_genes


def write_enriched_genes(
    configuration: Configuration,
    enriched_genes: dict[str, list[str]],
) -> None:
    """Write enriched genes JSON artifact."""

    enriched_genes_path = (
        configuration.results_directory / "cluster_enriched_genes.json"
    )
    with enriched_genes_path.open("w") as file_handle:
        json.dump(enriched_genes, file_handle, indent=2)
    logger.debug("wrote enriched gene sets to %s", enriched_genes_path)


def write_annotations(
    configuration: Configuration,
    annotations: dict[str, dict[str, str | float]],
    target: str,
) -> None:
    """Write cluster/domain annotation JSON artifacts."""

    annotation_paths = {
        "cluster": configuration.results_directory / "cluster_annotations.json",
        "domain": configuration.results_directory / "spatial_domain_annotations.json",
    }
    annotations_path = annotation_paths[target]
    with annotations_path.open("w") as file_handle:
        json.dump(annotations, file_handle, indent=2)
    logger.debug("wrote %s annotations to %s", target, annotations_path)


def write_labels(
    configuration: Configuration,
    annotated_data: AnnData,
    target: str,
) -> None:
    """Write cluster/domain label CSV artifacts."""

    if target == "cluster":
        label_column = "leiden"
        output_path = configuration.results_directory / "leiden_clusters.csv"
    elif target == "domain":
        label_column = "spatial_domain"
        output_path = configuration.results_directory / "spatial_domain_labels.csv"

    dataframe = annotated_data.obs[[label_column]].rename(
        columns={label_column: "group"}
    )
    dataframe.insert(0, "cell_id", annotated_data.obs["cell_id"])
    dataframe.to_csv(output_path, index=False)
    logger.debug("wrote %s labels to %s", target, output_path)


def save_state(
    path: Path,
    state_payload: dict[str, Any],
) -> None:
    """Write run configuration snapshot JSON."""

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(state_payload, file_handle, indent=2, sort_keys=True)
    logger.debug("saved run state to %s", path)
