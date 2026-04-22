from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import squidpy as sq

from .logging import get_logger

logger = get_logger(__name__)


def compute_neighborhood_composition(
    annotated_data: AnnData,
    radius: float,
) -> None:
    """Compute per-cell neighborhood cell-type proportions within a spatial radius."""

    logger.debug("computing neighborhood composition with radius=%s", radius)
    cell_types = annotated_data.obs["cell_type"].astype("category")
    categories = cell_types.cat.categories.astype(str)
    codes = cell_types.cat.codes.to_numpy()

    spatial_neighbors_kwargs = {
        "radius": radius,
        "coord_type": "generic",
        "delaunay": True,
    }
    if (
        "sample_id" in annotated_data.obs.columns
        and annotated_data.obs["sample_id"].nunique() > 1
    ):
        spatial_neighbors_kwargs["library_key"] = "sample_id"
    sq.gr.spatial_neighbors(annotated_data, **spatial_neighbors_kwargs)

    connectivities = annotated_data.obsp["spatial_connectivities"].tocsr().copy()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()

    one_hot = np.eye(len(categories), dtype=np.float32)[codes]
    one_hot[codes < 0] = 0

    neighbor_counts = connectivities @ one_hot
    totals = neighbor_counts.sum(axis=1, keepdims=True)
    composition = np.divide(
        neighbor_counts,
        totals,
        out=np.zeros_like(neighbor_counts),
        where=totals > 0,
    )

    annotated_data.obsm["neighborhood_composition"] = composition
    logger.debug(
        "stored neighborhood composition matrix with shape=%s",
        composition.shape,
    )


def assign_spatial_domains(
    annotated_data: AnnData,
    n_clusters: int = 10,
) -> None:
    """Cluster neighborhood composition vectors into spatial domain labels."""

    logger.debug("assigning %s spatial domains with k-means", n_clusters)
    composition_matrix = np.asarray(annotated_data.obsm["neighborhood_composition"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(composition_matrix).astype(str)

    annotated_data.obs["spatial_domain"] = pd.Categorical(labels)
    logger.info(
        "assigned %s spatial domains", annotated_data.obs["spatial_domain"].nunique()
    )


def build_domain_signatures(
    annotated_data: AnnData,
) -> dict[str, list[tuple[str, float]]]:
    """Summarize each spatial domain by dominant neighborhood components."""

    logger.debug("building domain signatures from neighborhood composition")
    component_labels = annotated_data.obs["cell_type"].astype(str).unique().tolist()
    composition = pd.DataFrame(
        annotated_data.obsm["neighborhood_composition"],
        index=annotated_data.obs_names,
        columns=component_labels,
    )
    domain_means = composition.groupby(
        annotated_data.obs["spatial_domain"].astype(str)
    ).mean()

    signatures = {
        str(domain_id): [
            (str(cell_type), float(frequency))
            for cell_type, frequency in row.sort_values(ascending=False).items()
        ]
        for domain_id, row in domain_means.iterrows()
    }
    logger.debug("built %s domain signatures", len(signatures))
    return signatures
