from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from spatialdata_io import xenium

from . import io
from .config import Configuration
from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_NUMBER_OF_CORES = 3
CORE_INFERENCE_SAMPLE_SIZE = 30_000
CORE_INFERENCE_MINIMUM_GAP = 250.0
CORE_INFERENCE_MINIMUM_FRACTION = 0.05


def run_ingest(
    configuration: Configuration,
) -> None:
    """Read each Xenium sample's table, concatenate into one AnnData, and write it out."""

    logger.info("ingesting %s sample(s)", len(configuration.samples))
    per_sample_tables = []
    for sample in configuration.samples:
        sample_table = xenium(sample.path)["table"]
        assign_core_labels(sample_table, sample.id)
        per_sample_tables.append(sample_table)

    sample_ids = [sample.id for sample in configuration.samples]
    merged = ad.concat(
        per_sample_tables,
        keys=sample_ids,
        label="sample_id",
        index_unique="_",
    )
    merged.obs.index.name = None
    io.write_processed_anndata(configuration, merged)


def assign_core_labels(
    annotated_data: ad.AnnData,
    sample_id: str,
    number_of_cores: int | None = None,
) -> None:
    """Assign deterministic core_id labels to a sample from spatial centroids."""

    coordinates = _spatial_coordinates(annotated_data)
    if coordinates is None or annotated_data.n_obs < DEFAULT_NUMBER_OF_CORES:
        logger.warning(
            "assigning all %s cells from %s to one core because spatial coordinates are unavailable or too sparse",
            annotated_data.n_obs,
            sample_id,
        )
        annotated_data.obs["core_id"] = f"{sample_id}_core_1"
        return

    if number_of_cores is None:
        number_of_cores = infer_number_of_cores(coordinates)

    annotated_data.obs["core_id"] = core_labels_from_coordinates(
        coordinates,
        sample_id,
        number_of_cores=number_of_cores,
    )
    logger.debug(
        "assigned %s core labels for %s: %s",
        number_of_cores,
        sample_id,
        annotated_data.obs["core_id"].value_counts().sort_index().to_dict(),
    )


def core_labels_from_coordinates(
    coordinates: np.ndarray,
    sample_id: str,
    number_of_cores: int | None = None,
) -> pd.Categorical:
    """Return core_id labels for cell centroid coordinates."""

    if number_of_cores is None:
        number_of_cores = infer_number_of_cores(coordinates)

    labels = _partition_spatial_cores(coordinates, number_of_cores)
    return pd.Categorical([f"{sample_id}_core_{label}" for label in labels])


def infer_number_of_cores(
    coordinates: np.ndarray,
    maximum_number_of_cores: int = DEFAULT_NUMBER_OF_CORES,
) -> int:
    """Infer whether a sample looks like one, two, or three spatial cores."""

    if coordinates.shape[0] < maximum_number_of_cores:
        return 1

    sampled_coordinates = _sample_coordinates_for_core_inference(coordinates)
    raw_labels = _fit_spatial_core_model(
        sampled_coordinates,
        maximum_number_of_cores,
    )
    summaries = _core_x_summaries(sampled_coordinates, raw_labels)
    if not all(
        summary["cell_count"]
        >= max(3, CORE_INFERENCE_MINIMUM_FRACTION * sampled_coordinates.shape[0])
        for summary in summaries
    ):
        return 1

    gaps = [
        summaries[index + 1]["x_low"] - summaries[index]["x_high"]
        for index in range(maximum_number_of_cores - 1)
    ]
    return max(1, 1 + sum(gap >= CORE_INFERENCE_MINIMUM_GAP for gap in gaps))


def _spatial_coordinates(
    annotated_data: ad.AnnData,
) -> np.ndarray | None:
    """Return cell centroid coordinates from obsm or obs when available."""

    if "spatial" in annotated_data.obsm:
        coordinates = np.asarray(annotated_data.obsm["spatial"])[:, :2]
    elif {"x_centroid", "y_centroid"}.issubset(annotated_data.obs.columns):
        coordinates = annotated_data.obs[["x_centroid", "y_centroid"]].to_numpy()
    else:
        return None

    finite_mask = np.isfinite(coordinates).all(axis=1)
    if not finite_mask.all():
        raise ValueError("spatial coordinates contain non-finite values")
    return coordinates.astype(np.float64, copy=False)


def _partition_spatial_cores(
    coordinates: np.ndarray,
    number_of_cores: int,
) -> np.ndarray:
    """Cluster centroids into left-to-right core numbers."""

    raw_labels = _fit_spatial_core_model(coordinates, number_of_cores)
    centers = np.asarray(
        [
            coordinates[raw_labels == label].mean(axis=0)
            for label in range(number_of_cores)
        ]
    )
    label_order = np.argsort(centers[:, 0])
    label_to_core = {
        old_label: core_number
        for core_number, old_label in enumerate(label_order, start=1)
    }
    return np.asarray([label_to_core[label] for label in raw_labels], dtype=int)


def _sample_coordinates_for_core_inference(
    coordinates: np.ndarray,
) -> np.ndarray:
    """Return a deterministic coordinate subsample for fast core inference."""

    if coordinates.shape[0] <= CORE_INFERENCE_SAMPLE_SIZE:
        return coordinates

    random_generator = np.random.default_rng(0)
    sampled_index = random_generator.choice(
        coordinates.shape[0],
        size=CORE_INFERENCE_SAMPLE_SIZE,
        replace=False,
    )
    return coordinates[sampled_index]


def _fit_spatial_core_model(
    coordinates: np.ndarray,
    number_of_cores: int,
) -> np.ndarray:
    """Fit the spatial k-means model and return raw cluster labels."""

    if number_of_cores == 1:
        return np.zeros(coordinates.shape[0], dtype=int)

    model = MiniBatchKMeans(
        n_clusters=number_of_cores,
        n_init=20,
        random_state=0,
        batch_size=8192,
    )
    return model.fit_predict(coordinates)


def _core_x_summaries(
    coordinates: np.ndarray,
    raw_labels: np.ndarray,
) -> list[dict[str, float]]:
    """Return left-to-right robust x ranges for raw spatial clusters."""

    summaries = []
    for label in sorted(np.unique(raw_labels)):
        cluster_coordinates = coordinates[raw_labels == label]
        summaries.append(
            {
                "cell_count": float(cluster_coordinates.shape[0]),
                "x_low": float(np.percentile(cluster_coordinates[:, 0], 1)),
                "x_high": float(np.percentile(cluster_coordinates[:, 0], 99)),
            }
        )
    return sorted(
        summaries,
        key=lambda summary: (summary["x_low"] + summary["x_high"]) / 2,
    )
