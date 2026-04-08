from __future__ import annotations

import argparse
from pathlib import Path

from spatialdata_io import xenium

from src import (
    analysis,
    annotation,
    colocalization,
    io,
    plotting,
    preprocessing,
    spatial_domains,
    state,
)
from src.config import Configuration
from src.logging import clear_active_log, get_logger, initialize_logging

CONFIG_PATH = Path("config.yaml").resolve()
logger = get_logger(__name__)

STAGE_ORDER = ("ingest", "preprocess", "annotate", "domains", "colocalization")


def parse_arguments() -> list[str]:
    """Parse CLI arguments and return the selected stages in pipeline order."""

    parser = argparse.ArgumentParser(
        description="Xenium spatial transcriptomics pipeline"
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=STAGE_ORDER,
        default=None,
        help="one or more pipeline stages to run (default: all)",
    )
    args = parser.parse_args()

    if args.stage is None:
        return list(STAGE_ORDER)

    return [stage for stage in STAGE_ORDER if stage in args.stage]


def load_configuration() -> Configuration:
    """Load and initialize top-level configuration."""

    configuration = Configuration()
    configuration.load_from_yaml(CONFIG_PATH)
    configuration.create_directories()
    initialize_logging(configuration.logs_directory, reset=True)
    return configuration


def _zarr_path(configuration: Configuration) -> Path:
    """Return the path to the processed zarr store."""

    return configuration.processed_data_directory / "processed.zarr"


def _validate_zarr_exists(configuration: Configuration) -> None:
    """Raise if the processed zarr does not exist."""

    path = _zarr_path(configuration)
    if not path.exists():
        raise FileNotFoundError(
            f"processed zarr not found at '{path}'. run the ingest stage first"
        )


def _validate_obs_column(
    configuration: Configuration,
    column: str,
    stage_name: str,
) -> None:
    """Raise if the zarr exists but is missing a required obs column."""

    _validate_zarr_exists(configuration)
    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]
    if column not in annotated_data.obs.columns:
        raise ValueError(
            f"'{column}' not found in obs. run the upstream stages before {stage_name}"
        )


def _validate_obsp_key(
    configuration: Configuration,
    key: str,
    stage_name: str,
) -> None:
    """Raise if the zarr exists but is missing a required obsp key."""

    _validate_zarr_exists(configuration)
    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]
    if key not in annotated_data.obsp:
        raise ValueError(
            f"'{key}' not found in obsp. run the upstream stages before {stage_name}"
        )


def run_ingest_stage(configuration: Configuration) -> None:
    """Ingest raw Xenium output into a processed SpatialData zarr."""

    logger.info("stage: ingest")
    if not configuration.raw_data_directory.exists():
        raise FileNotFoundError(
            f"raw data directory not found at '{configuration.raw_data_directory}'"
        )

    spatial_data = xenium(configuration.raw_data_directory)
    if "morphology_focus" in spatial_data:
        del spatial_data["morphology_focus"]
        logger.debug("removed morphology_focus element from ingested spatialdata")

    io.write_spatialdata_zarr(configuration, spatial_data)


def run_preprocess_stage(configuration: Configuration) -> None:
    """Run preprocessing and clustering steps on pre-ingested data."""

    logger.info("stage: preprocess and cluster")
    _validate_zarr_exists(configuration)
    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    plotting.plot_cell_and_nucleus_boundaries(configuration, spatial_data)
    plotting.plot_transcripts(
        configuration,
        spatial_data,
        list(configuration.plots.genes_to_plot),
        ["blue", "orange"],
    )

    if "counts" in annotated_data.layers:
        annotated_data.X = annotated_data.layers["counts"].copy()
    else:
        annotated_data.layers["counts"] = annotated_data.X.copy()

    plotting.plot_qc_histogram(
        configuration,
        annotated_data,
        [
            configuration.pipeline.minimum_counts,
            configuration.pipeline.maximum_counts_quantile,
        ],
        ["crimson", "goldenrod"],
    )
    preprocessing.filter_cells_and_genes(
        annotated_data,
        configuration.pipeline.minimum_counts,
        configuration.pipeline.maximum_counts_quantile,
        configuration.pipeline.minimum_cells,
    )
    preprocessing.normalize_and_scale(annotated_data)
    analysis.run_clustering(
        annotated_data,
        configuration.pipeline.pca_n_components,
    )
    analysis.run_umap(annotated_data)
    analysis.rank_genes(annotated_data)
    enriched_gene_lists = analysis.compute_enriched_genes(
        annotated_data,
        configuration.pipeline.rank_top_n,
        configuration.pipeline.minimum_logarithm_fold_change,
        configuration.pipeline.maximum_adjusted_p_value,
    )
    plotting.plot_rank_genes_dotplot(configuration, annotated_data)

    io.write_labels(configuration, annotated_data, "cluster")
    io.write_enriched_genes(configuration, enriched_gene_lists)

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(configuration, spatial_data)


def run_annotate_stage(configuration: Configuration) -> None:
    """Run LLM-driven cell-type annotation and persist updated zarr."""

    logger.info("stage: annotate clusters")
    _validate_obs_column(configuration, "leiden", "annotate")
    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    enriched_gene_lists = io.read_enriched_genes(configuration)
    cluster_annotations = annotation.annotate_clusters_with_llm(
        enriched_gene_lists,
        configuration.annotation_model,
        configuration.condition,
        "marker_genes",
    )
    io.write_annotations(configuration, cluster_annotations, "cluster")

    annotated_data.obs["cell_type"] = (
        annotated_data.obs["leiden"]
        .astype(str)
        .map(
            {
                cluster: value["cell_type"]
                for cluster, value in cluster_annotations.items()
            }
        )
    )

    plotting.plot_umap_leiden(
        configuration,
        spatial_data,
    )
    plotting.plot_cluster_overlay(
        configuration,
        spatial_data,
        cluster_key="cell_type",
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(configuration, spatial_data)


def run_domains_stage(configuration: Configuration) -> None:
    """Run neighborhood analysis and persist updated zarr."""

    logger.info("stage: spatial domains")
    _validate_obs_column(configuration, "cell_type", "domains")
    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    spatial_domains.compute_neighborhood_composition(
        annotated_data, configuration.pipeline.neighborhood_colocalization_radius
    )
    spatial_domains.assign_spatial_domains(
        annotated_data, configuration.pipeline.domain_n_clusters
    )
    domain_signatures = spatial_domains.build_domain_signatures(annotated_data)
    domain_annotations = annotation.annotate_clusters_with_llm(
        domain_signatures,
        configuration.annotation_model,
        configuration.condition,
        "neighborhood_cell_types",
    )

    io.write_annotations(configuration, domain_annotations, "domain")
    annotated_data.obs["spatial_domain_label"] = (
        annotated_data.obs["spatial_domain"]
        .astype(str)
        .map(
            {
                domain_id: value["cell_type"]
                for domain_id, value in domain_annotations.items()
            }
        )
    )

    io.write_labels(configuration, annotated_data, "domain")

    plotting.plot_cluster_overlay(
        configuration,
        spatial_data,
        cluster_key="spatial_domain_label",
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(configuration, spatial_data)


def run_colocalization_stage(configuration: Configuration) -> None:
    """Run observed cell-type contact colocalization and write artifacts."""

    logger.info("stage: colocalization")
    _validate_obs_column(configuration, "cell_type", "colocalization")
    _validate_obsp_key(configuration, "spatial_connectivities", "colocalization")
    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    counts, row_proportions = colocalization.compute_observed_contact_matrices(
        annotated_data,
    )
    plotting.plot_colocalization_contact_counts(configuration, counts)
    plotting.plot_colocalization_contact_row_proportions(
        configuration,
        row_proportions,
    )

    permutation_results = colocalization.compute_permutation_significance(
        annotated_data,
        number_of_permutations=(
            configuration.pipeline.colocalization_number_of_permutations
        ),
        minimum_cells=configuration.pipeline.colocalization_minimum_cells,
    )
    plotting.plot_colocalization_log2_fold_enrichment(
        configuration,
        permutation_results["log2_fold_enrichment"],
    )
    plotting.plot_colocalization_log2_fold_enrichment_significant_only(
        configuration,
        permutation_results["log2_fold_enrichment"],
        permutation_results["significant_mask"],
    )


STAGE_DISPATCH = {
    "ingest": run_ingest_stage,
    "preprocess": run_preprocess_stage,
    "annotate": run_annotate_stage,
    "domains": run_domains_stage,
    "colocalization": run_colocalization_stage,
}


def main() -> None:
    stages = parse_arguments()
    configuration = load_configuration()
    logger.info("pipeline start (stages: %s)", ", ".join(stages))

    for stage_name in stages:
        STAGE_DISPATCH[stage_name](configuration)

    state_path = state.build_state_path(configuration)
    configuration_snapshot = state.configuration_settings_snapshot(configuration)
    io.save_state(state_path, configuration_snapshot)
    logger.info("pipeline complete")
    clear_active_log(configuration.logs_directory)


if __name__ == "__main__":
    main()
