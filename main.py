from __future__ import annotations

import argparse
from pathlib import Path

from src import (
    analysis,
    annotation,
    colocalization,
    ingest,
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

STAGE_ORDER = (
    "ingest",
    "preprocess",
    "annotate",
    "domains",
    "colocalization",
)


def parse_arguments() -> list[str]:
    """Parse CLI arguments and return the selected stages in pipeline order."""

    # fmt: off
    parser = argparse.ArgumentParser(description="Xenium spatial transcriptomics pipeline")
    parser.add_argument("--stage", nargs="+", choices=STAGE_ORDER, default=None, help="one or more pipeline stages to run (default: all)")
    # fmt: on

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


def _validate_anndata_exists(configuration: Configuration) -> None:
    """Raise if the processed AnnData does not exist."""

    path = configuration.processed_data_directory / "processed.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"processed anndata not found at '{path}'. run the ingest stage first"
        )


def _validate_obs_column(
    configuration: Configuration,
    column: str,
    stage_name: str,
) -> None:
    """Raise if the processed AnnData is missing a required obs column."""

    _validate_anndata_exists(configuration)
    annotated_data = io.read_processed_anndata(configuration)
    if column not in annotated_data.obs.columns:
        raise ValueError(
            f"'{column}' not found in obs. run the upstream stages before {stage_name}"
        )


def _validate_obsp_key(
    configuration: Configuration,
    key: str,
    stage_name: str,
) -> None:
    """Raise if the processed AnnData is missing a required obsp key."""

    _validate_anndata_exists(configuration)
    annotated_data = io.read_processed_anndata(configuration)
    if key not in annotated_data.obsp:
        raise ValueError(
            f"'{key}' not found in obsp. run the upstream stages before {stage_name}"
        )


def _validate_single_sample_until_library_key_support(
    configuration: Configuration,
    stage_name: str,
) -> None:
    """Block multi-sample runs from the spatial stages until #16 adds library_key handling."""

    _validate_anndata_exists(configuration)
    annotated_data = io.read_processed_anndata(configuration)
    if "sample_id" in annotated_data.obs.columns:
        number_of_samples = int(annotated_data.obs["sample_id"].nunique())
        if number_of_samples > 1:
            raise NotImplementedError(
                f"{stage_name} does not yet support multi-sample runs "
                f"(found {number_of_samples} samples). "
                "Per-sample spatial graphs via squidpy's library_key land in issue #16."
            )


def run_ingest_stage(configuration: Configuration) -> None:
    """Ingest raw Xenium samples into a merged AnnData h5ad."""

    logger.info("stage: ingest")
    ingest.run_ingest(configuration)


def run_preprocess_stage(configuration: Configuration) -> None:
    """Run preprocessing and clustering steps on pre-ingested data."""

    logger.info("stage: preprocess and cluster")
    _validate_anndata_exists(configuration)
    annotated_data = io.read_processed_anndata(configuration)

    # TODO(#16): per-sample spatial overlay plots (boundaries, transcripts) move to notebook helpers

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

    io.write_processed_anndata(configuration, annotated_data)


def run_annotate_stage(configuration: Configuration) -> None:
    """Run LLM-driven cell-type annotation and persist the updated AnnData."""

    logger.info("stage: annotate clusters")
    _validate_obs_column(configuration, "leiden", "annotate")
    annotated_data = io.read_processed_anndata(configuration)

    enriched_gene_lists = io.read_enriched_genes(configuration)
    cluster_annotations = annotation.annotate_clusters_with_llm(
        enriched_gene_lists,
        configuration.annotation_model,
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

    # TODO(#16): UMAP-by-cluster and per-sample cluster overlay scatters

    io.write_processed_anndata(configuration, annotated_data)


def run_domains_stage(configuration: Configuration) -> None:
    """Run neighborhood analysis and persist the updated AnnData."""

    logger.info("stage: spatial domains")
    _validate_obs_column(configuration, "cell_type", "domains")
    _validate_single_sample_until_library_key_support(configuration, "domains")
    annotated_data = io.read_processed_anndata(configuration)

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

    # TODO(#16): per-sample spatial_domain_label overlay scatters

    io.write_processed_anndata(configuration, annotated_data)


def run_colocalization_stage(configuration: Configuration) -> None:
    """Run observed cell-type contact colocalization and write artifacts."""

    logger.info("stage: colocalization")
    _validate_obs_column(configuration, "cell_type", "colocalization")
    _validate_obsp_key(configuration, "spatial_connectivities", "colocalization")
    _validate_single_sample_until_library_key_support(configuration, "colocalization")
    annotated_data = io.read_processed_anndata(configuration)

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
