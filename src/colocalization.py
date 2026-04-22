from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .logging import get_logger

logger = get_logger(__name__)

FDR_ALPHA = 0.05
RANDOM_SEED = 0


def compute_observed_contact_matrices(
    annotated_data: AnnData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute observed contact counts and row-normalized proportions."""

    logger.debug("computing observed contact matrices")
    cell_types = annotated_data.obs["cell_type"].astype("category")
    categories = cell_types.cat.categories.astype(str)
    codes = cell_types.cat.codes.to_numpy()
    edges = _undirected_edges(annotated_data)

    row_codes = codes[edges.row]
    col_codes = codes[edges.col]
    valid_edge_mask = (row_codes >= 0) & (col_codes >= 0)
    counts = _symmetric_counts(
        row_codes[valid_edge_mask],
        col_codes[valid_edge_mask],
        len(categories),
    )

    totals = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(
        counts.astype(np.float64),
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )
    logger.info("computed contact matrix for %s cell types", len(categories))
    return _frame(counts, categories), _frame(proportions, categories)


def compute_permutation_significance(
    annotated_data: AnnData,
    number_of_permutations: int,
    minimum_cells: int,
) -> dict[str, pd.DataFrame]:
    """Compute permutation-based contact enrichment statistics."""

    logger.debug(
        "running colocalization permutation testing with number_of_permutations=%s and minimum_cells=%s",
        number_of_permutations,
        minimum_cells,
    )
    cell_type_labels = annotated_data.obs["cell_type"].astype(str)

    type_counts = cell_type_labels.dropna().value_counts().sort_index()
    tested_cell_types = (
        type_counts[type_counts >= minimum_cells].index.astype(str).tolist()
    )
    number_of_tested_types = len(tested_cell_types)

    if number_of_tested_types < 2:
        logger.warning(
            "skipping colocalization significance because only %s cell type(s) met the minimum cell count",
            number_of_tested_types,
        )
        nan_matrix = np.full((number_of_tested_types, number_of_tested_types), np.nan)
        return {
            "log2_fold_enrichment": _frame(nan_matrix, tested_cell_types),
            "significant_mask": _frame(
                np.zeros((number_of_tested_types, number_of_tested_types), bool),
                tested_cell_types,
            ),
        }

    cell_type_to_code = {
        cell_type: index for index, cell_type in enumerate(tested_cell_types)
    }
    cell_type_codes = cell_type_labels.map(cell_type_to_code).fillna(-1).to_numpy()

    edges = _undirected_edges(annotated_data)
    valid_edges_mask = (cell_type_codes[edges.row] >= 0) & (
        cell_type_codes[edges.col] >= 0
    )
    eligible_index = np.flatnonzero(cell_type_codes >= 0)
    eligible_cell_type_codes = cell_type_codes[eligible_index]

    position = np.full(cell_type_codes.size, -1, int)
    position[eligible_index] = np.arange(eligible_index.size)
    row_position = position[edges.row][valid_edges_mask]
    column_position = position[edges.col][valid_edges_mask]

    observed_contacts = _symmetric_counts(
        eligible_cell_type_codes[row_position],
        eligible_cell_type_codes[column_position],
        number_of_tested_types,
    )

    expected_contacts = np.zeros((number_of_tested_types, number_of_tested_types))
    enrich_exceed_count = np.zeros(
        (number_of_tested_types, number_of_tested_types), int
    )
    deplete_exceed_count = np.zeros(
        (number_of_tested_types, number_of_tested_types), int
    )

    random_generator = np.random.default_rng(RANDOM_SEED)
    sample_partitions = _sample_partitions(annotated_data, eligible_index)

    for _ in range(number_of_permutations):
        permuted_codes = eligible_cell_type_codes.copy()
        for partition in sample_partitions:
            permuted_codes[partition] = random_generator.permutation(
                permuted_codes[partition]
            )
        permuted_contacts = _symmetric_counts(
            permuted_codes[row_position],
            permuted_codes[column_position],
            number_of_tested_types,
        )
        expected_contacts += permuted_contacts
        enrich_exceed_count += permuted_contacts >= observed_contacts
        deplete_exceed_count += permuted_contacts <= observed_contacts

    expected_contacts /= number_of_permutations

    empirical_p_values_enrichment = (enrich_exceed_count + 1) / (
        number_of_permutations + 1
    )
    empirical_p_values_depletion = (deplete_exceed_count + 1) / (
        number_of_permutations + 1
    )

    fold_enrichment = np.divide(
        observed_contacts,
        expected_contacts,
        out=np.full((number_of_tested_types, number_of_tested_types), np.nan),
        where=expected_contacts > 0,
    )

    log2_fold_enrichment = np.where(
        fold_enrichment > 0,
        np.log2(fold_enrichment),
        np.nan,
    )

    upper_triangle_indices = np.triu_indices(number_of_tested_types, k=1)
    enrichment_fdr_values = _benjamini_hochberg(
        empirical_p_values_enrichment[upper_triangle_indices]
    )
    depletion_fdr_values = _benjamini_hochberg(
        empirical_p_values_depletion[upper_triangle_indices]
    )

    significant_mask = np.zeros((number_of_tested_types, number_of_tested_types), bool)

    enrichment_selection_mask = (
        (enrichment_fdr_values <= FDR_ALPHA)
        & np.isfinite(fold_enrichment[upper_triangle_indices])
        & (fold_enrichment[upper_triangle_indices] > 1)
    )
    depletion_selection_mask = (
        (depletion_fdr_values <= FDR_ALPHA)
        & np.isfinite(fold_enrichment[upper_triangle_indices])
        & (fold_enrichment[upper_triangle_indices] < 1)
    )
    selection_mask = enrichment_selection_mask | depletion_selection_mask

    significant_mask[upper_triangle_indices] = selection_mask
    significant_mask[(upper_triangle_indices[1], upper_triangle_indices[0])] = (
        selection_mask
    )

    results = {
        "log2_fold_enrichment": _frame(log2_fold_enrichment, tested_cell_types),
        "significant_mask": _frame(significant_mask, tested_cell_types),
    }
    logger.info(
        "colocalization tested %s cell types with %s enriched and %s depleted significant pairs",
        number_of_tested_types,
        int(enrichment_selection_mask.sum()),
        int(depletion_selection_mask.sum()),
    )
    return results


def _sample_partitions(
    annotated_data: AnnData,
    eligible_index: np.ndarray,
) -> list[np.ndarray]:
    """Partition eligible-cell positions by sample_id so permutations stay within sample."""

    if "sample_id" not in annotated_data.obs.columns:
        return [np.arange(eligible_index.size)]

    eligible_sample_ids = (
        annotated_data.obs["sample_id"].iloc[eligible_index].astype(str).to_numpy()
    )
    return [
        np.where(eligible_sample_ids == sample_value)[0]
        for sample_value in pd.unique(eligible_sample_ids)
    ]


def _undirected_edges(
    annotated_data: AnnData,
) -> sparse.coo_matrix:
    """Build the undirected radius-neighbor graph used for colocalization."""

    connectivities = annotated_data.obsp["spatial_connectivities"].tocsr().copy()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    return sparse.triu(connectivities, k=1, format="coo")


def _symmetric_counts(
    row_codes: np.ndarray,
    col_codes: np.ndarray,
    n_types: int,
) -> np.ndarray:
    """Build a symmetric contact matrix from undirected edge type pairs."""

    counts = np.zeros((n_types, n_types), dtype=np.int64)
    np.add.at(counts, (row_codes, col_codes), 1)
    off_diagonal = row_codes != col_codes
    np.add.at(counts, (col_codes[off_diagonal], row_codes[off_diagonal]), 1)
    return counts


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply BH correction to a flat p-value vector."""

    if p_values.size == 0:
        return np.array([], dtype=np.float64)

    return np.asarray(
        stats.false_discovery_control(p_values, method="bh"),
        dtype=np.float64,
    )


def _frame(
    values: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """Wrap a matrix in a labeled dataframe."""

    index = pd.Index(labels, name="cell_type")
    return pd.DataFrame(values, index=index, columns=index)
