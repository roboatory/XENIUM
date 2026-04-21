from __future__ import annotations

import logging as stdlib_logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from src import logging as pipeline_logging
from src.config import Configuration, PipelineConfiguration, PlotsConfiguration


@pytest.fixture(autouse=True)
def reset_pipeline_logging():
    """Reset the pipeline logging singleton so each test starts clean."""

    pipeline_logging._LOG_PATH = None
    pipeline_logger = stdlib_logging.getLogger("pipeline")
    for handler in list(pipeline_logger.handlers):
        pipeline_logger.removeHandler(handler)
    yield
    pipeline_logging._LOG_PATH = None
    pipeline_logger = stdlib_logging.getLogger("pipeline")
    for handler in list(pipeline_logger.handlers):
        handler.close()
        pipeline_logger.removeHandler(handler)


@pytest.fixture
def configuration(tmp_path: Path) -> Configuration:
    """Build a Configuration whose directories all live under tmp_path."""

    config = Configuration(
        raw_data_directory=tmp_path / "raw",
        output_directory=tmp_path / "output",
        processed_data_directory=tmp_path / "output" / "processed",
        results_directory=tmp_path / "output" / "analysis",
        figures_directory=tmp_path / "output" / "figures",
        logs_directory=tmp_path / "output" / "logs",
        annotation_model="llama3.1:8b",
        condition="prostate cancer",
        pipeline=PipelineConfiguration(
            minimum_counts=5,
            maximum_counts_quantile=0.99,
            minimum_cells=2,
            pca_n_components=5,
            neighborhood_colocalization_radius=25.0,
            colocalization_number_of_permutations=50,
            colocalization_minimum_cells=3,
            domain_n_clusters=3,
            rank_top_n=10,
            minimum_logarithm_fold_change=0.0,
            maximum_adjusted_p_value=1.0,
        ),
        plots=PlotsConfiguration(genes_to_plot=("gene_0", "gene_1")),
    )
    config.create_directories()
    return config


def build_synthetic_adata(
    seed: int = 0,
    n_per_type: int = 40,
    n_genes: int = 24,
    n_types: int = 3,
) -> AnnData:
    """Synthetic AnnData with distinct cell-type marker blocks and clustered coordinates."""

    rng = np.random.default_rng(seed)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    block_size = n_genes // n_types

    count_blocks = []
    for cell_type_index in range(n_types):
        baseline = rng.poisson(lam=1.0, size=(n_per_type, n_genes)).astype(np.float32)
        block_start = cell_type_index * block_size
        block_end = block_start + block_size
        baseline[:, block_start:block_end] += rng.poisson(
            lam=15.0, size=(n_per_type, block_size)
        ).astype(np.float32)
        count_blocks.append(baseline)

    counts = np.vstack(count_blocks)
    n_obs = counts.shape[0]

    cell_ids = [f"cell_{i:04d}" for i in range(n_obs)]
    centers = np.array(
        [[0.0, 0.0], [120.0, 0.0], [60.0, 120.0], [0.0, 120.0], [120.0, 120.0]]
    )[:n_types]
    coords = np.vstack(
        [
            centers[i] + rng.normal(0.0, 8.0, size=(n_per_type, 2))
            for i in range(n_types)
        ]
    )

    cell_types = np.concatenate(
        [np.full(n_per_type, f"type_{i}", dtype=object) for i in range(n_types)]
    )

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": pd.Categorical(cell_types),
            "region": pd.Categorical(["cell_boundaries"] * n_obs),
            "total_counts": counts.sum(axis=1),
        },
        index=pd.Index(cell_ids, name="cell_id"),
    )
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_id"))

    adata = AnnData(X=csr_matrix(counts), obs=obs, var=var)
    adata.obsm["spatial"] = coords.astype(np.float32)
    adata.layers["counts"] = adata.X.copy()
    return adata


@pytest.fixture
def tiny_adata() -> AnnData:
    """Synthetic AnnData for tests that do not need a SpatialData wrapper."""

    return build_synthetic_adata()


def _cell_boundary_polygons(coords: np.ndarray, cell_ids: list[str]) -> GeoDataFrame:
    """Build a GeoDataFrame of square cell boundaries centered on spatial coords."""

    side = 2.0
    polygons = [
        Polygon(
            [
                (x - side, y - side),
                (x + side, y - side),
                (x + side, y + side),
                (x - side, y + side),
            ]
        )
        for x, y in coords
    ]
    gdf = GeoDataFrame(
        {"geometry": polygons},
        index=pd.Index(cell_ids, name="cell_id"),
    )
    return ShapesModel.parse(gdf)


@pytest.fixture
def tiny_spatialdata(tiny_adata: AnnData) -> SpatialData:
    """Minimal SpatialData wrapping tiny_adata with cell_boundaries shapes."""

    cell_ids = tiny_adata.obs["cell_id"].tolist()
    cell_boundaries = _cell_boundary_polygons(tiny_adata.obsm["spatial"], cell_ids)

    table = TableModel.parse(
        tiny_adata.copy(),
        region="cell_boundaries",
        region_key="region",
        instance_key="cell_id",
    )
    return SpatialData(
        shapes={"cell_boundaries": cell_boundaries},
        tables={"table": table},
    )
