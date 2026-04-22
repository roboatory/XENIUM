from __future__ import annotations

import pytest
from anndata import AnnData

import main
from src import io
from src.config import Configuration


def _write_adata_with_sample_ids(
    configuration: Configuration,
    adata: AnnData,
    sample_ids: list[str],
) -> None:
    """Stamp a sample_id column on the AnnData and write it as the processed artifact."""

    assert len(sample_ids) == adata.n_obs
    adata.obs["sample_id"] = sample_ids
    io.write_processed_anndata(configuration, adata)


def test_domains_stage_raises_on_multi_sample_data(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """run_domains_stage refuses multi-sample runs until per-sample spatial graphs land."""

    sample_ids = ["patient_001"] * (tiny_adata.n_obs // 2) + ["patient_002"] * (
        tiny_adata.n_obs - tiny_adata.n_obs // 2
    )
    tiny_adata.obs["cell_type"] = tiny_adata.obs["cell_type"].astype(str)
    _write_adata_with_sample_ids(configuration, tiny_adata, sample_ids)

    with pytest.raises(NotImplementedError, match="multi-sample runs"):
        main.run_domains_stage(configuration)


def test_colocalization_stage_raises_on_multi_sample_data(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """run_colocalization_stage refuses multi-sample runs until per-sample spatial graphs land."""

    import squidpy as sq

    sq.gr.spatial_neighbors(
        tiny_adata, radius=30.0, coord_type="generic", delaunay=True
    )
    sample_ids = ["patient_001"] * (tiny_adata.n_obs // 2) + ["patient_002"] * (
        tiny_adata.n_obs - tiny_adata.n_obs // 2
    )
    tiny_adata.obs["cell_type"] = tiny_adata.obs["cell_type"].astype(str)
    _write_adata_with_sample_ids(configuration, tiny_adata, sample_ids)

    with pytest.raises(NotImplementedError, match="multi-sample runs"):
        main.run_colocalization_stage(configuration)


def test_guard_passes_for_single_sample_data(
    configuration: Configuration,
    tiny_adata: AnnData,
) -> None:
    """The guard is a no-op when only one sample_id is present."""

    sample_ids = ["patient_001"] * tiny_adata.n_obs
    _write_adata_with_sample_ids(configuration, tiny_adata, sample_ids)

    main._validate_single_sample_until_library_key_support(configuration, "domains")
