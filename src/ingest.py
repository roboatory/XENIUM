from __future__ import annotations

import anndata as ad
from spatialdata_io import xenium

from . import io
from .config import Configuration
from .logging import get_logger

logger = get_logger(__name__)


def run_ingest(
    configuration: Configuration,
) -> None:
    """Read each Xenium sample's table, concatenate into one AnnData, and write it out."""

    logger.info("ingesting %s sample(s)", len(configuration.samples))
    per_sample_tables = [
        xenium(sample.path)["table"] for sample in configuration.samples
    ]
    sample_ids = [sample.id for sample in configuration.samples]
    merged = ad.concat(
        per_sample_tables,
        keys=sample_ids,
        label="sample_id",
        index_unique="_",
    )
    merged.obs.index.name = None
    io.write_processed_anndata(configuration, merged)
