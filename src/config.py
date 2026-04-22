from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class Sample:
    """A single Xenium sample record loaded from config.yaml."""

    id: str
    path: Path

    @classmethod
    def from_dictionary(
        cls: type[Sample],
        data: dict[str, Any],
    ) -> Sample:
        """Create a Sample from a raw YAML record with id and path."""

        return cls(
            id=str(data["id"]),
            path=Path(data["path"]).resolve(),
        )


@dataclass(frozen=True)
class PipelineConfiguration:
    """Numeric parameters for the analysis pipeline."""

    minimum_cells: int
    pca_n_components: int
    neighborhood_colocalization_radius: float
    colocalization_number_of_permutations: int
    colocalization_minimum_cells: int
    domain_n_clusters: int
    rank_top_n: int
    minimum_logarithm_fold_change: float
    maximum_adjusted_p_value: float

    @classmethod
    def from_dictionary(
        cls: type[PipelineConfiguration],
        data: dict[str, Any],
    ) -> PipelineConfiguration:
        """Create from a raw dictionary (typically loaded from YAML)."""

        return cls(
            minimum_cells=int(data["minimum_cells"]),
            pca_n_components=int(data["pca_n_components"]),
            neighborhood_colocalization_radius=float(
                data["neighborhood_colocalization_radius"]
            ),
            colocalization_number_of_permutations=int(
                data["colocalization_number_of_permutations"]
            ),
            colocalization_minimum_cells=int(data["colocalization_minimum_cells"]),
            domain_n_clusters=int(data["domain_n_clusters"]),
            rank_top_n=int(data["rank_top_n"]),
            minimum_logarithm_fold_change=float(data["minimum_logarithm_fold_change"]),
            maximum_adjusted_p_value=float(data["maximum_adjusted_p_value"]),
        )


@dataclass(frozen=True)
class PlotsConfiguration:
    """Configuration for plotting behaviour."""

    genes_to_plot: tuple[str, ...] = ()

    @classmethod
    def from_dictionary(
        cls: type[PlotsConfiguration],
        data: dict[str, Any],
    ) -> PlotsConfiguration:
        """Create from a raw dictionary (typically loaded from YAML)."""

        return cls(
            genes_to_plot=tuple(sorted(data["genes_to_plot"])),
        )


@dataclass
class Configuration:
    """Top-level configuration loaded from YAML."""

    samples: list[Sample] = field(default_factory=list)
    output_directory: Path | None = None
    processed_data_directory: Path | None = None
    results_directory: Path | None = None
    figures_directory: Path | None = None
    logs_directory: Path | None = None
    annotation_model: str = "llama3.1:8b"
    pipeline: PipelineConfiguration | None = None
    plots: PlotsConfiguration | None = None

    def load_from_yaml(
        self,
        configuration_path: Path,
    ) -> None:
        """Load configuration from a YAML file and populate this instance."""

        with configuration_path.open("r") as f:
            configuration = yaml.safe_load(f)

        output_directory = Path(configuration["output_directory"]).resolve()
        self.output_directory = output_directory
        self.processed_data_directory = output_directory / "processed"
        self.results_directory = output_directory / "analysis"
        self.figures_directory = output_directory / "figures"
        self.logs_directory = output_directory / "logs"
        self.annotation_model = str(
            configuration.get("annotation_model", self.annotation_model)
        )

        self.samples = [
            Sample.from_dictionary(record) for record in configuration["samples"]
        ]
        self.pipeline = PipelineConfiguration.from_dictionary(configuration["pipeline"])
        self.plots = PlotsConfiguration.from_dictionary(configuration["plots"])
        logger.debug(
            "loaded configuration from %s with %s sample(s) and output root %s",
            configuration_path,
            len(self.samples),
            self.output_directory,
        )

    def create_directories(self) -> None:
        """Ensure all output directories exist, including per-sample subdirs."""

        for path in (
            self.processed_data_directory,
            self.results_directory,
            self.figures_directory,
            self.logs_directory,
        ):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug("ensured output directory exists: %s", path)

        for sample in self.samples:
            for parent in (self.figures_directory, self.results_directory):
                if parent is not None:
                    (parent / sample.id).mkdir(parents=True, exist_ok=True)
