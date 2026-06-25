# CLAUDE.md

This file provides guidance to coding agents (Claude Code, Codex-style agents, etc.) when working with code in this repository. It is kept byte-for-byte identical to `AGENTS.md` except for this top heading; update both together.

## What This Is

A spatial transcriptomics analysis pipeline for Xenium data. Single entry point (`main.py`) runs five stages sequentially: ingestion, preprocessing/clustering, LLM-based annotation (via local Ollama), spatial domain analysis, and colocalization with permutation significance testing. A run config must be provided via `--config`; stages can be run selectively via `--stage`.

## Commands

```bash
# Install dependencies
uv sync

# Run the full pipeline (all stages)
uv run main.py --config ../configs/bone.yaml

# Run specific stages (always execute in pipeline order)
uv run main.py --config ../configs/bone.yaml --stage ingest preprocess
uv run main.py --config ../configs/bone.yaml --stage colocalization

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Run tests
uv run pytest
```

Pre-commit hooks (ruff lint + ruff format + nbstripout) run on commit and push.

## Architecture

`main.py` orchestrates all pipeline stages, delegating to modules in `src/`. It requires `--config <path>` for the run configuration. Stages can be run selectively via `--stage` (accepts one or more of: `ingest`, `preprocess`, `annotate`, `domains`, `colocalization`). Whatever subset is requested is always reordered to run in canonical pipeline order (`STAGE_ORDER`). When omitted, all stages run. Each stage validates its preconditions (processed AnnData exists, required `obs` columns / `obsp` keys present) before executing.

### Pipeline stages in `main.py`

#### Stage 0: Ingest (`run_ingest_stage`)

Reads each sample's raw Xenium output via `spatialdata_io.xenium(path)["table"]`, assigns spatial `core_id` labels per sample, concatenates all samples into one AnnData (adding a `sample_id` obs column), and writes the merged file to `processed/processed.h5ad`.

Core inference: a sample is partitioned into one to three left-to-right spatial cores by k-means (`MiniBatchKMeans`) over cell centroids. The number of cores is inferred deterministically — candidate clusters must each hold enough cells (`CORE_INFERENCE_MINIMUM_FRACTION = 0.05` of cells, minimum 3) and be separated by a robust x-gap of at least `CORE_INFERENCE_MINIMUM_GAP = 250.0` units; otherwise the sample collapses to a single core. Inference runs on a deterministic subsample (`CORE_INFERENCE_SAMPLE_SIZE = 30_000`) for speed. `core_id` is later used to constrain colocalization label permutations.

#### Stage 1: Preprocess & Cluster (`run_preprocess_stage`)

QC metric computation, per-sample MAD-based cell filtering, gene filtering, normalization (Seurat v3 HVG selection), scaling, PCA, optional Harmony batch correction, neighbor graph, Leiden clustering, UMAP, and marker gene ranking. Raw counts are preserved in the `counts` layer and log-normalized values in the `log_normalized` layer (marker ranking runs on the latter). When more than one sample is present, HVG selection is batch-aware and `harmonypy` integrates the PCA embedding across `sample_id`; the pre-correction UMAP is retained for a before/after diagnostic. Writes per-cluster enriched gene lists, per-sample cluster-label CSVs, QC / Harmony / marker-gene figures, and the updated AnnData.

#### Stage 2: Annotation (`run_annotate_stage`)

Sends per-cluster enriched gene lists to a local Ollama LLM (`marker_genes` evidence mode), which returns a `cell_type` label (with a confidence score and rationale) for each Leiden cluster. Maps labels onto `obs["cell_type"]`, writes the annotations JSON, renders the Leiden UMAP and per-sample cell-type spatial overlays, and writes the updated AnnData. Requires the `leiden` obs column from Stage 1.

#### Stage 3: Spatial Domains (`run_domains_stage`)

Computes per-cell neighborhood composition (cell-type proportions among spatial neighbors within a radius), clusters those vectors with k-means into `domain_n_clusters` spatial domains, builds per-domain composition signatures, and sends them to the LLM (`neighborhood_cell_types` evidence mode) for microenvironment-style labeling. Maps labels onto `obs["spatial_domain_label"]`, writes the domain annotations JSON, per-sample domain-label CSVs, per-sample spatial-domain overlays, and the updated AnnData. Requires the `cell_type` obs column from Stage 2.

#### Stage 4: Colocalization (`run_colocalization_stage`)

Quantifies which cell-type pairs are spatially co-located beyond what random arrangement would predict. Requires the `cell_type` obs column and the `spatial_connectivities` obsp graph (produced in Stage 3).

**Observed contact matrix.** Builds a symmetric $T \times T$ matrix from undirected edges in the spatial neighbor graph, plus row-normalized proportions:

$$
C_{ij}^{\mathrm{obs}} = \text{number of edges between types } i \text{ and } j
$$

$$
P_{ij}^{\mathrm{obs}} = \frac{C_{ij}^{\mathrm{obs}}}{\sum_j C_{ij}^{\mathrm{obs}}}
$$

**Permutation significance testing.** Keeps coordinates and the graph fixed, shuffles cell-type labels for $B$ permutations (default 1000), recomputing $C^{(b)}$ each time. Labels are shuffled *within* spatial partitions — `core_id` when present, otherwise `sample_id` — so the null preserves per-core/per-sample composition. Cell types with fewer than `colocalization_minimum_cells` cells are excluded.

Expected contacts:

$$
\mu_{ij} = \operatorname{mean}_b\!\left(C_{ij}^{(b)}\right)
$$

Fold enrichment:

$$
\mathrm{FE}_{ij} = \frac{C_{ij}^{\mathrm{obs}}}{\mu_{ij}}
$$

One-sided empirical p-values per tail:

$$
p_{ij}^{\mathrm{enrich}} = \frac{1 + \#\{ C_{ij}^{(b)} \geq C_{ij}^{\mathrm{obs}} \}}{B + 1}
$$

$$
p_{ij}^{\mathrm{deplete}} = \frac{1 + \#\{ C_{ij}^{(b)} \leq C_{ij}^{\mathrm{obs}} \}}{B + 1}
$$

BH-FDR correction is applied across all upper-triangle pairs separately for each tail. A pair is flagged significant when its FDR $\leq 0.05$ in the enrichment tail with $\mathrm{FE} > 1$, or in the depletion tail with $\mathrm{FE} < 1$.

**Outputs.** Heatmaps of raw contact counts, row-normalized proportions, $\log_2(\mathrm{FE})$ for all pairs, and $\log_2(\mathrm{FE})$ for significant pairs only.

#### Finalization

After the requested stages complete, saves a configuration snapshot (`analysis/state.json`) for provenance and clears the active log pointer.

### Key modules in `src/`

- `config.py` — `Sample`, `PipelineConfiguration`, and `Configuration` dataclasses; loads a YAML config path, resolves all paths, and creates output directories
- `ingest.py` — Xenium reading, deterministic spatial `core_id` inference/assignment, and per-sample concatenation into the merged AnnData
- `io.py` — all read/write operations (AnnData, enriched-gene and annotation JSON, per-sample CSV labels, state snapshots)
- `preprocessing.py` — QC metrics, per-sample MAD cutoffs, cell/gene filtering, normalization, scaling
- `analysis.py` — PCA, optional Harmony integration, neighbor graph, Leiden clustering, UMAP, marker gene ranking, enriched gene computation
- `annotation.py` — Ollama API client; two evidence modes: marker-gene annotation and neighborhood-composition annotation
- `spatial_domains.py` — neighborhood composition, k-means domain assignment, domain signature building
- `colocalization.py` — observed contact matrices, partitioned permutation null distribution, fold enrichment, BH-FDR
- `plotting.py` — all visualization (QC histograms, UMAP, spatial overlays, heatmaps, dotplots); saves to figures dir at 300 DPI
- `logging.py` — centralized run-scoped logging with an active log pointer; writes to stdout under SLURM, otherwise to a timestamped log file
- `state.py` — configuration snapshot serialization for provenance

Notebook-only diagnostics and exploratory analyses should stay in `notebooks/` rather than being promoted into `src/` unless they are used by `main.py` pipeline stages or shared production code. For example, core-specific Harmony before/after UMAP diagnostics are local helpers in `notebooks/core_harmony_batch_effect_diagnostic.ipynb`; the pipeline itself only computes the standard sample-level Harmony embeddings in `analysis.run_clustering` and renders them with `plotting.plot_harmony_diagnostic`.

### Core data structures

- **AnnData** (`processed/processed.h5ad`) — persistent processed expression matrix and metadata store; carries `sample_id`, `core_id`, `leiden`, `cell_type`, and `spatial_domain` / `spatial_domain_label` obs columns as stages populate them, plus `counts` and `log_normalized` layers
- **AnnData in memory** (`adata`) — object passed through pipeline stages; gene expression in `.X`, metadata in `.obs`, embeddings in `.obsm`, neighbor graphs in `.obsp`

### Runtime dependency

The annotation stages (`annotate`, `domains`) require a running Ollama server (`ollama serve`) with the configured model (default `llama3.1:8b`) pulled. API endpoint: `http://localhost:11434/api/chat` (the host can be overridden via `OLLAMA_HOST`). Requests are deterministic (temperature 0, fixed seed).

## Configuration

`pipeline/config.example.yaml` documents the expected schema. Named run configs should live outside `pipeline/`, for example `configs/bone.yaml`, and be passed with `--config`. Key sections:

- `samples` — list of `{id, path}` records pointing at raw Xenium output directories
- `output_directory` — root for `processed/`, `figures/`, `analysis/`, `logs/`
- `annotation_model` — Ollama LLM model name (defaults to `llama3.1:8b` if omitted)
- `pipeline` — numeric parameters: `minimum_cells`, `mad_threshold`, `pca_n_components`, `leiden_resolution`, `neighborhood_colocalization_radius`, `colocalization_number_of_permutations`, `colocalization_minimum_cells`, `domain_n_clusters`, `rank_top_n`, `minimum_logarithm_fold_change`, `maximum_adjusted_p_value`

Relative paths in a YAML config (sample paths and `output_directory`) are resolved relative to the config file's directory.
