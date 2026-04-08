# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A spatial transcriptomics analysis pipeline for Xenium data. Single entry point (`main.py`) that runs five stages sequentially: ingestion, preprocessing/clustering, LLM-based annotation (via local Ollama), spatial domain analysis, and colocalization with permutation significance testing. Stages can be run selectively via `--stage`.

## Commands

```bash
# Install dependencies
uv sync

# Run the full pipeline (all stages)
uv run main.py

# Run specific stages (always execute in pipeline order)
uv run main.py --stage ingest preprocess
uv run main.py --stage colocalization

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Run tests
uv run pytest
```

Pre-commit hooks (ruff lint + ruff format) run on commit and push.

## Architecture

`main.py` orchestrates all pipeline stages, delegating to modules in `src/`. Stages can be run selectively via `--stage` (accepts one or more of: `ingest`, `preprocess`, `annotate`, `domains`, `colocalization`). When omitted, all stages run in order. Each stage validates its preconditions before executing.

### Pipeline stages in `main.py`

#### Stage 0: Ingest (`run_ingest_stage`)

Reads raw Xenium output via `spatialdata_io.xenium()` into a SpatialData Zarr (`processed/processed.zarr`).

#### Stage 1: Preprocess & Cluster (`run_preprocess_stage`)

QC filtering, normalization (Seurat v3 HVG selection), PCA, Leiden clustering, UMAP, and marker gene ranking. Writes cluster labels, enriched gene lists, and the updated Zarr.

#### Stage 2: Annotation (`run_annotate_stage`)

Sends per-cluster enriched gene lists to a local Ollama LLM, which returns a cell-type label for each Leiden cluster. Maps labels onto `obs["cell_type"]` and writes the updated Zarr.

#### Stage 3: Spatial Domains (`run_domains_stage`)

Computes per-cell neighborhood composition (cell-type proportions among spatial neighbors within a radius), clusters those vectors with k-means into spatial domains, and sends domain signatures to the LLM for microenvironment-style labeling. Writes domain labels and the updated Zarr.

#### Stage 4: Colocalization (`run_colocalization_stage`)

Quantifies which cell-type pairs are spatially co-located beyond what random arrangement would predict.

**Observed contact matrix.** Builds a symmetric $T \times T$ matrix from undirected edges in the spatial neighbor graph, plus row-normalized proportions:

$$
C_{ij}^{\mathrm{obs}} = \text{number of edges between types } i \text{ and } j
$$

$$
P_{ij}^{\mathrm{obs}} = \frac{C_{ij}^{\mathrm{obs}}}{\sum_j C_{ij}^{\mathrm{obs}}}
$$

**Permutation significance testing.** Keeps coordinates and graph fixed, shuffles cell-type labels for $B$ permutations (default 1000), recomputing $C^{(b)}$ each time.

Expected contacts:

$$
\mu_{ij} = \operatorname{mean}_b\!\left(C_{ij}^{(b)}\right)
$$

Fold enrichment:

$$
\mathrm{FE}_{ij} = \frac{C_{ij}^{\mathrm{obs}}}{\mu_{ij}}
$$

Two-sided empirical p-values:

$$
p_{ij}^{\mathrm{enrich}} = \frac{1 + \#\{ C_{ij}^{(b)} \geq C_{ij}^{\mathrm{obs}} \}}{B + 1}
$$

$$
p_{ij}^{\mathrm{deplete}} = \frac{1 + \#\{ C_{ij}^{(b)} \leq C_{ij}^{\mathrm{obs}} \}}{B + 1}
$$

BH-FDR correction is applied across all upper-triangle pairs for each tail. Cell types with fewer than `colocalization_minimum_cells` cells are excluded.

**Outputs.** Heatmaps of $\log_2(\mathrm{FE})$ for all pairs, $\log_2(\mathrm{FE})$ for significant pairs only (FDR $\leq 0.05$), raw contact counts, and row-normalized proportions.

#### Finalization

Saves a configuration snapshot (`state.json`) for provenance and clears the active log pointer.

### Key modules in `src/`

- `config.py` — `Configuration` dataclass that loads `config.yaml` and manages all paths
- `io.py` — all read/write operations (Zarr, JSON artifacts, CSV labels, state snapshots); uses atomic temp-then-rename for Zarr writes
- `preprocessing.py` — cell/gene QC filtering, normalization, scaling
- `analysis.py` — PCA, Leiden clustering, UMAP, marker gene ranking, enriched gene computation
- `annotation.py` — Ollama API client; two modes: marker gene annotation and neighborhood composition annotation
- `spatial_domains.py` — neighborhood composition, k-means domain assignment, domain signature building
- `colocalization.py` — observed contact matrices, permutation null distribution, fold enrichment, FDR
- `plotting.py` — all visualization (spatial overlays, UMAP, heatmaps, dotplots); saves to figures dir at 300 DPI
- `logging.py` — centralized logging with run-scoped log files and active log pointer
- `state.py` — configuration snapshot serialization for provenance

### Core data structures

- **SpatialData Zarr** (`processed.zarr`) — persistent store for spatial coordinates, transcripts, cell/nucleus boundaries, and the AnnData expression matrix
- **AnnData** (`adata`) — the in-memory object passed through pipeline stages; gene expression in `.X`, metadata in `.obs`, embeddings in `.obsm`

### Runtime dependency

The annotation stages require a running Ollama server (`ollama serve`) with model `llama3.1:8b` pulled. API endpoint: `http://localhost:11434/api/chat`.

## Configuration

All pipeline parameters live in `config.yaml` at the repo root. Key sections:
- `data_directory` / `output_directory` — input/output paths
- `annotation_model` — LLM model name
- `pipeline` — numeric parameters (min counts, PCA components, colocalization radius, permutation count, clustering params, significance thresholds)
- `plots.genes_to_plot` — genes for transcript visualization
