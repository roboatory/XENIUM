# Prostate Cancer Spatial Transcriptomics Pipeline

This repository runs a Xenium-based spatial transcriptomics pipeline. The workflow has two required execution steps:

1. Create or choose a run configuration YAML.
2. `uv run main.py --config <path-to-config>`

The pipeline ingests raw Xenium output, then runs preprocessing, clustering, marker analysis, LLM-based annotation, spatial domain analysis, and colocalization.

## Prerequisites

Install `uv` by following Astral's official instructions: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). If you are new to `uv`, it is a fast Python package and environment manager; in this repository, `uv run ...` will create/use the project environment and run the scripts with the locked dependencies.

Install Ollama using the official docs: [Ollama quickstart](https://docs.ollama.com/quickstart) or [download page](https://ollama.com/download). This pipeline uses a local Ollama-compatible server for annotation, so Ollama must be installed locally and running before `main.py`.

## Required model and server

The annotation stage requires the `llama3.1:8b` model.

Pull the model once:

```bash
ollama pull llama3.1:8b
```

Start the local Ollama server in a separate terminal before running the pipeline:

```bash
ollama serve
```

The pipeline expects the Ollama API at `http://localhost:11434`, which is the default server address used by `src/annotation.py`.

## Configure a run

Use `config.example.yaml` as the schema/reference config, then create a named run config outside `pipeline/`, such as `../configs/bone.yaml`.

- `samples`: list of sample IDs and raw Xenium output directories
- `output_directory`: path where processed data, analysis artifacts, figures, and logs should be written
- `annotation_model`: LLM model name, defaulting to `llama3.1:8b`
- `pipeline`: numeric analysis parameters

Relative paths inside a config file are resolved relative to that config file's directory.

## Run the pipeline end-to-end

From the `pipeline/` directory, run all stages:

```bash
uv run main.py --config ../configs/bone.yaml
```

### Run selected stages

Stages always execute in pipeline order, even if they are provided out of order:

```bash
uv run main.py --config ../configs/bone.yaml --stage ingest preprocess
uv run main.py --config ../configs/bone.yaml --stage annotate domains colocalization
```

Available stages are:

- `ingest`
- `preprocess`
- `annotate`
- `domains`
- `colocalization`

### Stage 0: ingest raw data

The ingest stage reads the raw Xenium samples from `samples` and writes the merged AnnData object:

- `processed/processed.h5ad`

### Stage 1: preprocess and cluster

The preprocess stage reads `processed/processed.h5ad`, computes QC metrics, filters cells/genes, normalizes and scales expression, runs PCA, optionally applies Harmony across `sample_id`, builds neighbors, computes UMAP and Leiden clusters, ranks marker genes, and writes cluster labels/enriched marker lists.

For multi-sample runs, the production pipeline stores the standard pre/post-Harmony sample-level diagnostic through `analysis.run_clustering` and `plotting.plot_harmony_diagnostic`. More exploratory Harmony diagnostics, such as core-specific before/after UMAPs, belong in `notebooks/` rather than `src/` unless they are used by `main.py` or shared production code.

### Stage 2: annotate clusters

The annotation stage sends per-cluster enriched gene lists to the local Ollama model and writes:

- `analysis/cluster_annotations.json`
- `figures/umap_leiden.png`
- per-sample `figures/cell_type_overlays/*.png`

It also maps the returned labels onto `obs["cell_type"]` in `processed/processed.h5ad`.

### Stage 3: spatial domains

The domain stage computes local neighborhood composition, clusters those vectors into spatial domains, asks the LLM for microenvironment-style domain labels, and writes:

- `analysis/spatial_domain_annotations.json`
- per-sample `analysis/<sample_id>/spatial_domain_labels.csv`
- per-sample `figures/spatial_domain_overlays/*.png`

### Stage 4: colocalization

The colocalization stage computes observed cell-type contact matrices and permutation-based enrichment/depletion statistics while keeping coordinates and the neighbor graph fixed. It writes heatmaps for raw contact counts, row-normalized contact proportions, log2 fold enrichment, and significant-only fold enrichment.

## Output layout

Under `output_directory`, the pipeline creates:

- `processed/`: persistent processed AnnData, especially `processed/processed.h5ad`
- `analysis/`: JSON/CSV analysis artifacts
- `figures/`: saved plots
- `logs/`: run logs

## Notebooks

The `notebooks/` directory contains exploratory or validation analyses that are not part of the production `main.py` stage graph. Examples include IHC/Xenium cell-type concordance analysis and core-specific Harmony diagnostics. Notebook-only helpers should stay local to notebooks unless they become reusable production logic.

## Common failure modes

- If a downstream stage fails because `processed.h5ad` is missing, include the `ingest` stage first.
- If annotation fails, confirm that `ollama serve` is running and that `llama3.1:8b` was downloaded with `ollama pull llama3.1:8b`.
- If a stage complains about a missing `obs` column such as `leiden` or `cell_type`, rerun the required upstream stage first.
- If paths are wrong, update the run config passed to `--config` before rerunning.
