from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .logging import get_logger

logger = get_logger(__name__)

_CONDITION = "prostate cancer"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_ANNOTATION_TIMEOUT_SECONDS = 900
_DEFAULT_NUM_PREDICT = 1200


def annotate_clusters_with_llm(
    annotation_evidence_by_cluster: dict[str, list[object]],
    model: str,
    evidence_type: str,
    host: str | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    timeout_seconds: int = _DEFAULT_ANNOTATION_TIMEOUT_SECONDS,
    num_predict: int = _DEFAULT_NUM_PREDICT,
) -> dict[str, dict[str, Any]]:
    """Annotate clusters using a local Ollama-compatible LLM."""

    logger.info(
        "annotating %s groups with %s (%s)",
        len(annotation_evidence_by_cluster),
        model,
        evidence_type,
    )
    is_marker_mode = evidence_type == "marker_genes"
    number_of_clusters = len(annotation_evidence_by_cluster)
    evidence_label = (
        "marker genes" if is_marker_mode else "neighboring cell-type composition"
    )
    uniqueness_instruction = (
        f"You must return exactly {number_of_clusters} annotations with exactly "
        f"{number_of_clusters} unique cell_type labels. Consider two labels "
        "duplicates if they become the same after lowercasing and removing "
        "spaces, punctuation, or singular/plural variation. Keep an internal "
        "set of already-used labels and rewrite any candidate label that would "
        "collide before you return JSON. "
    )
    condition_context = f"{_CONDITION} spatial transcriptomics"
    system_instruction = (
        f"You are a domain expert in {condition_context}. "
        "Use maximally specific labels (lineage + subtype + state), keep "
        f"labels biologically plausible from {evidence_label}. "
        f"{uniqueness_instruction}"
        "Return JSON only."
        if is_marker_mode
        else (
            f"You are a domain expert in {condition_context}. "
            "Given neighboring cell-type composition, assign spatial-domain "
            "microenvironment labels (niche/interface/transition zone), not raw "
            f"single-cell-type labels. {uniqueness_instruction}"
            "Return JSON only."
        )
    )

    response = _ollama_chat(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": _build_annotation_prompt(
                        annotation_evidence_by_cluster,
                        evidence_type,
                    ),
                },
            ],
            "format": _annotation_schema(set(annotation_evidence_by_cluster)),
            "stream": False,
            "options": _ollama_options(temperature, seed, num_predict),
        },
        _resolve_ollama_host(host),
        timeout_seconds,
    )

    raw_content = response["message"]["content"]
    parsed_response = json.loads(raw_content)
    annotations = {
        str(cluster_id): {
            "cell_type": annotation["cell_type"],
            "confidence": float(annotation["confidence"]),
            "rationale": annotation["rationale"],
        }
        for cluster_id, annotation in parsed_response["annotations"].items()
    }
    logger.info("annotated %s groups", len(annotations))
    return annotations


def _build_annotation_prompt(
    annotation_evidence_by_cluster: dict[str, list[object]],
    evidence_type: str,
) -> str:
    """Build prompt with schema and per-cluster annotation evidence."""

    is_marker_mode = evidence_type == "marker_genes"
    number_of_clusters = len(annotation_evidence_by_cluster)
    evidence_label = (
        "marker genes" if is_marker_mode else "dominant neighboring cell types"
    )
    support_phrase = "marker-supported" if is_marker_mode else "composition-supported"

    if is_marker_mode:
        lines = [
            f"Condition: {_CONDITION}.",
            f"Annotate each cluster with one unique cluster annotation label from {evidence_label}.",
            "Each cell_type must include lineage plus a distinguishing marker/state phrase.",
            "Do not return only broad labels like 'Basal epithelial cell' if more than one cluster is epithelial.",
            "Be as specific as possible (lineage + subtype + functional state).",
            f"If clusters are similar, disambiguate using {support_phrase} states.",
            f"The final JSON must contain exactly {number_of_clusters} annotations and exactly {number_of_clusters} unique cell_type labels.",
            "The annotations field must be a JSON object keyed by the exact cluster ids listed below.",
            "Do not include cluster_id inside each annotation object because the key is the cluster id.",
            "Cell type labels must be globally unique across clusters.",
            "Treat labels as duplicates if they differ only by case, spacing, punctuation, or singular/plural form.",
            "Maintain an internal used-label list as you assign labels; if a new label matches any earlier label after normalization, rewrite it before continuing.",
            "Do not reuse any label string across clusters.",
            "When disambiguating, use biologically meaningful qualifiers (lineage/subtype/state), not generic numbering.",
            "Bad: 'Basal epithelial cell' and 'Basal epithelial cell' (duplicate).",
            "Good: 'Basal epithelial cell, KRT5-high stress state' and 'Basal epithelial cell, KRT14-high proliferative state'.",
            f"Before returning JSON, verify all checks pass: annotation count = {number_of_clusters}, exact cluster id set matches the listed ids, unique normalized cell_type count = {number_of_clusters}, and duplicate_labels = []. If any check fails, rewrite and re-check.",
            "Return JSON using this schema only:",
            '{ "annotations": { "0": { "cell_type": "label", "confidence": 0.0, "rationale": "..." } }, "validation": { "annotation_count": 1, "unique_cell_type_count": 1, "duplicate_labels": [] } }',
            "Confidence must be a number in [0, 1].",
            f"Rationale should be 1-2 sentences with discriminating {evidence_label}.",
            "",
            f"Clusters and {evidence_label}:",
        ]
    else:
        lines = [
            f"Condition: {_CONDITION}.",
            "Annotate each spatial domain with one unique microenvironment annotation label from neighboring cell-type composition.",
            "Do not output only a single raw cell type name; use niche/interface/transition-zone wording.",
            "Each cell_type must include a distinguishing ecological or compositional phrase.",
            "Good label styles: 'Tumor-immune interface', 'Fibro-inflammatory stroma niche', 'Basal-luminal transition zone'.",
            f"If domains are similar, disambiguate with {support_phrase} context.",
            f"The final JSON must contain exactly {number_of_clusters} annotations and exactly {number_of_clusters} unique cell_type labels.",
            "The annotations field must be a JSON object keyed by the exact domain ids listed below.",
            "Do not include cluster_id inside each annotation object because the key is the domain id.",
            "Cell type labels must be globally unique across clusters.",
            "Treat labels as duplicates if they differ only by case, spacing, punctuation, or singular/plural form.",
            "Maintain an internal used-label list as you assign labels; if a new label matches any earlier label after normalization, rewrite it before continuing.",
            "Do not reuse any label string across clusters.",
            "When disambiguating, use ecological qualifiers (niche/interface/transition/perivascular/stromal-adjacent).",
            "Bad: 'Tumor-immune interface' and 'tumor immune interface' (duplicate). Good: 'Tumor-immune interface' and 'Perivascular tumor-immune niche'.",
            f"Before returning JSON, verify all checks pass: annotation count = {number_of_clusters}, exact domain id set matches the listed ids, unique normalized cell_type count = {number_of_clusters}, and duplicate_labels = []. If any check fails, rewrite and re-check.",
            "Return JSON using this schema only:",
            '{ "annotations": { "0": { "cell_type": "label", "confidence": 0.0, "rationale": "..." } }, "validation": { "annotation_count": 1, "unique_cell_type_count": 1, "duplicate_labels": [] } }',
            "Confidence must be a number in [0, 1].",
            f"Rationale should be 1-2 sentences with discriminating {evidence_label}, including interactions/co-occurrence when possible.",
            "",
            f"Clusters and {evidence_label}:",
        ]
    for cluster_id, evidence_items in sorted(
        annotation_evidence_by_cluster.items(), key=lambda item: item[0]
    ):
        formatted_items: list[str] = []
        for item in evidence_items:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = str(item[0])
                try:
                    value = float(item[1])
                except (TypeError, ValueError):
                    formatted_items.append(str(label))
                    continue
                percent = f"{value * 100:.1f}%"
                formatted_items.append(f"{label} ({percent})")
            else:
                formatted_items.append(str(item))
        lines.append(f"- {cluster_id}: {', '.join(formatted_items)}")
    return "\n".join(lines)


def _ollama_options(
    temperature: float,
    seed: int,
    num_predict: int,
) -> dict[str, int | float]:
    """Build deterministic Ollama generation options."""

    options: dict[str, int | float] = {
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
    }
    num_thread = _resolve_num_thread()
    if num_thread is not None:
        options["num_thread"] = num_thread
    return options


def _resolve_num_thread() -> int | None:
    """Return the requested Ollama CPU thread count from the environment."""

    for variable_name in ("OLLAMA_NUM_THREAD", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(variable_name)
        if value is None:
            continue
        try:
            num_thread = int(value)
        except ValueError:
            logger.warning("ignoring invalid %s=%s", variable_name, value)
            continue
        if num_thread > 0:
            return num_thread
        logger.warning("ignoring non-positive %s=%s", variable_name, value)
    return None


def _resolve_ollama_host(host: str | None) -> str:
    """Return the Ollama host from an explicit value or the environment."""

    resolved_host = host or os.environ.get("OLLAMA_HOST") or _DEFAULT_OLLAMA_HOST
    if "://" not in resolved_host:
        resolved_host = f"http://{resolved_host}"
    return resolved_host


def _ollama_chat(
    payload: dict[str, Any],
    host: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call an Ollama-compatible /api/chat endpoint."""

    url = f"{host.rstrip('/')}/api/chat"
    logger.debug("sending annotation request to %s", url)
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as error:
        message = f"failed to reach local LLM at {url}: {error}"
        logger.error(message)
        raise RuntimeError(message) from error

    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        message = "local LLM returned invalid JSON."
        logger.error(message)
        raise RuntimeError(message) from error


def _annotation_schema(
    expected_cluster_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Schema used by Ollama to enforce response structure."""

    annotation_item_schema = {
        "type": "object",
        "required": ["cell_type", "confidence", "rationale"],
        "properties": {
            "cell_type": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }
    annotations_schema: dict[str, Any]
    if expected_cluster_ids is None:
        annotations_schema = {
            "type": "object",
            "additionalProperties": annotation_item_schema,
        }
    else:
        cluster_ids = sorted(expected_cluster_ids)
        annotations_schema = {
            "type": "object",
            "required": cluster_ids,
            "properties": {
                cluster_id: annotation_item_schema for cluster_id in cluster_ids
            },
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "required": ["annotations", "validation"],
        "properties": {
            "annotations": annotations_schema,
            "validation": {
                "type": "object",
                "required": [
                    "annotation_count",
                    "unique_cell_type_count",
                    "duplicate_labels",
                ],
                "properties": {
                    "annotation_count": {"type": "integer"},
                    "unique_cell_type_count": {"type": "integer"},
                    "duplicate_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
