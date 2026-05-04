from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .logging import get_logger

logger = get_logger(__name__)

_CONDITION = "prostate cancer"


def annotate_clusters_with_llm(
    annotation_evidence_by_cluster: dict[str, list[object]],
    model: str,
    evidence_type: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    seed: int = 42,
    timeout_seconds: int = 120,
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
            "format": _annotation_schema(),
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "seed": seed,
            },
        },
        host,
        timeout_seconds,
    )

    raw_content = response["message"]["content"]
    parsed_response = json.loads(raw_content)
    annotations = {
        str(annotation["cluster_id"]): {
            "cell_type": annotation["cell_type"],
            "confidence": float(annotation["confidence"]),
            "rationale": annotation["rationale"],
        }
        for annotation in parsed_response["annotations"]
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
            f"Annotate each cluster with one main cell type label from {evidence_label}.",
            "Be as specific as possible (lineage + subtype + functional state).",
            f"If clusters are similar, disambiguate using {support_phrase} states.",
            f"The final JSON must contain exactly {number_of_clusters} annotations and exactly {number_of_clusters} unique cell_type labels.",
            "Cell type labels must be globally unique across clusters.",
            "Treat labels as duplicates if they differ only by case, spacing, punctuation, or singular/plural form.",
            "Maintain an internal used-label list as you assign labels; if a new label matches any earlier label after normalization, rewrite it before continuing.",
            "Do not reuse any label string across clusters.",
            "When disambiguating, use biologically meaningful qualifiers (lineage/subtype/state), not generic numbering.",
            "Bad: 'T cell' and 'T-cell' (duplicate). Good: 'Cytotoxic T-cell effector state' and 'Exhausted T-cell state'.",
            f"Before returning JSON, verify both checks pass: annotation count = {number_of_clusters} and unique normalized cell_type count = {number_of_clusters}. If either check fails, rewrite and re-check.",
            "Return JSON using this schema only:",
            '{ "annotations": [{ "cluster_id": "0", "cell_type": "label", "confidence": 0.0, "rationale": "..." }] }',
            "Confidence must be a number in [0, 1].",
            f"Rationale should be 1-2 sentences with discriminating {evidence_label}.",
            "",
            f"Clusters and {evidence_label}:",
        ]
    else:
        lines = [
            f"Condition: {_CONDITION}.",
            "Annotate each spatial domain with one microenvironment label from neighboring cell-type composition.",
            "Do not output only a single raw cell type name; use niche/interface/transition-zone wording.",
            "Good label styles: 'Tumor-immune interface', 'Fibro-inflammatory stroma niche', 'Basal-luminal transition zone'.",
            f"If domains are similar, disambiguate with {support_phrase} context.",
            f"The final JSON must contain exactly {number_of_clusters} annotations and exactly {number_of_clusters} unique cell_type labels.",
            "Cell type labels must be globally unique across clusters.",
            "Treat labels as duplicates if they differ only by case, spacing, punctuation, or singular/plural form.",
            "Maintain an internal used-label list as you assign labels; if a new label matches any earlier label after normalization, rewrite it before continuing.",
            "Do not reuse any label string across clusters.",
            "When disambiguating, use ecological qualifiers (niche/interface/transition/perivascular/stromal-adjacent).",
            "Bad: 'Tumor-immune interface' and 'tumor immune interface' (duplicate). Good: 'Tumor-immune interface' and 'Perivascular tumor-immune niche'.",
            f"Before returning JSON, verify both checks pass: annotation count = {number_of_clusters} and unique normalized cell_type count = {number_of_clusters}. If either check fails, rewrite and re-check.",
            "Return JSON using this schema only:",
            '{ "annotations": [{ "cluster_id": "0", "cell_type": "label", "confidence": 0.0, "rationale": "..." }] }',
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
            if payload.get("stream"):
                return _parse_streaming_ollama_response(response)
            body = response.read().decode("utf-8")
    except (HTTPError, URLError) as error:
        message = f"failed to reach local LLM at {url}: {error}"
        logger.error(message)
        raise RuntimeError(message) from error

    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        message = "local LLM returned invalid JSON."
        logger.error(message)
        raise RuntimeError(message) from error


def _parse_streaming_ollama_response(
    response: Any,
) -> dict[str, Any]:
    """Accumulate Ollama streamed chat chunks into the non-streaming response shape."""

    content_parts: list[str] = []
    final_chunk: dict[str, Any] = {}

    for line in response:
        decoded_line = line.decode("utf-8").strip()
        if not decoded_line:
            continue
        try:
            chunk = json.loads(decoded_line)
        except json.JSONDecodeError as error:
            message = "local LLM returned invalid JSON."
            logger.error(message)
            raise RuntimeError(message) from error

        final_chunk = chunk
        message = chunk.get("message", {})
        content = message.get("content")
        if content:
            content_parts.append(str(content))

    if not content_parts:
        message = "local LLM returned an empty streaming response."
        logger.error(message)
        raise RuntimeError(message)

    final_chunk["message"] = {
        **final_chunk.get("message", {}),
        "content": "".join(content_parts),
    }
    return final_chunk


def _annotation_schema() -> dict[str, Any]:
    """Schema used by Ollama to enforce response structure."""

    return {
        "type": "object",
        "required": ["annotations"],
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["cluster_id", "cell_type", "confidence", "rationale"],
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "cell_type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            }
        },
        "additionalProperties": True,
    }
