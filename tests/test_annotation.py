from __future__ import annotations

import json
from unittest.mock import patch
from urllib.error import URLError

import pytest

from src import annotation


def test_annotation_schema_shape() -> None:
    """_annotation_schema declares the required top-level and item keys."""

    schema = annotation._annotation_schema({"0", "1"})
    assert schema["type"] == "object"
    assert schema["required"] == ["annotations", "validation"]
    annotations_schema = schema["properties"]["annotations"]
    assert annotations_schema["required"] == ["0", "1"]
    assert annotations_schema["additionalProperties"] is False
    item_schema = annotations_schema["properties"]["0"]
    assert set(item_schema["required"]) == {
        "cell_type",
        "confidence",
        "rationale",
    }
    assert set(schema["properties"]["validation"]["required"]) == {
        "annotation_count",
        "unique_cell_type_count",
        "duplicate_labels",
    }


def test_build_annotation_prompt_marker_mode_includes_evidence() -> None:
    """Marker-mode prompt lists clusters, formatted evidence, and the hardcoded condition."""

    prompt = annotation._build_annotation_prompt(
        {"0": ["KRT8", "EPCAM"], "1": ["CD3D"]},
        evidence_type="marker_genes",
    )
    assert "Condition: prostate cancer." in prompt
    assert "marker genes" in prompt
    assert "JSON object keyed by the exact cluster ids" in prompt
    assert "unique normalized cell_type count" in prompt
    assert "validation" in prompt
    assert "- 0: KRT8, EPCAM" in prompt
    assert "- 1: CD3D" in prompt


def test_build_annotation_prompt_neighborhood_mode_uses_niche_wording() -> None:
    """Neighborhood-mode prompt instructs niche/interface-style labels and formats pairs."""

    prompt = annotation._build_annotation_prompt(
        {"0": [("Tumor", 0.6), ("Stroma", 0.4)]},
        evidence_type="neighborhood_cell_types",
    )
    assert "niche/interface" in prompt
    assert "JSON object keyed by the exact domain ids" in prompt
    assert "- 0: Tumor (60.0%), Stroma (40.0%)" in prompt


def _build_ollama_response_body(annotations: list[dict]) -> bytes:
    """Produce the bytes an Ollama /api/chat response would deliver."""

    annotations_by_cluster = {
        str(annotation["cluster_id"]): {
            key: value for key, value in annotation.items() if key != "cluster_id"
        }
        for annotation in annotations
    }
    payload = {
        "message": {
            "content": json.dumps(
                {
                    "annotations": annotations_by_cluster,
                    "validation": {
                        "annotation_count": len(annotations_by_cluster),
                        "unique_cell_type_count": len(
                            {
                                annotation["cell_type"]
                                for annotation in annotations_by_cluster.values()
                            }
                        ),
                        "duplicate_labels": [],
                    },
                }
            )
        }
    }
    return json.dumps(payload).encode("utf-8")


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        yield from self._body.splitlines()


def test_annotate_clusters_with_llm_parses_response() -> None:
    """A mocked Ollama response is parsed into a dict keyed by cluster id."""

    response_body = _build_ollama_response_body(
        [
            {
                "cluster_id": "0",
                "cell_type": "Luminal epithelial",
                "confidence": 0.92,
                "rationale": "KRT8 high",
            },
            {
                "cluster_id": "1",
                "cell_type": "T cell",
                "confidence": 0.87,
                "rationale": "CD3D high",
            },
        ]
    )

    with patch.object(
        annotation,
        "urlopen",
        return_value=_FakeResponse(response_body),
    ):
        result = annotation.annotate_clusters_with_llm(
            {"0": ["KRT8"], "1": ["CD3D"]},
            model="llama3.1:8b",
            evidence_type="marker_genes",
        )

    assert set(result) == {"0", "1"}
    assert result["0"]["cell_type"] == "Luminal epithelial"
    assert result["0"]["confidence"] == pytest.approx(0.92)
    assert result["1"]["rationale"] == "CD3D high"


def test_annotate_clusters_with_llm_raises_on_network_error() -> None:
    """A URLError from urlopen surfaces as a descriptive RuntimeError."""

    with patch.object(
        annotation, "urlopen", side_effect=URLError("connection refused")
    ):
        with pytest.raises(RuntimeError, match="failed to reach local LLM"):
            annotation.annotate_clusters_with_llm(
                {"0": ["KRT8"]},
                model="llama3.1:8b",
                evidence_type="marker_genes",
            )


def test_annotate_clusters_with_llm_raises_on_timeout() -> None:
    """A socket timeout surfaces as a descriptive RuntimeError."""

    with patch.object(annotation, "urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(RuntimeError, match="failed to reach local LLM"):
            annotation.annotate_clusters_with_llm(
                {"0": ["KRT8"]},
                model="llama3.1:8b",
                evidence_type="marker_genes",
            )


def test_annotate_clusters_with_llm_raises_on_invalid_json() -> None:
    """A non-JSON response body surfaces as a descriptive RuntimeError."""

    with patch.object(
        annotation,
        "urlopen",
        return_value=_FakeResponse(b"not json"),
    ):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            annotation.annotate_clusters_with_llm(
                {"0": ["KRT8"]},
                model="llama3.1:8b",
                evidence_type="marker_genes",
            )


def test_annotate_clusters_with_llm_sends_expected_payload() -> None:
    """The request payload includes the correct model and deterministic options."""

    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return _FakeResponse(
            _build_ollama_response_body(
                [
                    {
                        "cluster_id": "0",
                        "cell_type": "Luminal",
                        "confidence": 0.5,
                        "rationale": "",
                    }
                ]
            )
        )

    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(annotation, "urlopen", side_effect=fake_urlopen),
    ):
        annotation.annotate_clusters_with_llm(
            {"0": ["KRT8"]},
            model="llama3.1:8b",
            evidence_type="marker_genes",
        )

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "llama3.1:8b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"] == {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": 1200,
    }


def test_annotate_clusters_with_llm_uses_slurm_cpu_count() -> None:
    """The request payload pins Ollama threads to the SLURM allocation."""

    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            _build_ollama_response_body(
                [
                    {
                        "cluster_id": "0",
                        "cell_type": "Luminal",
                        "confidence": 0.5,
                        "rationale": "",
                    }
                ]
            )
        )

    with (
        patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "8"}, clear=True),
        patch.object(annotation, "urlopen", side_effect=fake_urlopen),
    ):
        annotation.annotate_clusters_with_llm(
            {"0": ["KRT8"]},
            model="llama3.1:8b",
            evidence_type="marker_genes",
        )

    assert captured["payload"]["options"]["num_thread"] == 8
    assert captured["payload"]["options"]["num_predict"] == 1200


def test_annotate_clusters_with_llm_uses_ollama_host_environment() -> None:
    """The request URL follows OLLAMA_HOST when no explicit host is provided."""

    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _FakeResponse(
            _build_ollama_response_body(
                [
                    {
                        "cluster_id": "0",
                        "cell_type": "Luminal",
                        "confidence": 0.5,
                        "rationale": "",
                    }
                ]
            )
        )

    with (
        patch.dict("os.environ", {"OLLAMA_HOST": "127.0.0.1:18234"}, clear=True),
        patch.object(annotation, "urlopen", side_effect=fake_urlopen),
    ):
        annotation.annotate_clusters_with_llm(
            {"0": ["KRT8"]},
            model="llama3.1:8b",
            evidence_type="marker_genes",
        )

    assert captured["url"] == "http://127.0.0.1:18234/api/chat"


def test_annotate_clusters_with_llm_sends_one_request_for_many_groups() -> None:
    """Annotation sends all groups in one non-streaming request."""

    captured_payloads: list[dict] = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured_payloads.append(payload)
        return _FakeResponse(
            _build_ollama_response_body(
                [
                    {
                        "cluster_id": str(cluster_id),
                        "cell_type": f"cell type {cluster_id}",
                        "confidence": 0.5,
                        "rationale": "",
                    }
                    for cluster_id in range(12)
                ]
            )
        )

    with patch.object(annotation, "urlopen", side_effect=fake_urlopen):
        result = annotation.annotate_clusters_with_llm(
            {str(cluster_id): ["KRT8"] for cluster_id in range(12)},
            model="llama3.1:8b",
            evidence_type="marker_genes",
        )

    assert len(captured_payloads) == 1
    assert len(result) == 12
