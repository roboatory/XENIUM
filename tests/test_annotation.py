from __future__ import annotations

import json
from unittest.mock import patch
from urllib.error import URLError

import pytest

from src import annotation


def test_annotation_schema_shape() -> None:
    """_annotation_schema declares the required top-level and item keys."""

    schema = annotation._annotation_schema()
    assert schema["type"] == "object"
    assert schema["required"] == ["annotations"]
    item_schema = schema["properties"]["annotations"]["items"]
    assert set(item_schema["required"]) == {
        "cluster_id",
        "cell_type",
        "confidence",
        "rationale",
    }


def test_build_annotation_prompt_marker_mode_includes_evidence() -> None:
    """Marker-mode prompt lists clusters and their formatted evidence items."""

    prompt = annotation._build_annotation_prompt(
        {"0": ["KRT8", "EPCAM"], "1": ["CD3D"]},
        condition="prostate cancer",
        evidence_type="marker_genes",
    )
    assert "Condition: prostate cancer." in prompt
    assert "marker genes" in prompt
    assert "- 0: KRT8, EPCAM" in prompt
    assert "- 1: CD3D" in prompt


def test_build_annotation_prompt_neighborhood_mode_uses_niche_wording() -> None:
    """Neighborhood-mode prompt instructs niche/interface-style labels and formats pairs."""

    prompt = annotation._build_annotation_prompt(
        {"0": [("Tumor", 0.6), ("Stroma", 0.4)]},
        condition="prostate cancer",
        evidence_type="neighborhood_cell_types",
    )
    assert "niche/interface" in prompt
    assert "- 0: Tumor (60.0%), Stroma (40.0%)" in prompt


def _build_ollama_response_body(annotations: list[dict]) -> bytes:
    """Produce the bytes an Ollama /api/chat response would deliver."""

    payload = {"message": {"content": json.dumps({"annotations": annotations})}}
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
            condition="prostate cancer",
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
                condition="prostate cancer",
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
                condition="prostate cancer",
                evidence_type="marker_genes",
            )


def test_annotate_clusters_with_llm_sends_expected_payload() -> None:
    """The request payload includes the correct model, stream flag, and deterministic options."""

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

    with patch.object(annotation, "urlopen", side_effect=fake_urlopen):
        annotation.annotate_clusters_with_llm(
            {"0": ["KRT8"]},
            model="llama3.1:8b",
            condition="prostate cancer",
            evidence_type="marker_genes",
        )

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "llama3.1:8b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"] == {"temperature": 0.0, "seed": 42}
