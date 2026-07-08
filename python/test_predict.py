from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from clasificacion_residuos_ejemplo import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    MAX_UPLOAD_SIZE,
    UNRELATED_IMAGE_DETAIL,
    app,
    build_local_reply,
    get_gemini_reply,
    validate_prediction_confidence,
    validate_upload_file,
)

VALID_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x02\x00\x01"
    b"\xe2\x21\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82"
)

client = TestClient(app)


def make_upload_file(content: bytes, filename: str, content_type: str) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def test_validate_upload_file_accepts_valid_png() -> None:
    upload = make_upload_file(VALID_PNG, "test.png", "image/png")
    validate_upload_file(upload)
    upload.file.seek(0)
    assert upload.content_type in ALLOWED_IMAGE_CONTENT_TYPES


def test_validate_upload_file_rejects_invalid_content_type() -> None:
    upload = make_upload_file(VALID_PNG, "test.gif", "image/gif")
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_file(upload)
    assert exc_info.value.status_code == 400
    assert "Formato de archivo no válido" in exc_info.value.detail


def test_validate_upload_file_rejects_empty_file() -> None:
    upload = make_upload_file(b"", "empty.png", "image/png")
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_file(upload)
    assert exc_info.value.status_code == 400
    assert "vacío" in exc_info.value.detail.lower()


def test_validate_upload_file_rejects_large_file() -> None:
    large_content = b"0" * (MAX_UPLOAD_SIZE + 1)
    upload = make_upload_file(large_content, "big.png", "image/png")
    with pytest.raises(HTTPException) as exc_info:
        validate_upload_file(upload)
    assert exc_info.value.status_code == 413
    assert "límite" in exc_info.value.detail


def test_validate_prediction_confidence_accepts_clear_prediction() -> None:
    validate_prediction_confidence([0.04, 0.96])


def test_validate_prediction_confidence_rejects_unrelated_or_uncertain_image() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_prediction_confidence([0.48, 0.52])

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == UNRELATED_IMAGE_DETAIL


def test_build_local_reply_handles_more_recycling_topics() -> None:
    assert "Pilas" in build_local_reply("Que hago con una pila usada?")
    assert "aceite" in build_local_reply("Donde tiro aceite de cocina?").lower()
    assert "organicos" in build_local_reply("Que hago con cascaras de fruta?").lower()


def test_get_gemini_reply_uses_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return b'{"candidates":[{"content":{"parts":[{"text":"Si, es reciclable."}]}}]}'

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("clasificacion_residuos_ejemplo.GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("clasificacion_residuos_ejemplo.GOOGLE_MODEL", "gemini-flash-lite-latest")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reply = get_gemini_reply("Una botella de vidrio limpia es reciclable?")

    assert reply == "Si, es reciclable."
    assert captured["url"].endswith("/v1beta/models/gemini-flash-lite-latest:generateContent")
    assert captured["headers"]["X-goog-api-key"] == "test-key"
    assert b"systemInstruction" in captured["body"]
    assert captured["timeout"] == 20


def test_predict_endpoint_rejects_corrupted_image() -> None:
    response = client.post(
        "/predict",
        files={"file": ("bad.png", b"notanimage", "image/png")},
    )
    assert response.status_code == 400
    assert "no válido" in response.json()["detail"].lower()
