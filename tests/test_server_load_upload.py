import importlib
import os
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import pycaret_mcp.adapter as pa
import pycaret_mcp.server as server


def test_load_endpoint(monkeypatch, tmp_path):
    # create a fake module and monkeypatch pa.load_model
    monkeypatch.setattr(pa, "load_model", lambda p, task="classification": object())
    client = TestClient(server.app)

    r = client.post("/load", json={"path": "models/foo", "task": "classification"})
    assert r.status_code == 200
    assert r.json()["loaded"] is True


def test_upload_endpoint(monkeypatch, tmp_path):
    # create a small dummy file and fake load_model
    content = b"dummy-model-bytes"
    fpath = tmp_path / "tmp.pkl"
    fpath.write_bytes(content)

    def fake_load(path, task="classification"):
        assert "tmp.pkl" in str(path)
        return object()

    monkeypatch.setattr(pa, "load_model", fake_load)

    client = TestClient(server.app)
    with open(fpath, "rb") as fh:
        files = {"file": ("tmp.pkl", fh, "application/octet-stream")}
        r = client.post("/upload", files=files)
    assert r.status_code == 200
    assert r.json()["uploaded"] is True
    # ensure file saved
    assert (Path("models") / "tmp.pkl").exists()
