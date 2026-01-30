import importlib
import os
import pandas as pd
from fastapi.testclient import TestClient

import pycaret_mcp.adapter as pa


def test_health():
    import app
    client = TestClient(app.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metadata_no_model(monkeypatch):
    # ensure no model loaded and no env var
    monkeypatch.delenv("PYCARET_MODEL_PATH", raising=False)
    import app
    importlib.reload(app)
    client = TestClient(app.app)
    r = client.get("/metadata")
    assert r.status_code == 200
    assert r.json() == {"loaded": False}


def test_predict_no_model(monkeypatch):
    monkeypatch.delenv("PYCARET_MODEL_PATH", raising=False)
    import app
    importlib.reload(app)
    client = TestClient(app.app)
    r = client.post("/predict", json={"data": [{"a": 1}]})
    assert r.status_code == 400


def test_predict_success(monkeypatch):
    # set env var and patch pycaret_adapter.load_model and predict
    monkeypatch.setenv("PYCARET_MODEL_PATH", "models/foo")

    def fake_load(path, task="classification"):
        return object()

    def fake_predict(model, data):
        return pd.DataFrame({"Label": [1]})

    monkeypatch.setattr(pa, "load_model", fake_load)
    monkeypatch.setattr(pa, "predict", fake_predict)

    import app
    importlib.reload(app)
    client = TestClient(app.app)
    r = client.post("/predict", json={"data": [{"a": 1}]})
    assert r.status_code == 200
    assert r.json() == {"predictions": [{"Label": 1}]}
