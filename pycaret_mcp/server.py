"""FastAPI server implementation for the pycaret_mcp package.

Provides endpoints:
- GET /health
- GET /metadata
- POST /predict
- POST /load    -> load model from existing path
- POST /upload  -> multipart upload model file (saves under models/) and loads it
"""
from __future__ import annotations

from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import threading
import shutil

from . import adapter as pa

app = FastAPI(title="PyCaret MCP Adapter Server")

_model_lock = threading.Lock()
_model: Any | None = None
_model_info: dict | None = None

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _set_model(m: Any, info: dict | None = None):
    global _model, _model_info
    with _model_lock:
        _model = m
        _model_info = info


def _get_model():
    # Lazy load from environment if present (helps tests and backward compatibility)
    with _model_lock:
        if _model is None:
            import os
            env_path = os.environ.get("PYCARET_MODEL_PATH")
            env_task = os.environ.get("PYCARET_MODEL_TASK", "classification")
            if env_path:
                try:
                    m = pa.load_model(env_path, task=env_task)
                    _set_model(m)
                except Exception:
                    # ignore load errors here
                    pass
        return _model


# Auto-load model from environment if provided for backwards compatibility
import os
_model_env_path = os.environ.get("PYCARET_MODEL_PATH")
_model_env_task = os.environ.get("PYCARET_MODEL_TASK", "classification")
if _model_env_path:
    try:
        # Try to load and set model; ignore errors during import/startup
        m = pa.load_model(_model_env_path, task=_model_env_task)
        _set_model(m)
    except Exception:
        pass


class PredictRequest(BaseModel):
    data: Any


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    model = _get_model()
    if model is None:
        return {"loaded": False}
    return {"loaded": True, "metadata": pa.model_metadata(model)}


@app.post("/predict")
def predict(req: PredictRequest):
    model = _get_model()
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Use /load or set PYCARET_MODEL_PATH")
    df = pa.predict(model, req.data)
    return {"predictions": df.to_dict(orient="records")}


@app.post("/load")
def load_model(body: dict):
    """Load a model from server-side path: {"path": "models/iris_model", "task": "classification"}"""
    path = body.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="'path' is required")
    task = body.get("task", "classification")
    try:
        m = pa.load_model(path, task=task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _set_model(m)
    return {"loaded": True, "path": path}


@app.post("/upload")
def upload_model(file: UploadFile = File(...), task: str = "classification"):
    """Upload a model file and load it. Saves file under `models/` retaining filename."""
    dest = MODELS_DIR / file.filename
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to save file: {exc}")

    # Try to load (path without extension if PyCaret expects it)
    p = str(dest)
    try:
        m = pa.load_model(p, task=task)
    except Exception:
        # If load_model expects path without extension, try stripping suffix
        try:
            m = pa.load_model(str(dest.with_suffix('')), task=task)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to load uploaded model: {exc}")

    _set_model(m)
    return {"uploaded": True, "filename": file.filename}
