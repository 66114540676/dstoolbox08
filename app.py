"""FastAPI server wrapping `pycaret_adapter`.

Endpoints:
- GET /health -> simple liveness
- GET /metadata -> model metadata if loaded
- POST /predict -> accepts JSON {"data": [...] } and returns predictions

Model is loaded lazily from environment variable `PYCARET_MODEL_PATH`.
Run server: `uvicorn app:app --reload --port 8080` or with process manager in production.
"""
from __future__ import annotations

# Backwards compatibility shim: import the server from package
from pycaret_mcp.server import app  # type: ignore
