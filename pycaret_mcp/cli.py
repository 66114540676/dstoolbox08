"""Simple CLI entrypoint for running the server."""
from __future__ import annotations

import os
import uvicorn


def serve(host: str = "127.0.0.1", port: int = 8080, reload: bool = False, model_path: str | None = None, task: str = "classification"):
    if model_path:
        os.environ.setdefault("PYCARET_MODEL_PATH", model_path)
        os.environ.setdefault("PYCARET_MODEL_TASK", task)
    uvicorn.run("pycaret_mcp.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
