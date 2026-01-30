"""pycaret_mcp package - minimal MCP-style wrapper for PyCaret models."""
from .adapter import load_model, predict, save_model, model_metadata
from .server import app

__all__ = ["load_model", "predict", "save_model", "model_metadata", "app"]
