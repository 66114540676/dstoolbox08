"""Minimal PyCaret adapter for an MCP-like server.

Provides simple helpers to load, predict, save and inspect metadata from
PyCaret models. Designed to mirror the interface used in pandas-mcp-server
so it can be integrated into an MCP server implementation.
"""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd
import importlib


def load_model(path: str, task: str = "classification") -> Any:
    """Load a PyCaret model saved with `save_model`.

    Args:
        path: Path to the saved PyCaret model file (without extension).
        task: 'classification' or 'regression'. Defaults to 'classification'.
    """
    task = (task or "classification").lower()
    if task.startswith("class"):
        module_name = "pycaret.classification"
    else:
        module_name = "pycaret.regression"

    mod = importlib.import_module(module_name)
    if not hasattr(mod, "load_model"):
        raise RuntimeError(f"pycaret module '{module_name}' has no load_model")

    return mod.load_model(path)


def predict(model: Any, input_data: pd.DataFrame | Dict | list) -> pd.DataFrame:
    """Run inference using a PyCaret model.

    - Prefers `predict_model(model, data=df)` from PyCaret (classification or regression)
    - Falls back to `model.predict(df)` if `predict_model` isn't available

    Returns:
        A pandas DataFrame with prediction outputs (the same structure PyCaret returns)
    """
    if not isinstance(input_data, pd.DataFrame):
        df = pd.DataFrame(input_data)
    else:
        df = input_data

    # Try classification then regression predict_model
    for module_name in ("pycaret.classification", "pycaret.regression"):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, "predict_model"):
            # pycaret.predict_model signature: predict_model(estimator, data=...)
            try:
                return mod.predict_model(model, data=df)
            except TypeError:
                # older/newer signatures might accept differently
                return mod.predict_model(model, df)

    # Fallback: assume model has predict
    if hasattr(model, "predict"):
        preds = model.predict(df)
        return pd.DataFrame({"Label": preds})

    raise RuntimeError("No available prediction function for provided model")


def save_model(model: Any, path: str, task: str = "classification") -> Any:
    """Save a PyCaret model using `pycaret.<task>.save_model`.

    Returns whatever PyCaret `save_model` returns (often a filepath or artefact).
    """
    task = (task or "classification").lower()
    module_name = "pycaret.classification" if task.startswith("class") else "pycaret.regression"
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "save_model"):
        raise RuntimeError(f"pycaret module '{module_name}' has no save_model")
    return mod.save_model(model, path)


def model_metadata(model: Any) -> Dict[str, Any]:
    """Extract minimal metadata from a PyCaret model/pipeline.

    This is intentionally lightweight: it returns the model class name and
    some safe attributes if present (e.g., summary or estimator info).
    """
    meta: Dict[str, Any] = {"class": model.__class__.__name__}
    # Common PyCaret pipeline attributes
    for attr in ("_name", "model_type", "estimator", "meta"):  # conservative list
        if hasattr(model, attr):
            try:
                meta[attr] = getattr(model, attr)
            except Exception:
                meta[attr] = "<unserializable>"
    return meta


__all__ = ["load_model", "predict", "save_model", "model_metadata"]
