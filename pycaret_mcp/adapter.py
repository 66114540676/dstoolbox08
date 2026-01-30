"""Adapter utilities for PyCaret models.

Moved from top-level `pycaret_adapter.py` into package layout and
kept functions nearly identical but with clearer docstrings.
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

    Prefers `pycaret.<task>.predict_model` when available. Returns a DataFrame.
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
            try:
                return mod.predict_model(model, data=df)
            except TypeError:
                return mod.predict_model(model, df)

    # Fallback: assume model has predict
    if hasattr(model, "predict"):
        preds = model.predict(df)
        return pd.DataFrame({"Label": preds})

    raise RuntimeError("No available prediction function for provided model")


def save_model(model: Any, path: str, task: str = "classification") -> Any:
    """Save a PyCaret model using `pycaret.<task>.save_model`.

    Returns whatever PyCaret `save_model` returns.
    """
    task = (task or "classification").lower()
    module_name = "pycaret.classification" if task.startswith("class") else "pycaret.regression"
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "save_model"):
        raise RuntimeError(f"pycaret module '{module_name}' has no save_model")
    return mod.save_model(model, path)


def model_metadata(model: Any) -> Dict[str, Any]:
    """Extract minimal metadata from a PyCaret model/pipeline."""
    meta: Dict[str, Any] = {"class": model.__class__.__name__}
    for attr in ("_name", "model_type", "estimator", "meta"):
        if hasattr(model, attr):
            try:
                meta[attr] = getattr(model, attr)
            except Exception:
                meta[attr] = "<unserializable>"
    return meta


__all__ = ["load_model", "predict", "save_model", "model_metadata"]
