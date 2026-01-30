import sys
from types import ModuleType
import pandas as pd
import pytest

import pycaret_mcp.adapter as pa


def make_module_with_funcs(name: str, **funcs):
    mod = ModuleType(name)
    for k, v in funcs.items():
        setattr(mod, k, v)
    return mod


def test_load_model_classification(monkeypatch):
    mod = make_module_with_funcs("pycaret.classification", load_model=lambda p: {"loaded": p})
    monkeypatch.setitem(sys.modules, "pycaret.classification", mod)

    res = pa.load_model("models/foo", task="classification")
    assert isinstance(res, dict) and res["loaded"] == "models/foo"


def test_predict_uses_predict_model(monkeypatch):
    # Predict model should return a DataFrame similar to PyCaret
    def fake_predict_model(model, data=None):
        # data should be the DataFrame passed
        assert isinstance(data, pd.DataFrame)
        return pd.DataFrame({"Label": [0, 1], "Score": [0.2, 0.8]})

    mod = make_module_with_funcs("pycaret.classification", predict_model=fake_predict_model)
    monkeypatch.setitem(sys.modules, "pycaret.classification", mod)

    fake_model = object()
    input_data = [{"a": 1}, {"a": 2}]
    df = pa.predict(fake_model, input_data)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Label", "Score"]
    assert df.shape[0] == 2


def test_predict_fallback_to_model_predict():
    # Ensure that if pycaret modules are not present, it falls back to model.predict
    class FakeModel:
        def predict(self, df):
            # return simple predictions list
            return [42] * len(df)

    fake = FakeModel()
    input_data = [{"x": 10}, {"x": 20}, {"x": 30}]
    df = pa.predict(fake, input_data)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Label"]
    assert df.shape[0] == 3
    assert df["Label"].tolist() == [42, 42, 42]


def test_save_model_calls_save_model(monkeypatch):
    mod = make_module_with_funcs("pycaret.classification", save_model=lambda m, p: f"saved:{p}")
    monkeypatch.setitem(sys.modules, "pycaret.classification", mod)
    res = pa.save_model(object(), "models/foo", task="classification")
    assert res == "saved:models/foo"


def test_model_metadata_extracts_attrs():
    class Dummy:
        def __init__(self):
            self._name = "dummy"
            self.estimator = "est"

    d = Dummy()
    meta = pa.model_metadata(d)
    assert meta["class"] == "Dummy"
    assert meta["_name"] == "dummy"
    assert meta["estimator"] == "est"
