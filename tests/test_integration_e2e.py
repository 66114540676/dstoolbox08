import os
import sys
import subprocess
import time
from pathlib import Path

import pytest
import httpx

pytestmark = pytest.mark.integration


def _is_model_present(path: Path) -> bool:
    # pycaret.save_model typically writes a .pkl file with the given name
    if path.with_suffix('.pkl').exists():
        return True
    if path.exists():
        return True
    return False


def test_integration_end_to_end(tmp_path):
    # Skip integration locally if pycaret not available
    pytest.importorskip("pycaret")
    # 1) Train and save model (script must exist)
    script = Path("scripts/train_save_pycaret.py")
    assert script.exists(), "training script not found"

    subprocess.check_call([sys.executable, str(script)])

    model_path = Path("models") / "iris_model"
    assert _is_model_present(model_path), "Model file not found after training"

    # 2) Start server with env pointing to model
    port = 8081
    env = os.environ.copy()
    env["PYCARET_MODEL_PATH"] = str(model_path)
    env["PYCARET_MODEL_TASK"] = "classification"

    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, env=env)

    try:
        client = httpx.Client(timeout=5.0)
        # wait for server to be ready
        for _ in range(60):
            try:
                r = client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("Server did not start in time")

        # 3) Call predict endpoint with a single iris sample
        sample = {"data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}
        r = client.post(f"http://127.0.0.1:{port}/predict", json=sample)
        assert r.status_code == 200, f"Predict failed: {r.status_code} {r.text}"
        body = r.json()
        assert "predictions" in body
        assert isinstance(body["predictions"], list)
        assert len(body["predictions"]) == 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
