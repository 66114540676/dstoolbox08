"""Train a small PyCaret model and save it to `models/iris_model`.

This script is intentionally minimal and intended for CI/dev integration tests.
It uses the `iris` dataset from `pycaret.datasets` and a classification workflow.
"""
from pathlib import Path


def main():
    try:
        from pycaret.datasets import get_data
        from pycaret.classification import setup, compare_models, finalize_model, save_model
    except Exception as exc:  # pragma: no cover - import-time environment-dependent
        raise RuntimeError("pycaret is required to run training script") from exc

    df = get_data("iris")
    # setup for a quick run (session_id for reproducibility)
    s = setup(df, target="species", session_id=123, silent=True, html=False)
    model = compare_models()
    final = finalize_model(model)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(final, str(out_dir / "iris_model"))
    print(f"Saved model to {str(out_dir / 'iris_model')}")


if __name__ == "__main__":
    main()
