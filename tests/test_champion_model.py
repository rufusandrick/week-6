from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

mlflow = pytest.importorskip("mlflow")
mlflow_sklearn = pytest.importorskip("mlflow.sklearn")
mlflow.sklearn = mlflow_sklearn

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
port = os.getenv("MLFLOW_PORT", "5050")
tracking_uri = f"http://{username}:{password}@localhost:{port}"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


def main() -> None:
    mlflow.set_tracking_uri(tracking_uri)

    model_name = "week-4-logreg@champion"
    model_uri = f"models:/{model_name}"

    model = mlflow.sklearn.load_model(model_uri)

    base_dir = Path(__file__).resolve().parent
    dialog_path = base_dir / "sample_dialog.json"
    with dialog_path.open(encoding="utf-8") as file:
        dialog_data = json.load(file)

    _dialog_id, messages = next(iter(dialog_data.items()))
    first = " ".join(message["text"] for message in messages["0"] if message["participant_index"] == "0")
    second = " ".join(message["text"] for message in messages["0"] if message["participant_index"] == "1")
    texts = [first, second]

    for _participant_index, msg in enumerate(texts):
        model.predict_proba(msg)[0, 1]


if __name__ == "__main__":
    main()
