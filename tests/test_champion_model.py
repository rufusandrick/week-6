import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

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

    dialog_id, messages = next(iter(dialog_data.items()))  # Берём первые попавшиеся ключ и значение из словаря
    first = " ".join([k["text"] for k in messages["0"] if k["participant_index"] == "0"])  # Фразы первого участника
    second = " ".join([k["text"] for k in messages["0"] if k["participant_index"] == "1"])  # Фразы второго участника
    messages = [first, second]

    for _participant_index, msg in enumerate(messages):
        model.predict_proba(msg)[0, 1]


if __name__ == "__main__":
    main()
