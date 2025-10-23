import json
import os

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
port = os.getenv("MLFLOW_PORT", 5050)
tracking_uri = f"http://{username}:{password}@localhost:{port}"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


def main() -> None:
    mlflow.set_tracking_uri(tracking_uri)

    model_name = "week-4-logreg@champion"
    model_uri = f"models:/{model_name}"

    model = mlflow.sklearn.load_model(model_uri)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dialog_path = os.path.join(base_dir, "sample_dialog.json")
    with open(dialog_path, encoding="utf-8") as f:
        dialog_data = json.load(f)

    dialog_id, messages = next(iter(dialog_data.items()))  # Берём первые попавшиеся ключ и значение из словаря
    first = " ".join([k["text"] for k in messages["0"] if k["participant_index"]=="0"])  # Фразы первого участника
    second = " ".join([k["text"] for k in messages["0"] if k["participant_index"]=="1"])  # Фразы второго участника
    messages = [first, second]

    participant_index = 0
    for msg in messages:
        model.predict_proba(msg)[0, 1]

        participant_index += 1


if __name__ == "__main__":
    main()
