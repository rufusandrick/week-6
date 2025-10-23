import logging
import os

import boto3
import cloudpickle
import mlflow
import mlflow.sklearn
import numpy as np
import onnxruntime as rt
from dotenv import load_dotenv
from skl2onnx import to_onnx
from sklearn.pipeline import Pipeline

load_dotenv()

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
port = os.getenv("MLFLOW_PORT", 5050)
tracking_uri = f"http://{username}:{password}@localhost:{port}"
s3_endpoint = "http://localhost:9000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    mlflow.set_tracking_uri(tracking_uri)

    model_name = "week-4-logreg@champion"
    model_uri = f"models:/{model_name}"

    logger.info("Loading model")
    model_pipeline = mlflow.sklearn.load_model(model_uri)
    logger.info("Loading complete")

    messages = [
        "Первое сообщение",
        "Второе",
        "Я — ChatGPT, искусственный интеллект"
    ]

    transform = Pipeline(model_pipeline.steps[:-1])
    model = model_pipeline.steps[-1][1]

    input_matrix = transform.transform(messages).toarray().astype(np.float32)
    options = {id(model): {"zipmap": False}}

    logger.info("Start convert")
    onx = to_onnx(model, input_matrix, options=options)
    with open("model_repository/logreg/1/model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    with open("model_repository/text_preprocess/1/transform.pkl", "wb") as f:
        cloudpickle.dump(transform, f)

    logger.info("Convertion is ended")

    logger.info("Start check")
    test_messages = [
        "Какое-то новое сообщение",
        "И тут же второе",
        "Я человек, не баньте"
    ]

    input_matrix = transform.transform(test_messages).toarray().astype(np.float32)

    onnx_model = rt.InferenceSession("model_repository/logreg/1/model.onnx", providers=["CPUExecutionProvider"])
    inp = onnx_model.get_inputs()[0].name
    outs = [o.name for o in onnx_model.get_outputs()]

    raw_proba = model.predict_proba(input_matrix)
    onnx_proba = onnx_model.run([outs[1]], {inp: input_matrix})[0]

    assert np.linalg.norm(raw_proba - onnx_proba) < 1e-5
    logger.info("Check is successful")

    logger.info("Uploading model to S3")
    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3_client.upload_file("model_repository/logreg/1/model.onnx", "models", "triton/model.onnx")
    s3_client.upload_file("model_repository/text_preprocess/1/transform.pkl", "models", "triton/transform.pkl")
    logger.info("Upload complete")


if __name__ == "__main__":
    main()
