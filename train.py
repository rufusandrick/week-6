"""Training pipeline for dialog bot classification."""

from __future__ import annotations

import os
from typing import Any, Self

# Third-party packages
import mlflow
import mlflow.catboost
import mlflow.sklearn
import pandas as pd
import psutil
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Локальные импорты
from prepare_data import DATA_PATH

load_dotenv()
username = os.getenv("MLFLOW_TRACKING_USERNAME", "")
password = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
port = os.getenv("MLFLOW_PORT", "5050")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
EXP_NAME = "week-4"
EXPERIMENT_LIMIT = 10
TOP_MODEL_COUNT = 2


class TextCleaner(BaseEstimator, TransformerMixin):
    """Lowercase text transformer for use within scikit-learn pipelines."""

    def fit(self, X: pd.Series | list[str], y: pd.Series | None = None) -> Self:  # noqa: N803 - keep sklearn signature
        del X, y
        return self

    def transform(self, X: pd.Series | list[str]) -> list[str]:  # noqa: N803 - keep sklearn signature
        return [str(x).lower() for x in X]


def run_single_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str,
    model_params: dict[str, Any],
    data_processing_params: dict[str, Any],
) -> tuple[str, float]:
    """Run a single experiment and log metrics to MLflow."""
    with mlflow.start_run() as run:
        mlflow.set_tag("experiment", EXP_NAME)

        mlflow.log_param("model_type", model_type)

        for k, v in model_params.items():
            mlflow.log_param(f"model_params_{k}", v)

        for k, v in data_processing_params.items():
            mlflow.log_param(f"data_processing_{k}", v)

        if model_type == "logreg":
            model = LogisticRegression(**model_params)
        elif model_type == "catboost":
            model = CatBoostClassifier(**model_params, verbose=0)
        else:
            msg = f"Unknown model_type: {model_type}"
            raise ValueError(msg)

        pipeline = Pipeline(
            [
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(**data_processing_params)),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", model),
            ]
        )

        pipeline.fit(train_df["text"], train_df["is_bot"])

        train_preds = pipeline.predict(train_df["text"])  # Предсказываем метки
        train_probas = pipeline.predict_proba(train_df["text"])[:, 1]  # Предсказываем вероятности

        train_accuracy = accuracy_score(train_df["is_bot"], train_preds)
        train_logloss = log_loss(train_df["is_bot"], train_probas)
        train_f1 = f1_score(train_df["is_bot"], train_preds)
        train_precision = precision_score(train_df["is_bot"], train_preds)
        train_recall = recall_score(train_df["is_bot"], train_preds)

        val_preds = pipeline.predict(val_df["text"])
        val_probas = pipeline.predict_proba(val_df["text"])[:, 1]

        val_accuracy = accuracy_score(val_df["is_bot"], val_preds)
        val_logloss = log_loss(val_df["is_bot"], val_probas)
        val_f1 = f1_score(val_df["is_bot"], val_preds)
        val_precision = precision_score(val_df["is_bot"], val_preds)
        val_recall = recall_score(val_df["is_bot"], val_preds)

        mlflow.log_metric("train_num_dialogs", len(train_df))
        mlflow.log_metric("val_num_dialogs", len(val_df))

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("train_logloss", train_logloss)
        mlflow.log_metric("val_logloss", val_logloss)

        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("val_recall", val_recall)

        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        mlflow.log_metric("cpu_usage", cpu_usage)
        mlflow.log_metric("memory_usage", memory_usage)

        mlflow.sklearn.log_model(pipeline, name="model")

        run_id = run.info.run_id
        return run_id, val_logloss


def build_experiment_grid(
    data_processing_params: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Create a list of experiments to run based on model grids."""
    experiments: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for model_type in ["logreg", "catboost"]:
        if model_type == "logreg":
            for penalty_strength in [0.5, 0.9, 1.0]:
                for max_iter in [200, 300]:
                    model_params = {"C": penalty_strength, "max_iter": max_iter}
                    experiments.append((model_type, model_params, data_processing_params))
        else:
            for iterations in [200, 300]:
                for depth in [3, 4, 5]:
                    model_params = {"iterations": iterations, "depth": depth}
                    experiments.append((model_type, model_params, data_processing_params))

    return experiments[:EXPERIMENT_LIMIT]


def ensure_registered_model(client: MlflowClient, reg_name: str) -> None:
    """Create a registered model if it does not exist."""
    try:
        client.get_registered_model(reg_name)
    except MlflowException:
        client.create_registered_model(reg_name)


def register_top_models(df_res: pd.DataFrame, client: MlflowClient, top_n: int) -> None:
    """Register champion and challenger models based on validation loss."""
    for model_type in df_res["model_type"].unique():
        top_runs = df_res[df_res["model_type"] == model_type].sort_values("val_logloss").head(top_n)
        if len(top_runs) < top_n:
            continue

        champion_run_id = top_runs.iloc[0]["run_id"]
        challenger_run_id = top_runs.iloc[1]["run_id"]

        reg_name = f"{EXP_NAME}-{model_type}"
        ensure_registered_model(client, reg_name)

        champion_model_uri = f"runs:/{champion_run_id}/model"
        mv_champion = client.create_model_version(name=reg_name, source=champion_model_uri, run_id=champion_run_id)
        client.set_registered_model_alias(name=reg_name, alias="champion", version=mv_champion.version)

        challenger_model_uri = f"runs:/{challenger_run_id}/model"
        mv_challenger = client.create_model_version(
            name=reg_name,
            source=challenger_model_uri,
            run_id=challenger_run_id,
        )
        client.set_registered_model_alias(name=reg_name, alias="challenger", version=mv_challenger.version)


def main() -> None:
    train_prepared = pd.read_csv(DATA_PATH / "train_prepared.csv", index_col=0)
    train_df, val_df = train_test_split(train_prepared, test_size=0.2)

    tracking_uri = (
        f"http://{username}:{password}@localhost:{port}" if username and password else f"http://localhost:{port}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXP_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    data_processing_params = {"ngram_range": (1, 2), "max_df": 0.95}
    experiments_to_run = build_experiment_grid(data_processing_params)

    results: list[tuple[str, str, float]] = []
    for model_type, model_params, processing_params in experiments_to_run:
        run_id, val_metric = run_single_experiment(train_df, val_df, model_type, model_params, processing_params)
        results.append((run_id, model_type, val_metric))

    df_res = pd.DataFrame(results, columns=["run_id", "model_type", "val_logloss"])
    register_top_models(df_res, client, TOP_MODEL_COUNT)


if __name__ == "__main__":
    main()
