# Стандартные библиотеки
import os

# Сторонние пакеты
import mlflow
import mlflow.catboost
import mlflow.sklearn
import pandas as pd
import psutil
from catboost import CatBoostClassifier
from dotenv import load_dotenv
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
port = os.getenv("MLFLOW_PORT", 5050)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
EXP_NAME = "week-4"


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [str(x).lower() for x in X]


def run_single_experiment(
        train_df, val_df,
        model_type,
        model_params,
        data_processing_params
):
    """Запускает один эксперимент, логирует в MLflow."""
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

        pipeline = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(**data_processing_params)),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", model),
        ])

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


def main() -> None:
    train_prepared = pd.read_csv(DATA_PATH / "train_prepared.csv", index_col=0)
    train_df, val_df = train_test_split(train_prepared, test_size=0.2)

    tracking_uri = f"http://{username}:{password}@localhost:{port}" if username and password else f"http://localhost:{port}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXP_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments_to_run = []
    data_processing_params = {
        "ngram_range": (1, 2),
        "max_df": 0.95
    }
    for mt in ["logreg", "catboost"]:
        if mt == "logreg":
            for C in [0.5, 0.9, 1.0]:
                for max_iter in [200, 300]:
                    model_params = {
                        "C": C,
                        "max_iter": max_iter
                    }
                    experiments_to_run.append(
                        (mt, model_params, data_processing_params)
                    )
        else:
            for iters in [200, 300]:
                for depth in [3, 4, 5]:
                    model_params = {
                        "iterations": iters,
                        "depth": depth
                    }
                    experiments_to_run.append(
                        (mt, model_params, data_processing_params)
                    )

    experiments_to_run = experiments_to_run[:10]

    results = []
    for (model_type,
         model_params,
         data_processing_params) in experiments_to_run:
        run_id, val_metric = run_single_experiment(
            train_df, val_df,
            model_type, model_params,
            data_processing_params
        )
        results.append((run_id, model_type, val_metric))

    df_res = pd.DataFrame(results, columns=["run_id", "model_type", "val_logloss"])

    for mt in df_res["model_type"].unique():
        subset = df_res[df_res["model_type"] == mt].sort_values("val_logloss")
        top2 = subset.head(2)
        if len(top2) < 2:
            continue

        champion_run_id = top2.iloc[0]["run_id"]
        challenger_run_id = top2.iloc[1]["run_id"]

        reg_name = f"{EXP_NAME}-{mt}"

        try:
            client.get_registered_model(reg_name)
        except Exception:
            client.create_registered_model(reg_name)

        champion_model_uri = f"runs:/{champion_run_id}/model"
        mv_champion = client.create_model_version(
            name=reg_name,
            source=champion_model_uri,
            run_id=champion_run_id
        )
        # Присваиваем alias champion
        client.set_registered_model_alias(
            name=reg_name,
            alias="champion",
            version=mv_champion.version
        )

        challenger_model_uri = f"runs:/{challenger_run_id}/model"
        mv_challenger = client.create_model_version(
            name=reg_name,
            source=challenger_model_uri,
            run_id=challenger_run_id
        )
        # Присваиваем alias challenger
        client.set_registered_model_alias(
            name=reg_name,
            alias="challenger",
            version=mv_challenger.version
        )




if __name__ == "__main__":
    main()
