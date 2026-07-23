"""
Skrip pelatihan model untuk dijalankan melalui MLflow Project (CI).

Dipanggil oleh workflow CI (`mlflow run MLProject`) setiap kali trigger
terpantik, sehingga proses re-training model berjalan otomatis.
"""

import argparse

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


class ChurnCIRetrainer:
    """Retraining model churn untuk dipanggil dari MLflow Project (CI)."""

    target_column = "Churn"

    def __init__(self, train_data: str, test_data: str, n_estimators: int, max_depth: int):
        self.train_data = train_data
        self.test_data = test_data
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _load_data(self):
        train_df = pd.read_csv(self.train_data)
        test_df = pd.read_csv(self.test_data)
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]
        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]
        return X_train, X_test, y_train, y_test

    def run(self):
        X_train, X_test, y_train, y_test = self._load_data()

        mlflow.sklearn.autolog()

        with mlflow.start_run(run_name="ci_retrain_random_forest"):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="telco_churn_preprocessing/train.csv")
    parser.add_argument("--test_data", type=str, default="telco_churn_preprocessing/test.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrainer = ChurnCIRetrainer(args.train_data, args.test_data, args.n_estimators, args.max_depth)
    retrainer.run()
