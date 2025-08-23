from pathlib import Path

import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():

    Path("models").mkdir(parents=True, exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])

    with mlflow.start_run(run_name="baseline-logreg"):
        mlflow.log_params({"model": "logreg", "max_iter": 1000})
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        f1 = f1_score(yte, preds)
        mlflow.log_metric("f1", f1)
        joblib.dump(pipe, "models/baseline_logreg.joblib")
        mlflow.log_artifact("models/baseline_logreg.joblib")
        print("F1=", f1)


if __name__ == "__main__":
    main()
