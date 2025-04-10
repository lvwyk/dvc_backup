import mlflow
from pathlib import Path
 
from box import ConfigBox
from ruamel.yaml import YAML
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
config = ConfigBox(YAML(typ="safe").load(Path("params.yaml").open(encoding="utf-8")))
print(config)
 
mlflow.set_experiment('mlflow_dvc_integration')
def evaluate(model: RandomForestClassifier, X, y, split: str):
    y_pred = model.predict(X)
    mlflow.log_metric(f"{split}/accuracy", accuracy_score(y, y_pred))
    mlflow.log_metric(
        f"{split}/average_precision",
        average_precision_score(y, model.predict_proba(X), average="macro"),
    )
    mlflow.sklearn.log_model(sk_model=model, artifact_path="models",signature=infer_signature(X_test, y_pred), registered_model_name=f"sk-learn-random-forest-{config.model.max_depth}-{config.model.n_estimators}-{config.model.random_state}")
    mlflow.log_metric(f"{split}/recall", recall_score(y, y_pred, average="macro"))
 
 
X, y = datasets.load_iris(as_frame=True, return_X_y=True)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", config.model.n_estimators)
    mlflow.log_param("max_depth", config.model.max_depth)
    mlflow.log_param("random_state", config.model.random_state)
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        random_state=config.model.random_state,
    )
    model.fit(X_train, y_train)
    
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_test, y_test, "test")
 