import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

X_train = pd.read_csv("hf://datasets/amitmzn/tourism-dataset/Xtrain.csv")
X_test = pd.read_csv("hf://datasets/amitmzn/tourism-dataset/Xtest.csv")
y_train = pd.read_csv("hf://datasets/amitmzn/tourism-dataset/ytrain.csv")
y_test = pd.read_csv("hf://datasets/amitmzn/tourism-dataset/ytest.csv")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

with mlflow.start_run():
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train.values.ravel())
    
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_f1", results['mean_test_score'][i])
    
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    })
    
    model_path = "best_tourism_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    
    repo_id = "amitmzn/tourism-model"
    repo_type = "model"
    
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")
    
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type
    )
    print("Model uploaded to HF!")
