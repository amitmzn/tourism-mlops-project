import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

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

# FIXED: Removed deprecated use_label_encoder parameter
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train.values.ravel())

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

model_path = "best_tourism_model.joblib"
joblib.dump(best_model, model_path)

repo_id = "amitmzn/tourism-model"
try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(path_or_fileobj=model_path, path_in_repo="best_tourism_model.joblib", repo_id=repo_id, repo_type="model")
print("Model uploaded to HF!")
