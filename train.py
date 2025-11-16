import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# paths
train_path = "adult/adult.data"
test_path = "adult/adult.test"

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]

# load
df_train = pd.read_csv(train_path, names=column_names, na_values=" ?", skipinitialspace=True)
df_test = pd.read_csv(test_path, names=column_names, na_values=" ?", skipinitialspace=True, skiprows=1)

# drop sampling weight column
df_train.drop(columns=["fnlwgt"], inplace=True)
df_test.drop(columns=["fnlwgt"], inplace=True)

# fix target format
df_test["income"] = df_test["income"].str.rstrip(".")
target_map = {"<=50K": 0, ">50K": 1}
df_train["income"] = df_train["income"].map(target_map)
df_test["income"] = df_test["income"].map(target_map)

# train / test split
X_train = df_train.drop("income", axis=1)
y_train = df_train["income"]
X_test = df_test.drop("income", axis=1)
y_test = df_test["income"]

# detect categorical columns and cast for LightGBM
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c] = X_test[c].astype("category")

def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }

# known best params (from your search)
best_params = {
    "subsample": 1.0,
    "num_leaves": 20,
    "n_estimators": 500,
    "min_child_samples": 20,
    "max_depth": -1,
    "learning_rate": 0.02517241379310345,
    "colsample_bytree": 0.6,
}

model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    **best_params,
)

# train on full training set
model.fit(X_train, y_train)

# get probabilities and apply optimized threshold
y_proba = model.predict_proba(X_test)[:, 1]
OPTIMAL_THRESHOLD = 0.386163
y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

# evaluate on test set with optimized threshold
metrics = evaluate(y_test, y_pred, y_proba)

print("="*70)
print("Trained LightGBM on full training set")
print("="*70)
print("\nHyperparameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\nPrediction threshold: {OPTIMAL_THRESHOLD} (optimized for accuracy)")
print("\nTest metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

results_df = pd.DataFrame({"model": ["LightGBM (threshold=0.386163)"], **{k: [v] for k, v in metrics.items()}})
print("\nResults dataframe:")
print(results_df)

metadata = {
    "threshold": OPTIMAL_THRESHOLD,
    "feature_names": X_train.columns.tolist(),
    "cat_cols": cat_cols,
    "params": best_params,
}

with open("adult_lgbm_model.pkl", "wb") as model_out:
    joblib.dump(model, model_out)

with open("adult_lgbm_metadata.pkl", "wb") as meta_out:
    joblib.dump(metadata, meta_out)

print("\nModel saved to adult_lgbm_model.pkl")
print("Metadata saved to adult_lgbm_metadata.pkl")