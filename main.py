import joblib
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import pandas as pd
import os

# ===== Pydantic Models =====
class AdultRecord(BaseModel):
    age: int = Field(..., ge=18, le=100)
    workclass: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ]
    education: Literal[
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", 
        "11th", "12th", "HS-grad", "Some-college", "Assoc-acdm", 
        "Assoc-voc", "Bachelors", "Masters", "Prof-school", "Doctorate"
    ]
    education_num: int = Field(..., ge=1, le=16)
    marital_status: Literal[
        "Married-civ-spouse", "Divorced", "Never-married", 
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ]
    occupation: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales", 
        "Exec-managerial", "Prof-specialty", "Protective-serv", 
        "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", 
        "Farming-fishing", "Armed-Forces", "Priv-house-serv", "Adm-clerical"
    ]
    relationship: Literal["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Male", "Female"]
    capital_gain: int = Field(..., ge=0)
    capital_loss: int = Field(..., ge=0)
    hours_per_week: int = Field(..., ge=0, le=168)
    native_country: Literal[
       'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
       'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
       'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
       'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
       'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
       'China', 'Japan', 'Yugoslavia', 'Peru',
       'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
       'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
       'Holand-Netherlands'
    ]


class PredictResponse(BaseModel):
    probability_over_50k: float
    prediction: Literal["<=50K", ">50K"]
    threshold: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as model_in:
        model = joblib.load(model_in)

    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as meta_in:
            metadata = joblib.load(meta_in)

    app.state.model = model
    app.state.metadata = metadata
    app.state.threshold = float(metadata.get("threshold", 0.5548))
    app.state.feature_names = metadata.get("feature_names", [])
    app.state.cat_cols = metadata.get("cat_cols", [])

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Threshold: {app.state.threshold:.4f}")

    try:
        yield  
    finally:
        pass

app = FastAPI(
    title="Adult Income Prediction",
    description="Predicts whether income is >50K based on demographic/employment features",
    version="1.0.0", 
    lifespan= lifespan
)

MODEL_PATH = "adult_lgbm_model.pkl"
META_PATH = "adult_lgbm_metadata.pkl"

def predict_single(record_dict: dict) -> float:
    """
    Convert record to DataFrame, enforce feature order/dtypes and return probability >50K.
    """
    model = app.state.model
    feature_names = app.state.feature_names or list(record_dict.keys())
    cat_cols = app.state.cat_cols or []

    # ensure all required features are present
    missing = [f for f in feature_names if f not in record_dict]
    if missing:
        raise ValueError(f"Missing features in input: {missing}")

    df = pd.DataFrame([record_dict])[feature_names]

    # cast categorical columns if any
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    proba = model.predict_proba(df)[:, 1][0]
    return float(proba)


@app.post("/predict", response_model=PredictResponse)
def predict(record: AdultRecord) -> PredictResponse:
    prob = predict_single(record.model_dump())
    thresh = app.state.threshold
    return PredictResponse(
        probability_over_50k=prob,
        prediction=">50K" if prob >= thresh else "<=50K",
        threshold=thresh
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "adult-income", "threshold": app.state.threshold}


@app.get("/info")
def model_info():
    return {
        "model": "LightGBM",
        "threshold": app.state.threshold,
        "features": app.state.feature_names,
        "categorical_features": app.state.cat_cols,
        "version": "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)