# Income Prediction Project

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green?style=flat&logo=xgboost)](https://xgboost.readthedocs.io/en/stable/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-purple?style=flat&logo=lightgbm)](https://lightgbm.readthedocs.io/en/latest/)
[![Docker](https://img.shields.io/badge/Docker-24.0.6-blue?style=flat&logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat&logo=render)](https://render.com/)

This project focuses on building and deploying a machine learning model to predict whether an individual's income exceeds $50,000 annually based on various demographic and employment-related features. The solution includes data exploration, model training, API development with FastAPI, and a user-friendly interface with Streamlit.

## Features

*   **Data Exploration (EDA):** Comprehensive analysis of the Adult Income dataset to understand feature distributions, relationships, and identify patterns.
*   **Machine Learning Models:** Implementation and comparison of several classification algorithms, including Logistic Regression, Decision Tree, XGBoost, and LightGBM.
*   **Hyperparameter Tuning:** Optimized model performance using `RandomizedSearchCV` with `StratifiedKFold` cross-validation.
*   **FastAPI Backend:** A robust and scalable API for serving model predictions.
*   **Streamlit Frontend:** An interactive web application for users to input data and receive real-time income predictions.
*   **Dockerization:** Containerized the FastAPI and Streamlit applications for easy deployment and reproducibility.
*   **Cloud Deployment:** The FastAPI backend is deployed on Render, providing a live prediction service.

## Live Demo

The FastAPI backend is deployed and accessible at: `https://predict-income-latest.onrender.com`

The Streamlit UI is configured to connect to this deployed backend by default.

## Tech Stack

*   **Python:** Core programming language.
*   **Data Manipulation:** `pandas`, `numpy`.
*   **Data Visualization:** `matplotlib`, `seaborn`, `missingno`.
*   **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`.
*   **Web Framework (Backend):** `FastAPI`.
*   **Web Framework (Frontend):** `Streamlit`.
*   **Environment Management:** `uv`.
*   **Containerization:** `Docker`.
*   **Cloud Platform:** `Render`.

## 📂 Project Structure

```
.
├── data/
│   └── adult/          # Raw data files (adult.data, adult.test, adult.names)
├── notebooks/
│   ├── eda.ipynb       # Exploratory Data Analysis
│   ├── model_train.ipynb # Initial model training and comparison
│   └── Xgb_lgbm_tune.ipynb # Detailed tuning for XGBoost and LightGBM
├── src/
│   ├── main.py         # FastAPI application for model serving
│   ├── train.py        # Script for training and saving the final model
│   └── test.py         # Script for testing the FastAPI endpoint
├── ui/
│   └── app.py          # Streamlit web application
├── .gitignore          # Specifies intentionally untracked files to ignore
├── Dockerfile          # Dockerfile for containerizing the application
├── pyproject.toml      # Project dependencies and metadata (managed by uv)
├── README.md           # Project README file
├── adult_lgbm_metadata.pkl # Metadata for the LightGBM model
└── adult_lgbm_model.pkl    # Trained LightGBM model
```

## 📊 Dataset

The project utilizes the **Adult Income Dataset** from the UCI Machine Learning Repository.

*   **Source:** Extracted from the 1994 Census database.
*   **Task:** Binary classification to predict whether an individual's income exceeds $50,000 annually.
*   **Instances:** 48,842 instances (32,561 for training, 16,281 for testing).
*   **Features:**
    *   `age`: continuous.
    *   `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    *   `fnlwgt`: continuous (final weight - represents the number of people the census believes the entry represents).
    *   `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    *   `education-num`: continuous (numerical representation of education level).
    *   `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    *   `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    *   `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    *   `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    *   `sex`: Female, Male.
    *   `capital-gain`: continuous.
    *   `capital-loss`: continuous.
    *   `hours-per-week`: continuous.
    *   `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    *   `income`: >50K, <=50K (target variable).

## 🔍 Exploratory Data Analysis (EDA)

The `notebooks/eda.ipynb` notebook provides a detailed EDA, covering:

*   **Data Loading and Initial Inspection:** Loading of train and test datasets, handling missing values (represented as ' ?').
*   **Data Overview:** Examination of data shapes, data types, and summary statistics for numerical features.
*   **Target Variable Distribution:** Analysis of the `income` class distribution, revealing class imbalance.
*   **Feature Distributions:** Histograms for numerical features and count plots for categorical features, often segmented by income level to observe relationships (e.g., `Income Distribution by Education`, `Income Distribution by Occupation`).
*   **Correlation Analysis:** Heatmaps to visualize correlations between numerical features.
*   **Feature Importance:** Univariate analysis using Pearson correlation, F-value, Mutual Information, and categorical separation to rank features by their predictive power.

## 🧠 Model Training & Comparison

The `notebooks/model_train.ipynb` and `notebooks/Xgb_lgbm_tune.ipynb` notebooks detail the machine learning pipeline:

### Data Preprocessing

*   The `fnlwgt` column is dropped as it's not relevant for individual income prediction.
*   The target variable `income` is converted to numerical (`<=50K` as 0, `>50K` as 1).
*   A `ColumnTransformer` is used for robust preprocessing:
    *   **Numerical Features:** Imputed with the median and scaled using `StandardScaler`.
    *   **Categorical Features:** Imputed with the most frequent value and encoded using `OneHotEncoder`.

### Models Evaluated

The following models were trained and evaluated:

1.  **Logistic Regression:** A linear model for binary classification.
2.  **Decision Tree:** A non-linear, tree-based model.
3.  **XGBoost Classifier:** A gradient boosting framework known for its performance.
4.  **LightGBM Classifier:** Another high-performance gradient boosting framework, often faster than XGBoost.

### Hyperparameter Tuning

`RandomizedSearchCV` with `StratifiedKFold` (5 folds) was employed for efficient hyperparameter tuning across all models, optimizing for the `f1_score` due to the class imbalance in the target variable.

### Performance Metrics

Models were evaluated on a dedicated test set using:

*   **Accuracy:** Overall correctness of predictions.
*   **Precision:** Proportion of positive identifications that were actually correct.
*   **Recall:** Proportion of actual positives that were identified correctly.
*   **F1-Score:** Harmonic mean of precision and recall, balancing both metrics.
*   **ROC AUC:** Area Under the Receiver Operating Characteristic Curve, indicating the model's ability to distinguish between classes.
*   **PR AUC:** Area Under the Precision-Recall Curve, particularly useful for imbalanced datasets.

### Results Summary

| Model               | Accuracy | Precision | Recall   | F1       | ROC AUC  | PR AUC   |
| :------------------ | :------- | :-------- | :------- | :------- | :------- | :------- |
| Logistic Regression | 0.8085   | 0.5637    | 0.8388   | 0.6743   | 0.9046   | 0.7595   |
| Decision Tree       | 0.8028   | 0.5532    | 0.8593   | 0.6731   | 0.8993   | 0.7416   |
| XGBoost             | 0.8334   | 0.6039    | 0.8562   | 0.7082   | 0.9272   | 0.8250   |
| LightGBM            | 0.8361   | 0.6094    | 0.8531   | **0.7109** | 0.9279   | **0.8262** |
| LightGBM  (tuned)   | **0.8727**   | **0.7717**    | **0.6555**   | **0.7088**   | **0.9284**   | **0.8278**   |

**Conclusion:** The **tuned LightGBM model** demonstrated the best overall performance, achieving the highest F1-score and PR AUC, making it the chosen model for deployment.



## ⚙️ Setup and Usage

### Prerequisites

*   Python 3.9+
*   `uv` (for environment management)
*   Docker (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/income-prediction.git
cd income-prediction
```

### 2. Environment Setup

This project uses `uv` for dependency management.

```bash
uv venv
uv sync
```

### 3. Train the Model (Optional)

The pre-trained LightGBM model (`adult_lgbm_model.pkl`) and its metadata (`adult_lgbm_metadata.pkl`) are already included in the repository. If you wish to retrain the model or experiment with different configurations:

```bash
uv run src/train.py
```

This script will train the LightGBM model and save the `adult_lgbm_model.pkl` and `adult_lgbm_metadata.pkl` files in the project root.

### 4. Run the FastAPI Backend

The FastAPI application serves the trained model for predictions.

```bash
uvicorn src.main:app --host 0.0.0.0 --port 9696
```

The API will be accessible at `http://localhost:9696`.

#### API Endpoints

*   **`/`**: Root endpoint, returns a welcome message.
*   **`/ping`**: Health check endpoint.
*   **`/info`**: Returns model metadata (features, threshold).
*   **`/predict`**: Accepts a JSON payload with user features and returns an income prediction (probability and class).

### 5. Run the Streamlit Frontend

The Streamlit application provides a user interface to interact with the prediction service.

```bash
uv run streamlit run ui/app.py
```

The Streamlit app will open in your web browser, usually at `http://localhost:8501`.

**Note:** By default, the Streamlit app connects to the deployed FastAPI backend (`https://predict-income-latest.onrender.com`). If you are running the FastAPI backend locally, you can set the `API_BASE_URL` environment variable before running the Streamlit app:

```bash
# On Linux/macOS
export API_BASE_URL="http://localhost:9696"
uv run streamlit run ui/app.py

# On Windows (Command Prompt)
set API_BASE_URL="http://localhost:9696"
uv run streamlit run ui/app.py

# On Windows (PowerShell)
$env:API_BASE_URL="http://localhost:9696"
uv run streamlit run ui/app.py
```

### 6. Docker Deployment (Optional)

You can build and run the entire application using Docker.

#### Build Docker Image

```bash
docker build -t income-prediction .
```

#### Run Docker Container

```bash
docker run -p 9696:9696 -p 8501:8501 income-prediction
```

This will start both the FastAPI backend (on port 9696) and the Streamlit frontend (on port 8501) within the Docker container. You can then access them via `http://localhost:9696` and `http://localhost:8501` respectively.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.