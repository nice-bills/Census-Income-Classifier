# Adult Income Prediction

This project provides a machine learning-powered service to predict whether an individual's income exceeds $50,000 per year based on census data. It includes a REST API backend built with FastAPI and a simple web-based user interface built with Streamlit.

## Features

- **REST API**: A FastAPI backend serves the income prediction model.
- **Web UI**: A Streamlit frontend provides a user-friendly way to interact with the model.
- **Machine Learning Model**: Utilizes a pre-trained LightGBM (LGBM) model.
- **Containerized**: Includes a `Dockerfile` for easy deployment.
- **Modern Tooling**: Uses `uv` for fast dependency and environment management.

## Tech Stack

- **Backend**: `FastAPI`, `Uvicorn`, `LightGBM`, `Scikit-learn`, `Pandas`
- **Frontend**: `Streamlit`
- **Tooling**: `uv`, Docker, Git
- **Language**: Python 3.12

## Project Structure

```
.
├── Dockerfile
├── main.py                 # FastAPI application
├── train.py                # Script for model training
├── test.py                 # Script for testing the API
├── requirements.txt        # Python dependencies
├── adult_lgbm_model.pkl    # Pre-trained model
├── streamlit-ui/
│   └── app.py              # Streamlit UI application
└── ...
```

## Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create and Activate Virtual Environment

This project uses `uv` to manage the virtual environment and dependencies.

```bash
# Create the virtual environment
uv venv

# Activate the environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages using `uv`.

```bash
uv sync
```

## Usage

This project has two main components: the FastAPI backend and the Streamlit frontend. You need to run them in separate terminals.

### 1. Run the FastAPI Backend

The API server provides the prediction logic.

```bash
uvicorn main:app --host 0.0.0.0 --port 9696
```
The API will be available at `http://localhost:9696`. You can access the interactive API documentation at `http://localhost:9696/docs`.

### 2. Run the Streamlit Frontend

The UI allows you to interact with the API through a web interface.

```bash
uv run streamlit run streamlit-ui/app.py
```
The Streamlit app will be available at `http://localhost:8501`.

## API Endpoints

The following endpoints are available:

- **`POST /predict`**: Submits data to get an income prediction.
  - **Payload**: A JSON object with features like `age`, `workclass`, `education`, etc.
  - **Response**: A JSON object with the prediction (`<=50K` or `>50K`), the probability, and the threshold used.

- **`GET /health`**: A health check endpoint.
  - **Response**: `{"status": "ok"}`

- **`GET /info`**: Provides information about the loaded model.

## Docker

You can also build and run the application using Docker. The Docker container runs the FastAPI server.

### 1. Build the Docker Image

```bash
docker build -t adult-income-prediction .
```

### 2. Run the Docker Container

```bash
docker run -p 9696:9696 adult-income-prediction
```
The API will be accessible at `http://localhost:9696`.
