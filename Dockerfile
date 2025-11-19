FROM python:3.12.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts to the root of the app directory
COPY src/main.py adult_lgbm_model.pkl adult_lgbm_metadata.pkl ./


EXPOSE 9696

# Update the command to run the app from the src module
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9696"]