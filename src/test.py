import requests
import sys

URL = "http://localhost:9696/predict"

# Example record must follow the AdultRecord schema in main.py
record = {
    "age": 37,
    "workclass": "Private",
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

def main():
    try:
        resp = requests.post(URL, json=record, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("Request failed:", e)
        sys.exit(1)

    data = resp.json()
    prob = data.get("probability_over_50k")
    pred = data.get("prediction")
    thresh = data.get("threshold")

    print(f"probability_over_50k: {prob:.4f}")
    print(f"prediction: {pred} (threshold={thresh})")

if __name__ == "__main__":
    main()