"""Download the IBM Telco Customer Churn dataset into data/."""
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
TARGET = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"

if __name__ == "__main__":
    TARGET.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, TARGET)
    print(f"Saved {TARGET}")
