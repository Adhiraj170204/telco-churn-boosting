"""Load, clean and feature-engineer the Telco churn data."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]


def load_clean():
    df = pd.read_csv(DATA)
    df = df.drop(columns="customerID")
    # 11 customers have a blank TotalCharges: all have tenure 0 (brand new), so they have paid nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Engineered features
    df["n_addon_services"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"].where(df["tenure"] > 0)).fillna(df["MonthlyCharges"])
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]  # > 0: paying more now than on average
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    return df


def split_columns(df):
    y = df["Churn"]
    X = df.drop(columns="Churn")
    categorical = X.select_dtypes("object").columns.tolist()
    numeric = X.select_dtypes("number").columns.tolist()
    return X, y, numeric, categorical
