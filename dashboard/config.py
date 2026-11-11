"""File locations, expected schemas and the shared segment colour system."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


def _candidate_dirs():
    env = os.environ.get("SEGMENTATION_DATA_DIR")
    dirs = [Path(env)] if env else []
    return dirs + [APP_DIR / "data" / "processed", APP_DIR, APP_DIR.parent / "03_Datasets_and_Results"]


def data_dir() -> Path:
    """Folder holding the project CSVs: $SEGMENTATION_DATA_DIR, data/processed, the app folder, or ../03_Datasets_and_Results."""
    return next((d for d in _candidate_dirs() if (d / "Phase4_Customer_RFM.csv").exists()), _candidate_dirs()[0])


def verification_dir() -> Path:
    for d in [APP_DIR / "verification", APP_DIR.parent / "04_Verification", data_dir() / "verification"]:
        if d.exists():
            return d
    return APP_DIR / "verification"


# name -> (file name, required columns, required for the app to start)
FILES = {
    "rfm": ("Phase4_Customer_RFM.csv", ["CustomerID", "Recency", "Frequency", "Monetary"], True),
    "labels": ("Phase5_Customer_Clusters.csv", ["CustomerID", "Cluster", "Segment"], True),
    "lines": ("Online_Retail_Cleaned_Purchase_Lines.csv",
              ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country", "LineValue"], False),
    "profiles7": ("Phase7_KMeans_Segment_Profiles.csv", ["KMeans_Cluster", "Customers", "Suggested_Segment_Name"], False),
    "eval5": ("Phase5_KMeans_Evaluation.csv", ["K", "Inertia", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"], False),
    "audit": ("Phase3_Preprocessing_Audit.csv", ["Step", "Rows"], False),
    "inspection": ("Phase3_Raw_Data_Inspection.csv", ["Attribute", "dtype", "missing"], False),
    "reversed": ("Product_Reversed_Purchase_Lines.csv", ["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "InvoiceDate", "CustomerID"], False),
}

EXPECTED = {  # documented, independently verified figures (verification/independent_check.py)
    "raw_rows": 541909, "duplicates": 5268, "missing_customerid": 135037, "cancellations": 8872,
    "nonpos_quantity": 0, "nonpos_price": 40, "clean_lines": 392692, "customers": 4338,
    "reference_date": "2011-12-10",
}

NON_PRODUCT_CODES = ["POST", "DOT", "C2", "M", "BANK CHARGES"]

# Colours follow the descriptive segment name, never the cluster number.
SEGMENT_ORDER = ["Recent, Frequent, High-Spend Customers", "Occasional / Lower-Spend Customers",
                 "At-Risk / Inactive, Low-Frequency Customers"]
SEGMENT_SHORT = {"Recent, Frequent, High-Spend Customers": "High-Spend",
                 "Occasional / Lower-Spend Customers": "Occasional",
                 "At-Risk / Inactive, Low-Frequency Customers": "At-Risk"}
SEGMENT_COLORS = {"Recent, Frequent, High-Spend Customers": "#2F9E77",
                  "Occasional / Lower-Spend Customers": "#4C6FD6",
                  "At-Risk / Inactive, Low-Frequency Customers": "#D08A2E",
                  "Unlabelled": "#9AA3B5"}
EXTRA_COLORS = ["#8B7FE0", "#C9A227", "#C25E8A", "#2E9BB0"]

UNITS = {"Recency": "days since last purchase", "Frequency": "distinct invoices", "Monetary": "GBP, gross positive purchases"}
