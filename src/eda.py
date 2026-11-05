import pandas as pd

def load_customer_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def generate_eda_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=["number"])
    return {
        "total_records": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": numeric_df.describe().round(2).to_dict()
    }
