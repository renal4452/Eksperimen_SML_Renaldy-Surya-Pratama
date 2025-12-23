# =========================
# IMPORT ZONE (dipisah rapi)
# =========================
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# =========================
# CONFIG
# =========================
RAW_DATA_PATH = "bank_raw/bank_raw.csv"
OUTPUT_DIR = "preprocessing/bank_preprocessing"
TARGET_COL = "deposit"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================
# HELPER FUNCTION
# =========================
def ensure_dir(path: str):
    """Ensure output directory exists"""
    os.makedirs(path, exist_ok=True)


def to_dense(x):
    """Convert sparse matrix to dense if needed"""
    return x.toarray() if hasattr(x, "toarray") else x


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("START AUTOMATED PREPROCESSING")

    # Load data
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Split feature & target
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Train-test split (stratified)
    print("Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Define feature groups
    numerical_features = [
        "age", "balance", "day", "duration",
        "campaign", "pdays", "previous"
    ]

    categorical_features = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]

    # Build preprocessor
    print("Building preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit & transform
    print("Applying preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Feature names
    feature_names = (
        preprocessor.named_transformers_["num"]
        .get_feature_names_out(numerical_features)
        .tolist()
        +
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )

    # Convert to DataFrame
    X_train_df = pd.DataFrame(
        to_dense(X_train_processed),
        columns=feature_names
    )

    X_test_df = pd.DataFrame(
        to_dense(X_test_processed),
        columns=feature_names
    )

    # Save outputs
    ensure_dir(OUTPUT_DIR)

    print("Saving preprocessed datasets...")
    X_train_df.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test_df.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    print("PREPROCESSING DONE SUCCESSFULLY")
    print(f"Output saved in: {OUTPUT_DIR}/")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()

