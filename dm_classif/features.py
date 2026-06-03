from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger
import typer

from dm_classif.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Removed {before - after} duplicate rows.")
    return df


@app.command()
def main():
    logger.info("Loading raw train and test data...")
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Remove duplicates
    train_df = remove_duplicates(train_df)
    test_df = remove_duplicates(test_df)

    # Target and Features
    target_col = "Diabetes_binary"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Step 1: Feature selection using SelectKBest
    logger.info("Applying SelectKBest to select top features...")
    selector = SelectKBest(score_func=mutual_info_classif, k=12)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Step 2: Identify non-binary columns from selected features
    selected_columns = selector.get_support(indices=True)
    important_features = X_train.columns[selected_columns].tolist()

    X_train_selected = pd.DataFrame(X_train_selected, columns=important_features, index=X_train.index)
    X_test_selected = pd.DataFrame(X_test_selected, columns=important_features, index=X_test.index)


    binary_cols = [col for col in important_features if train_df[col].nunique() == 2]
    scale_cols = [col for col in important_features if col not in binary_cols]

    logger.info(f"Top selected Features: {important_features}")

    # Step 3: Apply scaling
    logger.info("Scaling selected features...")
    scaler = RobustScaler()
    X_train_scaled = X_train_selected.copy()
    X_test_scaled = X_test_selected.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])


    # Step 4: Re-attach target column
    train_processed = pd.concat([X_train_scaled, y_train], axis=1)
    test_processed = pd.concat([X_test_scaled, y_test], axis=1)

    # Step 5: Save output
    train_out_path = PROCESSED_DATA_DIR / "train_processed_pipeline.csv"
    test_out_path = PROCESSED_DATA_DIR / "test_processed_pipeline.csv"

    train_processed.to_csv(train_out_path, index=False)
    test_processed.to_csv(test_out_path, index=False)

    logger.success(f"Saved processed train data to: {train_out_path}")
    logger.success(f"Saved processed test data to: {test_out_path}")

if __name__ == "__main__":
    app()
