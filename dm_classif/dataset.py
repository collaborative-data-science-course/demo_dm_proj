from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "raw_diabetes_data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    logger.info(f"Reading dataset from {input_path}")
    
    # Read dataset
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        return

    logger.info(f"Dataset shape: {df.shape}")

    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save to raw folder
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.success(f"Saved train data to {train_path}")
    logger.success(f"Saved test data to {test_path}")


if __name__ == "__main__":
    app()
