from pathlib import Path
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
)

from loguru import logger
import typer

from dm_classif.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test_processed_pipeline.csv",
    model_path: Path = MODELS_DIR / "best_model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    target_col: str = "Diabetes_binary"
):
    logger.info("Loading processed test data and best model...")
    df = pd.read_csv(test_path)
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Running inference...")
    preds = model.predict(X_test)

    # Save predictions
    pred_df = X_test.copy()
    pred_df["actual"] = y_test
    pred_df["predicted"] = preds
    pred_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")

    # Evaluation metrics
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, pos_label=1)
    precision = precision_score(y_test, preds, pos_label=1)
    report_str = classification_report(y_test, preds)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    conf_mat = confusion_matrix(y_test, preds, normalize='true')

    # Create figures directory
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save confusion matrix (%)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        conf_mat,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (%)")
    conf_matrix_path = figures_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()
    logger.success(f"Saved confusion matrix plot to {conf_matrix_path}")

    # Create markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "test_report.md"
    with open(summary_path, "w") as f:
        f.write("# Model Test Evaluation\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n" + report_str + "\n```\n")
        f.write(f"\n**Accuracy**: `{accuracy:.4f}`\n")
        f.write(f"\n**Recall**: `{recall:.4f}`\n")
        f.write(f"\n**Precision**: `{precision:.4f}`\n")
        f.write("\n## Confusion Matrix (Percentage)\n\n")
        f.write("![Confusion Matrix](figures/confusion_matrix.png)\n")

    logger.success(f"Evaluation report saved to {summary_path}")
    logger.success("Inference and reporting complete!")

if __name__ == "__main__":
    app()