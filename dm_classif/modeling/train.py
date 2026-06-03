from pathlib import Path
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score

from loguru import logger
import typer

from dm_classif.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR

app = typer.Typer()

def get_all_models():
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear"),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"]
            }
        ),
        "sgdclassifier": (
            SGDClassifier(random_state=42),
            {
                'loss': ['hinge', 'log_loss', 'modified_huber'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['optimal', 'adaptive', 'invscaling'],
                'eta0': [0.01, 0.1],
                'max_iter': [1000],
            }
        ),
        "linearsvc": (
            LinearSVC(random_state=42, dual=False),
            {
                'C': [0.01, 0.1, 1, 10],           
                'penalty': ['l1','l2'],                
                'loss': ['squared_hinge'],
                'tol': [1e-4, 1e-3, 1e-2],
                'max_iter': [15000, 20000]         
            }
        ),
        "gradientboostingclassifier": (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5],
                'subsample': [1.0, 0.8]
            }
        ),
        "votingclassifier": (
            VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(max_depth=20, min_samples_leaf=6, min_samples_split=15, n_estimators=30, random_state=42)),
                ('mlp', MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(50,), learning_rate='constant', max_iter=1500, solver='adam', random_state=42)),
                ('gb', GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0, random_state=42)),
                ('ada', AdaBoostClassifier(learning_rate=1.5, n_estimators=100, random_state=42))
            ]),
            {
                "voting": ["hard", "soft"]
            }
        ),
    }

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "train_processed_pipeline.csv",
    model_output: Path = MODELS_DIR / "best_model.pkl",
    summary_output: Path = REPORTS_DIR / "model_summaries.md",
    target_col: str = "Diabetes_binary",
    metric: str = "recall"
):
    logger.info("Loading processed training data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    best_score = 0
    best_model = None
    best_model_name = None
    results = []

    for model_name, (model, param_grid) in get_all_models().items():
        logger.info(f"Training {model_name} with scoring='{metric}'...")
        try:
            grid = GridSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1)
            grid.fit(X, y)
        except Exception as e:
            logger.warning(f"{model_name} failed: {e}")
            continue

        score = grid.best_score_
        logger.info(f"{model_name} best {metric}: {score:.4f}")

        results.append({
            "model": model_name,
            "best_params": grid.best_params_,
            "best_score": score
        })

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_model_name = model_name

    # Generate classification report on training data
    y_pred = best_model.predict(X)
    report = classification_report(y, y_pred)

    # Build Markdown summary
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Model Training Summary\n"]

    # Model comparison table
    lines.append("## Overall Model Comparison")
    lines.append(f"| Model | {metric.title()} |")
    lines.append(f"|-------|----------------|")
    for result in results:
        lines.append(f"| {result['model']} | {result['best_score']:.4f} |")
    lines.append("")

    # Best parameters section
    for result in results:
        lines.append(f"## {result['model'].replace('_', ' ').title()}")
        lines.append(f"- **Best {metric.title()}**: `{result['best_score']:.4f}`")
        lines.append(f"- **Best Parameters**:\n```python\n{result['best_params']}\n```\n")

    # Best model section with classification report
    lines.append(f"---\n\n### Best Model: `{best_model_name}` with **{metric}** = `{best_score:.4f}`\n")
    lines.append("#### Classification Report on entire test set")
    lines.append("```")
    lines.append(report)
    lines.append("```")

    # Save to Markdown
    with open(summary_output, "w") as f:
        f.write("\n".join(lines))

    logger.success(f"Saved model summary (with classification report) to {summary_output}")

    # Save best model separately as pickle
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as f:
        pickle.dump(best_model, f)
    logger.success(f"Saved best model to {model_output}")

if __name__ == "__main__":
    app()