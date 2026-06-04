from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loguru import logger
import typer

from dm_classif.config import FIGURES_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "raw_diabetes_data.csv",
    target_col: str = "Diabetes_binary"
):
    logger.info("Generating custom plots...")

    df = pd.read_csv(input_path)
    df_yes = df[df[target_col] == 1]
    df_no = df[df[target_col] == 0]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Target variable distribution
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x=target_col, hue=target_col, palette='Set2')
    plt.title("Target Variable Distribution (Diabetes vs Non-Diabetes)")
    plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.savefig(FIGURES_DIR / "1_target_distribution.png")
    plt.close()

    # 2. Gender-wise distribution for diabetes and non-diabetes
    gender_dist = df.groupby(['Sex', 'Diabetes_binary']).size().unstack(fill_value=0)
    # convert to percentage (0-100) instead of fraction (0-1)
    gender_percent = gender_dist.div(gender_dist.sum(axis=1), axis=0) * 100
    labels = ['Female', 'Male']
    no_diabetes = gender_percent[0].values  
    yes_diabetes = gender_percent[1].values  
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(labels, no_diabetes, label='No Diabetes', color='skyblue')
    ax.bar(labels, yes_diabetes, bottom=no_diabetes, label='Diabetes', color='salmon')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Diabetes Status by Gender (Percentage)')
    ax.legend(title='Diabetes Status')

    for i in range(len(labels)):
        # show one decimal place for clarity (e.g. 52.3%)
        ax.text(i, no_diabetes[i] / 2, f'{no_diabetes[i]:.1f}%', ha='center', va='center', color='black')
        ax.text(i, no_diabetes[i] + yes_diabetes[i] / 2, f'{yes_diabetes[i]:.1f}%', ha='center', va='center', color='black')

    # ensure y axis spans 0-100 for percentage plots
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2_gender_distribution.png")
    plt.close()

    # 3. Distribution of diabetes across age groups
    age_diabetes_counts = (
        df.groupby(['Age', 'Diabetes_binary']).size()
        .unstack(fill_value=0)
        .rename(columns={0: 'No Diabetes', 1: 'Diabetes'})
    )
    age_diabetes_percent = age_diabetes_counts.div(age_diabetes_counts.sum(axis=1), axis=0) * 100
    ax= age_diabetes_percent.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6),
        color=['skyblue', 'orange']
    )
    age_labels = [
        '18-24', '25-29', '30-34', '35-39', '40-44',
        '45-49', '50-54', '55-59', '60-64',
        '65-69', '70-74', '75-79', '>80'
    ]
    plt.title('Age Distribution by Diabetes Status (Percentage Stacked)')
    plt.xlabel('Age Group')
    plt.ylabel('Percentage')
    ax.set_xticklabels(age_labels, rotation=45)
    plt.legend(title='Diabetes Status')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_age_distribution_diabetics.png")
    plt.close()

    # 4. BMI distribution for diabetics
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_yes, x='BMI', bins=30, kde=True)
    plt.title("BMI Distribution for Diabetics")
    plt.xlim(15, 60)
    plt.savefig(FIGURES_DIR / "4_bmi_distribution_diabetics.png")
    plt.close()

    # 5. BMI comparison for diabetic and non-diabetic
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=df, x='Diabetes_binary', hue='Diabetes_binary', y='BMI', palette='Set3')
    ax.set_title("BMI Distribution: No-Diabetes vs Diabetics")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Diabetes', 'Diabetic'])
    plt.ylim(15, 60)
    plt.savefig(FIGURES_DIR / "5_bmi_comparison.png")
    plt.close()

    # 6. Comparison plots for related features
    comparison_cols = ['HighChol', 'HighBP', 'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 'DiffWalk']
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))
    axs = axs.flatten()

    for i, col in enumerate(comparison_cols):
        ax = axs[i]
        ctab = pd.crosstab(df['Diabetes_binary'], df[col], normalize='index') * 100
        ctab = ctab[[0, 1]] if 0 in ctab.columns and 1 in ctab.columns else ctab
        bottom = None
        for label in ctab.columns:
            values = ctab[label]
            ax.bar(['No Diabetes', 'Diabetes'], values, bottom=bottom, label=str(label), alpha=0.8)
            bottom = values if bottom is None else bottom + values

        ax.set_title(f"Diabetes Status by {col} (Percentage)")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
        ax.legend(title=col)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "6_diabetes_comparisons.png")
    plt.close()

    logger.success(f"All plots generated and saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    app()
