# Diabetes Risk Prediction🩺

This project demonstrates how a team can collaboratively work on a machine learning project using industry best practices, version control with Git, and the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure.

The goal is to predict whether an individual is diabetic or not based on health-related features.

---

## Project Structure

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

We follow the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) standard:


```
├── LICENSE            
├── Makefile           
├── README.md          
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- The leaning modules will be inside this folder, any documentation related to the project are stored here.    
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, description and version all with a short `_` delimiter e.g.
│                         `01_jqp_initial_data_exploration_v0.1.ipynb`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         diabetes_prediction and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as MarkDown, HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
|
|── scripts            <- Folder for Additional scripts (optional)
│
└── dm_classif         <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes diabetes_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

---


## Dataset

We used the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), which includes:

Each column and their encodings:

|Column Name|Encoding|
|-----------|--------|
|1. **Diabetes_binary**| 0 = no diabetes, 1 = prediabetes/diabetes|
|2. **HighBP**| 0 = no high BP, 1 = high BP|
|3. **HighChol**| 0 = no high cholesterol, 1 = high cholesterol|
|4. **CholCheck**| 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years|
|5. **BMI**| Body Mass Index|
|6. **Smoker**| Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no, 1 = yes|
|7. **Stroke**| (Ever told) you had a stroke. 0 = no, 1 = yes|
|8. **HeartDiseaseorAttack**| coronary heart disease (CHD) or myocardial infarction (MI) 0 = no, 1 = yes|
|9. **PhysActivity**| physical activity in past 30 days - not including job 0 = no, 1 = yes|
|10. **Fruits**| Consume Fruit 1 or more times per day 0 = no, 1 = yes|
|11. **Veggies**| Consume Vegetables 1 or more times per day 0 = no, 1 = yes|
|12. **HvyAlcoholConsump**| (adult men >=14 drinks per week and adult women>=7 drinks per week) 0 = no, 1 = yes|
|13. **AnyHealthcare**| Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no, 1 = yes|
|14. **NoDocbcCost**| Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no, 1 = yes|
|15. **GenHlth**| Would you say that in general your health is: scale 1-5, 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor|
|16. **MentHlth**| days of poor mental health scale 1-30 days|
|17. **PhysHlth**| physical illness or injury days in past 30 days scale 1-30|
|18. **DiffWalk**| Do you have serious difficulty walking or climbing stairs? 0 = no, 1 = yes|
|19. **Sex**| 0 = female, 1 = male|
|20. **Age**| 13-level age category (_AGEG5YR see codebook) 1 = 18-24, 9 = 60-64, 13 = 80 or older|
|21. **Education**| Education level (EDUCA see codebook) scale 1-6, 1 = Never attended school or only kindergarten, 2 = elementary etc.|
|22. **Income**| Income scale (INCOME2 see codebook) scale 1-8, 1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more|

---

## Team Members & Responsibilities

| Name    | Responsibilities |
|---------|------------------|
| **Bob (Team Lead)**   | Data Inspection (EDA), Preprocessing, Baseline Model, Logistic Regression, Decision Tree, Data Ingestion|
| **Ricky** | Univariate Analysis (EDA), Random Forest, XGB, KNN, Training code for Pipeline |
| **Sarah** | Hypothesis Testing (EDA), SGDClassifier, Perceptron, SVC, Feature Selection and Transformation |
| **Amy**   | Multivariate Analysis (EDA), MLPClassifier, AdaBoost, Voting, Stacking, Feature Selection and Transformation Pipeline |
| **All Members**  | GitHub workflows, Makefile automation, Documentation, Report |

---

## Phase-wise Responsibilities

### 1️. Exploratory Data Analysis
- **Bob and Ricky**: Built code for data-ingestion.
- **Bob**: Performed Data Inspection, outlier detection and finding anomalies.
- **Ricky**: Performed Univariate Analysis.
- **Amy**: Performed Bi-variate and Multivariate Analysis.
- **Sarah**: Performed Hypothesis Testing.

### 2️. Data Cleaning and Feature Engineering
- **Amy**: Removed Duplicated and Handled outliers.
- **Bob**: Performed Feature Selection and Scaling.

### 3️. Model Development and Evaluation
- **Bob**: Tried DummyClassifier, LogisticRegression and DecisionTree.
- **Ricky**: Tried RandomForestClassifier, XGBClassifier, KNN.
- **Sarah**: Tried SVC, perceptron, SGDClassifiers.
- **Amy**: Tried MLPClassifier and Ensemble models.

### 4️. Creating End to End Pipeline
- **Sarah**: Built code for pipeline data transformation and feature selection.
- **Ricky**: Built pipeline code for model training.
- **Amy**: Built code for predicting and evaluating models.

### 5. Documentation
- **All**: Created Projected related documentation, such as this.

---

##  Setup Instructions

After cloning the project to your local machine, Do the following to replicate the work.

1. Navigate to `demo_dm_proj/` 

```bash
cd demo_dm_proj
```

2. Create Virtual Environment:

```bash
python3 -m venv .venv
```

3. Activate the virtual environment:

    - On Windows:
        ```
        .venv\Scripts\activate
        ``` 
    
    - on Linux

        ```bash
        source .venv/bin/activate
        ```

4. Install the required Package:

```bash
pip install -r requirements.txt
```

Once successfully installed all requirements, you can run the project using various `make` commands.

---

## How to Run the project:
(If you close the project and come back again, always activate the virtual environment:

1. Navigate to the folder `demo_dm_proj`

2. Activate the virtual environment:

    - On Windows:
        ```
        .venv\Scripts\activate
        ``` 
    
    - on Linux

        ```bash
        source .venv/bin/activate
        ```


Then you can continue..
)

To streamline development and collaboration, this project uses a Makefile to automate common tasks. Below are the available commands and what each one does:

1. `Make dataset`: Ingests the dataset from the raw folder, splits it into training and testing sets, and stores them back in `data/raw/`.

```bash
make dataset
```

2. `make preprocess`: Runs the feature selection and scaling pipeline. It performs:

- Duplicate removal
- Feature selection using SelectKBest
- Scaling using StandardScaler

Outputs are stored as `train_processed_pipeline.csv` and `test_processed_pipeline.csv` in `data/processed/`.

```bash
make preprocess
```

3. We can train models and obtain the best model based on 3 different metrics : `accuracy`, `precision` and `recall`. The below commands will do the same.

    All the command does the following:
    - Compares model performances
    - Logs a summary of all models
    - Saves the best model as a `.pkl` file in `models/`

        - `make train-accuracy`: Trains multiple machine learning models with hyperparameter tuning using GridSearchCV, and finds the best model using evalauation metric as `accuracy`.

        ```bash
        make train-accuracy
        ```

        - `make train-precision`: Trains multiple machine learning models with hyperparameter tuning using GridSearchCV, and finds the best model using evalauation metric as `precision`.

        ```bash
        make train-precision
        ```

        - `make train-recall`: Trains multiple machine learning models with hyperparameter tuning using GridSearchCV, and finds the best model using evalauation metric as `recall`.

        ```bash
        make train-recall
        ```

4. `make predict`: Loads the saved best model and evaluates it on the test set, logging:

- Accuracy
- Classification report

Results are saved in `reports/test_evaluation.txt`.

```bash
make predict
```

5. We can run the entire pipeline in one go using the below commands: 
    The Following commands will run: 

    - Data ingestion
    - Feature engineering
    - Model training

        - `make full-accuracy`: Performs Data-ingestion, Feature engineering and Trains multiple machine learning models with evalauation metric as `accuracy`.

        ```bash
        make full-accuracy
        ```

        - `make full-precision`: Performs Data-ingestion, Feature engineering and Trains multiple machine learning models with evalauation metric as `precision`.

        ```bash
        make full-precision
        ```

        - `make full-recall`: Performs Data-ingestion, Feature engineering and Trains multiple machine learning models with evalauation metric as `recall`.

        ```bash
        make full-recall
        ```


6. `make clean-artifacts`: Removes all generated files, including:

- Processed datasets (`train_processed_pipeline.csv`, `test_processed_pipeline.csv`)
- Predictions
- Pickled model files
- Report files (but retains folder structure and .gitkeep files)

```bash
make clean-artifacts
```

7. `make plots`: Generates visualizations from the raw data, including:

- Distribution of target variable
- BMI comparisons
- Age and gender distribution
- Correlation heatmaps

Outputs are stored in `reports/figures/`.

```bash
make plots
```
---
## Learning Path

1. Navigate to docs folder: 

    ```bash
    cd docs/
    ```

2. start serving:

    ```bash
    mkdocs serve
    ```

Visit `http://127.0.0.1:8000` to access the full educational guide.


or


Navigate to the `docs/` folder and start learning right away.

---