# Startup Funding Stage Prediction & Market Analysis (Bay Area Focus)

## 1. Summary

This repository contains the codebase for a comprehensive project focused on
**startups in the San Francisco Bay Area**.
The initial ambition was to predict startup success and failure. However, this
proved challenging due to the scarcity of publicly available, detailed
financial data crucial for such predictions. While funding information was
accessible, comprehensive financial statements (revenue, profit margins, burn
rate, etc.) for early-stage private companies are typically not public.

We then explored an alternative approach: developing five distinct predictive
models (covering different facets of startup data) and using their outputs as
weighted inputs for a meta-model to predict success/failure. This multi-model
strategy, while innovative, introduced significant complexity in integration,
weighting, and interpretation.

Recognizing these hurdles and the robustness of the available funding-related
data, the project pivoted to a more feasible yet highly valuable goal:
**predicting startup funding stages** (e.g., Seed, Series A, Series B). This
prediction offers tangible insights for entrepreneurs, investors, and market
analysts in the Bay Area. The current system leverages machine learning to
predict these stages based on funding amounts, employee counts, industry, and
other key features. Beyond stage prediction, the project also analyzes market
trends specific to this region, detects data anomalies, and provides insights
through an interactive web application.

## 2. Website

**Link to the deployed web application:** https://oval-sunset-450610-h4.uc.r.appspot.com


## 3. Directory Structure

Below is an overview of the main project directory (`cs163-main`) and related
top-level folders:

```
.
├── cs163-main/
│   ├── backend/
│   │   ├── AdataCollection/        # Scripts for scraping Bay Area startup data
│   │   │   ├── growthlistscrapper.py
│   │   │   ├── fundraiserstartup50.py
│   │   │   ├── topstartupio50.py
│   │   │   └── JSONFolder/         # Output for raw scraped JSON/CSV data
│   │   ├── ML/
│   │   │   └── funding_stage_predictionORIGINAL.py # Core ML pipeline script (We moved this ML folder in appengine folder due to issues occuring)
│   │   └── MainOutput/             # Default output for ML pipeline (models, summaries, viz)
│   │
│   ├── appengine/                  # Frontend and Google App Engine deployment configuration
│   │   ├── appPredictionEngine.py  # Core Dash web application logic (or app163.py)
│   │   ├── app163.py               # frontend Python file
│   │   ├── assets/                 # Static files for the Dash app (CSS, images, plots)
│   │   └── app.yaml                # App Engine configuration file
│   │
│   ├── .ipynb_checkpoints/         # Auto-saves & checkpoints for Jupyter Notebooks (planning, exploration)
│   │   ├── MachineLearningPlan.ipynb # Notebook for initial ML planning
│   │   ├── MachineLearningPlan.pdf   # PDF version of the ML plan
│   │   └── ... (other checkpoint files for notebooks like OutputOf5MainpyFiles.ipynb, VisualizationResults.ipynb)
│   │
│   ├── requirements.txt            # Python package dependencies for the cs163-main project
│   └── README.md                   # This file
│
├── ExperimentsForML/               # Development area for ML experiments
│   ├── JSONFolder/                 # Contains raw and processed data (CSV/JSON) used during experimentation
│   ├── MLPredictiveAnalysis/       # Notebooks/scripts for ML model experimentation & development
│   ├── MainOutput*/                # Timestamped/versioned outputs from experimental pipeline runs
│   ├── ... (other experimental scripts, data, logs)
│
├── ExperimentationsNotUsefull/     # Archive of older/deprecated experiments, datasets, and logs
│
├── .venv/                          # Virtual environment folder (if used)
├── funding_prediction.log          # Main log file for the system
└── ... (other configuration files like .gitignore)
```

## 4. The Pipeline Explained

The project follows a multi-stage pipeline, from raw data collection to
delivering insights via a web interface, with a focus on the Bay Area
startup scene:

### 4.1. Data Collection
*   **Source:** Bay Area startup data is gathered from online platforms like
    `growthlist.co`, `topstartups.io`, and 'FundraiseInsider.com'.
*   **Process:** Python scripts located in `cs163-main/backend/AdataCollection/`
    use Selenium for web scraping. They extract company information such as
    funding amounts, employee counts, industry, location, and founding dates
    specific to Bay Area companies.
*   **Output:** The raw extracted data is saved as CSV or JSON files within
    `cs163-main/backend/AdataCollection/JSONFolder/`.

### 4.2. Data Preprocessing & Feature Engineering
*   **Location:** `cs163-main/backend/ML/funding_stage_predictionORIGINAL.py`
    (within `DataLoader` and `FeatureEngineering` classes).
*   **Process:**
    *   **Loading & Merging:** The pipeline loads data from the scraper outputs
        in `cs163-main/backend/AdataCollection/JSONFolder/` It merges these
        datasets, handling duplicates and standardizing company information.
    *   **Cleaning:** Data is cleaned by parsing funding amounts into numerical
        USD values, standardizing industry categories, processing dates, and
        handling missing values.
    *   **Feature Creation:** A rich set of features is engineered, focusing on
        aspects relevant to funding stage prediction:
        *   Log-transformed funding amounts.
        *   Company age, time since last funding.
        *   Funding velocity (funding amount per unit of time).
        *   Employee efficiency (funding per employee).
        *   Categorical features (industry, location) are one-hot encoded.
        *   Temporal features (e.g., funding month/year with cyclical encoding).
        *   Interaction terms between key features.
        *   And more
    *   **Scaling:** Numerical features are scaled (e.g., using StandardScaler).

### 4.3. Model Training & Evaluation
*   **Location:** `cs163-main/backend/ML/funding_stage_predictionORIGINAL.py`
    (within `ModelTrainer`, `EnhancedModelTrainer`, and `EnhancedPipeline`
    classes).
*   **Process:**
    *   A variety of machine learning models (Random Forest, Decision Trees,
        XGBoost, LightGBM, Stacking Ensembles, etc.) are trained to predict Bay
        Area startup funding stages.
    *   Hyperparameter tuning optimizes model performance.
    *   Models are evaluated (accuracy, F1-score, precision, recall, ROC AUC).
    *   The best performing model is selected, calibrated (e.g., using
        `CalibratedClassifierCV`), and retrained.
    *   Calibration plots assess the reliability of model probabilities.

### 4.4. Anomaly Detection
*   **Location:** `cs163-main/backend/ML/funding_stage_predictionORIGINAL.py`
    (`AnomalyDetector` class).
*   **Process:** An anomaly detection model (e.g., Isolation Forest) is trained.
    During prediction, it flags input data for Bay Area startups that is
    significantly different from the training data.

### 4.5. Time Series Forecasting (Bay Area Market Trends)
*   **Location:** `cs163-main/backend/ML/funding_stage_predictionORIGINAL.py`
    (`TimeSeriesForecaster` class).
*   **Process:** A Prophet model forecasts overall Bay Area funding trends,
    providing market context.

### 4.6. Backend Output & Artifacts
*   **Location:** Primarily `cs163-main/backend/MainOutput/`.
*   **Process:** The pipeline saves:
    *   **Trained Model:** The best calibrated model (`.joblib`), scaler, and
        feature names.
    *   **Anomaly Detector:** The trained anomaly model.
    *   **Summary JSON:** (`summary_<timestamp>.json`) with metrics, class
        mappings, feature importances, and Bay Area specific benchmarks (e.g.,
        median funding/employees per stage).
    *   **Visualizations:** Feature importance, calibration plots (PNGs), and an
        interactive Bay Area funding trend forecast (HTML).

### 4.7. Frontend Web Application & Publication
*   **Location:** `cs163-main/appengine/` (containing `appPredictionEngine.py`
    or `app163.py`).
*   **Process:**
    *   A Dash web application, configured via `app.yaml` for potential Google
        App Engine deployment.
    *   Loads artifacts from `cs163-main/backend/MainOutput/`.
    *   Users input Bay Area startup features.
    *   The app predicts funding stage, performs anomaly detection, and displays
        results with confidence scores, probabilities, and comparisons to Bay
        Area benchmarks.
    *   Embeds the Bay Area market trend forecast and model calibration plots.

*   **Initial Planning & Exploration Notebooks:**
    *   `cs163-main/.ipynb_checkpoints/`
    *   **Purpose:** This directory contains checkpoints and auto-saves from
        Jupyter Notebooks used during the initial planning and exploratory
        phases of the project. Files like `MachineLearningPlan.ipynb` (and its
        PDF versions) document early strategies, and other notebooks (e.g.,
        `OutputOf5MainpyFiles.ipynb`, `VisualizationResults.ipynb` checkpoints)
        reflect preliminary data analysis, visualization tests, and the
        exploration of combining multiple predictive outputs, which was part of
        the initial project ideation before focusing on funding stage prediction.

## 5. Location of Key Processing Code

*   **Data Collection (Bay Area Scrapers):**
    *   `cs163-main/backend/AdataCollection/` (contains `growthlistscrapper.py`,
        `fundraiserstartup50.py`, `topstartupio50.py`)
    *   **Purpose:** Scripts for gathering raw data on Bay Area startups. Output
        is in `cs163-main/backend/AdataCollection/JSONFolder/`.

*   **Core Backend ML Pipeline:**
    *   `cs163-main/backend/ML/funding_stage_predictionORIGINAL.py`
    *   **Purpose:** Central script for the entire ML workflow: data loading,
        preprocessing, feature engineering for Bay Area startups, model training,
        evaluation, and artifact generation.

*   **Frontend Web Application:**
    *   `cs163-main/appengine/appPredictionEngine.py` (or `app163.py`)
    *   **Purpose:** Dash application for user interaction, displaying predictions
        and insights for Bay Area startups. Configuration for deployment is in
        `cs163-main/appengine/app.yaml`.

*   **Experimental Code & Notebooks:**
    *   `ExperimentsForML/`
    *   **Purpose:** Contains Jupyter notebooks, Python scripts, experimental
        data, and versioned outputs used for development, testing, and iteration
        of ML approaches and data processing before integration into the main
        pipeline. This is a sandbox for research and development.
    *   `ExperimentationsNotUsefull/`
    *   **Purpose:** An archive for older or deprecated experimental work,
        datasets, and logs that are no longer part of the active development
        pipeline. Basically extensive amount of scratch work and testing for both
        Experimentation folders. We wanted to keep our scratch work folders instead
        of deleting it to show our work and progress. It is very messy but its
        our scratch work and versions of folders, files, and code.

## 6. Setup Instructions

### 6.1. Prerequisites
*   Python (version 3.8+ recommended)
*   `pip` (Python package installer)
*   Git (for cloning the repository)
*   Google Chrome and ChromeDriver (for Selenium-based scrapers).

### 6.2. Environment Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HKcode22/DSFinalYearProject.git
    cd DSFinalYearProject
    ```

2.  **Navigate to the main project directory:**
    ```bash
    cd cs163-main
    ```

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

4.  **Install required Python packages:**
    The primary `requirements.txt` is in `cs163-main/`. The `appengine`
    directory also has its own `requirements.txt` for deployment.
    ```bash
    pip install -r requirements.txt 
    # For App Engine deployment, ensure dependencies in appengine/requirements.txt are also considered.
    ```

5.  **ChromeDriver (for Data Collection):**
    *   Download ChromeDriver from
        [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
        matching your Chrome version.
    *   Place `chromedriver` in your system's PATH or update scraper scripts.

### 6.3. Running the Application & Pipeline

*   **Running Data Scrapers (from `cs163-main/` directory):**
    ```bash
    python backend/AdataCollection/topstartupio50.py
    # Run other scrapers as needed
    ```
    *Output will be in `cs163-main/backend/AdataCollection/JSONFolder/`.*

*   **Running the Backend ML Pipeline (from `cs163-main/` directory):**
    ```bash
    python backend/ML/funding_stage_predictionORIGINAL.py
    ```
    *This executes the pipeline, saving artifacts to
    `cs163-main/backend/MainOutput/`.*

*   **Running the Frontend Web Application (from `cs163-main/` directory):**
    ```bash
    python appengine/appPredictionEngine.py 
    # or: python appengine/app163.py
    ```
    *Starts the Dash server (usually `http://127.0.0.1:8057/`). Ensure the
    backend pipeline has run first.*

---
*This README provides a general guide. Specific configurations or execution
details might vary based on recent changes or local setup.*
Because we did do alot of changes here and there.

