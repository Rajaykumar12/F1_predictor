# Enterprise-Level F1 Race Prediction System

## Overview
This project is an enterprise-level F1 race prediction system designed to provide accurate predictions for race winners, lap times, and full race positions. Built using FastAPI and XGBoost, the system leverages robust feature engineering, data cleaning, and model evaluation techniques to ensure high-quality predictions. The system is modular, scalable, and ready for deployment in production environments.

## Key Features

- **FastAPI Integration**: Provides multiple prediction endpoints for race winner, lap time, and full race positions.
- **Robust Feature Engineering**: Includes rolling windows, DNF counts, podiums, tire degradation metrics, categorical encodings, and performance metrics.
- **Comprehensive Data Cleaning**: Handles outlier removal using IQR, missing value imputation, and normalization.
- **Advanced Model Architecture**: Utilizes XGBoost for classification, regression, and multi-target regression tasks.
- **Detailed Documentation**: Includes technical details, feature engineering, data cleaning, visualization, and model architecture.
- **Deployment Ready**: Supports Docker, AWS, and Azure deployment.

## Directory Structure
```
F1/
├── app.py
├── data_cleaning.py
├── data_collection.py
├── python_script.py
├── README.md
├── requirements.txt
├── __pycache__/
├── data/
│   ├── f1_laps_cleaned.csv
│   ├── f1_laps_features.csv
│   ├── f1_laps_simple.csv
│   ├── f1_laps.csv
│   ├── f1_results_cleaned.csv
│   ├── f1_results_features.csv
│   ├── f1_results_simple.csv
│   ├── f1_results.csv
├── models/
│   ├── race_prediction_pipeline.pk1
│   ├── xgb_laptime_pipeline.pk1
│   ├── xgb_laptime.pk1
│   ├── xgb_racewin_pipeline.pk1
│   ├── xgb_racewin.pk1
├── notebooks/
│   ├── feature_and_eda.ipynb
│   ├── lap_time_model.ipynb
│   ├── next_race_predict.ipynb
│   ├── race_winner_model.ipynb
```

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Running Locally](#How-to-Run-Locally)
- [Example Input](#Example-Inputs-and-Possible-Values)
- [Data Collection](#data-collection)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering and EDA](#feature-engineering-and-eda)
- [Models](#models)
- [Notebooks](#notebooks)
- [Race Winner Model](#race-winner-model)
- [Lap Time Model](#lap-time-model)
- [Next Race Prediction](#next-race-prediction)
- [FastAPI Application](#fastapi-application)
- [Pipeline Orchestration](#pipeline-orchestration)
- [Deployment](#deployment)
- [Requirements](#requirements)
- [Future Work](#future-work)
- [License](#license)


## How to Run Locally

To run the project on your local machine after cloning it, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd F1
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Application**:
   Run the FastAPI application using Uvicorn:
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the API Endpoints**:
   Open your browser or use a tool like Postman to access the following endpoints:
   - `/predict`: Predicts race winners.
   - `/predict_laptime`: Predicts lap times.
   - `/predict_next_race`: Predicts full race positions.
   - `/health`: Checks the health of the API.

5. **Explore the Notebooks**:
   Open the Jupyter notebooks in the `notebooks/` directory to explore feature engineering, model training, and predictions.

6. **Run the Pipeline**:
   Execute the `python_script.py` to orchestrate the entire pipeline:
   ```bash
   python python_script.py
   ```

This will ensure the project is set up and running locally for development and testing purposes.

For more details, refer to the individual notebooks and scripts included in the project.


## Example Inputs and Possible Values

### Predict Endpoint
#### Input Fields:
- **Team**: Possible values include `Red Bull Racing`, `Mercedes`, `Ferrari`, `McLaren`, `Alpine`, `AlphaTauri`, `Aston Martin`, `Williams`, `Haas`, `Alfa Romeo`.
- **Position**: Integer values ranging from `1` to `20`.
- **GridPosition**: Integer values ranging from `1` to `20`.
- **driver_win_rate**: Float values, e.g., `0.75`.
- **team_reliability**: Float values, e.g., `0.95`.

### Lap Time Endpoint
#### Input Fields:
- **Driver**: Possible values include `Max Verstappen`, `Lewis Hamilton`, `Charles Leclerc`, `Sergio Perez`, `Lando Norris`, `Fernando Alonso`, `Pierre Gasly`, `Sebastian Vettel`, `George Russell`, `Valtteri Bottas`.
- **Team**: Same as the `Team` field in the Predict Endpoint.
- **Position**: Integer values ranging from `1` to `20`.
- **TireCompound**: Possible values include `Soft`, `Medium`, `Hard`, `Intermediate`, `Wet`.
- **TireAge**: Integer values, e.g., `10`.
- **driver_win_rate**: Float values, e.g., `0.75`.
- **team_reliability**: Float values, e.g., `0.95`.


## Data Collection

The `data_collection.py` script is responsible for collecting and saving F1 race data. Key functionalities include:

- **Single Race Data Collection**: Fetches lap and race result data for individual races using the FastF1 library.
- **Multiple Race Data Collection**: Automates the collection of data for multiple races across seasons.
- **Data Saving**: Saves lap and race result data into CSV files for further analysis.
- **Caching**: Utilizes FastF1's caching mechanism to speed up data retrieval

## Data Cleaning

The `data_cleaning.py` script is responsible for preparing the collected F1 race data for analysis and modeling. Key functionalities include:

- **Missing Value Imputation**: Handles missing values using strategies like median imputation for lap times, mode imputation for tire compounds, and forward filling for positions.
- **Outlier Detection and Removal**: Identifies outliers in lap times using the Interquartile Range (IQR) method and removes them to ensure data integrity.
- **Categorical Encoding**: Converts categorical features such as driver names, team names, and race statuses into appropriate formats for analysis.
- **Data Saving**: Saves the cleaned data into CSV files for use in feature engineering and model training.

### Key Features
- **Lap Data Cleaning**: Ensures lap times, tire compounds, and positions are consistent and free of missing values.
- **Race Results Cleaning**: Prepares race result data by handling missing values and encoding categorical features.
- **Outlier Analysis**: Provides detailed insights into outlier bounds and their impact on the dataset.

### Usage
1. Run the script to clean the collected data.
2. Save the cleaned data into CSV files for use in feature engineering and modeling.

The cleaned data is essential for building accurate and reliable prediction models, ensuring the integrity of the feature engineering and modeling processes.

## Feature Engineering and EDA

The `feature_and_eda.ipynb` notebook is dedicated to feature engineering and exploratory data analysis (EDA) for the F1 race prediction system. Key functionalities include:

- **Driver and Team Analysis**: Visualizes driver wins, average positions, consistency, team points, and position changes.
- **Lap Time Analysis**: Explores lap time distributions, tire compound effects, and position vs lap time relationships.
- **Correlation Analysis**: Generates heatmaps to identify relationships between numerical features in lap and race result data.
- **Advanced Feature Engineering**: Creates features such as tire degradation, race phases, team reliability, and driver win rates.
- **Performance Trends**: Analyzes driver and team performance trends across races.

### Key Features
- **Visualization**: Provides detailed plots for understanding driver and team performance, lap time trends, and race strategies.
- **Feature Creation**: Develops advanced features like positions gained, tire degradation, and race phases to enhance model predictions.
- **Data Preparation**: Ensures the data is clean and ready for modeling by handling missing values and creating new features.

### Usage
1. Open the notebook to explore the visualizations and feature engineering steps.
2. Use the generated features for training prediction models.

The insights and features developed in this notebook are crucial for building accurate models and understanding the dynamics of F1 races.

## Models

The project includes multiple XGBoost models tailored for specific prediction tasks:
- **Race Winner Prediction**: A classification model trained to predict the race winner based on historical data and engineered features.
- **Lap Time Prediction**: A regression model designed to predict lap times using tire degradation, driver performance, and track conditions.
- **Full Race Position Prediction**: A multi-target regression model that predicts the positions of all drivers at the end of the race.

## Notebooks

The following Jupyter notebooks were created to support the development and evaluation of the models:
- **Feature Engineering and EDA**: Contains detailed feature engineering and exploratory data analysis.
- **Race Winner Model**: Implements and evaluates the race winner prediction model.
- **Lap Time Model**: Develops the lap time regression model.
- **Next Race Prediction**: Predicts full race positions for upcoming races.

## Race Winner Model

The `race_winner_model.ipynb` notebook focuses on predicting the race winner using advanced machine learning techniques. Key functionalities include:

- **Model Selection**: Implements multiple models including Random Forest, Logistic Regression, Gradient Boosting, and SVM for classification tasks.
- **Feature Engineering**: Utilizes features such as team reliability, driver win rates, grid positions, and race positions.
- **Cross-Validation**: Evaluates model performance using Stratified K-Fold cross-validation for metrics like accuracy and F1 score.
- **Visualization**: Provides detailed plots comparing model performance based on accuracy and F1 score.
- **Best Model Selection**: Identifies the best-performing model based on F1 score and saves it for deployment.

### Key Features
- **Pipeline Integration**: Combines preprocessing and model training into a single pipeline for efficiency.
- **Advanced Metrics**: Uses accuracy and F1 score to evaluate model performance.
- **Visualization**: Offers boxplots and bar charts for comparing model metrics.
- **Deployment Ready**: Saves the best model and pipeline for use in the FastAPI application.

### Usage
1. Open the notebook to explore the model training and evaluation process.
2. Use the saved model and pipeline for race winner predictions.

This notebook is essential for building a robust race winner prediction system, leveraging advanced machine learning techniques and feature engineering.

## Lap Time Model

The `lap_time_model.ipynb` notebook is dedicated to predicting lap times using advanced regression techniques. Key functionalities include:

- **Feature Engineering**: Utilizes features such as tire compound, tire age, driver win rates, team reliability, and race positions.
- **Model Selection**: Implements XGBoost for regression tasks, leveraging its ability to handle complex relationships and large datasets.
- **Pipeline Integration**: Combines preprocessing steps like scaling and encoding with model training into a single pipeline.
- **Model Saving**: Saves the trained model and pipeline for deployment in the FastAPI application.

### Key Features
- **Advanced Regression**: Uses XGBoost for accurate lap time predictions.
- **Preprocessing**: Handles categorical and numerical features using one-hot encoding and standard scaling.
- **Deployment Ready**: Saves the trained model and pipeline for use in production.

### Usage
1. Open the notebook to explore the lap time prediction process.
2. Use the saved model and pipeline for lap time predictions.

This notebook is essential for building a reliable lap time prediction system, leveraging advanced regression techniques and feature engineering.

## Next Race Prediction

The `next_race_predict.ipynb` notebook is designed to predict full race positions for upcoming races using historical data and advanced regression techniques. Key functionalities include:

- **Feature Engineering**: Creates historical features such as average positions, grid positions, DNF counts, positions gained, and podium finishes over previous races.
- **Model Selection**: Implements XGBoost for multi-target regression tasks, leveraging its ability to handle complex relationships and large datasets.
- **Pipeline Integration**: Combines preprocessing steps like scaling and encoding with model training into a single pipeline.
- **Model Saving**: Saves the trained model and pipeline for deployment in the FastAPI application.

### Key Features
- **Advanced Regression**: Uses XGBoost for accurate race position predictions.
- **Historical Features**: Incorporates rolling averages and counts to capture trends and patterns in driver and team performance.
- **Deployment Ready**: Saves the trained model and pipeline for use in production.

### Usage
1. Open the notebook to explore the race position prediction process.
2. Use the saved model and pipeline for predicting race positions.

This notebook is essential for building a reliable race position prediction system, leveraging advanced regression techniques and feature engineering.

## FastAPI Application

The `app.py` script serves as the backend API for the F1 race prediction system, built using FastAPI. It provides multiple endpoints for race winner prediction, lap time prediction, and full race position prediction.

### Key Features
- **Race Winner Prediction Endpoint**: Accepts input features such as team reliability, driver win rates, and grid positions to predict the likelihood of a driver winning the race.
- **Lap Time Prediction Endpoint**: Predicts lap times based on features like tire compound, tire age, driver performance, and team reliability.
- **Next Race Position Prediction Endpoint**: Predicts full race positions for upcoming races using historical data and advanced regression techniques.
- **Health Check Endpoint**: Provides a simple health check to ensure the API is running.

### Functionality
- **Model Integration**: Loads pre-trained models for race winner, lap time, and race position predictions.
- **Feature Engineering**: Includes functions for creating historical features such as average positions, DNF counts, and podium finishes.
- **Error Handling**: Implements robust error handling for missing data, invalid inputs, and prediction failures.
- **Logging**: Provides detailed logging for debugging and monitoring API requests.

### Usage
1. Start the FastAPI application using `uvicorn app:app --reload`.
2. Access the endpoints for predictions:
   - `/predict`: Predicts race winners.
   - `/predict_laptime`: Predicts lap times.
   - `/predict_next_race`: Predicts full race positions.
   - `/health`: Checks the health of the API.

This script is essential for deploying the F1 race prediction system, providing a user-friendly interface for accessing prediction models.

## Pipeline Orchestration

The `python_script.py` script is responsible for orchestrating the execution of all components in the F1 race prediction system. It ensures that data collection, cleaning, feature engineering, model training, and API setup are performed sequentially.

### Key Features
- **Sequential Execution**: Runs Python scripts and Jupyter notebooks in the correct order to ensure smooth pipeline execution.
- **Notebook Execution**: Executes Jupyter notebooks programmatically using `nbconvert` to automate feature engineering and model training.
- **Error Handling**: Captures and logs errors during script and notebook execution for debugging.
- **Scalability**: Supports adding new scripts or notebooks to the pipeline with minimal changes.

### Functionality
- **Script Execution**: Runs Python scripts for data collection, cleaning, and API setup.
- **Notebook Execution**: Processes Jupyter notebooks for feature engineering, exploratory data analysis, and model training.
- **Logging**: Provides detailed logs for each step in the pipeline.

### Usage
1. Add the required scripts and notebooks to the `files_to_run` list.
2. Run the script using `python python_script.py`.
3. Monitor the logs for progress and errors.

This script is essential for automating the workflow of the F1 race prediction system, ensuring all components are executed in the correct order.

## Deployment

The system is ready for deployment using Docker, AWS, or Azure. Detailed deployment instructions are provided in the documentation.

## Requirements

The project dependencies are listed in `requirements.txt`. Key libraries include:
- FastAPI
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- FastF1
- Matplotlib
- Seaborn
- Jupyter

## Usage
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the FastAPI application using `uvicorn app:app --reload`.
3. Access the API endpoints for predictions.

### Key Features
- **Lap Data**: Includes details such as lap times, positions, tire compounds, and tire age.
- **Race Results**: Captures information like driver names, team names, grid positions, final positions, points, and race status.
- **Error Handling**: Ensures robust handling of API errors and missing data.
- **Scalability**: Supports data collection for multiple races and seasons.

### Usage
1. Run the script to collect data for specified races.
2. Save the collected data into CSV files for use in feature engineering and modeling.

The collected data serves as the foundation for feature engineering and model training, enabling accurate predictions for race winners, lap times, and full race positions.

## Future Work

Potential enhancements include:
- Integration of real-time data (e.g., weather conditions).
- Advanced deployment strategies.
- Additional feature engineering techniques.

## License

This project is licensed under the MIT License.

