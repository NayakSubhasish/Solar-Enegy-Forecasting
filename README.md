# Solar-Enegy-Forecasting
IV. B.TECH FINAL YEAR PROJECT
README for Solar Energy Forecasting Project
# Solar Energy Forecasting Using Machine Learning

![Solar Energy Forecasting](https://img.shields.io/badge/Project-Solar%20Energy%20Forecasting-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

This repository contains the implementation of a machine learning-based system for forecasting solar energy power output, as detailed in the project *"Using Machine Learning Algorithms to Forecast Solar Energy Power Output"*. The project leverages meteorological data (2018–2023) to predict solar power generation (W/m²) using a suite of machine learning models, culminating in an ensemble model for enhanced accuracy. The goal is to support renewable energy integration and grid management with precise, data-driven forecasts.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
The project aims to forecast solar energy power output using machine learning, addressing the challenge of variable solar generation due to weather conditions. By combining data from sources like the Indian Meteorological Department (IMD), NASA, and IoT sensors, we preprocess and engineer features to train multiple models. An ensemble of the top-performing models achieves superior accuracy, reducing forecasting errors by 4–5% compared to individual models. The system is designed for real-time applications in grid management and renewable energy optimization.

Key objectives:
- Develop accurate solar power forecasting models using meteorological data.
- Compare performance of diverse machine learning algorithms.
- Create an ensemble model for improved prediction reliability.
- Enable scalable deployment for energy management systems.

## Features
- **Comprehensive Pipeline**: Data collection, preprocessing, feature engineering, model training, and evaluation.
- **Multiple Models**: Linear Regression, KNN, Decision Tree, Random Forest, XGBoost, AdaBoost, FFNN, LSTM, and an Ensemble Model.
- **Robust Preprocessing**: Handles missing values, outliers (IQR method), normalization (MinMaxScaler), and feature selection (RFE).
- **Feature Engineering**: Time-based features (hour, day, month), lagged power outputs, and rolling mean of irradiance.
- **Performance Metrics**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R².
- **Ensemble Model**: Weighted combination of XGBoost (40%), Random Forest (35%), and LSTM (25%) for optimal predictions.
- **Visualizations**: Predicted vs. actual plots, residual plots, and feature importance analysis.

## Dataset
The dataset spans 2018–2023, sourced from:
- **IMD**: Weather data (irradiance G, temperature T, humidity H, precipitation P).
- **NASA**: Satellite-based solar and meteorological data.
- **IoT Sensors**: Ground-level power output measurements (y, W/m²).

**Features**:
- Raw: G, T, H, P, y.
- Engineered: Hour, day, month, lagged outputs (y_{t-1}, y_{t-2}, y_{t-3}), rolling mean of G (24-hour window).
- Preprocessed: Missing values removed, outliers filtered (IQR, threshold=1.5), normalized (MinMaxScaler), top 10 features selected via RFE.
- Split: 80% training, 20% testing, with 3D input for LSTM (samples, timesteps=24, features).

**Note**: Due to data privacy, the dataset is not included. Sample data is provided in `data/sample_data.csv`.

## Models
The project implements and compares nine machine learning models:
1. **Linear Regression**: Baseline model with Ridge regularization.
2. **K-Nearest Neighbors (KNN)**: k=5, Euclidean distance.
3. **Decision Tree**: max_depth=10, MSE criterion.
4. **Random Forest**: n_trees=100, max_depth=10, max_features=sqrt(n).
5. **XGBoost**: n_estimators=100, learning_rate=0.1, max_depth=6.
6. **AdaBoost**: n_estimators=50, learning_rate=1.0.
7. **Feed-Forward Neural Network (FFNN)**: 3 layers ([64, 32, 16]), ReLU activation, 20% dropout.
8. **Long Short-Term Memory (LSTM)**: 2 layers ([50, 50]), timesteps=24, tanh activation.
9. **Ensemble Model**: Weighted average of XGBoost (40%), Random Forest (35%), LSTM (25%).

Hyperparameters were tuned using GridSearchCV (5-fold CV, neg_mean_absolute_error scoring).

## Installation
### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/solar-energy-forecasting.git
   cd solar-energy-forecasting


Create a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt

requirements.txt:numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.7.0
matplotlib>=3.4.0
seaborn>=0.11.0



Usage

Prepare Data:

Place your dataset in data/ or use data/sample_data.csv.
Update config.yaml with dataset paths and parameters.


Run the Pipeline:
python main.py

This executes the full pipeline: data preprocessing, model training, prediction, and evaluation.

Key Scripts:

preprocess.py: Data cleaning, normalization, feature engineering.
train.py: Model training and hyperparameter tuning.
predict.py: Generate predictions using trained models.
evaluate.py: Compute MSE, RMSE, MAE, R²; generate visualizations.
ensemble.py: Combine XGBoost, Random Forest, LSTM predictions.


Output:

Predictions: outputs/predictions.csv
Metrics: outputs/metrics.csv
Plots: outputs/plots/ (e.g., predicted vs. actual, feature importance)


Deploy Model:

Use deploy.py to save the ensemble model for real-time forecasting:python deploy.py





Results
The ensemble model outperformed individual models, achieving:

MAE: Reduced by 4–5% compared to standalone models.

R²: Consistently >0.8, indicating strong fit.

Key Metrics:



Model
MSE
RMSE
MAE
R²



Linear Regression
0.12
0.35
0.28
0.75


Random Forest
0.08
0.28
0.22
0.82


XGBoost
0.07
0.26
0.20
0.85


LSTM
0.09
0.30
0.23
0.80


Ensemble
0.06
0.24
0.19
0.87



Visualizations: Show strong alignment between predicted and actual power output, with minimal residuals.

Feature Importance: Irradiance (G) and temperature (T) were the most influential predictors.


Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please follow the Code of Conduct and ensure tests pass (pytest tests/).
License
This project is licensed under the MIT License. See LICENSE for details.
Acknowledgements

Data Sources: Indian Meteorological Department (IMD), NASA, IoT sensor providers.
Libraries: scikit-learn, TensorFlow, XGBoost, Matplotlib, Seaborn.
References:
[27] S. Aslam et al., Renewable Energy, 2022.
[63] L. Breiman, Machine Learning, 2001.
[72] T. Chen and C. Guestrin, SIGKDD, 2016.





