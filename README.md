# Bus Travel Time Prediction

## Overview
This project applies machine learning models to predict bus travel time. The study aims to build a predictive system for estimating bus arrival times, helping passengers manage their schedules efficiently and assisting transportation authorities in optimizing routes and improving service quality.

## Problem Statement
The goal is to predict bus arrival times using various machine learning models, considering factors such as traffic conditions, weather, and historical trip data. 

### **Input Data:**
- **Schedule Data:** Planned departure times, past stop durations.
- **Traffic Data:** Current road conditions, accidents, construction work.
- **Weather Data:** Conditions such as rain, temperature, and humidity.
- **Date Information:** Current time, day of the week, holidays, and special events.
- **Sensor Data:** GPS location, speed, and vehicle status.

### **Output:**
- Estimated Time of Arrival (ETA) for upcoming bus stops.

## Applied Methods
### 1. **Support Vector Machine (SVM)**
- Used for regression (Support Vector Regression - SVR)
- Constructs a hyperplane to separate data optimally
- Kernel trick applied for non-linear relationships

### 2. **Random Forest**
- Uses multiple decision trees for prediction
- Handles noisy data and prevents overfitting

### 3. **Gradient Boosting**
- An ensemble learning technique that builds weak learners iteratively
- Applied to improve prediction accuracy

## Dataset
- **Source:** GPS data collected every 15 seconds from buses on two routes (654 and 690) in Kandy.
- **Data Collection Period:**
  - 01/10/2021 - 28/02/2022
  - 01/07/2022 - 01/11/2022
- **Number of trips:**
  - Kandy to Digana: 14,128 trips
  - Kandy to Kadugannawa: 11,200 trips
- **Weather data** was retrieved from the Sri Lanka Meteorological Department.

## Data Analysis & Preprocessing
This project demonstrates strong **data analysis** and **data engineering** skills, including:
1. **Exploratory Data Analysis (EDA):**
   - Data visualization to understand distribution and trends.
   - Correlation analysis to identify key predictive features.
2. **Cleaning & Feature Engineering:**
   - Handling missing values and detecting outliers using statistical methods (Z-score analysis).
   - Creating new time-based features like `part_of_day` and `date_in_week`.
3. **Encoding & Scaling:**
   - One-Hot Encoding for categorical variables.
   - Min-Max Scaling for numerical feature normalization.
4. **Model Evaluation & Interpretation:**
   - Comparing multiple regression models using performance metrics (MAE, MSE, RMSE, R²).
   - Using visualization techniques (Residual Plots, Error Distribution) to assess model effectiveness.

## Model Implementation
### Feature Selection
- **X (Features):** Includes trip_id, device_id, direction, weather, and encoded time-related variables.
- **y (Target Variable):** Travel time between two bus stops (in seconds).

### Training and Testing
- Dataset split into **80% training** and **20% testing**.
- Models used:
  1. **SVR** (Radial Basis Function kernel, C=170, epsilon=24)
  2. **Random Forest Regressor** (200 trees, random_state=48)
  3. **Gradient Boosting Regressor** (500 estimators, max_depth=4, learning_rate=0.01)

## Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-Squared (R²)**

## Results and Future Work
- Performance comparison of models using error metrics.
- Future work includes:
  - Exploring **Unsupervised Learning** for clustering similar travel patterns.
  - Incorporating **real-time data streaming** to improve accuracy.

## Repository Structure
```
📂 Bus-Travel-Time-Prediction
├── 📁 dataset        # Dataset files
├── 📁 preprocess     # Jupyter notebooks for analysis
├── 📁 model          # Trained models
├── README.md        # Project documentation
```
## Required Libraries

Make sure you have these libraries installed to run this project:

* **math:**  Python's built-in math module. This provides essential mathematical functions like trigonometry, logarithms, and more.
* **numpy:** The fundamental package for numerical operations in Python. NumPy introduces powerful arrays and matrices, along with a wide range of mathematical functions to operate on them.
* **matplotlib:** The primary plotting library for creating static, animated, and interactive visualizations in Python.
* **pandas:** A high-performance library for data analysis and manipulation. pandas introduces DataFrames, a versatile data structure for working with tabular data.
* **scikit-learn (sklearn):** A comprehensive machine learning library offering tools for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing.

## Installation

You can install these libraries using `pip`, Python's package installer:

```bash
pip install numpy matplotlib pandas scikit-learn
