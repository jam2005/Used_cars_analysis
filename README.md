# Indian Used Car Price Predictor

This project provides a machine learning solution for predicting used car prices in India using a RandomForest Regressor model, with a Streamlit web interface for user-friendly predictions. The model is trained on a dataset of 150 used car listings from cities like Bangalore, Pune, Mumbai, Chennai, Kolkata, and Jaipur, capturing key features such as car company, kilometers driven, fuel type, ownership history, model year, body style, and warranty status.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Performance Metrics](#performance-metrics)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
The Indian Used Car Price Predictor estimates the market price of used cars based on key attributes. The project includes:
- Data preprocessing and exploratory data analysis (EDA) to clean and prepare the dataset.
- A RandomForest Regressor model trained on 150 car listings.
- A Streamlit web app for users to input car details and receive price predictions in lakhs.
- Saved model components (model, label encoders, and column order) for reproducibility.

## Features
- **Data Preprocessing**: Cleans missing values, removes outliers, and encodes categorical features.
- **Feature Selection**: Uses `Company`, `Kilometer`, `FuelType`, `Owner`, `Age` (derived from `ModelYear`), `BodyStyle`, and `Warranty`.
- **Model**: RandomForest Regressor with 200 estimators for robust predictions.
- **Web Interface**: Streamlit app with a two-column input form and error handling for invalid inputs.
- **Error Handling**: Manages unseen categorical values and ensures valid numeric inputs.

## Dataset
The dataset (`usedCars.csv`) contains 150 used car listings with the following key columns:
- **Columns Used**:
  - `Company`: Car manufacturer (e.g., Maruti Suzuki, Hyundai, Tata, Honda).
  - `Kilometer`: Distance driven (range: 101 to 195,575 km).
  - `FuelType`: Fuel type (Petrol, Diesel, CNG).
  - `Owner`: Ownership history (1st, 2nd, 3rd Owner).
  - `ModelYear`: Year of manufacture (2003–2022).
  - `BodyStyle`: Car type (Hatchback, Sedan, SUV, MPV, MUV, Van).
  - `Warranty`: Warranty availability (0 or 1).
  - `Price`: Sale price in lakhs (₹1–31.9 Lakhs).
- **Other Columns**: `Id`, `Model`, `Variant`, `Colour`, `TransmissionType`, `ManufactureDate`, `CngKit`, `DealerState`, `DealerName`, `City`, `QualityScore` (dropped during preprocessing).
- **Preprocessing**:
  - Dropped irrelevant columns (e.g., `Id`, `Model`, `Variant`, `Colour`, `TransmissionType`, `CngKit`, `DealerName`, `City`, `DealerState`, `QualityScore`).
  - Converted `Price` from string (e.g., "5.75 Lakhs") to float.
  - Derived `Age` from `ModelYear` (Age = 2025 - ModelYear, assuming current year is 2025).
  - Removed outliers using the 99th percentile for `Price` (threshold ~₹23.5 Lakhs).
  - Applied label encoding to categorical columns (`Company`, `FuelType`, `Owner`, `BodyStyle`).
  - Dropped rows with missing values (none observed in the provided dataset).

The dataset is not included in the repository but must be placed in the project root for model training and analysis.

## Installation
To set up the project locally:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/indian-used-car-price-predictor.git
   cd indian-used-car-price-predictor
   ```
2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Place the Dataset**:
   - Ensure `usedCars.csv` is in the project root or update file paths in `create_model.py` and `Cleaning & EDA.ipynb`.
5. **Train the Model** (optional, if not using pre-trained model):
   ```bash
   python create_model.py
   ```
6. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Training the Model**:
   - Run `create_model.py` to preprocess the dataset, train the RandomForest model, and save components (`random_forest_model.joblib`, `label_encoders.joblib`, `model_columns.joblib`).
   - Outputs R² and MAE metrics for model performance.
2. **Running the Web App**:
   - Launch with `streamlit run app.py` and access at `http://localhost:8501`.
   - Input car details (e.g., Company: Maruti Suzuki, Kilometer: 50,000, FuelType: Petrol, etc.) and click "Predict Price" for an estimated price.
3. **Exploratory Data Analysis**:
   - Use `Cleaning & EDA.ipynb` to visualize data distributions (e.g., price vs. age, fuel type counts) and preprocess the dataset.

## File Structure
```
indian-used-car-price-predictor/
├── app.py                    # Streamlit app for price predictions
├── create_model.py           # Script for data preprocessing and model training
├── Cleaning & EDA.ipynb      # Notebook for data cleaning and EDA
├── requirements.txt          # Python dependencies
├── random_forest_model.joblib # Trained RandomForest model
├── label_encoders.joblib     # Label encoders for categorical features
├── model_columns.joblib      # Model input column order
├── README.md                 # Project documentation
└── usedCars.csv              # Dataset (not included, must be provided)
```

## Model Training
The `create_model.py` script processes the dataset as follows:
1. **Data Cleaning**:
   - Load `usedCars.csv` and drop irrelevant columns.
   - Convert `Price` to numeric (remove "Lakhs").
   - Calculate `Age` from `ModelYear` and drop `ModelYear`.
   - Remove outliers (prices > 99th percentile, ~₹23.5 Lakhs).
   - Drop rows with missing values (if any).
2. **Feature Encoding**:
   - Label encode categorical features (`Company`, `FuelType`, `Owner`, `BodyStyle`).
   - Save encoders to `label_encoders.joblib`.
3. **Feature Selection**:
   - Use `Company`, `Kilometer`, `FuelType`, `Owner`, `Age`, `BodyStyle`, `Warranty`.
   - Save column order to `model_columns.joblib`.
4. **Training**:
   - Train RandomForest Regressor (200 estimators, `min_samples_leaf=2`, `max_features=0.75`).
   - Save model to `random_forest_model.joblib`.
5. **Evaluation**:
   - Compute R² and Mean Absolute Error (MAE) on test data.

## Web Application
The Streamlit app (`app.py`) offers:
- **Interface**: Two-column form for inputs (dropdowns for categorical features, numeric fields for `Kilometer` and `ModelYear`).
- **Prediction**: Encodes inputs, aligns with model features, and displays predicted price in lakhs.
- **Error Handling**: Manages invalid inputs (e.g., negative kilometers) and unseen categorical values with user-friendly messages.
  ![Screenshot_23-7-2025_163712_localhost](https://github.com/user-attachments/assets/a5043f35-1190-4c94-a5c1-4560972954af)


## Performance Metrics
Based on the original project, the RandomForest model achieves:
- **R-squared (R²)**: ~0.36 (indicating moderate fit to price variation).
- **Mean Absolute Error (MAE)**: ~₹0.60 Lakhs (average prediction error).

These metrics are based on the full dataset; retraining on the provided 150-row subset may yield different results due to the smaller sample size.

## Dependencies
Listed in `requirements.txt`:
```
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib
seaborn
```
Install with:
```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

## Future Improvements
- **Model Improvements**:
  - Test Gradient Boosting or XGBoost for better accuracy.
  - Tune hyperparameters (e.g., grid search for `n_estimators`, `max_depth`).
  - Incorporate additional features like `TransmissionType` or `Colour` with proper encoding.
- **Data Enhancements**:
  - Expand dataset beyond 150 records for better generalization.
  - Include newer cars (post-2022) and more diverse `FuelType` (e.g., only 2 CNG cars in current data).
- **Web App Enhancements**:
  - Add input validation (e.g., restrict `Kilometer` to positive values).
  - Display price distribution charts or feature importance.
  - Deploy to a cloud platform (e.g., Streamlit Cloud).
- **EDA Improvements**:
  - Analyze correlations (e.g., `Price` vs. `Kilometer`, `Price` vs. `Age`).
  - Visualize trends by `Company` or `BodyStyle`.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
- **Dataset**: Sourced from used car listings in India (150 records provided).
- **Libraries**: Thanks to pandas, scikit-learn, Streamlit, and other open-source contributors.
- **Purpose**: Built to help car buyers and sellers estimate fair market prices.
