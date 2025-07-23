# create_model.py (FINAL LOGIC-CHECKED VERSION)

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

print("--- Starting FINAL LOGIC-CHECKED Model Creation Script ---")

# --- STEP 1: Load, Clean, and Select Final Features ---
print("1. Loading, cleaning, and selecting final features...")
try:
    df = pd.read_csv('usedCars.csv')

    # --- FINAL FEATURE SELECTION ---
    # We remove 'QualityScore' because it had an inverse correlation with price.
    cols_to_drop = [
        'Unnamed: 0', 'Id', 'Model', 'Variant', 'ManufactureDate',
        'DealerName', 'City', 'TransmissionType', 'CngKit', 'QualityScore'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print("Dropped noisy and illogical columns.")
    # --- END OF FEATURE SELECTION ---

    df['Price'] = df['Price'].str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
    df.dropna(subset=['Price'], inplace=True)

    current_year = pd.to_datetime('today').year
    df['Age'] = current_year - df['ModelYear']
    df.drop('ModelYear', axis=1, inplace=True, errors='ignore')

    price_cap = df['Price'].quantile(0.99)
    df = df[df['Price'] <= price_cap]
    
    df.dropna(inplace=True)
    print("✅ Data processing complete.")

except Exception as e:
    print(f"❌ An error occurred during data processing: {e}")
    exit()

# --- The rest of the script remains the same ---

# --- STEP 2: Encode Remaining Categorical Columns ---
print("\n2. Encoding categorical data...")
encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
joblib.dump(encoders, 'label_encoders.joblib')
print("✅ Encoders saved.")

# --- STEP 3: Define Features, Target & Column Order ---
print("\n3. Defining features and saving column order...")
X = df.drop('Price', axis=1)
y = df['Price']
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.joblib')
print("✅ Column order saved.")

# --- STEP 4: Train the Final, Stable Model ---
print("\n4. Training the final, stable model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2, max_features=0.75, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- STEP 5: Save the Final Model ---
print("\n5. Saving the final model...")
joblib.dump(model, 'random_forest_model.joblib')
print("✅ Final model saved.")

# --- STEP 6: Evaluate the Final Model ---
print("\n6. Evaluating the final model...")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- FINAL MODEL PERFORMANCE ---")
print(f"✅ R-squared (R²): {r2:.2f}")
print(f"✅ Mean Absolute Error (MAE): ₹ {mae:.2f} Lakhs")
print("-----------------------------")