import streamlit as st
import joblib
import pandas as pd

# Load the saved components
model = joblib.load('random_forest_model.joblib')
encoders = joblib.load('label_encoders.joblib')
model_columns = joblib.load('model_columns.joblib')

# --- UI Layout ---
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")
st.title("🚗 Indian Used Car Price Predictor")
st.markdown("Enter the details of the car to get an estimated price.")

# We need to get the unique values for the dropdowns from the original dataset
# In a real-world scenario, you'd save these unique values during preprocessing
# For now, let's define them manually or load a small file.
# NOTE: Replace these with the actual unique values from your 'usedCars.csv' for better accuracy.
companies = ['MARUTI SUZUKI', 'HYUNDAI', 'TATA', 'MAHINDRA', 'HONDA', 'TOYOTA', 'FORD', 'RENAULT', 'KIA', 'VOLKSWAGEN']
fuel_types = ['PETROL', 'DIESEL', 'CNG']
body_styles = ['HATCHBACK', 'SUV', 'SEDAN', 'MUV']
owner_types = ['1st Owner', '2nd Owner', '3rd Owner']


# --- Create Input Fields in Two Columns ---
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", options=companies)
    kilometer = st.number_input("Kilometers Driven", min_value=0, step=1000)
    fuel_type = st.selectbox("Fuel Type", options=fuel_types)
    owner = st.selectbox("Owner", options=owner_types)

with col2:
    model_year = st.number_input("Model Year", min_value=1990, max_value=2025, step=1)
    body_style = st.selectbox("Body Style", options=body_styles)
    warranty = st.selectbox("Warranty Available?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")


# --- Prediction Logic ---
if st.button("Predict Price", use_container_width=True):
    try:
        # Create a dictionary from the user inputs
        # IMPORTANT: The keys must match the column names your model was trained on!
        input_data = {
            'Company': company,
            'Kilometer': kilometer,
            'FuelType': fuel_type,
            'Owner': owner,
            'ModelYear': model_year,
            'BodyStyle': body_style,
            'Warranty': warranty
            # Add other features your model expects, setting a default if not in UI
        }
        
        # Create a DataFrame from the input
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the categorical features using the loaded encoders
        for column, encoder in encoders.items():
            if column in input_df.columns:
                # Use a lambda function to handle unseen labels gracefully
                input_df[column] = input_df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

        # Ensure all model columns are present and in the correct order
        final_df = pd.DataFrame(columns=model_columns)
        final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)
        final_df = final_df[model_columns] # Reorder columns to match model's expectation


        # Make prediction
        prediction = model.predict(final_df)
        predicted_price = prediction[0]

        # Display the result
        st.success(f"**Predicted Price: ₹ {predicted_price:,.2f} Lakhs**")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure all inputs are correct. Some values might not have been seen during training.")