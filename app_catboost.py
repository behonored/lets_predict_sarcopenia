import streamlit as st
import pandas as pd
import joblib  # For loading model

# Load the saved CatBoost model
model = joblib.load("catboost_model.pkl")
# model = joblib.load("LGBM_model.pkl")
# model = joblib.load("LR_model.pkl")

# Load dataset to get feature columns
df = pd.read_csv("preprocessing_241126.csv")
feature_columns = df.drop('sarco', axis=1).columns

# Main app title and description
st.title("Sarcopenia Prediction Program")
st.write("""
    This program predicts sarcopenia using an AI model based on the 2022 Korea National Health and Nutrition Examination Survey (2022 KNHNES).
    Enter values for the features in the sidebar and click the "Predict" button to determine whether the condition is **Sarcopenia** or **Normal**.
""")

# Sidebar inputs using st.form for better organization
st.sidebar.header("Input Features")
with st.sidebar.form(key="input_form"):
    user_input = {}
    for feature in feature_columns:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

    # Submit button inside the form
    submit_button = st.form_submit_button(label="Predict")

# Predict button logic
if submit_button:
    # Prepare input as a DataFrame
    input_data = pd.DataFrame([user_input])

    # Make predictions with the CatBoost model
    predictions = model.predict_proba(input_data)
    pred_label = model.predict(input_data)[0]  # Label prediction (0 or 1)
    pred_score = predictions[0][pred_label]  # Probability of the predicted class

    # Display results
    st.subheader("Prediction Result")
    result = "Sarcopenia" if pred_label == 0 else "Normal"
    st.markdown(f"<h3 style='color: #1f77b4;'>{result}</h3>", unsafe_allow_html=True)  # Styled result
    st.write(f"**Prediction Probability (Score):** {pred_score * 100:.2f}%")


# Footer
st.markdown("""
    <hr>
    <p style="text-align: center;">
    Copyright 2024. Kyungmo Kang Ph.D. All rights reserved.
    </p>
""", unsafe_allow_html=True)
        

    
# streamlit run app_catboost.py
