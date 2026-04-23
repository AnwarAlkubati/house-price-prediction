import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🏠 House Price Prediction App")
st.markdown("Estimate house prices instantly using Machine Learning.")

st.divider()

# Model info
st.subheader("Model Information")
st.write("""
This app uses a trained Log-Transformed Linear Regression model
built on the Kaggle House Prices dataset.
""")

st.divider()

# Sidebar inputs
st.sidebar.header("House Features")

gr_liv_area = st.sidebar.number_input(
    "Living Area (sq ft)", 500, 5000, 2000
)

overall_qual = st.sidebar.slider(
    "Overall Quality", 1, 10, 7
)

garage_cars = st.sidebar.slider(
    "Garage Capacity", 0, 5, 2
)

full_bath = st.sidebar.slider(
    "Full Bathrooms", 0, 5, 2
)

bedrooms = st.sidebar.slider(
    "Bedrooms", 1, 10, 3
)

year_built = st.sidebar.number_input(
    "Year Built", 1900, 2025, 2000
)

# House summary
st.subheader("Selected House")

col1, col2, col3 = st.columns(3)
col1.metric("Living Area", f"{gr_liv_area} sqft")
col2.metric("Bedrooms", bedrooms)
col3.metric("Bathrooms", full_bath)

col1, col2, col3 = st.columns(3)
col1.metric("Garage", garage_cars)
col2.metric("Quality", overall_qual)
col3.metric("Year Built", year_built)

st.write("")

# Prediction
if st.button("Predict Price", key="predict_btn"):

    new_house = pd.DataFrame({
        "GrLivArea": [gr_liv_area],
        "OverallQual": [overall_qual],
        "GarageCars": [garage_cars],
        "FullBath": [full_bath],
        "BedroomAbvGr": [bedrooms],
        "YearBuilt": [year_built]
    })

    new_house = pd.get_dummies(new_house)
    new_house = new_house.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction_log = model.predict(new_house)
    prediction = np.expm1(prediction_log)[0]

    st.markdown("## Predicted Price")

    st.metric(
        label="Estimated House Value",
        value=f"${prediction:,.0f}"
    )

    st.success("Prediction generated successfully.")
    st.caption("Prediction based on historical housing data.")