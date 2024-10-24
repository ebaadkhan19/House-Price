import streamlit as st
import pickle as pickle
import numpy as np
import pandas as pd

# Load data and model
path = "F:\House Prediction"
houses = pd.read_csv(path + 'Entities.csv')
with open(path + "house_prediction.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.markdown(
    """
    <div style="background-color:SkyBlue; padding:30px">
        <h2 style="color:black; text-align:center;">House Price Prediction</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Input fields
baths = st.number_input("Enter number of baths", min_value=1, max_value=10, step=1)
bedrooms = st.number_input("Enter number of bedrooms", min_value=1, max_value=10, step=1)
total_area = st.number_input("Enter total area", min_value=0.0, step=0.1)

property_types = ['Farm House', 'Flat', 'House', 'Lower Portion', 'Penthouse', 'Room', 'Upper Portion']
selected_property_type = st.selectbox("Select Property Type", property_types)

cities = ['Faisalabad', 'Islamabad', 'Karachi', 'Lahore', 'Rawalpindi']
selected_city = st.selectbox("Select City", cities)

# Prepare input for prediction
property_type_features = [selected_property_type == prop_type for prop_type in property_types]
city_features = [selected_city == city for city in cities]

input_features = [baths, bedrooms, total_area] + property_type_features + city_features

# Predict
prediction = model.predict([input_features])

# Display prediction
st.write("Predicted House Price:", prediction)
