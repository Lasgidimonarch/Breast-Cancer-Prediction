#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pickle
import numpy as np

model_path = r"C:\Users\LasgidiMonarch\Breast Cancer Project\Breastdetection.pkl"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Breast Cancer Prediction by Ayoade")
st.text("Note: M = Malignant and B = Benign")

# Define unique keys for each input widget
radius_mean = st.number_input("radius_mean", min_value=0.1, max_value=1000.0, value=10.0, key="radius_mean")
perimeter_mean = st.number_input("perimeter_mean", min_value=0.1, max_value=1000.0, value=10.0, key="perimeter_mean")
area_mean = st.number_input("area_mean", min_value=0.1, max_value=1000.0, value=10.0, key="area_mean")
compactness_mean = st.number_input("compactness_mean", min_value=0.1, max_value=1000.0, value=10.0, key="compactness_mean")
concavity_mean = st.number_input("concavity_mean", min_value=0.1, max_value=1000.0, value=10.0, key="concavity_mean")
concave_point_mean = st.number_input("concave points_mean", min_value=0.1, max_value=1000.0, value=10.0, key="concave_point_mean")
radius_se = st.number_input("radius_se", min_value=0.1, max_value=1000.0, value=10.0, key="radius_se")
perimeter_se = st.number_input("perimeter_se", min_value=0.1, max_value=1000.0, value=10.0, key="perimeter_se")
area_se = st.number_input("area_se", min_value=0.1, max_value=1000.0, value=10.0, key="area_se_2")
radius_worst = st.number_input("radius_worst", min_value=0.1, max_value=1000.0, value=10.0, key="radius_worst")
perimeter_worst = st.number_input("perimeter_worst", min_value=0.1, max_value=1000.0, value=10.0, key="perimeter_worst")
area_worst = st.number_input("area_worst", min_value=0.1, max_value=1000.0, value=10.0, key="area_worst")
compactness_worst = st.number_input("compactness_worst", min_value=0.1, max_value=1000.0, value=10.0, key="compactness_worst")
concavity_worst = st.number_input("concavity_worst", min_value=0.1, max_value=1000.0, value=10.0, key="concavity_worst")
concave_point_worst = st.number_input("concave points_worst", min_value=0.1, max_value=1000.0, value=10.0, key="concave_point_worst")

if st.button("Predict"):
    input_data = np.array([[radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean,
                             concave_point_mean, radius_se, perimeter_se, area_se, radius_worst,
                             perimeter_worst, area_worst, compactness_worst, concavity_worst,
                             concave_point_worst]])

    prediction = model.predict(input_data)

    # Map prediction back to original labels
    predicted_label = 'M' if prediction[0] == 1 else 'B'
   
    st.write(f"Predicted Breast Cancer Status: {predicted_label}")


# In[ ]:





# In[ ]:




