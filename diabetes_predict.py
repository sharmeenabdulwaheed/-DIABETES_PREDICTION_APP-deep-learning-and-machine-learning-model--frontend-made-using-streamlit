import numpy as np
import pandas as pd
import streamlit as st
import pickle



st.title('Diabetes Prediction App')
st.header("Enter Patient Details")
model=pickle.load(open("C:\\Users\\U$ER\\Documents\\AKTI\\model.pkl", "rb"))
prob=pickle.load(open("C:\\Users\\U$ER\\Documents\\AKTI\\lr_mode.pkl", "rb"))
st.sidebar.title("Contact Us")
name = st.sidebar.text_input("Name")
email = st.sidebar.text_input("Email")
message = st.sidebar.text_input("contact no: ")

if st.sidebar.button("Submit"):
    st.sidebar.success("Thank you for reaching out! We'll get back to you soon.")



col1, col2= st.columns(2)

with col1:
    no_pregnancies = st.number_input("Number of pregnancies:" ,min_value=1, step=1)
    glucose = st.number_input("Glucose level:", min_value=0.0, step=0.1)
    insulin = st.number_input("Insulin level:", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI:", min_value=0.0, step=0.1)

with col2:
    bp = st.number_input("Blood pressure:", min_value=0.0, step=0.1)
    skin_thickness = st.number_input("Skin thickness:", min_value=0.0, step=0.1)
    db_func = st.number_input("Diabetes pedigree function:", min_value=0.0, step=0.01)
    age = st.number_input("Age:", min_value=0, step=1)


if st.button("Predict"):
    # Create input array for the model
    input_features = np.array([no_pregnancies, glucose, bp, skin_thickness, insulin, bmi, db_func, age]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)

    # Display result
    if prediction[0] == 1:
        st.success("The patient is likely to have diabetes.")
        probability=(prob.predict_proba(input_features)[0][1])*100
        st.success(f"The patient is {probability:.2f}% susceptible to having diabetes")
        

        st.markdown("[Click here for precautions and management for diabetes](https://www.mayoclinic.org/diseases-conditions/diabetes/in-depth/diabetes-management/ART-20045803)")
        

    else:
        st.success("The patient is unlikely to have diabetes.")

st.sidebar.title("Learn More")
st.sidebar.markdown("[Diabetes Symptoms](https://www.diabetes.org/diabetes/symptoms)")
st.sidebar.markdown("[Healthy Diet](https://www.eatright.org/)")
st.sidebar.markdown("[Mayo Clinic Diabetes Information](https://www.mayoclinic.org/diseases-conditions/diabetes/)")
