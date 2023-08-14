import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl1', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict():
    st.title("IT Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Switzerland",
        "Sweden",
        "Denmark",
        "Norway",
        "Israel",
        "Other",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",)

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 1)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.fit_transform(X[:,0])
        X[:, 1] = le_education.fit_transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        estimated_salary = salary[0]
        st.subheader(f"The estimated salary is ${estimated_salary:.2f}")