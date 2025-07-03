import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------------
# 🌊 Custom Background & Style
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .block-container {
        backdrop-filter: blur(4px);
        background-color: rgba(255, 255, 255, 0.65);
        padding: 2rem 2rem 2rem 2rem;
        border-radius: 1rem;
        margin-top: 1rem;
    }

    h1 {
        color: #003366;
        text-align: center;
        font-size: 2.8em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }

    .stButton>button {
        background-color: #006699;
        color: white;
        font-weight: bold;
        padding: 0.5em 1.5em;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ----------------------------------

# 💠 App Title
st.markdown("<h1>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("Enter passenger details to predict survival based on historical Titanic data.")

# 📋 Input Form
pclass = st.selectbox("Passenger Class (1 = 1st, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare Paid (£)", 0.0, 512.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=6, value=0)

# 🧠 Preprocess Input
sex = 0 if sex == "male" else 1
embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
embarked = embarked_map[embarked]
family_size = sibsp + parch + 1

# 🎯 Predict
if st.button("Predict Survival"):
    input_data = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]],
                              columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize'])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger **would survive!**")
    else:
        st.error("😢 The passenger **would not survive.**")
