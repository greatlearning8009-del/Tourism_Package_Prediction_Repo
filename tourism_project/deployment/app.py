"""
Streamlit UI for Tourism Package Prediction.

Loads the trained model from tourism_project/models/ in the repo. The
model file is produced by train.py and, in CI, committed back to the
repo by the pipeline workflow.
"""

import os
import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_PATH = os.path.join(
    REPO_ROOT, "tourism_project", "models", "best_tourism_package_model_v1.joblib"
)

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🧳")
st.title("🧳 Tourism Package Prediction")
st.write("Fill in the customer details to predict whether they'll purchase the Wellness Tourism Package.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Either push to `main` to let the pipeline train and commit it, "
        "or run locally:\n\n"
        "```\n"
        "python tourism_project/model_building/prep.py\n"
        "python tourism_project/model_building/train.py\n"
        "```"
    )
    st.stop()

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model(MODEL_PATH)

col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    NumberOfTrips = st.slider("Number of Trips per Year", 1, 20, 3)
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 5, 1)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Senior Manager"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0, step=1000.0)

input_data = pd.DataFrame([{
    "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch, "Occupation": Occupation,
    "Gender": Gender, "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups, "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation, "MonthlyIncome": MonthlyIncome,
}])

THRESHOLD = 0.45

if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_data)[0, 1])
    pred = int(prob >= THRESHOLD)
    st.metric("Purchase probability", f"{prob:.1%}")
    if pred == 1:
        st.success("✅ Customer is likely to purchase the travel package.")
    else:
        st.info("ℹ️ Customer is unlikely to purchase the travel package.")
    with st.expander("Model input"):
        st.dataframe(input_data)
