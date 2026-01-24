import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(repo_id="amitmzn/tourism-model", filename="best_tourism_model.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")
st.write("Predict if a customer will purchase the Wellness Tourism Package.")

age = st.number_input("Age", 18, 100, 35)
gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
income = st.number_input("Monthly Income", 5000, 100000, 20000)
city_tier = st.selectbox("City Tier", [1, 2, 3])
passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Owns Car", [0, 1])
num_persons = st.number_input("Persons Visiting", 1, 10, 2)
num_children = st.number_input("Children Visiting", 0, 5, 0)
num_trips = st.number_input("Annual Trips", 1, 20, 3)
prop_star = st.selectbox("Preferred Hotel Star", [3, 4, 5])
contact_type = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
pitch_duration = st.number_input("Pitch Duration (min)", 1, 60, 15)
followups = st.number_input("Followups", 0, 10, 3)
pitch_score = st.selectbox("Pitch Satisfaction", [1, 2, 3, 4, 5])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

gender_enc = 1 if gender == "Male" else 0
marital_enc = {"Single": 2, "Married": 1, "Divorced": 0, "Unmarried": 3}.get(marital, 0)
occ_enc = {"Salaried": 2, "Small Business": 3, "Large Business": 1, "Free Lancer": 0}.get(occupation, 0)
contact_enc = 1 if contact_type == "Self Enquiry" else 0
product_enc = {"Basic": 0, "Standard": 3, "Deluxe": 1, "Super Deluxe": 4, "King": 2}.get(product, 0)
desig_enc = {"Executive": 0, "Manager": 2, "Senior Manager": 3, "AVP": 1, "VP": 4}.get(designation, 0)

input_data = pd.DataFrame([[
    age, contact_enc, city_tier, pitch_duration, occ_enc, gender_enc,
    num_persons, followups, product_enc, prop_star, marital_enc,
    num_trips, passport, pitch_score, own_car, num_children, desig_enc, income
]], columns=['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender',
             'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
             'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
             'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    if prediction == 1:
        st.success(f"Customer is LIKELY to purchase! (Probability: {probability[1]:.2%})")
    else:
        st.warning(f"Customer is UNLIKELY to purchase. (Probability: {probability[0]:.2%})")
