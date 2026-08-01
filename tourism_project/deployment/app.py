import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# The model is expected to be a Pipeline that includes preprocessing steps
model = joblib.load('tourism_project/deployment/best_xgboost_model.joblib')

st.set_page_config(
    page_title="Tourism Package Purchase Prediction",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title('✈️ Tourism Package Purchase Prediction')
st.markdown("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for customer details
st.header('Customer Information')

# Numerical inputs
age = st.slider('Age', min_value=18, max_value=90, value=30)
number_of_person_visiting = st.slider('Number of Persons Visiting', min_value=1, max_value=10, value=2)
number_of_trips = st.slider('Number of Trips Annually', min_value=0, max_value=20, value=5)
number_of_children_visiting = st.slider('Number of Children Visiting (under 5)', min_value=0, max_value=5, value=0)
monthly_income = st.number_input('Monthly Income', min_value=0.0, value=25000.0, format="%.2f")
pitch_satisfaction_score = st.slider('Pitch Satisfaction Score (1-5)', min_value=1, max_value=5, value=3)
number_of_followups = st.slider('NumberOfFollowups', min_value=0, max_value=10, value=3)
duration_of_pitch = st.number_input('Duration of Pitch (minutes)', min_value=0.0, value=10.0, format="%.2f")

# Categorical inputs
type_of_contact = st.selectbox('Type of Contact', ['Company Invited', 'Self Inquiry'])
city_tier = st.selectbox('City Tier', [1, 2, 3])
occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Unemployed'])
gender = st.selectbox('Gender', ['Male', 'Female'])
preferred_property_star = st.selectbox('Preferred Property Star Rating', [3, 4, 5])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director', 'Employee'])
product_pitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
passport = st.selectbox('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.selectbox('Owns Car?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')


# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'Age': [age],
    'NumberOfPersonVisiting': [number_of_person_visiting],
    'NumberOfTrips': [number_of_trips],
    'NumberOfChildrenVisiting': [number_of_children_visiting],
    'MonthlyIncome': [monthly_income],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'NumberOfFollowups': [number_of_followups],
    'DurationOfPitch': [duration_of_pitch],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'Occupation': [occupation],
    'Gender': [gender],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'Designation': [designation],
    'ProductPitched': [product_pitched],
    'Passport': [passport],
    'OwnCar': [own_car]
})

# Predict button
if st.button('Predict Purchase'):
    try:
        prediction_proba = model.predict_proba(input_data)[:, 1]
        prediction = model.predict(input_data)[0]

        st.subheader('Prediction Result:')
        if prediction == 1:
            st.success(f"The customer is LIKELY to purchase the package! (Probability: {prediction_proba[0]:.2f})")
        else:
            st.info(f"The customer is UNLIKELY to purchase the package. (Probability: {prediction_proba[0]:.2f})")

        st.write("**Features used for prediction:**")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are correctly filled and the model is loaded properly.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn.")
