import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("delivery_model.pkl")

# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="centered"
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📋 Project Information")
st.sidebar.write("**Project:** Delivery Time Prediction")
st.sidebar.write("**Model:** LightGBM")
st.sidebar.write("**Developer:** Rakesh Dadoda")
st.sidebar.success("Status: Ready")

# ----------------------------
# Main Title
# ----------------------------
st.title("🚚 Delivery Time Prediction System")

st.markdown("""
Predict delivery time for Quick Commerce orders using a trained
**LightGBM Machine Learning Model**.
""")

# ----------------------------
# User Inputs
# ----------------------------
distance = st.number_input(
    "Distance (km)",
    min_value=1.0,
    max_value=20.0,
    value=5.0
)

traffic = st.selectbox(
    "Traffic",
    ["Low", "Medium", "High"]
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rain"]
)

batch = st.selectbox(
    "Batch Order",
    ["No", "Yes"]
)

experience = st.slider(
    "Rider Experience",
    1,
    10,
    5
)

vehicle = st.selectbox(
    "Vehicle",
    ["Bike", "Scooter"]
)

rating = st.slider(
    "Rider Rating",
    1.0,
    5.0,
    4.5
)

prep_time = st.number_input(
    "Store Preparation Time",
    min_value=5,
    max_value=30,
    value=15
)

city = st.selectbox(
    "City",
    ["Mumbai", "Delhi"]
)

# ----------------------------
# Encoding
# ----------------------------
traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

weather_map = {
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2
}

batch_map = {
    "No": 0,
    "Yes": 1
}

vehicle_map = {
    "Bike": 0,
    "Scooter": 1
}

city_map = {
    "Mumbai": 0,
    "Delhi": 1
}

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict Delivery Time"):

    input_df = pd.DataFrame({
        "order_id": [1001],
        "store_id": [101],
        "rider_id": [201],
        "distance_km": [distance],
        "traffic": [traffic_map[traffic]],
        "weather": [weather_map[weather]],
        "batch_order": [batch_map[batch]],
        "experience": [experience],
        "vehicle": [vehicle_map[vehicle]],
        "rating": [rating],
        "city": [city_map[city]],
        "prep_time": [prep_time],
        "distance_per_prep": [distance / prep_time],
        "long_distance": [1 if distance > 5 else 0]
    })

    prediction = model.predict(input_df)

    st.success("Prediction Completed Successfully!")

    st.subheader("📊 Input Summary")

    st.write(f"📍 Distance : {distance} km")
    st.write(f"🚦 Traffic : {traffic}")
    st.write(f"🌦 Weather : {weather}")
    st.write(f"📦 Batch Order : {batch}")
    st.write(f"🛵 Vehicle : {vehicle}")
    st.write(f"⭐ Rating : {rating}")
    st.write(f"👨‍💼 Experience : {experience} Years")
    st.write(f"🏪 Prep Time : {prep_time} Minutes")
    st.write(f"🏙 City : {city}")

    st.metric(
        label="🚚 Estimated Delivery Time",
        value=f"{prediction[0]:.2f} Minutes"
    )

st.markdown("---")
st.caption("Developed using Python, Streamlit, Pandas and LightGBM")