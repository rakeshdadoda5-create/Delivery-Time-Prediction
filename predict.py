import joblib

model = joblib.load("delivery_model.pkl")

print("Model loaded successfully!")

import joblib
import pandas as pd

# Load trained model
model = joblib.load("delivery_model.pkl")

# Sample input
new_order = pd.DataFrame({
    "order_id": [1001],
    "store_id": [101],
    "rider_id": [201],
    "distance_km": [4.5],
    "traffic": [0],      # Low=0, Medium=1, High=2 (example)
    "weather": [0],      # Clear=0, Cloudy=1, Rain=2 (example)
    "batch_order": [1],  # No=0, Yes=1
    "experience": [5],
    "vehicle": [0],      # Bike=0, Scooter=1
    "rating": [4.8],
    "city": [0],         # Mumbai=0 (example)
    "prep_time": [15],
    "distance_per_prep": [4.5/15],
    "long_distance": [0]
})

prediction = model.predict(new_order)

print(f"Predicted Delivery Time: {prediction[0]:.2f} minutes")