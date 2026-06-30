import pandas as pd
import random

riders = []

vehicles = ["Bike", "Scooter"]

for rider_id in range(201, 261):
    experience = random.randint(1, 10)
    vehicle = random.choice(vehicles)
    rating = round(random.uniform(3.5, 5.0), 1)

    riders.append([
        rider_id,
        experience,
        vehicle,
        rating
    ])

df = pd.DataFrame(
    riders,
    columns=[
        "rider_id",
        "experience",
        "vehicle",
        "rating"
    ]
)

df.to_csv("data/riders.csv", index=False)

print("✅ Riders dataset created successfully!")