import pandas as pd
import random

cities = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Lucknow"
]

stores = []

for store_id in range(101, 121):
    city = random.choice(cities)
    prep_time = random.randint(8, 25)

    stores.append([
        store_id,
        city,
        prep_time
    ])

df = pd.DataFrame(
    stores,
    columns=[
        "store_id",
        "city",
        "prep_time"
    ]
)

df.to_csv("data/stores.csv", index=False)

print("✅ Stores dataset created successfully!")