import pandas as pd
import random

orders = []

traffic = ["Low", "Medium", "High"]
weather = ["Clear", "Cloudy", "Rain"]

for i in range(1, 1001):
    store_id = random.randint(101, 120)
    rider_id = random.randint(201, 260)
    distance = round(random.uniform(1, 10), 1)

    t = random.choice(traffic)
    w = random.choice(weather)

    batch = random.choice(["Yes", "No"])

    delivery_time = (
        distance * 5
        + (10 if t == "High" else 5 if t == "Medium" else 0)
        + (5 if w == "Rain" else 0)
        + random.randint(0, 5)
    )

    orders.append([
        i,
        store_id,
        rider_id,
        distance,
        t,
        w,
        batch,
        int(delivery_time)
    ])

df = pd.DataFrame(
    orders,
    columns=[
        "order_id",
        "store_id",
        "rider_id",
        "distance_km",
        "traffic",
        "weather",
        "batch_order",
        "delivery_time",
    ],
)

df.to_csv("data/orders.csv", index=False)

print("✅ 1000 orders generated successfully!")