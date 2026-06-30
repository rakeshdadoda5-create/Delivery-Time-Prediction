import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

orders = pd.read_csv("data/orders.csv")
riders = pd.read_csv("data/riders.csv")
stores = pd.read_csv("data/stores.csv")

df = orders.merge(riders, on="rider_id")
df = df.merge(stores, on="store_id")

print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.bar(df["order_id"], df["delivery_time"])

plt.title("Delivery Time by Order")
plt.xlabel("Order ID")
plt.ylabel("Delivery Time (Minutes)")

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_style("whitegrid")

# 1. Traffic vs Delivery Time
plt.figure(figsize=(8,5))
sns.boxplot(x="traffic", y="delivery_time", data=df)
plt.title("Traffic vs Delivery Time")
plt.show()

# 2. Weather vs Delivery Time
plt.figure(figsize=(8,5))
sns.boxplot(x="weather", y="delivery_time", data=df)
plt.title("Weather vs Delivery Time")
plt.show()

# 3. Vehicle Distribution
plt.figure(figsize=(6,6))
df["vehicle"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Vehicle Distribution")
plt.ylabel("")
plt.show()

# 4. Distance vs Delivery Time
plt.figure(figsize=(8,5))
sns.scatterplot(x="distance_km", y="delivery_time", data=df)
plt.title("Distance vs Delivery Time")
plt.show()

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# Feature Engineering

# Distance per preparation minute
df["distance_per_prep"] = df["distance_km"] / df["prep_time"]

# Long distance order
df["long_distance"] = (df["distance_km"] > 5).astype(int)

print("\nNew Features:")
print(df[["distance_km", "prep_time", "distance_per_prep", "long_distance"]].head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Copy dataframe
model_df = df.copy()

# Convert text columns into numbers
encoder = LabelEncoder()

model_df["traffic"] = encoder.fit_transform(model_df["traffic"])
model_df["weather"] = encoder.fit_transform(model_df["weather"])
model_df["batch_order"] = encoder.fit_transform(model_df["batch_order"])
model_df["vehicle"] = encoder.fit_transform(model_df["vehicle"])
model_df["city"] = encoder.fit_transform(model_df["city"])

# Features
X = model_df.drop("delivery_time", axis=1)

# Target
y = model_df["delivery_time"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("\nModel Results")
print("----------------------")
print("MAE :", mean_absolute_error(y_test, pred))
print("R2 Score :", r2_score(y_test, pred))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("\n===== Model Evaluation =====")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

import joblib

joblib.dump(model, "delivery_model.pkl")

print("Model saved successfully!")