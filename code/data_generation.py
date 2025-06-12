import pandas as pd
import numpy as np
import random

# This script generates a synthetic dataset for urban happiness index based on various factors
# such as noise levels, traffic density, green space area, air quality, and more.

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define synthetic city names
cities = [
    "Auckland", "Barcelona", "Berlin", "Copenhagen", "Dublin", "Helsinki", "Lisbon", "London",
    "Melbourne", "Montreal", "New York", "Oslo", "Paris", "Prague", "Rome", "San Francisco",
    "Stockholm", "Sydney", "Tokyo", "Toronto", "Vienna", "Warsaw", "Zurich"
]

# Define years and months
years = list(range(2014, 2024))  # 2014 to 2024
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Traffic density categories
traffic_levels = ["Low", "Medium", "High", "Very High"]

# Function to generate the data
def generate_urban_happiness_data(cities, years, months):
    records = []
    for city in cities:
        for year in years:
            for month in months:
                decibel = random.randint(45, 85)
                traffic = random.choices(traffic_levels, weights=[0.25, 0.35, 0.25, 0.15])[0]
                green_space = random.randint(10, 85)
                air_quality = random.randint(30, 150)
                happiness = round(
                    10 - 0.03 * decibel + 0.04 * green_space - 0.02 * air_quality
                    + random.uniform(-1.5, 1.5), 2
                )
                happiness = min(max(happiness, 1.0), 10.0)
                cost_living = random.randint(50, 130)
                healthcare = random.randint(60, 100)

                records.append({
                    "City": city,
                    "Month": month,
                    "Year": year,
                    "Decibel_Level": decibel,
                    "Traffic_Density": traffic,
                    "Green_Space_Area": green_space,
                    "Air_Quality_Index": air_quality,
                    "Happiness_Score": happiness,
                    "Cost_of_Living_Index": cost_living,
                    "Healthcare_Index": healthcare
                })
    return pd.DataFrame(records)

# Generate and save the dataset
df_synthetic = generate_urban_happiness_data(cities, years, months)
df_synthetic.to_csv("C:/Users/julia/OneDrive/Desktop/Projects/Project-City-Happiness-Index/dataset/urban_happiness_data.csv", index=False)

print("Synthetic dataset generated with shape:", df_synthetic.shape)
df_synthetic.head()