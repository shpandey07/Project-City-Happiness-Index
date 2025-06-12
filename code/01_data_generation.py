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
                # Add noise in month format
                if random.random() < 0.05:
                    month_val = str(random.randint(1, 12))  # e.g. "3" instead of "March"
                elif random.random() < 0.03:
                    month_val = month.upper()  # e.g. "MARCH"
                else:
                    month_val = month

                # Inject typos or casing issues in traffic
                traffic = random.choices(traffic_levels, weights=[0.25, 0.35, 0.25, 0.15])[0]
                if random.random() < 0.05:
                    traffic = traffic.lower()
                elif random.random() < 0.02:
                    traffic = traffic.replace("High", "Hgh")  # typo

                # Introduce nulls in some columns
                green_space = random.randint(10, 85)
                if random.random() < 0.03:
                    green_space = np.nan

                air_quality = random.randint(30, 150)
                if random.random() < 0.02:
                    air_quality = np.nan

                # Add outliers
                decibel = random.randint(45, 85)
                if random.random() < 0.01:
                    decibel = random.randint(90, 120)  # extreme noise

                cost_living = random.randint(50, 130)
                if random.random() < 0.01:
                    cost_living = random.randint(200, 300)  # extreme cost

                healthcare = random.randint(60, 100)
                if random.random() < 0.02:
                    healthcare = np.nan

                # Happiness Score (with randomness + clipping)
                happiness = round(
                    10 - 0.03 * decibel + 0.04 * (green_space if not pd.isna(green_space) else 50)
                    - 0.02 * (air_quality if not pd.isna(air_quality) else 90)
                    + random.uniform(-1.5, 1.5), 2
                )
                happiness = min(max(happiness, 1.0), 10.0)

                records.append({
                    "City": city,
                    " month": month_val,  # messy name
                    "Year ": year,  # messy name
                    "decibel_Level": decibel,  # inconsistent casing
                    "Traffic Density": traffic,  # inconsistent name
                    "Green_Space_area": green_space,
                    "Air Quality Index": air_quality,
                    "Happiness_Score ": happiness,  # extra space
                    "Cost_of_living_index": cost_living,
                    "Healthcare_Index": healthcare
                })

    df = pd.DataFrame(records)

    # Add 1% duplicate rows
    dupes = df.sample(frac=0.01, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)

    return df

# Generate and save the dataset
df_synthetic = generate_urban_happiness_data(cities, years, months)
df_synthetic.to_csv("C:/Users/julia/OneDrive/Desktop/Projects/Project-City-Happiness-Index/dataset/urban_happiness_data.csv", index=False)

print("Synthetic dataset generated with shape:", df_synthetic.shape)
df_synthetic.head()