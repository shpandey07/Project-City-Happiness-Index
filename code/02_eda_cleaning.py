# Import libraries
import calendar
import pandas as pd
import numpy as np
import seaborn as sns   
import matplotlib.pyplot as plt

# Load the dataset  
df = pd.read_csv('C:/Users/julia/OneDrive/Desktop/Projects/Project-City-Happiness-Index/dataset/urban_happiness_data.csv')

# Standardize column names (remove spaces, convert to lowercase, replace spaces with underscores)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convert 'month' and 'year' columns to datetime format
df['date'] = pd.to_datetime(df['month'] + ' ' + df['year'].astype(str))

# Convert ´traffic_density´ to categorical type
df['traffic_density'] = df['traffic_density'].astype('category')

# Check column names
print(df.columns.tolist())

# Display the first few rows of the dataset
print(df.head())

# Display the shape of the dataset  
print(df.shape)

# Display the data info
df.info()

# Display the data types of the columns
# print(df.dtypes)

# Display the summary statistics for numeric columns (to detect outliers (very large/small values), unexpected distributions, suspicious zeros or extreme values, suspicious zeros or extreme values)
print(df.describe(include='number'))

# # Display the summary statistics for categorical columns
print(df.describe(include='object'))

# Display the number of missing values and duplicates in each column
print("Null values:\n", df.isnull().sum())
print("Duplicate values:", df.duplicated().sum())

# Check unique values for categorical columns
for col in ['city', 'traffic_density', 'month']:
    print(f"{col} → {df[col].unique()}")

# Visualize the data

# Plotting the distribution of numeric columns (histograms with KDE)
numeric_cols = ['decibel_level', 'green_space_area', 'air_quality_index',
                'happiness_score', 'cost_of_living_index', 'healthcare_index']

# for col in numeric_cols:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[col], bins=30, kde=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()

# Plotting outliers in numeric columns (box plots)
# for col in numeric_cols:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=df[col])
#     plt.title(f'Box plot of {col}')
#     plt.show()

# # Plotting categorical columns (bar plots)S
categorical_cols = ['city', 'traffic_density', 'month']
# for col in categorical_cols:
#     plt.figure(figsize=(8, 4))
#     sns.countplot(y=df[col], order=df[col].value_counts().index)
#     plt.title(f'Count of {col}')
#     plt.show()

# Visualize correlations to discover relationships between variables (escpecially happiness_score) make the text on x-axis slant
# plt.figure(figsize=(10, 6))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

# Plotting scatter plots for pairwise patterns
features_to_compare = ['decibel_level', 'green_space_area', 'air_quality_index',
                       'cost_of_living_index', 'healthcare_index']

# for col in features_to_compare:
#     plt.figure(figsize=(6, 4))
#     sns.scatterplot(data=df, x=col, y='happiness_score', hue='traffic_density')
#     plt.title(f'Happiness Score vs. {col}')
#     plt.show()


# Plotting time trends in happiness score over time
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df, x='date', y='happiness_score', hue='city', legend=False)
# plt.title("Happiness Score Trends Over Time")
# plt.show()

#  Loop through columns with mixed types
for col in df.columns:
    print(f"{col}: {df[col].unique()[:5]}")

# Display the first few rows of the dataset after cleaning
# print(df.head())