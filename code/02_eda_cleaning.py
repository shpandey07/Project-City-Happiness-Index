
# # Import necessary libraries
# import warnings
# warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output
# # Standard libraries
# import sys
# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")  # Ignore all warnings
# # Data manipulation and analysis libraries
# import pandas as pd
# import numpy as np
# # Visualization libraries
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import plotly.express as px
# # Fuzzy matching library for typo correction
# from rapidfuzz import process, fuzz
# # Time measurement library
# import time
# # File and directory management library
# import os
# # Set default plot style
# sns.set(style="whitegrid")  # Set seaborn style for plots
# # Set default matplotlib parameters for better readability
# plt.rcParams.update({
#     'figure.figsize': (10, 6),  # Default figure size
#     'axes.titlesize': 'large',   # Title size for axes
#     'axes.labelsize': 'medium',   # Label size for axes
#     'xtick.labelsize': 'small',   # X-tick label size
#     'ytick.labelsize': 'small',   # Y-tick label size
#     'legend.fontsize': 'medium',  # Legend font size
#     'grid.linestyle': '--'         # Grid line style
# })


# -*- coding: utf-8 -*-
"""
This script performs data loading, initial inspection, cleaning, and exploratory data analysis (EDA) on a dataset.
It includes functions for loading data, inspecting columns, cleaning categorical columns, handling missing values,
removing duplicates, detecting and visualizing outliers, and performing EDA with visualizations.
It also includes a main function to execute the data loading and initial inspection.
"""


import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rapidfuzz import process, fuzz
from sklearn.linear_model import LinearRegression


# DATA LOADING AND INITIAL INSPECTION

# Function for data loading  
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    Prints the loading time.
    """
    start_time = time.time()
    df = pd.read_csv(filepath)
    elapsed = (time.time() - start_time) / 60
    print(f"Dataset loaded in: {elapsed:.3f} mins")
    return df

# Function for nitial inspection of the DataFrame
def initial_inspection(df: pd.DataFrame) -> None:
    """
    Performs initial inspection of the DataFrame and prints summary information.
    """
    print(f"Data shape: {df.shape}")
    print(df.head(10))
    print(f"Data columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Data description:\n{df.describe(include='all')}")
    print(f"Unique values:\n{df.nunique()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Memory usage:\n{df.memory_usage(deep=True)}")


# Function to inspect categorical columns
def inspect_categorical_columns(df: pd.DataFrame) -> None:
    """
    Identifies categorical columns and prints their unique values.
    """
    categorical_cols = df.select_dtypes(include='object').columns
    print(f"Categorical columns: {categorical_cols.tolist()}")
    for col in categorical_cols:
        print(f"\nUnique values in '{col}':")
        print(df[col].unique())

# Function to check for column anomalies
def check_column_anomalies(df: pd.DataFrame) -> None:
    """
    Checks for columns with mixed types or numeric-like strings in object columns.
    Prints warnings if anomalies are found.
    """
    for col in df.columns:
        unique_types = df[col].map(type).value_counts()
        if len(unique_types) > 1:
            print(f"\n⚠️ Column '{col}' has mixed types:")
            print(unique_types)

        if df[col].dtype == 'object':
            numeric_like = df[col][df[col].apply(lambda x: str(x).isdigit())]
            if not numeric_like.empty:
                print(f"\n⚠️ Column '{col}' has numeric-like strings:")
                print(numeric_like)


# DATA CLEANING 

# Function to clean column names by stripping whitespace, converting to lowercase and adding underscores
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe column names by:
    - Stripping whitespace
    - Converting to lowercase
    - Replacing spaces with underscores
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with cleaned column names.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()              # Remove leading/trailing whitespace
        .str.lower()              # Convert to lowercase
        .str.replace(' ', '_')    # Replace spaces with underscores
    )
    return df

# Function to clean categorical columns with typo fixing and numeric mapping
def clean_categorical_columns(
    df: pd.DataFrame,
    canonical_values_map = None,
    numeric_to_category_map = None,
    threshold = 80,
    columns_to_skip_numeric_filter = None,
    verbose = True
) -> pd.DataFrame:
    """
    Cleans categorical columns by:
    - Lowercasing and stripping whitespace
    - Mapping numeric codes to string categories (if mapping provided)
    - Removing numeric-like strings only if no mapping exists and column not in skip list
    - Fixing typos using fuzzy matching against canonical valid values
    - Converting columns to categorical dtype

    Args:
        df (pd.DataFrame): Input DataFrame.
        canonical_values_map (dict): {col: [list_of_valid_values]} for fuzzy correction.
        numeric_to_category_map (dict): {col: {numeric_code_str: category_str}} for mapping numeric codes.
        threshold (int): Minimum fuzzy match score (0-100) to accept typo correction.
        columns_to_skip_numeric_filter (list): Columns to preserve numeric-like strings.
        verbose (bool): Whether to print value counts after cleaning.

    Returns:
        pd.DataFrame: Cleaned DataFrame with categorical columns processed.
    """
    df = df.copy()
    
    canonical_values_map = canonical_values_map or {}
    numeric_to_category_map = numeric_to_category_map or {}
    columns_to_skip_numeric_filter = columns_to_skip_numeric_filter or []

    # Select columns of object (string) type only
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    def fuzzy_correct(val, valid_values):
        if pd.isna(val):
            return val
        val_str = str(val).strip()
        
        # Exact match ignoring case
        for valid in valid_values:
            if val_str.lower() == valid.lower():
                return valid
        
        # Fuzzy match
        best_match = process.extractOne(val_str, valid_values, scorer=fuzz.ratio)
        if best_match and best_match[1] >= threshold:
            return best_match[0]
        
        # No good match found
        return val_str

    for col in cat_cols:
        # Lowercase and strip whitespace
        df[col] = df[col].astype(str).str.strip().str.lower()

        # Map numeric codes to categories if mapping provided for this column
        if col in numeric_to_category_map:
            mapping = numeric_to_category_map[col]
            df[col] = df[col].apply(lambda x: mapping.get(x, x))

        # Remove numeric-like strings only if no mapping for column and column not in skip list
        if col not in numeric_to_category_map and col not in columns_to_skip_numeric_filter:
            df[col] = df[col].apply(lambda x: np.nan if x.replace('.', '', 1).isdigit() else x)

        # Capitalize for better readability
        df[col] = df[col].str.title()

        # Fix typos with fuzzy matching if canonical values provided
        if col in canonical_values_map:
            valid_values = canonical_values_map[col]
            df[col] = df[col].apply(lambda x: fuzzy_correct(x, valid_values))

        # Convert to categorical type
        df[col] = df[col].astype('category')

        if verbose:
            print(f"\n✅ Cleaned column: '{col}'")
            print(df[col].value_counts(dropna=False))

    return df


# Function to handle missing values in numeric columns
def handle_missing_values(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Fills missing values in numeric columns using the column median.
    
    Returns:
      A cleaned copy of the DataFrame.    
    """
    df = df.copy()
    for col in numeric_cols:
        missing_before = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].median())
        print(f"✅ Filled {missing_before} missing values in '{col}' with median.")
    return df

# Function to remove duplicate rows from the DataFrame
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows and returns a new DataFrame.
    """
    df = df.copy()
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✅ Removed {duplicates_before} duplicate rows.")
    return df

# Function to detect outliers using the IQR method
def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detects outliers in a numeric column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to check for outliers.
        threshold (float): IQR multiplier to define outliers.
        
    Returns:
        pd.Series: Boolean Series indicating outliers (True if value is an outlier).
    """
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    if outliers.sum() > 0:
        print(f"⚠️ Detected {outliers.sum()} outliers in '{column}' using IQR method.")
    else:
        print(f"✅ No outliers detected in '{column}' using IQR method.")
    
    return outliers

# Function to visualize outliers in numeric columns using boxplots
def visualize_outliers(df: pd.DataFrame):
    """
    Visualizes outliers in numeric columns using boxplots.
    
    Args:
        df (pd.DataFrame): Input DataFrame with numeric columns.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist() ####
    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns found for outlier visualization.")
        return
    
    print(f"🔍 Visualizing outliers for numeric columns: {list(numeric_cols)}")
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot for {col}')
        plt.show()
        print(f"✅ Visualized outliers for column '{col}'\n")


# Function to remove outliers across all numeric columns using the IQR method
def remove_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()###
    outlier_mask = pd.Series(False, index=df.index)  # start with no outliers
    
    for col in numeric_cols:
        outliers = detect_outliers_iqr(df, col, threshold)
        outlier_mask = outlier_mask | outliers  # combine outlier masks
    
    print(f"Removing {outlier_mask.sum()} rows with outliers across all numeric columns.")
    df_cleaned = df[~outlier_mask].copy()
    return df_cleaned


# Function to add a 'period' column and sort the DataFrame by date
def add_period_and_sort(df):
    """
    Adds a 'period' column (monthly) from the 'date' column and sorts the DataFrame by 'date'.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a 'date' column.
    
    Returns:
        pd.DataFrame: Modified DataFrame with 'period' column and sorted by 'date'.
    """
    df = df.copy()
    
    if 'date' in df.columns:
        df['period'] = df['date'].dt.to_period('M')
        df = df.sort_values(by='date')
        print("📅 'period' column added using 'date' column (monthly granularity).")
        print("📊 DataFrame sorted by 'date'.")
    else:
        print("⚠️ 'date' column not found. Skipping period creation and sorting.")
    
    return df


# EXPLORATORY DATA ANALYSIS (EDA)

# Function to run the EDA pipeline
def eda_pipeline(df):
    """
    Perform an exploratory data analysis (EDA) on the given DataFrame.
    
    This function provides a comprehensive overview of the dataset by:
    - Displaying basic information about the DataFrame.
    - Showing descriptive statistics.
    - Visualizing the distribution of numeric columns.
    - Visualizing outliers using boxplots for numeric columns.
    - Visualizing counts for categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze.
    
    Returns:
        None: The function displays output and plots but does not return a value.
    """

    print("=== Starting EDA Pipeline ===\n")
    
    df = df.copy() 
    # Create output directory for EDA plots
    parent_dir = 'output'
    output_dir = os.path.join(parent_dir, 'eda_output')
    print(f"\nOutput directory for EDA plots: {output_dir}")
    # Create output directory if it doesn't exist   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    # 1. Initial Data Overview
    print("Data shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())
    
    # 2. Distribution Analysis
    print("\n--- Distribution Analysis ---")
    numeric_cols = df.select_dtypes(include='number').columns.tolist() 
    cat_cols = df.select_dtypes(include='category').columns.tolist()
    
    # Histograms and KDE for numeric columns
    for col in numeric_cols:
        print(f"\nHistogram and KDE for '{col}':")
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.xlabel(col, fontsize=6)
        plt.ylabel('Value', fontsize=6)
        plt.title(f'Distribution of {col}')
        # plt.show()
        plt.tight_layout()  # <-- To fix label cutoff
        plt.savefig(f"{output_dir}/histogram_{col}.png", dpi=300)
        plt.close()
       
    
    # Boxplots for numeric columns
    for col in numeric_cols:
        print(f"\nBoxplot for '{col}':")
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col])
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        plt.xlabel(col, fontsize=4)
        plt.ylabel('Value', fontsize=4)
        plt.title(f'Boxplot of {col}')
        # plt.show()
        plt.tight_layout()  # <-- To fix label cutoff
        plt.savefig(f"{output_dir}/boxplot_{col}.png", dpi=300)
        plt.close()
        
    
    # 3. Correlation Analysis
    print("\n--- Correlation Analysis ---")
    corr = df[numeric_cols].corr()
    print("\nCorrelation matrix:\n", corr)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.xticks(fontsize=7, rotation=30)
    plt.yticks(fontsize=8)
    plt.xlabel(col, fontsize=8)
    plt.ylabel('Value', fontsize=8)
    plt.title('Correlation Matrix')
    # plt.show()
    plt.tight_layout()  # <-- To fix label cutoff
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300)
    plt.close()

    # Pairplot for numeric variables (can be heavy for many columns)
    print("\n--- Pairplot for Numeric Variables ---")    
    if len(numeric_cols) <= 10:  # Limit to 10 numeric columns for pairplot
        print(f"Pairplot will be generated for {len(numeric_cols)} numeric columns.")
        print("\nPairplot for numeric variables:")
        pairplot_fig = sns.pairplot(df[numeric_cols])
        
        # Adjust font size of tick labels for all axes
        for ax in pairplot_fig.axes.flatten():
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
        
        pairplot_fig.fig.savefig(f"{output_dir}/pairplot_numeric.png", dpi=300)
        plt.close(pairplot_fig.fig)
    else:
        print("Skipping pairplot (too many numeric columns).")


    # 4. Time Series Analysis (Monthly) --- keep updating this

    if 'date' in df.columns:
        print("\n Time Series Analysis (Monthly Aggregates):")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')

        def plot_time_series_monthly(df, column, window=3):  # window = months for rolling mean
            """
            Plots a time series of a numeric column with monthly aggregation and rolling mean.
            Args:
                df (pd.DataFrame): DataFrame containing the data.
                column (str): Column name to plot.
                window (int): Number of months for rolling mean.
            """

            temp = df[['date', column]].dropna().copy()
            temp['date'] = pd.to_datetime(temp['date'])
            temp = temp.set_index('date').sort_index()

            monthly = temp.resample('M').mean()
            monthly['rolling'] = monthly[column].rolling(window=window).mean()

            if monthly[[column, 'rolling']].dropna(how='all').empty:
                print(f"⚠️ Skipping '{column}' — not enough data for monthly aggregation.")
                return

            plt.figure(figsize=(12, 4))
            plt.plot(monthly.index, monthly[column], label='Monthly Mean', alpha=0.5)
            plt.plot(monthly.index, monthly['rolling'], label=f'{window}-Month Rolling Mean', color='red')
            plt.title(f"Time Series of '{column}' (Monthly) with Rolling Mean")
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.xlabel(col, fontsize=6)
            plt.ylabel('Value', fontsize=6)

            ax = plt.gca()
            # Show ticks every 3 months to reduce clutter
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.tight_layout()  # <-- To fix label cutoff
            plt.savefig(f"{output_dir}/timeseries_{column}.png", dpi=300)
            plt.close()
            print(f"✅ Plotted monthly time series for '{column}'")

        # Select top 3 numeric columns by std dev
        numeric_cols = df.select_dtypes(include='number').columns.tolist() #####
        top3_cols = df[numeric_cols].std().sort_values(ascending=False).head(3).index
        for col in top3_cols:
            plot_time_series_monthly(df, col)

    else:
        print("⚠️ No 'date' column found. Skipping time series analysis.")


    # 5. Category-Based EDA
    print("\n--- Categorical Feature Analysis ---")
    for col in cat_cols:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts())
        plt.figure(figsize=(8,4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.xlabel(col, fontsize=6)
        plt.ylabel('Value', fontsize=6)
        plt.title(f'Distribution of {col}')
        # plt.show()
        plt.tight_layout()  # <-- To fix label cutoff
        plt.savefig(f"{output_dir}/Distribution of {col}.png", dpi=300)
        
        # Numeric distribution by category if numeric columns exist
        if numeric_cols:
            for num_col in numeric_cols:
                plt.figure(figsize=(8,4))
                sns.boxplot(x=df[col], y=df[num_col])
                plt.title(f'{num_col} distribution by {col}')
                plt.xticks(fontsize=6, rotation=30)
                plt.yticks(fontsize=6)
                plt.xlabel(col, fontsize=6)
                plt.ylabel('Value', fontsize=6)
                # plt.show()
                filename = f"{output_dir}/boxplot_{num_col}_by_{col}.png"
                plt.tight_layout()  # <-- To fix label cutoff
                plt.savefig(filename, dpi=300)
                plt.close()
        print(f"✅ Completed analysis for categorical column '{col}'")
        
    # Print categorical and numeric columns
    print("\n--- Categorical and Numeric Columns ---")
    print("\nCategorical columns:", cat_cols)
    print("Numeric columns:", numeric_cols)
    print("Running categorical feature analysis on:", cat_cols)
    print("Running numeric feature analysis on:", numeric_cols)
    print("Data preview:\n", df.head())

    # Save EDA results to CSV
    eda_results_path = os.path.join(output_dir, 'eda_results.csv')
    df.to_csv(eda_results_path, index=False)
    print(f"\nEDA results saved to: {eda_results_path}")


    # Summary of findings
    print("\n--- Summary of Findings ---")
    print("1. Data loaded successfully with shape:", df.shape)
    print("2. Categorical columns inspected and cleaned.")
    print("3. Numeric columns analyzed with distributions and outliers visualized.")
    print("4. Time series analysis performed on monthly aggregates.")
    print("5. Categorical features analyzed with counts and distributions.")
    print("6. Custom KPIs calculated for happiness scores by year.")
    print("7. EDA results saved to CSV file.")
    print("8. EDA plots saved to output directory:", output_dir)
    
        
    print("\n=== EDA Pipeline Completed ===")

# Function to run KPI analysis on the DataFrame
def run_kpi_analysis(df):
    # extract year-wise happiness KPIs, plots etc.
    """    Perform KPI analysis on the DataFrame to extract year-wise happiness KPIs and generate plots.
    This function calculates the average happiness score per year, aggregates statistics, and generates visualizations.
    Args:
        df (pd.DataFrame): Input DataFrame with necessary columns.
        output_dir (str): Directory to save the generated plots.
    Returns:
        None: The function saves plots to the specified directory.
    """

    # Create output directory for kpi plots
    parent_dir = 'output'
    output_dir = os.path.join(parent_dir, 'kpi_output')
    print(f"\nOutput directory for KPI plots: {output_dir}")
    # Create output directory if it doesn't exist   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")


     # 1. Custom KPIs & Insights
    print("\n--- Custom KPIs & Insights ---")

    overall_avg = None

    if 'year' in df.columns and 'happiness_score' in df.columns:
        happiness_by_year = df.groupby('year')['happiness_score'].mean().reset_index()
        # 1.1. Average happiness score per year
        print("Average happiness score per year:\n", happiness_by_year)
        plt.figure(figsize=(8,6))
        plt.bar(happiness_by_year['year'], happiness_by_year['happiness_score'], color='skyblue')
        plt.title('Average Happiness Score by Year')
        plt.xlabel('Year', fontsize=6)
        plt.ylabel('Average Happiness Score',fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.xticks(happiness_by_year['year'])  # ensure all years show
        plt.tight_layout()
        plt.savefig(f"{output_dir}/kpi_happiness_score_by_year.png", dpi=300)
        plt.close()


        happiness_by_year.to_csv(f"{output_dir}/happiness_summary_by_year.csv", index=False)
        print(f"Saved happiness summary by year to: {output_dir}/happiness_summary_by_year.csv")

    if 'year' in df.columns and 'happiness_score' in df.columns:

        # 1.2. Aggregate stats per year: mean, median, std, 25th and 75th percentile
        happiness_stats = df.groupby('year')['happiness_score'].agg([
            'mean', 'median', 'std',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ]).reset_index()

        happiness_stats.columns = ['year', 'mean', 'median', 'std', '25th_percentile', '75th_percentile']
        print("\nHappiness stats by year:\n", happiness_stats)

        # 1.3. Year-over-Year % Change in mean happiness
        happiness_stats['YoY_pct_change'] = happiness_stats['mean'].pct_change() * 100
        print("\nYear-over-Year % change in average happiness:\n", happiness_stats[['year', 'YoY_pct_change']])

        # 1.4. Plot Mean and Median Happiness per Year
        plt.figure(figsize=(8,6))
        plt.plot(happiness_stats['year'], happiness_stats['mean'], marker='o', label='Mean Happiness')
        plt.plot(happiness_stats['year'], happiness_stats['median'], marker='x', label='Median Happiness')
        plt.title('Mean and Median Happiness Score by Year')
        plt.xlabel('Year')
        plt.ylabel('Happiness Score')
        plt.xticks(happiness_stats['year'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mean_median_happiness_by_year.png", dpi=300)
        plt.close()

        # 1.5. Boxplot: Happiness distribution by year
        plt.figure(figsize=(10,6))
        sns.boxplot(x='year', y='happiness_score', data=df)
        plt.title('Happiness Score Distribution per Year')
        plt.xlabel('Year')
        plt.ylabel('Happiness Score')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/happiness_distribution_per_year.png", dpi=300)
        plt.close()

        # 1.6. Top 5 happiest years by mean happiness
        top_5_years = happiness_stats.sort_values(by='mean', ascending=False).head(5)
        print("\nTop 5 Happiest Years (by Mean Happiness):\n", top_5_years[['year', 'mean']])

        # 1.7. Overall average happiness score (2014-2024)
        overall_avg = df['happiness_score'].mean()
        print(f"\nOverall Average Happiness Score (2014-2024): {overall_avg:.3f}")

    return {
        'happiness_by_year': happiness_by_year,
        'happiness_stats': happiness_stats,
        'overall_avg_happiness': overall_avg,
        'top_5_years': top_5_years
    }

# Function to run regression analysis on the DataFrame
def run_multiple_regression_analysis(df, outcome='happiness_score', output_dir='eda_output'):
    """
    Run regression plots for multiple predictors against the outcome variable.

    Args:
        df (pd.DataFrame): Input data.
        outcome (str): The outcome (Y-axis) column name. Default is 'happiness_score'.
        output_dir (str): Directory to save the plots. Default is 'eda_output'.

    Returns:
        None
    """
    print("\n--- Regression Analysis ---")

     # Create output directory for regression plots
    parent_dir = 'output'
    output_dir = os.path.join(parent_dir, 'regression_output')
    print(f"\nOutput directory for KPI plots: {output_dir}")
    # Create output directory if it doesn't exist   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    traffic_mapping = {
    "Very Low": 20,
    "Low": 40,
    "Medium": 60,
    "High": 80,
    "Very High": 100
    }

    df['traffic_density_numeric'] = df['traffic_density'].map(traffic_mapping).astype(float)

    regression_pairs = [
    ("green_space_area", "Green Space Area"),
    ("air_quality_index", "Air Quality Index"),
    ("traffic_density_numeric", "Traffic Density"),
    ("decibel_level", "Decibel Level"),
    ("healthcare_index", "Healthcare Index"),
    ("cost_of_living_index", "Cost of Living Index"),
    ]

    for predictor, label in regression_pairs:
        if predictor in df.columns and outcome in df.columns:
            print(f"Running regression analysis: {label} vs {outcome.replace('_', ' ').title()}")

            sns.lmplot(
                x=predictor,
                y=outcome,
                hue='region' if 'region' in df.columns else None,
                data=df,
                aspect=2,
                height=6
            )

            plt.title(f"Regression Analysis: {outcome.replace('_', ' ').title()} vs {label}")
            plt.xlabel(label)
            plt.ylabel(outcome.replace('_', ' ').title())
            # plt.xticks(rotation=30)
            plt.tight_layout()

            filename = f"{output_dir}/regression_{outcome}_vs_{predictor}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            print(f"⚠️ Skipping: Missing '{predictor}' or '{outcome}'")
    print("✅ Regression analysis completed. Plots saved to:", output_dir)


# FEATURE ENGINEERING AND DATA TRANSFORMATION

# Function to perform feature engineering on the DataFrame
def feature_engineering(df):
    """
    Perform feature engineering on the DataFrame to create new features and composite scores.

    Adds:
    - 'infrastructure_score': Based on traffic density, decibel level, and healthcare index.
    - 'environment_score': Based on green space and air quality.

    Args:
        df (pd.DataFrame): Input DataFrame with required columns.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    # Map categorical traffic density to numeric values
    traffic_mapping = {
        "Very Low": 20,
        "Low": 40,
        "Medium": 60,
        "High": 80,
        "Very High": 100
    }

    # Apply mapping
    df['traffic_density_numeric'] = (
    df['traffic_density']
    .map(traffic_mapping)
    .astype(float)  # Convert to float for calculations
    )

    # Handle missing or unmapped traffic values
    if df['traffic_density_numeric'].isnull().any():
        print("⚠️ Warning: Some traffic_density values could not be mapped and are set to NaN.")

    # Create infrastructure_score
    df['infrastructure_score'] = (
        (100 - df['traffic_density_numeric']) +                         
        (100 - df['decibel_level']) +
        df['healthcare_index']
    ) / 3
    # Create environment_score  
    df['environment_score'] = (
        df['green_space_area'] + (100 - df['air_quality_index'])  # assuming lower AQI = better air
    ) / 2

    return df

# BUSINESS QUESTION ANALYSIS

# Analysis of business questions based on the dataset

def generate_business_question_plots(df, output_dir="business_output"):
    """
    Generate plots to answer business questions based on the dataset.
    This function creates visualizations to explore relationships between various features and the happiness score.

    Args:
        df (pd.DataFrame): Input DataFrame with necessary columns.
        output_dir (str): Directory to save the generated plots.

    Returns:
        None: The function saves plots to the specified directory.
    """
    # Ensure the DataFrame is a copy to avoid modifying the original
    df = df.copy()

    # Create output directory for Business Analysis plots#
    parent_dir = 'output'
    output_dir = os.path.join(parent_dir, 'business_output')
    # Create output directory if it doesn't exist
    print(f"\nOutput directory for Business Analysis plots: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    print("\nGenerating Business Question Plots...")

    # Map categorical traffic density to numeric values
    traffic_mapping = {
        "Very Low": 20,
        "Low": 40,
        "Medium": 60,
        "High": 80,
        "Very High": 100
    }
    df['traffic_density_numeric'] = df['traffic_density'].map(traffic_mapping).astype(float)

    # Drop rows with missing values in the numeric traffic density for regression
    df = df.dropna(subset=['traffic_density_numeric'])

    # Q1: Correlation Matrix - Drivers of Happiness
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr[['happiness_score']].sort_values(by='happiness_score', ascending=False),
                annot=True, cmap='coolwarm')
    plt.title('Correlation with Happiness Score')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/q1_correlation_matrix.png", dpi=300)
    plt.close()

    # Regression helper function
    def plot_regression(x_var, label, q_name):
        if x_var in df.columns:
            sns.lmplot(x=x_var, y='happiness_score',
                       hue='region' if 'region' in df.columns else None,
                       data=df, aspect=2, height=6)
            plt.title(f'Happiness vs. {label}')
            plt.xlabel(label)
            plt.ylabel("Happiness Score")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{q_name}_{x_var}_regression.png", dpi=300)
            plt.close()
        else:
            print(f"⚠️ Column '{x_var}' not found for regression plot.")

    # Q1 continued: Regression Plots - use numeric traffic density column
    for col, label in [('green_space_area', 'Green Space Area'),
                       ('air_quality_index', 'Air Quality Index'),
                       ('traffic_density_numeric', 'Traffic Density'),  # <-- use numeric here
                       ('decibel_level', 'Decibel Level')]:
        plot_regression(col, label, 'q1')

    # Q2: Infrastructure Score vs Happiness (Scatter + Residual)
    if 'infrastructure_score' in df.columns:
        sns.scatterplot(data=df, x='infrastructure_score', y='happiness_score',
                        hue='region' if 'region' in df.columns else None)
        plt.title("Q2: Happiness vs. Infrastructure Score")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q2_infrastructure_vs_happiness.png", dpi=300)
        plt.close()

        # Residuals
        model = LinearRegression()
        X = df[['infrastructure_score']].dropna()
        y = df.loc[X.index, 'happiness_score']
        model.fit(X, y)
        df.loc[X.index, 'predicted_happiness'] = model.predict(X)
        df.loc[X.index, 'residual'] = df.loc[X.index, 'happiness_score'] - df.loc[X.index, 'predicted_happiness']

        sns.scatterplot(data=df.loc[X.index], x='infrastructure_score', y='residual',
                        hue='region' if 'region' in df.columns else None)
        plt.axhline(0, linestyle='--', color='gray')
        plt.title("Q2: Residuals (Underperformers in Happiness)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q2_residuals.png", dpi=300)
        plt.close()

    # Q3: Happiness over Time
    if 'year' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='year', y='happiness_score', hue='city' if 'city' in df.columns else None, legend=False)
        plt.title("Q3: Happiness Trends Over Time (by City)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q3_happiness_trends.png", dpi=300)
        plt.close()

    # Q4: Green & Clean Impact
    if 'region' in df.columns and 'environment_score' in df.columns:
        sns.boxplot(data=df, x='region', y='environment_score')
        plt.title("Q4: Environment Score Distribution by Region")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q4_environment_boxplot.png", dpi=300)
        plt.close()

        plot_regression('environment_score', 'Environment Score', 'q4')

    # Q5: Cost vs Happiness
    plot_regression('cost_of_living_index', 'Cost of Living Index', 'q5')

    # Q6: Traffic & Noise Impact - use numeric traffic density
    plot_regression('traffic_density_numeric', 'Traffic Density', 'q6')
    plot_regression('decibel_level', 'Decibel Level', 'q6')

    # Q7: Healthcare
    plot_regression('healthcare_index', 'Healthcare Index', 'q7')

    # Q8: Best & Worst Cities
    if 'city' in df.columns:
        city_avg = df.groupby('city')['happiness_score'].mean().sort_values()
        plt.figure(figsize=(10, 6))
        city_avg.head(5).plot(kind='barh', color='crimson')
        plt.title("Q8: Bottom 5 Happiest Cities")
        plt.xlabel("Average Happiness Score")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q8_bottom5_cities.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        city_avg.tail(5).plot(kind='barh', color='seagreen')
        plt.title("Q8: Top 5 Happiest Cities")
        plt.xlabel("Average Happiness Score")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/q8_top5_cities.png", dpi=300)
        plt.close()

    print("✅ Business question plots generated and saved to:", output_dir)


# Main function to execute the data loading and initial inspection
def main():
    data_path = 'C:/Users/julia/OneDrive/Desktop/Projects/Project-City-Happiness-Index/dataset/urban_happiness_data.csv'
    
    # Data loading and initial inspection
    print("#" * 88 + "\n" +
          "START OF DATA LOADING AND INITIAL INSPECTION!\n" +
          "#" * 88)
    total_start = time.time()
    
    df = load_data(data_path)
    print("Columns after loading:", df.columns.tolist())

    # Initial inspection
    print("\nInitial inspection of the DataFrame:")
    initial_inspection(df)
    check_column_anomalies(df)
    inspect_categorical_columns(df)
    
    print("\n✅ Initial inspection completed. DataFrame is ready for cleaning and EDA.")

    # End the timer to measure the run time of the code
    total_elapsed = (time.time() - total_start) / 60
    print(f"\nTotal run time: {total_elapsed:.3f} mins")

    print("\nEND OF DATA LOADING AND INITIAL INSPECTION!\n" +
          "#" * 88)
    
    # Data cleaning and EDA
    print("#" * 88 + "\n" +
          "START OF DATA CLEANING!\n" +
          "#" * 88)
    
    # Clean column names
    df = clean_column_names(df)    
    print("Columns after cleaning:", df.columns.tolist())
    print("Column names are cleaned by stripping whitespace and converting to lowercase!!")

    canonical_values = {
    'month': ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'],
    'traffic_density': ['Low', 'Medium', 'High', 'Very High']
    }

    numeric_to_category = {
        'month': {
            '1': 'January', '2': 'February', '3': 'March', '4': 'April',
            '5': 'May', '6': 'June', '7': 'July', '8': 'August',
            '9': 'September', '10': 'October', '11': 'November', '12': 'December'
        }
    }

    # Call the cleaning function for categorical columns
    df = clean_categorical_columns(
        df,
        canonical_values_map=canonical_values, 
        numeric_to_category_map=numeric_to_category,
        columns_to_skip_numeric_filter=[],  # We don't skip numeric filter on 'month' because mapping handles it
        verbose=True
    )

    df['date'] = pd.to_datetime(df['month'].astype(str).str.strip() + ' ' + df['year'].astype(str).str.strip())
    print(f"\n✅ Converted 'month' and 'year' columns to datetime in 'date' column.")
    print(df[['month', 'year', 'date']].head(5))

    # Print column names after cleaning
    print("Columns after cleaning:", df.columns.tolist())
    # Display the data types of each column after cleaning
    print("\nData types after cleaning:")
    print(f"Data types:\n{df.dtypes}")

    # Handle missing values
    numeric_cols_to_fill = ['green_space_area', 'air_quality_index', 'healthcare_index']
    df = handle_missing_values(df, numeric_cols_to_fill)

    # Remove duplicates
    df = remove_duplicates(df)   # Dont remove duplicates for now, as it may be needed for EDA or they might be important
    print("\n✅ Duplicates removed from the DataFrame.")

    # Detect and remove outliers using IQR method
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()   #####
    for col in numeric_cols:
        outliers = detect_outliers_iqr(df, col)
    
    # Visualize outliers in numeric columns
    # visualize_outliers(df)

    # df = remove_outliers_iqr(df)
    # print(f"\n✅ Outliers removed from numeric columns using IQR method.")

    # Add 'period' column and sort by 'date'
    print("\nAdding 'period' column and sorting by 'date'...")
    df = add_period_and_sort(df)

    print("\nEND OF DATA CLEANING INSPECTION!\n" +
          "#" * 88)
    
    print("#" * 88 + "\n" +
          "START OF EXPLORATORY DATA ANALYSIS (EDA)!\n" +
          "#" * 88)
  
    # Run the EDA pipeline
    print("\nRunning EDA pipeline...")
    eda_pipeline(df)
    
    print("\nEND OF EXPLORATORY DATA ANALYSIS (EDA)!\n" +
          "#" * 88)
    
    run_kpi_analysis(df)    
    run_multiple_regression_analysis(df)
    feature_engineering(df)
    generate_business_question_plots(df)

    # Print final DataFrame info
    print("\nFinal DataFrame info:")    
    print(df.info())
    print(f"Data shape: {df.shape}")    
    print(f"Data columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")  
    print(f"Data description:\n{df.describe(include='all')}")
    print(f"Unique values:\n{df.nunique()}")    
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Outliers: {df.isna().sum().sum()}")
    print(f"Memory usage:\n{df.memory_usage(deep=True)}")
    print("\nEND OF DATA CLEANING, EDA, Fe!\n" +
          "#" * 88)

    print("\nEND OF DATA CLEANING, EDA, FEATURE ENGINEERING, REGRESSION ANALYSIS AND BUSINESS QUESTIONS!\n" +
          "#" * 88)
    
    
    # End the timer to measure the run time of the code
    total_elapsed = (time.time() - total_start) / 60    
    print(f"\nTotal run time: {total_elapsed:.3f} mins")
    print("#" * 88 + "\n" +
          "END OF SCRIPT EXECUTION!\n" +
          "#" * 88)
    

# Run the main function if this script is executed
if __name__ == "__main__":
    print("#" * 88 + "\n" +
          "START OF SCRIPT EXECUTION!\n" +
          "#" * 88)
    main()
