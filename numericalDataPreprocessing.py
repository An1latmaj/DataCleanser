import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

path=input("Enter the path of the CSV file: ")

try:
    df = pd.read_csv(path)
    print("File loaded successfully. Here's a preview of your data:")
    print(df.head())
except FileNotFoundError:
    print("Please enter a valid CSV file path.")
    exit()

# Allow the user to select columns for numerical and categorical processing
print("\nHere are the columns in your dataset:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Get numerical and categorical columns from the user
num_cols_input = input("Enter the column names or indices (comma-separated) for numerical features: ")
cat_cols_input = input("Enter the column names or indices (comma-separated) for categorical features: ")


# Convert user input into column lists
def get_columns(input_string, all_columns):
    columns = []
    for item in input_string.split(","):
        item = item.strip()
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(all_columns):
                columns.append(all_columns[idx])
            else:
                print(f"Invalid index: {idx}")
        else:
            if item in all_columns:
                columns.append(item)
            else:
                print(f"Invalid column name: {item}")
    return columns


numerical_features = get_columns(num_cols_input, df.columns.tolist())
categorical_features = get_columns(cat_cols_input, df.columns.tolist())

if not numerical_features and not categorical_features:
    print("No valid columns selected for preprocessing. Exiting.")
    exit()

print(f"\nNumerical features selected: {numerical_features}")
print(f"Categorical features selected: {categorical_features}")

# Numerical feature transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())  # Standardize features
])

# Categorical feature transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])

# Combine preprocessing steps into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#Create a complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

#Apply the pipeline to your dataset
print("\nProcessing the data...")
cleaned=pipeline.fit_transform(df)
print("Data processed successfully.")

# Preview the transformed data
print("\nOriginal DataFrame:")
print(df.head())
print("\nProcessed Data (transformed):")
print(cleaned)
