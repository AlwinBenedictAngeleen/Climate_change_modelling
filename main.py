# CO2 Emissions Prediction Project
# Real Machine Learning Code for Mac

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

print("=== CO2 Emissions Prediction Project ===\n")
print("This project uses REAL machine learning to predict CO2 emissions.")
print("It will load your data, clean it, train a model, and show results.\n")

# 1. Check for the data file
data_file_path = './data/train.csv'
print("1. Checking for the data file...")
if not os.path.exists(data_file_path):
    print("   ERROR: The data file 'train.csv' was not found in the ./data/ folder.")
    print("   Please make sure you have:")
    print("   1. Created a folder named 'data' here.")
    print("   2. Downloaded the file from Kaggle:")
    print("      https://www.kaggle.com/competitions/playground-series-s3e20/data")
    print("   3. Placed the file inside the 'data' folder.")
    print("\n   The program cannot continue without the data.")
    exit()
else:
    print("   ✅ Data file found! Proceeding...")

# 2. Load the data
print("\n2. Loading the dataset (this may take a moment)...")
try:
    df = pd.read_csv(data_file_path)
    print(f"   ✅ Success! Loaded {df.shape[0]} rows and {df.shape[1]} columns of data.")
except Exception as e:
    print(f"   ❌ Failed to load the data: {e}")
    exit()

# 3. Show a quick preview
print("\n3. Data Preview:")
print(f"   First few column names: {list(df.columns[:5])}")
print("   Target variable 'emission' statistics:")
print(f"     - Minimum: {df['emission'].min():.2f}")
print(f"     - Maximum: {df['emission'].max():.2f}")
print(f"     - Average: {df['emission'].mean():.2f}")

# 4. Data Cleaning
print("\n4. Cleaning the data...")
# Select only numeric columns to keep things simple
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_clean = df[numeric_cols].copy()

# Handle missing values by filling them with the column's median value
missing_before = df_clean.isnull().sum().sum()
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
missing_after = df_clean.isnull().sum().sum()

print(f"   ✅ Fixed {missing_before - missing_after} missing values.")
print(f"   ✅ Remaining missing values: {missing_after}")

# 5. Prepare for Machine Learning
print("\n5. Preparing for machine learning...")
X = df_clean.drop('emission', axis=1)  # Features (everything except emission)
y = df_clean['emission']               # Target (what we want to predict)

print(f"   We have {X.shape[1]} features to train the model with.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"   ✅ Split data into:")
print(f"      - Training set: {X_train.shape[0]} samples")
print(f"      - Testing set:  {X_test.shape[0]} samples")

# 6. Scale the features (important for many ML models)
print("\n6. Scaling the features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Feature scaling completed.")

# 7. Train the Machine Learning Model
print("\n7. Training the Random Forest model...")
# Use a slightly smaller model for speed on most Macs
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)
print("   ✅ Model training completed!")

# 8. Make predictions and evaluate the model
print("\n8. Evaluating the model's performance...")
y_pred = model.predict(X_test_scaled)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("   Model Performance Results:")
print(f"   - Mean Absolute Error (MAE):    {mae:.4f}")
print(f"   - Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"   - R-squared (R²) Score:         {r2:.4f}")

# 9. Explain the results
print("\n" + "="*55)
print("INTERPRETING THE RESULTS:")
print("="*55)
print("• MAE: Average prediction error. Lower is better.")
print("• RMSE: Average error, punishes large mistakes. Lower is better.")
print("• R² Score: How well the model fits the data.")
print("  1.0 = Perfect fit.")
print("  0.0 = No better than guessing the average.")
print("  Negative = Worse than guessing the average.\n")

print(f"Your model's R² score is: {r2:.4f}")

if r2 > 0.7:
    print("✅ Excellent! The model explains most of the variation in the data.")
elif r2 > 0.5:
    print("✅ Good! The model is a useful predictor.")
elif r2 > 0.3:
    print("⚠️  Fair. The model has some predictive power.")
else:
    print("⚠️  The model needs significant improvement.")

print("\n" + "="*55)
print("PROJECT EXECUTION COMPLETE!")
print("="*55)