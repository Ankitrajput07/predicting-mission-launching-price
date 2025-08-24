import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

print("--- Starting Model Training for Mission Launches ---")

# --- 1. Load and Clean Data ---
print("Step 1: Loading and Cleaning Data...")
try:
    df = pd.read_csv('mission_launches.csv')
except FileNotFoundError:
    print("Error: 'mission_launches.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Drop unnecessary columns that are just indices
df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)

# Clean and impute the 'Price' column
# This is our target variable, so it's crucial to clean it properly.
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
price_median = df['Price'].median()
df['Price'].fillna(price_median, inplace=True)

# Ensure there are no remaining missing values in our target
df.dropna(subset=['Price'], inplace=True)

print("Data Loading and Cleaning Complete.")

# --- 2. Feature Selection and Engineering ---
print("Step 2: Selecting Features and Preparing for Encoding...")
# These categorical features will be used to predict the price.
features = ['Organisation', 'Location', 'Rocket_Status', 'Mission_Status']
target = 'Price'

X = df[features]
y = df[target]

# Initialize the OneHotEncoder.
# This will convert our text categories into a numerical format (0s and 1s).
# handle_unknown='ignore' prevents errors if the app sees a category it wasn't trained on.
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit the encoder to our features and transform them into numerical data.
X_encoded = encoder.fit_transform(X)

print("Feature Engineering (One-Hot Encoding) Complete.")

# --- 3. Train the Linear Regression Model ---
print("Step 3: Training the Linear Regression Model...")
# We split our data: 80% for training the model, 20% for testing its performance.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance on the test set (R^2 score)
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.2f}")
print("Model Training Complete.")

# --- 4. Save the Model and Supporting Files ---
print("Step 4: Saving the model and other necessary files...")
# These files will be loaded by our Flask web application.

# Save the trained model
joblib.dump(model, 'launch_price_model.joblib')

# Save the encoder so we can apply the exact same transformation in the app
joblib.dump(encoder, 'one_hot_encoder.joblib')

# Save the list of feature columns to ensure consistency
joblib.dump(features, 'feature_columns.joblib')

# Save the unique values from each column to populate the dropdown menus in the HTML form
unique_values = {col: df[col].unique().tolist() for col in features}
joblib.dump(unique_values, 'unique_values.joblib')

print("--- Model training and all files have been saved successfully. ---")
