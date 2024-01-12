import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load your data (replace 'your_data_file.xlsx' with your actual file path)
data = pd.read_excel('MasterDataFile.xlsx')

# Data Preprocessing: Drop rows with missing values
cleaned_data = data.dropna()

# Feature Selection: Exclude non-numeric and target variable columns
features = cleaned_data.drop(['Voyage', 'Year', 'Departure_Arrival', 'Time', 'Total_Fuel_Consumption', 'Avg Fuel Consumption','QM2_Boiler_Aft_Usage_Mass_Flow', 'QM2_Boiler_Fwd_Usage_Mass_Flow', 'QM2_DG01_Usage_Mass_Flow','QM2_DG02_Usage_Mass_Flow', 'QM2_DG03_Usage_Mass_Flow', 'QM2_DG04_Usage_Mass_Flow','QM2_GT01_Usage_Mass_Flow', 'QM2_GT02_Usage_Mass_Flow', 'QM2_Val_POD01_Power', 'QM2_Val_POD02_Power', 'QM2_Val_POD03_Power', 'QM2_Val_POD04_Power'], axis=1)
target = cleaned_data['Total_Fuel_Consumption']

# Splitting the Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Building: Use RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Model Training
model.fit(X_train, y_train)

# Function to make predictions based on input features
def predict_total_fuel_consumption(input_features):
    """
    Predicts the Total Fuel Consumption based on the input features.
    
    Args:
    input_features (dict): A dictionary containing the values of the features.
    
    Returns:
    float: The predicted Total Fuel Consumption.
    """
    # Convert input features to DataFrame
    input_df = pd.DataFrame([input_features])
    
    # Ensure the order of columns matches the model's training data
    input_df = input_df[features.columns]
    
    # Make a prediction
    predicted_value = model.predict(input_df)
    
    return predicted_value[0]

# Example Usage
input_features = {
      'QM2_Ship_Outside_Temperature': 15,
    'QM2_NAV_STW_Longitudinal': 18.0,
    'Distance_nautical_miles': 40,
    'QM2_Ship_Outside_Pressure': 1000,
    'QM2_NAV_Latitude': 42.0,
    'QM2_NAV_Longitude': -60.0,
}

predicted_consumption = predict_total_fuel_consumption(input_features)

# Calculate RMSE
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print RMSE and Predicted Total Fuel Consumption
print(f"Predicted Total Fuel Consumption: {predicted_consumption}")
print(f"RMSE: {rmse}")

# Generate and print the correlation matrix
correlation_matrix = cleaned_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)
