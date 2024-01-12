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
features = cleaned_data.drop(['Voyage', 'Year', 'Departure_Arrival', 'Time', 'Total_Fuel_Consumption','Avg Fuel Consumption','QM2_NAV_Latitude',
    'QM2_NAV_Longitude'], axis=1)
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
    'QM2_Boiler_Aft_Usage_Mass_Flow': 0.2,
    'QM2_Boiler_Fwd_Usage_Mass_Flow': 0.1,
    'QM2_DG01_Usage_Mass_Flow': 2.5,
    'QM2_DG02_Usage_Mass_Flow': 2.4,
    'QM2_DG03_Usage_Mass_Flow': 2.3,
    'QM2_DG04_Usage_Mass_Flow': 2.2,
    'QM2_GT01_Usage_Mass_Flow': 1.1,
    'QM2_GT02_Usage_Mass_Flow': 1.0,
    # 'QM2_NAV_Latitude': 42.0,
    # 'QM2_NAV_Longitude': -60.0,
    'QM2_Ship_Outside_Pressure': 1000,
    'QM2_Val_POD01_Power': 6.5,
    'QM2_Val_POD02_Power': 6.6,
    'QM2_Val_POD03_Power': 6.7,
    'QM2_Val_POD04_Power': 6.8,
    'QM2_NAV_STW_Longitudinal': 18.0,
    'QM2_Ship_Outside_Temperature': 15,
    'Distance_nautical_miles': 40,
}

predicted_consumption = predict_total_fuel_consumption(input_features)


# create a correlation matrix






y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

print(f"Predicted Total Fuel Consumption: {predicted_consumption}")
