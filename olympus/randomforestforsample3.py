import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Function to make predictions and calculate statistics
def predict_and_evaluate(custom_inputs=None):
    # Load the dataset
    file_path = 'MasterDataFile.xlsx'  # Replace with your file path
    data = pd.read_excel(file_path)

    # Selecting relevant features for the model
    excluded_features = ['Year', 'Voyage', 'Time', 'Departure_Arrival']  # Exclude 'Departure_Arrival' here
    feature_columns = [col for col in data.columns if col not in excluded_features and col != 'Total_Fuel_Consumption']
    target_column = 'Total_Fuel_Consumption'

    # Creating feature and target datasets
    X = data[feature_columns]
    y = data[target_column]

    # Handling missing values - filling with the mean for numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X.loc[:, numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Creating and training the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Preparing the input for prediction
    sample_input = {col: X[col].mean() for col in feature_columns}
    if custom_inputs:
        sample_input.update(custom_inputs)

    # Convert sample to DataFrame and standardize
    sample_df = pd.DataFrame([sample_input])
    sample_scaled = scaler.transform(sample_df)

    # Make a prediction
    predicted_fuel_consumption = model.predict(sample_scaled)[0]

    # Calculate RMSE, MAE, and R-squared
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return predicted_fuel_consumption, rmse, mae, r2

# Example usage
custom_inputs = {
    'QM2_Ship_Outside_Temperature': 15,
    'QM2_NAV_STW_Longitudinal': 18.0,
    'Distance_nautical_miles': 40,
    'QM2_Ship_Outside_Pressure': 1000,
    'QM2_NAV_Latitude': 42.0,
    'QM2_NAV_Longitude': -60.0
}

predicted_fuel_consumption, rmse, mae, r2 = predict_and_evaluate(custom_inputs)
print("Predicted Fuel Consumption:", predicted_fuel_consumption)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)


