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
features = cleaned_data.drop(['Voyage', 'Year', 'Departure_Arrival', 'Time', 'Total_Fuel_Consumption'], axis=1)
target = cleaned_data['Total_Fuel_Consumption']

# Splitting the Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Building: Use RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Model Training
model.fit(X_train, y_train)

# Function to create an Excel file with predicted values for a specific Departure_Arrival
def create_predicted_values_excel(departure_arrival, data, model):
    """
    Creates an Excel file with predicted values for a specific Departure_Arrival.

    Args:
    departure_arrival (str): The departure and arrival port code to filter the data.
    data (DataFrame): The full dataset.
    model (Model): The trained model for prediction.

    Returns:
    None
    """
    # Filter the dataset for the specified Departure_Arrival
    filtered_data = data[data['Departure_Arrival'] == departure_arrival]

    # Check if there are data for the specified voyage
    if len(filtered_data) == 0:
        print(f"No data available for {departure_arrival}. Please check the code.")
        return

    # Prepare the data for prediction
    features_to_predict = filtered_data.drop(['Voyage', 'Year', 'Departure_Arrival', 'Time', 'Total_Fuel_Consumption'], axis=1)

    # Make predictions
    predicted_values = model.predict(features_to_predict)

    # Add the predicted values as a new column in the DataFrame
    filtered_data['Predicted Total Fuel Consumption'] = predicted_values

    # Export the DataFrame to an Excel file
    output_file = f'Predicted_Values_{departure_arrival}.xlsx'
    filtered_data.to_excel(output_file, index=False)

    print(f"Predicted values for {departure_arrival} have been saved to {output_file}")

# Example usage
departure_arrival = 'GBSOU - USNYC'  # Replace with the Departure_Arrival you're interested in
create_predicted_values_excel(departure_arrival, cleaned_data, model)
