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

# Function to predict Total Fuel Consumption for a specific voyage
def predict_for_voyage(voyage_code, data, model):
    """
    Predicts the Total Fuel Consumption for all rows of a specified voyage.

    Args:
    voyage_code (str): The voyage code to filter the data.
    data (DataFrame): The full dataset.
    model (Model): The trained model for prediction.

    Returns:
    DataFrame: A DataFrame with predictions for each row of the specified voyage.
    """
    # Filter the dataset for the specified voyage
    voyage_data = data[data['Voyage'] == voyage_code]

    # Prepare the data for prediction
    features_to_predict = voyage_data.drop(['Voyage', 'Year', 'Departure_Arrival', 'Time', 'Total_Fuel_Consumption'], axis=1)

    # Make predictions
    voyage_data['Predicted Total Fuel Consumption'] = model.predict(features_to_predict)

    return voyage_data


voyage_code = 'M210' 
predicted_voyage_data = predict_for_voyage(voyage_code, cleaned_data, model)
print(predicted_voyage_data)

# Function to create an Excel file with predicted values for a specific Departure_Arrival



# Display or analyze the predicted data
print(predicted_voyage_data)