import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Load your data (replace 'MasterDataFile.xlsx' with your actual file path)
# data = pd.read_excel('MasterDataFile.xlsx')
data = pd.read_excel('1hr-step-MasterDataFile.xlsx')



data['Departure_Arrival'] = data['Departure_Arrival'].map({'GBSOU - USNYC': 0, 'USNYC - GBSOU': 1})

#filtra
data = data[data['QM2_NAV_STW_Longitudinal'] >= 8]

# Data Preprocessing: Drop rows with missing values
cleaned_data = data.dropna()

# Feature Selection: Exclude non-numeric and target variable columns
features = cleaned_data.drop(['Voyage', 'Time', 'Total_Fuel_Consumption', 'Avg Fuel Consumption', 'QM2_Boiler_Aft_Usage_Mass_Flow','QM2_Boiler_Fwd_Usage_Mass_Flow', 'QM2_DG01_Usage_Mass_Flow',
    'QM2_DG02_Usage_Mass_Flow',
    'QM2_DG03_Usage_Mass_Flow',
    'QM2_DG04_Usage_Mass_Flow',
    'QM2_GT01_Usage_Mass_Flow',
    'QM2_GT02_Usage_Mass_Flow','QM2_Val_POD01_Power',
    'QM2_Val_POD02_Power',
    'QM2_Val_POD03_Power',
    'QM2_Val_POD04_Power'], axis=1)
target = cleaned_data['Total_Fuel_Consumption']

# Splitting the Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Building: Use RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Model Training
model.fit(X_train, y_train)

# Predictions and Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
r2adjusted = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print(f"R2 Adjusted: {r2adjusted}")

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
# Example Usage
input_features = {
   
    'QM2_NAV_Latitude': 49.8,
    'QM2_NAV_Longitude': -4.07,
    'QM2_Ship_Outside_Pressure': 1011.85,
    'QM2_NAV_STW_Longitudinal': 17.0,
    'QM2_Ship_Outside_Temperature': 22.8,
    'Distance_nautical_miles': 21.46,
    'Departure_Arrival':0,
    'Year':2022
}


predicted_consumption = predict_total_fuel_consumption(input_features)
print(f"Predicted Total Fuel Consumption: {predicted_consumption}")

# Visualization of the Correlation Matrix
correlation_matrix = features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()


# Visualization of a Single Tree from the Random Forest
plt.figure(figsize=(25, 15))  # Increased figure size
tree = model.estimators_[0]
plot_tree(tree, filled=True, feature_names=features.columns, max_depth=3, fontsize=12)  # Increased font size
plt.title('Visualization of a Tree in Random Forest', fontsize=14)  # Increased title font size
plt.show()