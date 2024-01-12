import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'results.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Handle missing values (Example: Fill missing values with the mean of the column)
df.fillna(df.mean(), inplace=True)

# Feature selection
features = ['QM2_NAV_STW_Longitudinal', 'QM2_Ship_Outside_Temperature', 'QM2_Ship_Outside_Pressure',
            'QM2_Boiler_Aft_Usage_Mass_Flow', 'QM2_Boiler_Fwd_Usage_Mass_Flow',
            'QM2_DG01_Usage_Mass_Flow', 'QM2_DG02_Usage_Mass_Flow', 'QM2_DG03_Usage_Mass_Flow', 'QM2_DG04_Usage_Mass_Flow',
            'QM2_GT01_Usage_Mass_Flow', 'QM2_GT02_Usage_Mass_Flow',
            'QM2_Val_POD01_Power', 'QM2_Val_POD02_Power', 'QM2_Val_POD03_Power', 'QM2_Val_POD04_Power']
X = df[features]
y = df['Distance_nautical_miles']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest Root Mean Squared Error: {rf_rmse}')
