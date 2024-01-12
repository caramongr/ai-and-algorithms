import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
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

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Visualization using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Actual Distances (nautical miles)')
plt.ylabel('Predicted Distances (nautical miles)')
plt.title('Actual vs Predicted Distances')
plt.show()
