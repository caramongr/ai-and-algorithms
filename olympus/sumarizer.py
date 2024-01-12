import pandas as pd

# Load the dataset
file_path = 'MasterDataFile.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Define aggregation functions
agg_functions = {
    'Year': 'first',  # Keep the first year
    'Time': ['first', 'last'],  # Create two columns Time_Started and Time_Ended
    'Departure_Arrival': 'first',  # Keep the first Departure_Arrival value
    'QM2_Boiler_Aft_Usage_Mass_Flow': 'sum',
    'QM2_Boiler_Fwd_Usage_Mass_Flow': 'sum',
    'QM2_DG01_Usage_Mass_Flow': 'sum',
    'QM2_DG02_Usage_Mass_Flow': 'sum',
    'QM2_DG03_Usage_Mass_Flow': 'sum',
    'QM2_DG04_Usage_Mass_Flow': 'sum',
    'QM2_GT01_Usage_Mass_Flow': 'sum',
    'QM2_GT02_Usage_Mass_Flow': 'sum',
    'QM2_Ship_Outside_Pressure': 'mean',
    'QM2_Val_POD01_Power': 'mean',
    'QM2_Val_POD02_Power': 'mean',
    'QM2_Val_POD03_Power': 'mean',
    'QM2_Val_POD04_Power': 'mean',
    'QM2_NAV_STW_Longitudinal': 'mean',
    'QM2_Ship_Outside_Temperature': 'mean',
    'Distance_nautical_miles': 'sum',
    'Avg Fuel Consumption': 'sum',
    'Total_Fuel_Consumption': 'sum'
}

# Group the data by 'Voyage' column and apply aggregation functions
summary_data = data.groupby(['Voyage']).agg(agg_functions)

# Rename columns as per your requirements
summary_data.columns = ['Year', 'Time_Started', 'Time_Ended', 'Departure_Arrival', 'QM2_Boiler_Aft_Usage_Mass_Flow',
                        'QM2_Boiler_Fwd_Usage_Mass_Flow', 'QM2_DG01_Usage_Mass_Flow', 'QM2_DG02_Usage_Mass_Flow',
                        'QM2_DG03_Usage_Mass_Flow', 'QM2_DG04_Usage_Mass_Flow', 'QM2_GT01_Usage_Mass_Flow',
                        'QM2_GT02_Usage_Mass_Flow', 'QM2_Ship_Outside_Pressure', 'QM2_Val_POD01_Power',
                        'QM2_Val_POD02_Power', 'QM2_Val_POD03_Power', 'QM2_Val_POD04_Power',
                        'QM2_NAV_STW_Longitudinal', 'QM2_Ship_Outside_Temperature', 'Distance_nautical_miles',
                        'Avg Fuel Consumption', 'Total_Fuel_Consumption']

# Export the summarized data to a new Excel file
output_file_path = 'SummaryDataFile.xlsx'  # Replace with your desired output file path
summary_data.to_excel(output_file_path, index=False)

print("Summary data exported to", output_file_path)
