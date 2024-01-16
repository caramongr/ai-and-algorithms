import math

import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6371 * c  # Radius of earth in kilometers
    return km

# Load the dataset
df = pd.read_excel('M238b.xlsx')

# Convert 'Time' column to datetime and calculate the time difference (step) in seconds
df['Time'] = pd.to_datetime(df['Time'])
df['step'] = df['Time'].diff().dt.total_seconds().fillna(0)

# Calculate the shifted values for latitude and longitude for the previous row
prev_lat = df['QM2_NAV_Latitude'].shift(1)
prev_lon = df['QM2_NAV_Longitude'].shift(1)

# Calculate distances using the Haversine formula
df['Distance_km'] = df.apply(lambda row: haversine(prev_lat[row.name], prev_lon[row.name], 
                                                   row['QM2_NAV_Latitude'], row['QM2_NAV_Longitude']), axis=1)

# Convert distance from kilometers to nautical miles
df['Distance_nautical_miles'] = df['Distance_km'] * 0.539957

# Calculate average fuel consumption (renaming the column)
fuel_columns = ['QM2_Boiler_Aft_Usage_Mass_Flow', 'QM2_Boiler_Fwd_Usage_Mass_Flow',
                'QM2_DG01_Usage_Mass_Flow', 'QM2_DG02_Usage_Mass_Flow',
                'QM2_DG03_Usage_Mass_Flow', 'QM2_DG04_Usage_Mass_Flow',
                'QM2_GT01_Usage_Mass_Flow', 'QM2_GT02_Usage_Mass_Flow']
df['Avg Fuel Consumption'] = df[fuel_columns].sum(axis=1)

# Calculate total fuel consumption in metric tons (mT)
df['Total_Fuel_Consumption'] = df['Avg Fuel Consumption'] * df['step'] / 3600

# Drop the temporary columns
df.drop(columns=['Distance_km','step'], inplace=True)

# Save the results to a new Excel file
df.to_excel('M238b-cal.xlsx', index=False)
