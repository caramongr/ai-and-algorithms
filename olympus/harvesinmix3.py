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

# Read the Excel file
df = pd.read_excel('input5.xlsx')

# Calculate the shifted values for latitude and longitude for the previous row
prev_lat = df['QM2_NAV_Latitude'].shift(1)
prev_lon = df['QM2_NAV_Longitude'].shift(1)

# Calculate distances using the Haversine formula
df['Distance_km'] = df.apply(lambda row: haversine(prev_lat[row.name], prev_lon[row.name], 
                                                   row['QM2_NAV_Latitude'], row['QM2_NAV_Longitude']), axis=1)

# Convert distance from kilometers to nautical miles
df['Distance_nautical_miles'] = df['Distance_km'] * 0.539957

# Drop the temporary 'Distance_km' column and keep all original columns plus 'Distance_nautical_miles'
df = df.drop(columns=['Distance_km'])

# Save the results to a new Excel file
df.to_excel('output8.xlsx', index=False)
