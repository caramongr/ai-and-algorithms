import numpy as np
import pandas as pd


def convert_to_degrees(raw_value):
    """
    Convert the latitude and longitude values from the raw format to degrees.
    Assuming the format is in a decimal representation (e.g., 49635555 -> 49.635555).
    """
    value_in_degrees = raw_value / 1000000
    print(value_in_degrees)

    return round(value_in_degrees, 5)  # rounding to 5 decimal places

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    R = 6371  # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

# Read the Excel file
df = pd.read_excel('input.xlsx')

# Convert latitude and longitude to degrees
df['QM2_NAV_Latitude_deg'] = df['QM2_NAV_Latitude'].apply(convert_to_degrees)
df['QM2_NAV_Longitude_deg'] = df['QM2_NAV_Longitude'].apply(convert_to_degrees)

# Calculate the shifted values for latitude and longitude for the previous row
df['Prev_QM2_NAV_Latitude_deg'] = df['QM2_NAV_Latitude_deg'].shift(1)
df['Prev_QM2_NAV_Longitude_deg'] = df['QM2_NAV_Longitude_deg'].shift(1)

# Calculate distances using the Haversine formula (comparing each row to its previous row)
df['Distance_km'] = df.apply(lambda row: haversine(row['Prev_QM2_NAV_Latitude_deg'], row['Prev_QM2_NAV_Longitude_deg'], 
                                                   row['QM2_NAV_Latitude_deg'], row['QM2_NAV_Longitude_deg']), axis=1)

# Convert distance from kilometers to nautical miles
df['Distance_nautical_miles'] = df['Distance_km'] * 0.539957

# Save the results to a new Excel file
# df.to_excel('output.xlsx', index=False)
