import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

# Read the CSV file
df = pd.read_csv('input.xlsx')

# Assuming you want to calculate the distance from one row to the next
df['Distance'] = df.apply(lambda row: haversine(row['QM2_NAV_Latitude'], row['QM2_NAV_Longitude'], 
                                                row['QM2_NAV_Latitude'].shift(-1), row['QM2_NAV_Longitude'].shift(-1)), axis=1)

# Save the results
df.to_csv('output.xlsx', index=False)
