import math

import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """

    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
 
    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 
    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
             math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    km = rad * c
    return km

# Read the Excel file
df = pd.read_excel('input5.xlsx')

# Convert latitude and longitude to degrees
df['QM2_NAV_Latitude_deg'] = df['QM2_NAV_Latitude']
df['QM2_NAV_Longitude_deg'] = df['QM2_NAV_Longitude']

# Calculate the shifted values for latitude and longitude for the previous row
df['Prev_QM2_NAV_Latitude_deg'] = df['QM2_NAV_Latitude_deg'].shift(1)
df['Prev_QM2_NAV_Longitude_deg'] = df['QM2_NAV_Longitude_deg'].shift(1)

# Calculate distances using the Haversine formula
df['Distance_km'] = df.apply(lambda row: haversine(row['Prev_QM2_NAV_Latitude_deg'], row['Prev_QM2_NAV_Longitude_deg'], 
                                                   row['QM2_NAV_Latitude_deg'], row['QM2_NAV_Longitude_deg']), axis=1)

# Convert distance from kilometers to nautical miles
df['Distance_nautical_miles'] = df['Distance_km'] * 0.539957

# Save the results to a new Excel file
df.to_excel('output6.xlsx', index=False)
