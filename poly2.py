#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#download the csv file
url = 'https://www.stats.govt.nz/assets/Uploads/Population/Population-estimates-at-30-June-2020/Download-data/population-estimates-at-30-june-2020-csv.csv'
df = pd.read_csv(url)

#get the year and population values
x = df['Year']
y = df['Population']

#find the best polynomial fit of degree 4
coeffs = np.polyfit(x, y, 4)

#generate the fitted values
yfit = np.polyval(coeffs, x)

#plot the data points and the line of best fit
plt.scatter(x, y, label='data')
plt.plot(x, yfit, color='red', label='best fit')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()