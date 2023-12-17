import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df_la = pd.read_csv('ave_hi_la_jan_1895-2018.csv')


print(f"\tLos Angeles Average January High Temperatures for 1885 through 2018:")
print("Head:\n", df_la.head())
print("\nTail:\n", df_la.tail())

# Cleaning the Data
df_la.columns = ['Date', 'Temperature', 'Anomaly']

df_la.Date = df_la.Date.floordiv(100)

print("\n\tCleaning the Date")
print(df_la.head(3))

pd.set_option('display.precision', 2)

print("\n\tCalculating Basic Descriptive Statistics for the Dataset")
print(df_la.Temperature.describe())
import seaborn as sns

print("\n\n\tPlotting the Average High Temperatures and a Regression Line")
sns.set_style('whitegrid')

axes = sns.regplot(x=df_la.Date, y=df_la.Temperature)

axes.set_ylim(10, 70)

plt.show()

linear_regression = stats.linregress(x=df_la.Date, y=df_la.Temperature)

print(linear_regression)

predict = linear_regression.slope * 2019 + linear_regression.intercept

print(f"\n\tThe Predicted Average High Temperature for 2019 is {predict:.2f} degrees Fahrenheit.")

print("\n\n\tLinear Regression Results")




