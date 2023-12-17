import pandas as pd
import matplotlib.pyplot as plt

csv_df = pd.read_csv('data/itunes_data.csv')
csv_df.head()
print(csv_df.head(2))
print(csv_df.columns)
print("-------------")
print(csv_df.iloc[2])
print("-------------")
print(csv_df.shape)
# print(csv_df.info)
print(csv_df.isna().sum())
#print(csv_df.describe())
print(csv_df.corr())
# csv_df['Milliseconds'].hist(bins=30)
# plt.show()

# csv_df['Genre'].value_counts().plot.bar()
# plt.show()
print("-------------")
# print(csv_df['Milliseconds'] > 4e6)
# print(csv_df[csv_df['Milliseconds'] > 4e6])

print(csv_df[(csv_df['Milliseconds'] > 2e6) 
 & (csv_df['Bytes'] < 0.4e9)]['Genre'].value_counts)
