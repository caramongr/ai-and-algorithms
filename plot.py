import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/itunes_data.csv')
df['Minutes'] = df['Milliseconds'] / (1000 * 60)
df['MB'] = df['Bytes'] / 1000000
df.drop(['Milliseconds', 'Bytes'], axis=1, inplace=True)
# df['Minutes'].plot.box()
# sns.boxenplot(y=df['Minutes'])
#sns.histplot(x=df['Minutes'], kde=True)
sns.heatmap(df.corr(), annot=True)

plt.show()