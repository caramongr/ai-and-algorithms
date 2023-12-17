import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression

df = pd.read_excel('data/default of credit card clients.xls',
                   skiprows=1,
                   index_col=0)


#print(df.head())

#print(df.info())

#report = ProfileReport(df, interactions=None)
#report.to_file('cc_defaults.html')

train_features = df.drop('default payment next month', axis=1)
#print(train_features)
train_targets = df['default payment next month']
print(train_targets)

lr_sklearn = LogisticRegression(random_state=42, max_iter=30000)
lr_sklearn.fit(train_features, train_targets)
print(lr_sklearn.score(train_features, train_targets))
predictions = lr_sklearn.predict(train_features)
print(predictions)