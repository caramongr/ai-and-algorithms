from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

#print(california.DESCR)

#print(california)

print(california.data.shape)

print(california.target.shape)

print(california.feature_names)

print(california.target_names)
import pandas as pd

pd.set_option('precision', 4)

pd.set_option('max_columns', 9)

pd.set_option('display.width', None)

california_df = pd.DataFrame(california.data, columns=california.feature_names)

#print(california_df.head())

california_df['MedHouseValue'] = pd.Series(california.target)

print(california_df.head())

#print(california.data[0])

print(california_df.describe())

sample_df = california_df.sample(frac=0.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)

sns.set_style('whitegrid')

#for feature in california.feature_names:
   # plt.figure(figsize=(16, 9))
    #sns.scatterplot(data=sample_df, x=feature, y='MedHouseValue', hue='MedHouseValue', palette='cool', legend=False)
   # plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, random_state=11)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X=X_train, y=y_train)



for i, name in enumerate(california.feature_names):
    print(f'{name:>10}: {linear_regression.coef_[i]}')


print(linear_regression.intercept_)

predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}')

#predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)
#print(predict(1))

from sklearn import metrics

print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(expected, predicted):.2f}')

print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(expected, predicted):.2f}')

print(f'Root Mean Squared Error (RMSE): {metrics.mean_squared_error(expected, predicted, squared=False):.2f}')

print(f'R2: {metrics.r2_score(expected, predicted):.2f}')

import matplotlib.pyplot as plt2

plt2.figure(figsize=(16, 9))

plt2.scatter(x=predicted, y=expected)

plt2.plot([0, 50], [0, 50], '--k')

plt2.axis('tight')

plt2.xlabel('Predicted price')

plt2.ylabel('Expected price')

plt2.tight_layout()

plt2.show()



    