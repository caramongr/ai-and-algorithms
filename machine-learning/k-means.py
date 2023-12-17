from sklearn.datasets import load_iris

iris = load_iris()

print(iris.DESCR)
print(iris.data.shape)

print(iris.data)

print(iris.target.shape)

print(iris.target_names)

print(iris.feature_names)

import pandas as pd

pd.set_option('precision', 4)

pd.set_option('max_columns', 5)

pd.set_option('display.width', None)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

iris_df['species'] = [iris.target_names[i] for i in iris.target]

print(iris_df.head())

print(iris_df.describe())

print(iris_df['species'].describe())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# sns.set(font_scale=1.1)

# sns.set_style('whitegrid')

# grid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4], hue='species')

# plt.show()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=11)

from sklearn.cluster import KMeans

Kmeans = KMeans(n_clusters=3, random_state=11)

Kmeans.fit(iris.data)

print(Kmeans.labels_[0:50])

print(Kmeans.labels_[50:100])

print(Kmeans.labels_[101:150])

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=11)

pca.fit(iris.data)

iris_pca = pca.transform(iris.data)

print(iris_pca.shape)

iris_pca_df = pd.DataFrame(iris_pca, columns=['Component1', 'Component2'])

iris_pca_df['species'] = iris_df.species

axes = sns.scatterplot(data=iris_pca_df, x='Component1', y='Component2', hue='species', legend='brief', palette='cool')

iris_centers = pca.transform(Kmeans.cluster_centers_)

import matplotlib.pyplot as plt2

dots = plt2.scatter(iris_centers[:, 0], iris_centers[:, 1], s=100, c='k')    

plt2.show()