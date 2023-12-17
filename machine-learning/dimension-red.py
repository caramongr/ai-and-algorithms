from sklearn.datasets import load_digits

digits = load_digits()

print(digits.data.shape)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=11)

reduced_data = tsne.fit_transform(digits.data)

print(reduced_data.shape)

print(reduced_data[0:1])

import matplotlib.pyplot as plt

# dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='black')

dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=digits.target, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))

colorbar = plt.colorbar(dots)

plt.show()