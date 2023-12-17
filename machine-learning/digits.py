import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
print(digits.data[18])

print(digits.target[18])

print(digits.target.shape)

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()

# axes = plt.subplots()
# image = plt.imshow(digits.images[22], cmap=plt.cm.gray_r)

# xticks = plt.xticks(ticks=[])

# yticks = plt.yticks(ticks=[])


# plt.show()

X_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=11)

# print(x_train.shape)

# print(y_train.shape)

# print(x_test.shape)

# print(y_test.shape)

print("----------------------------------------------------------------")

knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=x_test)

excepted = y_test

print("Predicted: ", predicted[:20])
print("Expected: ", excepted[:20])


wrong = [(p, e) for (p, e) in zip(predicted, excepted) if p != e]

print("Wrong: ", wrong)

print("Accuracy: ", knn.score(X=x_test, y=y_test))
print(f'{knn.score(X=x_test, y=y_test):.2%}')

print("----------------------------------------------------------------")

confusion = confusion_matrix(y_true=excepted, y_pred=predicted)

print(confusion)

print("----------------------------------------------------------------")

names = [str(digit) for digit in digits.target_names]

print(classification_report(y_true=excepted, y_pred=predicted, target_names=names))

print("----------------------------------------------------------------")



import seaborn as sns

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

axes = plt.subplot()
heatmap = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r', ax=axes)
heatmap.set_xlabel('Predicted')
heatmap.set_ylabel('Actual')

plt.show()

print("----------------------------------------------------------------")