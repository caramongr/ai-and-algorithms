from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

knn = KNeighborsClassifier()  # Define the KNN classifier
digits = load_digits()

scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=10)

print(scores)

print(f'Mean accuracy: {scores.mean():.2%}')
print(f'Standard deviation: {scores.std():.2%}')

svc = SVC()

estimators = {
    'KNeighborsClassifier': knn,
    'SVC': svc,
    'GaussianNB': GaussianNB()
}


for estimator_name, estimator_object in estimators.items():
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    scores = cross_val_score(estimator=estimator_object, X=digits.data, y=digits.target, cv=kfold)
    print(f'{estimator_name:>20}: ' + f'mean accuracy={scores.mean():.2%}; ' + f'standard deviation={scores.std():.2%}')


