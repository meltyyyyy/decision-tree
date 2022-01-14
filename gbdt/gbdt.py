from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def execute():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    gbrt = GradientBoostingClassifier(random_state=0,max_depth=1)
    gbrt.fit(X_train, y_train)

    print("Accuracy of training set: {:.3f}".format(gbrt.score(X_train, y_train)))
    print("Accuracy of test set: {:.3f}".format(gbrt.score(X_test, y_test)))

