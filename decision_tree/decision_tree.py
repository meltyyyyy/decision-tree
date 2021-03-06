import graphviz as graphviz
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def plot_feature_importance_cancer(model, dataset):
    fig = plt.figure()
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    fig.savefig("decision_tree/feature_importance.png")


def execute():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
    print("Feature importance:\n{}".format(tree.feature_importances_))

    export_graphviz(tree, out_file="decision_tree/tree.dot", class_names=["malignant", "benign"],
                    feature_names=cancer.feature_names,
                    impurity=False, filled=True)
    with open("decision_tree/tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)

    plot_feature_importance_cancer(tree, cancer)
