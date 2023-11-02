import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.decomposition import PCA

def load_file(file_name):
    x=[]
    y=[]
    file_link_path = f".\\work\\{file_name}.txt"
    with open(file_link_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split(",")
        x.append(parts[1:])
        y.append(parts[0])

    X = np.array(x)
    Y = np.array(y)
    return X,Y

def train(X_train, X_test, y_train, y_test):
    classifier_DecisionTree = DecisionTreeClassifier()
    classifier_DecisionTree.fit(X_train, y_train)
    y_test_pred = classifier_DecisionTree.predict(X_test)

    # compute accuracy of the classifier
    accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
    print("Accuracy of the classifier =", round(accuracy, 2), "%")

def scaler_data(X_train, X_test):
    X_train_scaler =PCA.fit_transform(X_train)
    X_test_scaler =PCA.fit_transform(X_test)

    return X_train_scaler,X_test_scaler

if __name__ == "__main__":
    file_name="wine"
    x,y=load_file(file_name)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=5)
    X_train, X_test=scaler_data(X_train, X_test)
    train(X_train, X_test, y_train, y_test)
