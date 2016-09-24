import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0, 50, 100]

# printing meta-data i.e. the name of features and labels (which are given a real-value in the datasets)
print iris.feature_names
print iris.target_names

# print all the training data
for i in range(len(iris.target)):
    print 'example', i, ' label ', iris.target[i], ', feature ', iris.data[i]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# actual correct answer
print test_target

# prediction
print clf.predict(test_data)