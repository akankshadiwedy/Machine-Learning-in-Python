from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# dividing iris datasets to training dataset and testing dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# classifier
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier()
clf2 = clf2.fit(x_train, y_train)

print y_test
predictionB = clf2.predict(x_test)
print predictionB

# accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictionB)
