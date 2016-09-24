from sklearn import datasets
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = learn.DNNClassifier(hidden_units=[10,20,10], m_classes=3)
clf = clf.fit(x_train, y_train, steps=200)
predictionA = clf.predict(x_test)
print predictionA

print accuracy_score(y_test, predictionA)
