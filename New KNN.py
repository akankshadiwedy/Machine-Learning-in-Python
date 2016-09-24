from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class NewKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions =[]
        for row in x_test:
            label = self.closet(row)
            predictions.append(label)
        return predictions

    def closet(self,row):
        best_dist = euc(row,self.x_train[0])
        best_idx = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return self.y_train[best_idx]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# dividing iris datasets to training dataset and testing dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# classifier
clf = NewKNN()
clf.fit(x_train, y_train)

print y_test
predictions = clf.predict(x_test)

# accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
