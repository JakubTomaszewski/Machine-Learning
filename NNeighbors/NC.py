'''working with the iris dataset'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

#creating my classifier which works as the Neighbors classifier
class Classifier():
    def fit(self, features, labels):
        self.x_train = features
        self.y_train = labels

    def closest(self, row):
        distances = [distance.euclidean(r, row) for r in self.x_train]
        idx = distances.index(min(distances))
        return self.y_train[idx]

    def predict(self, x_test):
        preds = [self.closest(row) for row in x_test]
            #for each point(row) choose the feature with
            #the min distance and append to the
            #predictions array the label where the
            #distance is min
        return preds


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.5)

clf = Classifier()

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print('The accuracy is {}'.format(accuracy_score(y_test, predictions)))
