import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

data = pd.read_csv('heart_men.csv')
#print(data)

arr = np.array(data)

#print("Shape of arr: {}".format(arr.shape))

np.random.shuffle(arr)

#print(arr)

column_labels = data.columns
samples = arr[:, :-1]
labels = arr[:, -1]

x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.20)

#print(x_train)

#print(y_train)

#input("waiting...")

neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(x_train, y_train)

print("\nK-Nearest Neighbor Model")


print("Test set accuracy: {}".format(neigh.score(x_test,y_test)))


logreg = LogisticRegression(C=10, max_iter = 1000).fit(x_train, y_train)

print("\nLogistic Regression Model")

print("Training set score {:.3f}".format(logreg.score(x_train, y_train)))

print("Test set score {:.3f}".format(logreg.score(x_test, y_test)))


logreg100 = LogisticRegression(C=100,max_iter = 1000).fit(x_train, y_train)

print("\nLogistic Regression Model with more regularization")

print("Training set score {:.3f}".format(logreg100.score(x_train, y_train)))

print("Test set score {:.3f}".format(logreg100.score(x_test, y_test)))


tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x_train,y_train)


print("\nDecision Tree Classifier Model")

print("Accuracy on training set: {:.3f}".format(tree.score(x_train,y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(x_test,y_test)))


forest = RandomForestClassifier(n_estimators=100,random_state=1)

forest.fit(x_train,y_train)

print("\nRandom Forest Classifier Model")

print("Accuracy on training set: {:.3f}".format(forest.score(x_train,y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(x_test,y_test)))
