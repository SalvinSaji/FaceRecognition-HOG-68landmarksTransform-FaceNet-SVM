from sklearn import svm
import numpy as np

# load the embeddings and their corresponding labels
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# train the SVM classifier on the training set
clf = svm.LinearSVC()
clf.fit(X_train, y_train)

# evaluate the performance on the test set
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

# make predictions on new data
new_data_embeddings = np.load('test_embeddings.npy')
new_data_labels = clf.predict(new_data_embeddings)
print('Predicted labels:', new_data_labels)
