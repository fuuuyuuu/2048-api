import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 0: left, 1: down, 2: right, 3: up

csv_data = pd.read_csv('Train.csv')
csv_data = csv_data.values
board_data = csv_data[:,0:16]
# print(board_data[0:4,:])
direction_data = csv_data[:,16]

# print(type(board_data))
X = np.int32(board_data)/11.0
Y = np.int32(direction_data)
# print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=0.3)
clf = SGDClassifier(penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)
clf.fit(X_train, Y_train)
train_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

# k_range = range(1,31)
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k, n_jobs=8)
#     knn.fit(X_train, Y_train)
#     scores = knn.score(X_test, Y_test)
#     k_scores.append(scores)
#
# plt.plot(k_range, k_scores)
# plt.xlabel('KNN_value')
# plt.ylabel('accuracy')
# plt.show()


print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
