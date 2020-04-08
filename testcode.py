import DataPrep
from DataPrep import create_training_data, x_data, y_data

import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

create_training_data()

X=x_data
Y=y_data

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# n_components = 100
# X_std = [StandardScaler().fit_transform(data) for data in x_data]
 
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Compute a PCA 
n_components = 100
X_train_pca = [PCA(n_components=n_components, whiten=True).fit_transform(X_train, y = 0)]
X_test_pca = [PCA(n_components=n_components, whiten=True).fit_transform(data) for data in X_test]
# apply PCA transformation
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=y_train))

# Visualization
h=250
w=250
def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
 
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)
 
prediction_titles = list(titles(y_pred, y_test, y_train))
plot_gallery(X_test, prediction_titles, h, w)   