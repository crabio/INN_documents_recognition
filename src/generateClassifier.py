# Import the modules
import datetime

import cv2
from pandas import DataFrame
from sklearn import svm
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from collections import Counter
import os
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

hog_orientations = 2
hog_pixels_per_cell = (4, 4)


# Load the dataset
# dataset = datasets.fetch_mldata("MNIST Original")


def load_images_from_mnist():
    return loadlocal_mnist(
        images_path='digitRecognition/mnist/train-images.idx3-ubyte',
        labels_path='digitRecognition/mnist/train-labels.idx1-ubyte')


def load_images_from_dir(images_dir):
    X = []
    y = []

    images_file_names_list = os.listdir(train_images_dir)
    images_count = len(images_file_names_list)

    fig = plt.figure()
    for i, image_file_name in enumerate(images_file_names_list):
        label = image_file_name[:1]

        image_rgb = cv2.imread(train_images_dir + '/' + image_file_name)

        image_gray = np.array(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY))

        image_gray_resized = cv2.resize(image_gray, (28, 28), interpolation=cv2.INTER_AREA)
        image_gray_delated = cv2.dilate(image_gray_resized, (3, 3))

        ax1 = fig.add_subplot(2, images_count, i + 1)
        ax1.set_title('Image: %s' % label)
        plt.imshow(image_gray, cmap='gray')
        ax2 = fig.add_subplot(2, images_count, i + 1 + images_count * 1)
        plt.imshow(image_gray_delated, cmap='gray')

        X.append(image_gray_delated)
        y.append(label)

    plt.show()

    return np.array(X), np.array(y)


X, y = load_images_from_mnist()
# X, y = load_images_from_dir('image/train')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0].shape)

# Extract the features and labels
features = np.array(X, 'int8')
labels = np.array(y, 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=4, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print(hog_features.shape)

print("Count of digits in dataset", Counter(labels))

# Create an linear SVM object
clf = svm.SVC(kernel='poly', degree=3, gamma='auto', probability=True)

start = datetime.datetime.now()

# Perform the training
clf.fit(hog_features, labels)

end = datetime.datetime.now()
print(end - start)

# Save the classifier
joblib.dump(clf, "digitRecognition/digits_cls.pkl")
