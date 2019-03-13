import cv2
import matplotlib.pyplot as plt
import os
from sklearn import svm, metrics
import numpy as np


# Read digits templates for TRAIN
train_dir = 'templates/train'

train_img_list = []
train_target_list = []

for filename in os.listdir(train_dir):
    # Get file name without extension
    # File name is symbol inside image
    symbol = os.path.splitext(filename)[0]

    # Try to convert name to digit
    try:
        symbol = int(symbol)
    except ValueError:
        pass

    train_img_list.append(cv2.imread(train_dir + '/' + filename, 0))
    train_target_list.append(symbol)


subplot_index = 0
for image in train_img_list:
    plt.subplot(4, 5, subplot_index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title('Training: %i' % train_target_list[subplot_index])
    subplot_index += 1


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(train_img_list)
train_data_array = np.array([img.flatten() for img in train_img_list])
train_target_array = np.array(train_target_list)

# Create a classifier: a support vector classifier
classifier = svm.SVC(kernel='poly', degree=3, gamma='auto', probability=True, tol=1e-5)

# We learn the digits on the first half of the digits
classifier.fit(train_data_array, train_target_array)





# Read digits templates for TEST
test_dir = 'templates/test'

test_img_list = []
test_target_list = []

for filename in os.listdir(test_dir):
    # Get file name without extension
    # File name is symbol inside image
    symbol = os.path.splitext(filename)[0]

    # Try to convert name to digit
    try:
        symbol = int(symbol)
    except ValueError:
        pass

    test_img_list.append(cv2.imread(test_dir + '/' + filename, 0))
    test_target_list.append(symbol)



# Now predict the value of the digit on the second half:
test_data_array = np.array([img.flatten() for img in test_img_list])
test_target_array = np.array(test_target_list)

predicted = classifier.predict(test_data_array)

print(test_target_array)
print(predicted)

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_target_list, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_target_list, predicted))

for image in test_img_list:
    plt.subplot(4, 5, subplot_index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title('Test: {0} Prediction: {1}'.format(test_target_list[subplot_index - len(train_img_list)], predicted[subplot_index - len(train_img_list)]))
    subplot_index += 1

plt.show()