import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import cv2
import pickle
import glob
# List of all possible classes
letters = ['1', '2', '4', 'Z', '0', 'C']

def read_training_data(training_directory):
    """
    Return array with images and their labels for fiting
    :param training_directory:
    :return: (images, labels)
    """
    images = []
    labels = []

    for each_letter in letters:
        # get num of images in each letter(class) directory
        images_list = glob.glob(training_directory + '/' + each_letter + '/*.jpg')
        for i in range(len(images_list)):
            # read each image of each character
            image = cv2.imread(images_list[i], cv2.IMREAD_UNCHANGED)
            _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
            image = np.concatenate(image)
            images.append(image)
            labels.append(each_letter)

    return (np.array(images), np.array(labels))

def cross_validation(model, num_of_fold, train_data, train_label):
    """
    This uses the concept of cross validation to measure the accuracy
    of a model, the num_of_fold determines the type of validation
    e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    it will divide the dataset into 4 and use 1/4 of it for testing
    and the remaining 3/4 for the training
    :param model:
    :param num_of_fold:
    :param train_data:
    :param train_label:
    :return:
    """
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)

    print("Cross Validation Result for ", str(num_of_fold), " -fold")
    print(accuracy_result * 100)

print('reading data')
training_dataset_dir = 'dataset'
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svc_model = SVC(kernel='linear', probability=True)

#cross_validation(svc_model, 1, image_data, target_data)

print('training model')

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

print("model trained.saving model..")
filename = 'model/train_russian_ocr.svm'
pickle.dump(svc_model, open(filename, 'wb'))
print("model saved")