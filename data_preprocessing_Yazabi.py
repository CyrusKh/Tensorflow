"""
Download the HAPT dataset (UCI HAR Dataset.zip) manually from the following link:

https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Then unzip the file. Since we are in a Windows environment, copy the directory of the UCI_HAR_Dataset
folder manually and paste in the root_folder variable right below the docstring and we are good to go!
"""
root_folder = "D:/yazabi/tf/tf-classification/"

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

y_label = 'activity'
    
def get_training_set():
    """
    This function reads the files and combines them into dataframes for data exploration. 
        
    """
    features_URL = root_folder + "UCI_HAR_Dataset/features.txt"
    har_data_labels = pd.read_csv(features_URL, sep="\s+", names=['id', 'feature_name']) #file_in_zip('features.txt')

    # Read the activity names
    activity_URL = root_folder + "UCI_HAR_Dataset/activity_labels.txt"
    activity_map = { str(row.id):row.label_name for i, row in pd.read_csv(activity_URL, sep="\s+", names=['id', 'label_name']).iterrows() }

    # Read the X
    X_train_URL = root_folder + "UCI_HAR_Dataset/train/X_train.txt"
    har_data_x = pd.read_csv(X_train_URL, sep="\s+", names=har_data_labels.feature_name.tolist())

    # Read the Y
    y_train_URL = root_folder + "UCI_HAR_Dataset/train/y_train.txt"
    har_data_y = pd.read_csv(y_train_URL, sep="\s+", names=[y_label])

    # Map the label into a readable form
    har_data_y[y_label] = har_data_y[y_label].apply(lambda x: activity_map[str(x)])

    # Training data, merge X and Y
    har_data_train = pd.concat([har_data_y, har_data_x], axis=1)

    # Read the X test
    X_test_URL = root_folder + "UCI_HAR_Dataset/test/X_test.txt"
    har_test_x = pd.read_csv(X_test_URL, sep="\s+", names=har_data_labels.feature_name.tolist())

    # Read the Y test
    y_test_URL = root_folder + "UCI_HAR_Dataset/test/y_test.txt"
    har_test_y = pd.read_csv(y_test_URL, sep="\s+", names=['activity'])

    # Map the label into a readable form
    har_test_y[y_label] = har_test_y[y_label].apply(lambda x: activity_map[str(x)])

    # Test data, merge X and Y
    har_data_test = pd.concat([har_test_y, har_test_x], axis=1)

    # Return the two frames as tuple
    return (har_data_train, har_data_test, sorted(activity_map.items()))

(train_data, test_data, activity_map) = get_training_set()

X_data = lambda df: df.copy().drop([y_label], axis=1)
Y_data = lambda df: df[y_label]

# The original test data already had the "real" outcomes.
y_numbers = pd.concat([Y_data(train_data), Y_data(
    test_data)])  # Get all possible activity names again from the data
    
from sklearn.preprocessing import LabelEncoder

# Turn the activity labels into integers
activity_encoder = LabelEncoder().fit(y_numbers)

# We do not need to scale the values really because they are already in the -1 to +1 range.

#Map the activity names (like Walking, Standing, Sitting into 0, 1, 2 etc)
Y_train_numeric = activity_encoder.transform(
    Y_data(train_data))  # Training labels
Y_test_numeric = activity_encoder.transform(Y_data(test_data))  # Test labels

#Number of distinct classes in the multiclass classifier
n_classes = len(activity_encoder.classes_)

num_labels = len(np.unique(Y_test_numeric)) #6
train_x = X_data(train_data)
train_y = Y_train_numeric
act = np.unique(train_y)
for i in np.arange(num_labels):
    np.put(train_y, np.where(train_y==act[i]), i)
train_y = np.eye(num_labels)[train_y.astype('int')] # one-hot encoding

test_x = X_data(test_data)
test_y = Y_test_numeric
for i in np.arange(num_labels):
    np.put(test_y, np.where(test_y==act[i]), i)
test_y = np.eye(num_labels)[test_y.astype('int')]

# load data
print('Train X Data', train_x.shape,'\n', train_x.columns)
print('Train y Data', train_y.shape,'\n')
print('\nTest X Data', test_x.shape)
print('Test y Data', test_y.shape,'\n')
