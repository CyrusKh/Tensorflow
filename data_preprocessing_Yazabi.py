import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests, zipfile, io, stat, inspect, os

def make_sure_data_is_downloaded():
    """
    Checks the directory of the current file for the "UCI HAR Dataset" folder. If it
    doesn't exist, tries to download and unzip the datafile.
    """
    print("")
    save_path = os.path.dirname(os.path.abspath(__file__))
    extract_directory = os.path.abspath("UCI HAR Dataset")
    data_file = "UCI HAR Dataset.zip"

    if os.path.exists(extract_directory):
        print("UCI HAR Dataset already prepared")
        print("")
        return
    else:
        if os.path.exists(data_file):
            print("UCI HAR Dataset.zip already downloaded")
        else:
            print("Downloading UCI HAR Dataset...")
            if not os.path.exists("UCI HAR Dataset"):
                os.mkdir("UCI HAR Dataset")
            zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"
            r = requests.get(zip_file_url)
            print("UCI HAR Dataset.zip downloaded to ", save_path)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(save_path)    
            print("UCI HAR Dataset prepared")
        print("")
    
    return 


def read_data():
    """
    Given the save_path as current directory, reads the files necessary for the model 
    
    :return: har_data_x, har_data_y, har_test_x, har_test_y, activity_map
    :rtype: data frame, data frame, data frame, data frame, dictionary
    
    Note: har_data_x and har_test_x are indeed ready to be fed into the NN. They will later
    be referred to as train_x and test_x
    """
    y_label = 'activity'
    save_path = str()
    for _ in os.path.dirname(os.path.abspath(__file__)).split("\\"):
        save_path = save_path + _ + "/"
        
    features_URL = save_path + "UCI HAR Dataset/features.txt"
    har_data_labels = pd.read_csv(features_URL, sep="\s+", names=['id', 'feature_name']) #file_in_zip('features.txt')

    # Read the activity names
    activity_URL = save_path + "UCI HAR Dataset/activity_labels.txt"
    activity_map = { str(row.id):row.label_name for i, row in pd.read_csv(activity_URL, sep="\s+", names=['id', 'label_name']).iterrows() }

    # Read the X
    X_train_URL = save_path + "UCI HAR Dataset/train/X_train.txt"
    har_data_x = pd.read_csv(X_train_URL, sep="\s+", names=har_data_labels.feature_name.tolist())

    # Read the Y
    y_train_URL = save_path + "UCI HAR Dataset/train/y_train.txt"
    har_data_y = pd.read_csv(y_train_URL, sep="\s+", names=[y_label])
    
    # Read the X test
    X_test_URL = save_path + "UCI HAR Dataset/test/X_test.txt"
    har_test_x = pd.read_csv(X_test_URL, sep="\s+", names=har_data_labels.feature_name.tolist())

    # Read the Y test
    y_test_URL = save_path + "UCI HAR Dataset/test/y_test.txt"
    har_test_y = pd.read_csv(y_test_URL, sep="\s+", names=['activity'])
    return har_data_x, har_data_y, har_test_x, har_test_y, activity_map


def combine_into_df(train_x, har_data_y, test_x, har_test_y, activity_map): 
    """
    Given the iput params, combines the feature and label columns into one dataframe
    
    :param train_x: train df containing feature space
    :param har_data_y: df containing labels for train set
    :param test_x: test df containing feature space
    :param har_test_y: df containing labels for test set
    :param activity_map: 
    :return: har_data_train, har_data_test,  sorted(activity_map.items())
    :rtype: df, df, list
    """
    # Map the label into a readable form
    har_data_y['activity'] = har_data_y['activity'].apply(lambda x: activity_map[str(x)])

    # Training data, merge X and Y
    har_data_train = pd.concat([har_data_y, train_x], axis=1)

    # Map the label into a readable form
    har_test_y['activity'] = har_test_y['activity'].apply(lambda x: activity_map[str(x)])

    # Test data, merge X and Y
    har_data_test = pd.concat([har_test_y, test_x], axis=1)

    # Return the two frames as tuple
    return har_data_train, har_data_test,  sorted(activity_map.items())


def encoding(train_data, test_data):
    """
    Encodes the categorical variable (activity labels)
    
    :param train_data: combined train set with x and y
    :param test_data: combined test set with x and y
    :return: Y_train_numeric, Y_test_numeric
    :rtype: np array, ap array
    """
    from sklearn.preprocessing import LabelEncoder
    y_numbers = pd.concat([train_data['activity'], 
        test_data['activity']])  # Get all possible activity names again from the data
    # Turn the activity labels into integers
    activity_encoder = LabelEncoder().fit(y_numbers)

    # We do not need to scale the values really because they are already in the -1 to +1 range.

    #Map the activity names (like Walking, Standing, Sitting into 0, 1, 2 etc)
    Y_train_numeric = activity_encoder.transform(
        train_data["activity"])  # Training labels Y_data(train_data) har_data_y.values.ravel()
    Y_test_numeric = activity_encoder.transform(test_data['activity'])  # Test labels
    return Y_train_numeric, Y_test_numeric



def one_hot(Y_train_numeric, Y_test_numeric):
    """
    One-hots the encoded labels
    
    :param Y_train_numeric: numpy array of labels with 6 unique values [0, 1, 2, 3, 4, 5]
    :param Y_test_numeric: numpy array of labels 
    :return: train_y, test_y
    :rtype: np array, np array 
    """
    num_labels = len(np.unique(Y_test_numeric)) #6
    train_y = Y_train_numeric
    act = np.unique(train_y)
    for i in np.arange(num_labels):
        np.put(train_y, np.where(train_y==act[i]), i)
    train_y = np.eye(num_labels)[train_y.astype('int')] # one-hot encoding
    
    test_y = Y_test_numeric
    for i in np.arange(num_labels):
        np.put(test_y, np.where(test_y==act[i]), i)
    test_y = np.eye(num_labels)[test_y.astype('int')]
    return train_y, test_y
