import requests
import pandas
import json

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def get_data(sensor):

    # Get data from the API
    sensor_data = requests.get('https://augeias-post.ece.uowm.gr/v1/' + sensor + '/all')

    # Jsonfy them
    json_data = json.loads(sensor_data.content)

    # Create a dataframe from the data
    df = pandas.DataFrame.from_records(json_data)

    # Drop the 2 columns we don't want
    df.drop('timestamp', inplace=True, axis=1)
    df.drop('application_group', inplace=True, axis=1)

    #
    df.dropna(inplace=True)

    # Sort based on index
    df.sort_index(inplace=True)

    # Split dataset
    train, test = train_test_split(df)

    return train, test


def train_data(train_data, test_data, clf):

    # fit the dataset to the model
    clf.fit(train_data)

    # raw outlier scores
    train_scores = clf.decision_scores_

    # Get the test score with teh confidence
    test_scores = clf.decision_function(test_data)  # outlier scores

    # it is possible to get the prediction confidence as well, predict and get confidences
    test_pred, test_pred_confidence = clf.predict(test_data, return_confidence=True)

    test_results = {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'test_pred': test_pred,
        'test_pred_confidence': test_pred_confidence
    }

    return test_results


def plot_results(test_data, y_test_pred):

    # Get the column names from the dataset in order to create the plot
    columns = test_data.columns.values

    # Create the put
    plt.scatter(test_data[columns[0]], test_data[columns[1]], s=10, c=y_test_pred, cmap='brg')

    # Put the labels
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])

    # Plot the plot
    plt.show()


if __name__ == '__main__':

    # Create the variables for defining the classifiers
    outliers_fraction = 0.01
    random_state = np.random.RandomState(42)

    # The names of all the sensors for the API, later we will parse through them
    sensors = ['atmos', 'aquatroll', 'triscan', 'scanchlori', 'teros', 'addvantage']

    # The main classifiers to be used for training, they will also parse thought them in later state
    classifiers = {
        'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'OneClassSVM': OCSVM(),
        'Autoencoder':  AutoEncoder(contamination=0.1, random_state=random_state, hidden_neurons=[1, 2, 2, 1], verbose=0)
    }

    results = {}

    num_of_test = 0

    # Parse through the classifiers
    for clf in classifiers:

        # Parse through the sensors
        for s in sensors:
            # Get the data from sensor
            train_dataset, test_dataset = get_data(s)

            # Train the data and get the results
            data_results = train_data(train_dataset, test_dataset, classifiers[clf])

            results[num_of_test] = {'classifier': clf,
                                    'sensor': s,
                                    'data_results': data_results}

            num_of_test = num_of_test + 1
