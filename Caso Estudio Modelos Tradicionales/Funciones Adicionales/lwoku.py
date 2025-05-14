# !/usr/bin/env python
"""Utility script with functions to be used in Tactic series:

https://www.kaggle.com/juanmah/tactic-00-baseline
https://www.kaggle.com/juanmah/tactic-01-test-classifiers

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pip._internal as pip
pip.main(['install', '--upgrade', 'numpy==1.17.2'])
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

__author__ = "Juanma Hernández"
__copyright__ = "Copyright 2019"
__credits__ = ["Juanma Hernández", "Kwabena"]
__license__ = "GPL"
__maintainer__ = "Juanma Hernández"
__email__ = "https://twitter.com/juanmah"
__status__ = "Utility script"

print('> NumPy version: {}'.format(np.__version__))

# Class names
CLASS_NAMES = np.array(
    [None, 'Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'])

"""" Set model parameters:

- random_state = 42. To get always the same results. And get always the same random split. 42 is the answer to the ultimate question of life, the universe, and everything.
- n_jobs = -1. Use all processors.
- verbose = 0. Per default, not nag this notebook. It could be change for testing.
"""
RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 0

# noinspection PyPep8Naming
def get_accuracy(estimator, X, y, cv=5, n_jobs=-1):
    """Wrapper to get the accuracy

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    accuracy: float
        The average score of all cross validation scores.
    """
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    return np.mean(scores)


# noinspection PyPep8Naming
def get_prediction(estimator, X, y, cv=5, n_jobs=-1):
    """Wrapper to get the prediction

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    prediction: ndarray
        The cross-validated prediction.
    """

    y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs)
    return y_pred


def plot_confusion_matrix(y, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    # Only use the labels that appear in the data
    classes = CLASS_NAMES[unique_labels(y, y_pred)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    np.set_printoptions(precision=2)

    plt.show()


def plot_features_importance(features, model):
    importances = pd.DataFrame({'Features': features,
                                'Importances': model.feature_importances_})

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

    plt.figure(figsize=(18, 6))
    plt.bar(importances['Features'], importances['Importances'])
    plt.xticks(rotation='vertical')
    plt.show()


# noinspection PyPep8Naming
def add_features(X):
    """Add features to the independent variables list
    The new features are created by calculations on the original features
    
    Most of the new features are get from Kwabena.

    Parameters
    ----------
    X : array-like
        The data to fit. Can be for example a list, or an array.

    Returns
    -------
    X : array-like
        Original data plus the new features.

    """
    X = X.copy()

    X['Hydro_Elevation_sum'] = X[['Elevation',
                                  'Vertical_Distance_To_Hydrology']
    ].sum(axis='columns')

    X['Hydro_Elevation_diff'] = X[['Elevation',
                                   'Vertical_Distance_To_Hydrology']
                                ].diff(axis='columns').iloc[:, [1]]

    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology'] ** 2 +
                                   X['Vertical_Distance_To_Hydrology'] ** 2)

    X['Hydro_Manhattan'] = (X['Horizontal_Distance_To_Hydrology'] +
                            X['Vertical_Distance_To_Hydrology'].abs())

    X['Hydro_Distance_sum'] = X[['Horizontal_Distance_To_Hydrology',
                                 'Vertical_Distance_To_Hydrology']
    ].sum(axis='columns')

    X['Hydro_Distance_diff'] = X[['Horizontal_Distance_To_Hydrology',
                                  'Vertical_Distance_To_Hydrology']
                               ].diff(axis='columns').iloc[:, [1]]

    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Fire_Points']
    ].sum(axis='columns')

    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
                           ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
    ].mean(axis='columns')

    X['Hydro_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Fire_Points']
    ].median(axis='columns')

    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Roadways']
    ].sum(axis='columns')

    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
                           ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Road_mean'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
    ].mean(axis='columns')

    X['Hydro_Road_median'] = X[['Horizontal_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Roadways']
    ].median(axis='columns')

    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',
                            'Horizontal_Distance_To_Fire_Points']
    ].sum(axis='columns')

    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
                          ].diff(axis='columns').iloc[:, [1]].abs()

    X['Road_Fire_mean'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
    ].mean(axis='columns')

    X['Road_Fire_median'] = X[['Horizontal_Distance_To_Roadways',
                               'Horizontal_Distance_To_Fire_Points']
    ].median(axis='columns')

    X['Hydro_Road_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',
                                   'Horizontal_Distance_To_Roadways',
                                   'Horizontal_Distance_To_Fire_Points']
    ].mean(axis='columns')

    X['Hydro_Road_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',
                                     'Horizontal_Distance_To_Roadways',
                                     'Horizontal_Distance_To_Fire_Points']
    ].median(axis='columns')

    X['Hillshade_sum'] = X[['Hillshade_9am',
                            'Hillshade_Noon',
                            'Hillshade_3pm']
    ].sum(axis='columns')

    X['Hillshade_mean'] = X[['Hillshade_9am',
                             'Hillshade_Noon',
                             'Hillshade_3pm']
    ].mean(axis='columns')

    X['Hillshade_median'] = X[['Hillshade_9am',
                               'Hillshade_Noon',
                               'Hillshade_3pm']
    ].median(axis='columns')

    X['Hillshade_min'] = X[['Hillshade_9am',
                            'Hillshade_Noon',
                            'Hillshade_3pm']
    ].min(axis='columns')

    X['Hillshade_max'] = X[['Hillshade_9am',
                            'Hillshade_Noon',
                            'Hillshade_3pm']
    ].max(axis='columns')

    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?
    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1,
                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0,
                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4,
                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

    # Compute Soil_Type number from Soil_Type binary columns
    X['Stoneyness'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))

    # Replace Soil_Type number with "stoneyness" value
    X['Stoneyness'] = X['Stoneyness'].replace(range(1, 41), stoneyness)

    rocks = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
             1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 1, 1, 0, 1,
             0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    X['Rocks'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
    X['Rocks'] = X['Rocks'].replace(range(1, 41), rocks)
    
    # 1=lower montane dry, 2=lower montane, 3=montane dry, 4=montane
    # 5=montane dry and montane, 6=montane and subalpine, 7=subalpine, 8=alpine  
    climatic_zone = [2, 2, 2, 2, 2, 2, 3, 3, 4, 4,
                     4, 4, 4, 5, 5, 6, 6, 6, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 8, 8, 8, 8, 8, 8]
    X['Climatic_zone'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
    X['Climatic_zone'] = X['Climatic_zone'].replace(range(1, 41), rocks)

    # 1=alluvium, 2=glacial, 3=shale, 4=sandstone
    # 5=mixed sedimentary, 6=unspecified, 7=igneous and metamorphic, 8=volcanic
    geologic_zone = [7, 7, 7, 7, 7, 7, 5, 5, 2, 7,
                     7, 7, 7, 1, 1, 1, 1, 7, 1, 1,
                     1, 2, 2, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    X['Geologic_zone'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
    X['Geologic_zone'] = X['Geologic_zone'].replace(range(1, 41), rocks)
    
    return X