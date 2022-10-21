import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from sklearn.metrics import roc_auc_score


def download(url):
    """Downloads a file if it doesn't already exist.

    Args:
        url: string or Path

    Returns: string filename
    """
    pr = urlparse(url)
    path = Path(pr.path)
    filename = path.name

    if not Path(filename).exists():
        local_filename, headers = urlretrieve(url, filename)
        assert local_filename == filename
        print(f"Downloaded {filename}")

    return filename


def download_data_files():
    path = "https://raw.githubusercontent.com/drivendataorg/tutorial-flu-shot-learning/main/data/"
    filenames = ["training_set_features.csv", 
                 "training_set_labels.csv", 
                 "test_set_features.csv", 
                 "submission_format.csv"]
    
    for filename in filenames:
        url = f"{path}/{filename}"
        download(url)


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()

    
def crosstab(x, y):
    """Make a cross tabulation and normalize the columns as percentages.

    Args:
        x: sequence of values that go in the index
        y: sequence of values that go in the columns

    returns: DataFrame
    """
    return pd.crosstab(x, y, normalize="columns") * 100


def value_counts(seq, **options):
    """Version of value_counts that works with any sequence type.

    Args:
        seq: sequence
        options: passed to pd.Series.value_counts

    Returns: pd.Series
    """
    return pd.Series(seq).value_counts(**options)


def score_model(model, features_df, labels_df):
    """Compute the average AUC score for the two labels.

    Args:
        model: A fitted Sklearn model
        features_df: DataFrame of features
        labels_df: DataFrame of labels

    Returns: float AUC score
    """

    pred1, pred2 = model.predict_proba(features_df)

    y_pred1 = pred1.T[1]
    score1 = roc_auc_score(labels_df["h1n1_vaccine"], y_pred1)

    y_pred2 = pred2.T[1]
    score2 = roc_auc_score(labels_df["seasonal_vaccine"], y_pred2)

    return (score1 + score2) / 2


def make_submission(model, test_features_df):
    """Make a DataFrame ready for submisstion to the competition.

    Args:
        model: fitted Scikit-learn model
        test_features_df: DataFrame of features

    Returns: DataFrame of predicted probabilities
    """
    pred1, pred2 = model.predict_proba(test_features_df)
    d = dict(h1n1_vaccine=pred1.T[1], seasonal_vaccine=pred2.T[1])
    return pd.DataFrame(d, index=test_features_df.index)

