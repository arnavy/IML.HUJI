import os
import random

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


# columns:
#   id
#   date - date of sale
#   price
#   bedrooms
#   bathrooms
#   sqft_living
#   sqft_lot
#   floors
#   waterfront
#   view
#   condition
#   grade
#   sqft_above
#   sqft_basement
#   yr_built
#   yr_renovated
#   zipcode
#   lat
#   long
#   sqft_living15   - avg sqft_living of closes 15 houses
#   sqft_lot15      - avg sqft_lot of closest 15 houses
def _preprocess(df):
    """
    1. drop id, date cols
    2. drop samples with negative price
    3. change categorical cols ("zipcode") to one-hot representation and drop ("long", "lat") which are contained in it
    4. accumulate ("yr_renovated", "yr_built") to a weighted average ("yr_eval")
    :param df: pandas data frame containing samples and their response
    :return: manipulated samples
    """
    df.drop(columns=["id", "date"], inplace=True)
    df.drop(np.where(df["price"] <= 0)[0], axis="index", inplace=True)

    # remove (lat, long) and change zipcode to 1 hot enc, where zipcode that appear in the lower third percentile are marked 'other_zip'
    df.loc[:, "zipcode"] = (df["zipcode"] / 10).astype(int)
    zip_counts = df["zipcode"].value_counts()
    upper_counts = (zip_counts > np.percentile(zip_counts, 33)).loc[lambda x: x].index
    df.loc[~df["zipcode"].isin(upper_counts), "zipcode"] = 'other_zip'
    zip_dummies = pd.get_dummies(df["zipcode"])
    df.drop(columns=["zipcode", "long", "lat"], inplace=True)

    # bathroom per floor average
    bath_avg = pd.DataFrame(df["bathrooms"] / df["floors"], columns=["bathrooms_floor"])
    df.drop(columns=["bathrooms"], inplace=True)

    df.fillna(0)

    return pd.concat([df, bath_avg, zip_dummies], axis=1)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if not os.path.isfile(filename):
        print("ERROR: file path doesn't exist or isn't a file! returning empty data frame :(")
        return pd.DataFrame()

    df = pd.read_csv(filename)

    df = _preprocess(df).apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)

    return df.loc[:, df.columns != "price"], df["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    ydev = np.std(y)

    for feature in X.columns:
        fvec = X[feature].to_numpy()

        # if np.all(np.isin(fvec, [0, 1])):
        #     continue

        pcor = np.cov(fvec, y) / (np.std(fvec) * ydev)

        plt.scatter(fvec, y)
        plt.title(f"{feature} to response")
        plt.xlabel(f"{feature} values")
        plt.ylabel("response values")
        plt.legend(loc='best', mode="expand",
                   handles=[Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)],
                   labels=[f"Pearson Correlation: {pcor}"])
        plt.savefig(os.path.join(output_path, f"{feature}.png"))


def play():
    df = pd.read_csv("../datasets/kc_house_data.csv")
    a = (df["zipcode"] / 10).astype(int).value_counts()
    b = (a > np.percentile(a, 33)).loc[lambda x: x].index
    c = (a < np.percentile(a, 33)).loc[lambda x: x].index
    a.plot(kind="barh")
    print(c)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/kc_house_data.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y, "/home/eran/dumps")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    loss = []
    variance = []
    sample_size = []
    train_size = X_train.shape[0]
    lr = LinearRegression(True)
    for p in range(10, 101):
        temp_loss = np.zeros(10)
        if p % 10 == 0:
            print(f"{p}. fitting")
        for i in range(10):
            current_train = np.array(random.sample(range(X_train.shape[0]), int(np.floor(p * train_size / 100))))
            lr.fit(X_train.to_numpy()[current_train], y_train.to_numpy()[current_train])

            temp_loss[i] = lr.loss(X_test.to_numpy(), y_test.to_numpy())
        sample_size.append(int(np.floor(p * train_size / 100)))
        loss.append(np.average(temp_loss))
        variance.append(np.var(temp_loss))

    loss = np.array(loss)
    variance = np.array(variance)
    data = (go.Scatter(x=sample_size, y=loss, mode="markers+lines", name="loss avg", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
            go.Scatter(x=sample_size, y=loss-2*variance, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=sample_size, y=loss+2*variance, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),)
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        title="are we gettig better?",
                        xaxis={"title": "sample size"},
                        yaxis={"title": "avg loss with error ribbon"},
                    ))
    fig.show()