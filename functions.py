import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import time
import sklearn
import seaborn as sns
import keras.backend as K
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers import SimpleRNN as RNN
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
from functions import *

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

### LONG ONLY

def max_drawdown(vec):
    drawdown = 0.
    max_seen = vec[0]
    for val in vec[1:]:
        max_seen = max(max_seen, val)
        drawdown = max(drawdown, 1 - val / max_seen)
    return drawdown

def trader(predictions, true, threshold):

    cap_list = []

    final_capital = 1000

    price_list = []

    for i in range(len(predictions)):

        if i == 0:

            None

        else:

            initial_capital = final_capital

            price_changes = predictions[i, :] - predictions[i-1, :]

            w_3 = softmax(price_changes)



            try:
                previous_weights

            except NameError:
                previous_weights = w_3

            if np.sum(np.abs((w_3[1:] - previous_weights[1:]))) < threshold:

              w_3 = previous_weights
            weights = w_3 * initial_capital
            cost = np.abs((w_3 - previous_weights)).sum() * initial_capital * 0.0025

            units = weights/true[i-1]

            change = units * (true[i] - true[i-1])

            final_capital = sum(change) + initial_capital - cost

            cap_list.append(final_capital)

            price_list.append(true[i, 1])

            previous_weights = w_3

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Oil price', color=color)
    ax1.plot(price_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Portfolio Value', color=color)  # we already handled the x-label with ax1
    ax2.plot(cap_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    print("Return on capital:", round(final_capital/1000*100,2), "%")

    cap_list = pd.DataFrame(np.asarray(cap_list))

    capital_returns = (cap_list/cap_list.shift(periods=1))-1

    sharpe = capital_returns.mean() / capital_returns.std()  * np.sqrt(365.25)

    print("Sharpe Ratio:", round(float(sharpe),2)  )

    return final_capital


### WITH SHORTING

def short_trader(predictions, true, threshold, mode, predictions2, threshold2):

    cap_list = []

    final_capital = 1000

    price_list = []

    cost = 0

    for i in range(len(predictions)):


        if i == 0:

            None

        else:

            initial_capital = final_capital

            price_changes = predictions[i, :] - predictions[i-1, :]

            if mode == "return":

              price_changes = predictions[i,:]


            w_1 = np.abs(np.tanh(price_changes))

            w_2 = softmax(w_1)

            for j in range(len(w_2)):

                if price_changes[j] < 0:

                    w_2[j] = -w_2[j]

            w_3 = np.asarray(w_2)

            try:
                previous_weights

            except NameError:
                previous_weights = w_3

            if np.sum(np.abs((w_3[1:] - previous_weights[1:]))) < threshold:

                w_3 = previous_weights

            # if correl[i] < threshold2:
            #
            #   w_3 = previous_weights

            weights = w_3 * initial_capital

            cost = np.sum(np.abs((w_3[1:] - previous_weights[1:])))* initial_capital * 0.0025

            units = weights/true[i-1]

            change = units * (true[i] - true[i-1])

            final_capital = sum(change) + initial_capital - cost

            cap_list.append(final_capital)

            price_list.append(true[i, 1])


            previous_weights = w_3

    max_drawdown2 = max_drawdown(cap_list)
    print(max_drawdown2)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Oil price', color=color)
    ax1.plot(price_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Portfolio Value', color=color)  # we already handled the x-label with ax1
    ax2.plot(cap_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    print("Return on capital:", round(final_capital/1000*100,2), "%")

    cap_list = pd.DataFrame(np.asarray(cap_list))

    capital_returns = (cap_list/cap_list.shift(periods=1))-1

    sharpe = capital_returns.mean() / capital_returns.std()  * np.sqrt(365.25)

    print("Sharpe Ratio:", round(float(sharpe),2)  )

    return cap_lists
