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
param1 = 1.60
param2 = 0.15
## TAKE OPEN PRICE OF EVERYTHING EXCEPT FUTURES PRICE, FOR WHICH WE TRY TO PREDICT THE CLOSE


one_month = pd.read_excel('data/1 MONTH.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
two_month = pd.read_excel('data/2 MONTH.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
three_month = pd.read_excel('data/3 MONTH.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
four_month = pd.read_excel('data/4 month.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
five_month = pd.read_excel('data/5 month.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
six_month = pd.read_excel('data/6 month.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
fed = pd.read_excel('data/fed.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
dollar_index = pd.read_excel('data/dollar index 3.xlsx', sheet_name = 0, index_col = 0).iloc[:,1]
inventory = pd.read_excel('data/inventory.xlsx', sheet_name = 0, index_col = 0).iloc[:,0]
sp500 = pd.read_excel('data/s&p.xlsx', sheet_name = 0, index_col = 0).iloc[:,1]
production = pd.read_excel('data/WCRFPUS2w.xls', sheet_name = 0, index_col = 0).iloc[:,0]
inven_r = pd.read_excel('data/inven_r.xls', sheet_name = 0, index_col =0).iloc[:,0]

features = [one_month, two_month, three_month, four_month, five_month, six_month, fed, dollar_index, inventory, sp500, production]

df = one_month.copy()
features_2 = [two_month, three_month, four_month, five_month, six_month, fed, dollar_index, inventory, sp500, production]
for i in features_2:
    df = pd.concat([df, i], axis = 1)
df.columns = ['one_month', 'two_month', 'three_month', 'four_month', 'five_month', 'six_month', 'fed', 'dollar_index', 'inventory', 'sp500', 'production']
### FILLING IN BLANKS WITH PREVIOUS VALUES
df = df.fillna(method='ffill')
df = df[1000:5850]
### NEW FEATURE, PRODUCTION/INVENTORY
df['inventory'] = df['inventory'].shift(periods=1)
df['production'] = df['production'].shift(periods=1)
df['Ratio'] = df['inventory']/df['production']
### ONE DAY AHEAD RETURN (DAILY REBALANCING)

column_names = ['one_month', 'two_month', 'three_month', 'four_month', 'five_month', 'six_month', 'fed', 'dollar_index', 'inventory', 'sp500', 'production']

for d in column_names:
    df[d+' return'] = (df[d]/df.shift(periods=1)[d])-1


for d in column_names:
    df[d+' change'] = df[d]-df.shift(periods=1)[d]

df = df[1:-1]

### MOVING AVERAGE, 2 DAYS

### NOT USED

for d in column_names:
    df[d+' Movavg'] = df[d+' return'].rolling(window=2).mean().shift(periods=1)

df = df[5:-1]


column_names = ['one_month', 'two_month', 'three_month', 'four_month', 'five_month', 'six_month']
for d in column_names:
    df['Label '+d] = df[d+' return'].shift(periods=-1)
df = df[9:-1]


def rnn(month, features, true_price, d):

  train_length = int(0.8*len(month))

  x_train = features[:train_length]
  x_test = features[train_length:]
  label_train = month[:train_length].reshape(-1, 1)
  label_test = month[train_length:].reshape(-1, 1)
  true = true_price[train_length:]

  scaler = StandardScaler()
  scaler = scaler.fit(x_train)

  x_train = scaler.transform(x_train)
  x_test = scaler.transform(x_test)

#   scaler2 = StandardScaler()
#   scaler2 = scaler2.fit(label_train)

#   label_train = scaler2.transform(label_train)
#   label_test = scaler2.transform(label_test)

  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  K.clear_session()

  model_rnn = Sequential()
  model_rnn.add(RNN(20, input_shape=(x_train.shape[1], 1), activation='relu', return_sequences=True))
  model_rnn.add(Dropout(0.2))
#   model_rnn.add(RNN(64, activation='relu', return_sequences=True))
#   model_rnn.add(Dropout(0.2))
  model_rnn.add(RNN(10, activation='relu', return_sequences=False))
  model_rnn.add(Dropout(0.2))
  model_rnn.add(Dense(1))


  model_rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
  early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

  model_rnn.fit(x_train, label_train, epochs = 1000, batch_size = 32, verbose = 0)

  # evaluate the keras model
  predicted_y = model_rnn.predict(x_test)

  predicted_y = predicted_y.reshape(len(predicted_y),)

  return true, predicted_y


  number = 0

final_rnn = np.ones((967))

true_rnn = np.ones((967))

column_names = ['one_month', 'two_month', 'three_month', 'four_month', 'five_month', 'six_month']

for d in column_names:

    number = number + 1

    features_dataset = df[['fed', 'dollar_index', 'sp500', 'inventory', 'production', 'Ratio']].values

    label = df[d].values

    label2 = df[d].values

    true, prediction = rnn(label, features_dataset, label2, d + 'new23')

    final_rnn = np.vstack((final_rnn, prediction))

    true_rnn = np.vstack((true_rnn, true))

final_rnn = np.transpose(final_rnn)
true_rnn = np.transpose(true_rnn)

rnn_trader = short_trader(final_rnn[:600], true_rnn[:600], param1, "price", inven_r[:600], param2)
