# =============================================================================
# Model
# =============================================================================
import tensorflow
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import datetime
from datetime import datetime, timedelta, date
from matplotlib import ticker
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import branca
# import calmap
# import folium
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
# import pycountry_convert as pc
import random
import seaborn as sns
import tensorflow as tf
import tqdm
import os
import matplotlib.pylab as pylab
tf.__version__


def build():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(),
        config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)


def evaluate_mulistep(true, prediction):
    scores = []
    for i in range(true.shape[1]):
        mse = mean_squared_error(true[:, i], prediction[:, i])
        rmse = np.sqrt(mse)
        scores.append(mse)
        # scores.append(rmse)
    total_score = 0
    for row in range(true.shape[0]):
        for col in range(prediction.shape[1]):
            total_score = total_score + \
                true[row, col] - prediction[row, col]**2
    total_score = np.sqrt(total_score/(true.shape[0]*prediction.shape[1]))
    return total_score, mse.mean()
def train():

    lstm = layers.LSTM(32)
    timesteps = 6
    n_features = 136

    state1Input = Input(shape=(timesteps, n_features))
    state1Output = lstm(state1Input)

    state2Input = Input(shape=(timesteps, n_features))
    state2Output = lstm(state2Input)

    merged = layers.concatenate([state1Output, state2Output], axis=-1)
    predictions = layers.Dense(2, activation='sigmoid')(merged)

    modelembedded2 = Model([state1Input, state2Input], predictions)
    modelembedded2.compile(optimizer="adam", loss="mse")

    # historyembedded2 = modelembedded2.fit([X_train1Urban[:split_2],X_train1Rural[:split_2]], y_train1Urban[:split_2], epochs=20, batch_size=128, verbose=2, shuffle=True) #Works
    historyembedded2 = modelembedded2.fit(x=[X_train1Urban[:split_2], X_train1Rural[:split_2]],
                                          y=np.concatenate(
                                              [y_train1Urban[:split_2], y_train1Rural[:split_2]], axis=-1),
                                          epochs=100, batch_size=128, verbose=2, shuffle=True)
    # model.fit([state1Data,state2Data],targets)

    timesteps = 6
    n_features = 136

    # Store A and B time-series inputs
    a_inputs = Input(shape=(timesteps, n_features))
    b_inputs = Input(shape=(timesteps, n_features))

    # Stacked LSTM
    lstm_1 = LSTM(32, return_sequences=True)
    lstm_2 = LSTM(32)
    # Stacked LSTM on Store A
    a_embedding = lstm_1(a_inputs)
    a_embedding = lstm_2(a_embedding)
    # Stacked LSTM on Store B
    b_embedding = lstm_1(b_inputs)
    b_embedding = lstm_2(b_embedding)

    # Concatenate embeddings and define model
    outputs = Concatenate()([a_embedding, b_embedding])
    outputs = Dense(64)(outputs)
    outputs = Dense(2, activation="relu")(outputs)
    model = Model([a_inputs, b_inputs], outputs)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    model.fit(x=[X_train1Urban[:split_2], X_train1Rural[:split_2]],
              y=np.concatenate([y_train1Urban[:split_2],
                                y_train1Rural[:split_2]], axis=-1), epochs=10,
              batch_size=forecastBatchSize, verbose=2, shuffle=True)
    # historyUrban = model1Urban.fit(X_train1Urban[:split_2], y_train1Urban[:split_2],validation_data=(X_train1Urban[split_2:], y_train1Urban[split_2:]), epochs=forecastEpochs, batch_size=forecastBatchSize, verbose=2, shuffle=True)
    # sequenceUrbanAE.compile(optimizer='adam', loss=lossmetric, metrics=[lossmetric])
    df['date_'] = 0
    for j in range(len(df)):
        df['date_'].loc[j] = df['Date'][j].strftime('%m/%d')
    params = {'legend.fontsize': 'small',
              'figure.figsize': (8, 4),
              'axes.labelsize': '4',
              'axes.titlesize': '4',
              'xtick.labelsize': '4',
              'ytick.labelsize': '4',
              'font.family': 'Calibri'}
    pylab.rcParams.update(params)
    split_test = int(len(X_test1Urban)/4)
    for z in range(10):
        # run0 = z
        for j in range(4):
            true = y_test1Urban[split_test*j:split_test*(j+1)]
            pred = model1Urban.predict(X_test1Urban[split_test*j:split_test*(j+1)])
            pred = pred.reshape(len(pred), forecast_window, 1)
            evaluate_mulistep(true, pred)
            ax = plt.plot(true[-1]*100, marker='.', label='Original')
            ax = plt.plot(pred[-1]*100, marker='x', label='Predicted')
            plt.title('Food industry working hours -'+testStates[j]+' state - Time window: '+str(forecast_window)+' - MSE: '+str(evaluate_mulistep(true, pred)[1])
                      )
            wandb.log({'test_mse_'+testStates[j]
                      : evaluate_mulistep(true, pred)[1]})
            plt.xticks(plt.xticks()[0], df.loc[(df.State == 'FL') & (
                df.Industry == 'Food & Drink')].date_[-forecast_window:])
            plt.xticks(np.arange(forecast_window)[::5], df.loc[(df.State == 'FL') & (
                df.Industry == 'Food & Drink')].date_[-forecast_window:][::5])
            plt.xlabel('Dates')
            plt.ylabel('Percentage change relativ to January')
            plt.legend()
            plt.savefig('/content/drive/My Drive/Colab Notebooks/coviddata/fig12/' +
                        testStates[j]+'_Multi-Horizon_'+str(forecast_window)+'__'+str(evaluate_mulistep(true, pred)[1])+'.png')
            plt.show()
