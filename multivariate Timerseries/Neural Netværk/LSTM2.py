# -*- coding: utf-8 -*-
"""
Lstm Multivariate Multi-Step

Created on Wed Oct 18 23:58:55 2023

@author: Mads Hansen
"""

###############################################################################
############################ Libraries ########################################
###############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
from sklearn.preprocessing import LabelEncoder
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

###############################################################################
############################## Custom functions ###############################
###############################################################################

def metrics(y_test, y_pred):
    """
    Computes and prints several evaluation metrics for regression models.
    
    Parameters:
    - y_test (array): True target values.
    - y_pred (array): Predicted target values from the model.
    
    Prints:
    - MSE, RMSE, MAE, R^2, and MAPE values.
    """
    # Previous metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = 100 * (sum(abs((y_test - y_pred) / y_test)) / len(y_test))
    
    print(f" MSE = {mse}\nRMSE = {rmse}\n MAE = {mae}\n  %R^2% = {r2}\n MAPE = {mape}%")

###############################################################################
############################ Load Data ########################################
###############################################################################

# Load the electricity consumption dataset
filepath = 'C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Speciale/Forecasting-energy-consumption-in-Denmark/Data/Combined data/'
dataset = pd.read_csv(filepath + 'combined_daily_flagged.csv')

# Put HourDK as DataFrame index
dataset.set_index('HourDK', inplace=True)
t = dataset.columns.tolist()

dataset = dataset[['GrossConsumptionMWh','flagged','t2m','step_days',
                   'Is_Holiday','Day_Monday','Day_Tuesday','Day_Wednesday',
                   'Day_Thursday','Day_Friday','Day_Saturday','Day_Sunday',
                   'Month_January','Month_February','Month_March','Month_April',
                   'Month_May','Month_June','Month_July','Month_August',
                   'Month_September','Month_October','Month_November',
                   'Month_December']]

#else slice is invalid for use in labelEncoder
dataset = dataset.values

# integer encode direction
encoder = LabelEncoder()
dataset[:,3] = encoder.fit_transform(dataset[:,3])

#convert to pd.Dataframe else slices error
dataset = pd.DataFrame(dataset)
dataset.columns = ['GrossConsumptionMWh','flagged','t2m','step_days',
                   'Is_Holiday','Day_Monday','Day_Tuesday','Day_Wednesday',
                   'Day_Thursday','Day_Friday','Day_Saturday','Day_Sunday',
                   'Month_January','Month_February','Month_March','Month_April',
                   'Month_May','Month_June','Month_July','Month_August',
                   'Month_September','Month_October','Month_November',
                   'Month_December']

###############################################################################
####################### Data Pre-processing Step  #############################
###############################################################################

#dataset[['GrossConsumptionMWh','flagged','t2m','step_days',
#                   'Is_Holiday','Day_Monday','Day_Tuesday','Day_Wednesday',
#                   'Day_Thursday','Day_Friday','Day_Saturday','Day_Sunday',
#                   'Month_January','Month_February','Month_March','Month_April',
#                   'Month_May','Month_June','Month_July','Month_August',
#                   'Month_September','Month_October','Month_November',
#                   'Month_December']]

#Data Pre-processing step--------------------------------
x_1  = dataset['flagged'].values
x_2  = dataset['t2m'].values
x_3  = dataset['step_days'].values
x_4  = dataset['Is_Holiday'].values
x_5  = dataset['Day_Monday'].values
x_6  = dataset['Day_Tuesday'].values
x_7  = dataset['Day_Wednesday'].values
x_8  = dataset['Day_Thursday'].values
x_9  = dataset['Day_Friday'].values
x_10  = dataset['Day_Saturday'].values
x_11 = dataset['Day_Sunday'].values
x_12 = dataset['Month_January'].values
x_13 = dataset['Month_February'].values
x_14 = dataset['Month_March'].values
x_15 = dataset['Month_April'].values
x_16 = dataset['Month_May'].values
x_17 = dataset['Month_June'].values
x_18 = dataset['Month_July'].values
x_19 = dataset['Month_August'].values
x_20 = dataset['Month_September'].values
x_21 = dataset['Month_October'].values
x_22 = dataset['Month_November'].values
x_23 = dataset['Month_December'].values

y = dataset['GrossConsumptionMWh'].values

# Step 1 : convert to [rows, columns] structure
x_1  = x_1.reshape((len(x_1), 1))
x_2  = x_2.reshape((len(x_2), 1))
x_3  = x_3.reshape((len(x_3), 1))
x_4  = x_4.reshape((len(x_4), 1))
x_5  = x_5.reshape((len(x_5), 1))
x_6  = x_6.reshape((len(x_6), 1))
x_7  = x_7.reshape((len(x_7), 1))
x_8  = x_8.reshape((len(x_8), 1))
x_9  = x_9.reshape((len(x_9), 1))
x_10 = x_10.reshape((len(x_10), 1))
x_11 = x_11.reshape((len(x_11), 1))
x_12 = x_12.reshape((len(x_12), 1))
x_13 = x_13.reshape((len(x_13), 1))
x_14 = x_14.reshape((len(x_14), 1))
x_15 = x_15.reshape((len(x_15), 1))
x_16 = x_16.reshape((len(x_16), 1))
x_17 = x_17.reshape((len(x_17), 1))
x_18 = x_18.reshape((len(x_18), 1))
x_19 = x_19.reshape((len(x_19), 1))
x_20 = x_20.reshape((len(x_20), 1))
x_21 = x_21.reshape((len(x_21), 1))
x_22 = x_22.reshape((len(x_22), 1))
x_23 = x_23.reshape((len(x_23), 1))

y = y.reshape((len(y), 1))
print ("x_1.shape" , x_1.shape) 
print ("x_2.shape" , x_2.shape) 
print ("y.shape" , y.shape)

# Step 2 : normalization (only on continues)
scaler = MinMaxScaler(feature_range=(0, 1))
x_2_scaled = scaler.fit_transform(x_2)
x_3_scaled = scaler.fit_transform(x_3)

y_scaled = scaler.fit_transform(y)

# Step 3 : horizontally stack columns
dataset_stacked = hstack((x_1, x_2_scaled,x_2_scaled, x_3_scaled,
                          x_4, x_5,x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, 
                          x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, 
                          x_23, y_scaled))
print ("dataset_stacked.shape" , dataset_stacked.shape)

###############################################################################
########################### Split the sequence  ###############################
###############################################################################

#1. n_steps_in : Specify how much data we want to look back for prediction
#2. n_step_out : Specify how much multi-step data we want to forecast

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequences)):
  # find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out-1
  # check if we are beyond the dataset
  if out_end_ix > len(sequences):
   break
  # gather input and output parts of the pattern
  seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)

# choose a number of time steps #change this accordingly
n_steps_in, n_steps_out = 365 , 184

# covert into input/output
X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)
print ("X.shape" , X.shape) 
print ("y.shape" , y.shape)

###############################################################################
###################### Do the Train and Test split  ###########################
###############################################################################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train_X, test_X,train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

#split_point = 1258*25
#train_X , train_y = X[:split_point, :] , y[:split_point, :]
#test_X , test_y = X[split_point:, :] , y[split_point:, :]

train_X.shape #[n_datasets,n_steps_in,n_features]
train_y.shape #[n_datasets,n_steps_out]
test_X.shape 
test_y.shape 
n_features = 24

#number of features
#n_features = 2
#optimizer learning rate
from tensorflow.keras.layers import LeakyReLU
opt = keras.optimizers.Adam(learning_rate=0.0001)

# define model
model = Sequential()
model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.compile(loss='mse' , optimizer=opt , metrics=['accuracy'])

# Fit network
history = model.fit(train_X, train_y, epochs=5000, batch_size=256, verbose=1, validation_split=0.2, shuffle=False)


###############################################################################
############################## time to predict  ###############################
###############################################################################

y_pred = model.predict(test_X)
y_pred = loaded_model.predict(test_X)
y_pred_inv = scaler.inverse_transform(y_pred)

test_y_inv = scaler.inverse_transform(test_y)

#print("y_pred :",y_pred.shape)
#print("y_pred_inv :",test_y_inv.shape)

# Get the prediction for the first sequence
single_sequence_pred = y_pred_inv[1205]
single_sequence_true = test_y_inv[1205]

metrics(single_sequence_true,single_sequence_pred)


#test_y_inv
#y_pred_inv



# Plotting the real and predicted values for better visualization
plt.figure(figsize=(14, 7))

# Plotting the real values
plt.plot(np.array(single_sequence_true), label='Real Values', marker='o')

# Plotting the predicted values
plt.plot(single_sequence_pred, label='Predicted Values', marker='x')

# Adding labels, title, and legend
plt.xlabel('HourDK')
plt.ylabel('Gross Consumption (MWh)')
plt.title('Real vs Predicted Gross Consumption')
plt.legend()

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

# Show only every 7th date
#plt.xticks(np.arange(0, len(test_X), 7), test_X.index[::7]) # Replace 'hour_dk_values' with your actual array of dates

# Show the plot
plt.show()



#Save
save_path = r'C:\Users\madsh\OneDrive\Dokumenter\kandidat\Fællesmappe\Speciale\Forecasting-energy-consumption-in-Denmark\multivariate Timerseries\Neural Netværk'
model_path = os.path.join(save_path, 'my_model.2L_50-tahn-50-tahn-V2.keras')
loaded_model.save(model_path)

#Load
from tensorflow.keras.models import load_model
loaded_model = load_model(model_path)

###############################################################################
########################### Retrain saved model  ##############################
###############################################################################
# Model summary
loaded_model.summary()

# Details about layers
for layer in loaded_model.layers:
    print("Layer name:", layer.name)
    print("Layer type:", layer.__class__.__name__)
    print("Activation function:", getattr(layer, "activation", None).__name__ if getattr(layer, "activation", None) is not None else None)
    print("Number of nodes (units):", getattr(layer, "units", None))
    print("-------")
    
#Restart training
# Continue training
history = loaded_model.fit(train_X, train_y, epochs=5000, batch_size=256, verbose=1, validation_split=0.2, shuffle=False)
