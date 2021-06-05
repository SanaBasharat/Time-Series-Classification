import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from collections import OrderedDict

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("TSLA.csv")
print('Number of rows and columns:', df.shape)
df.head(5)
print(df.shape[0])

training_set = df.iloc[:2000, 1:2].values    #first 2000 samples in training set
test_set = df.iloc[2000:, 1:2].values        #2000 onwards in testing set

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))   #scales each value to between 0 and 1
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

# Creating a data structure with 60 time-steps and 1 output
def create_dataset(df):
    x = []
    y = []
    for i in range(60, df.shape[0]):
        x.append(df[i-60:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

X_train, y_train = create_dataset(training_set_scaled)
X_train[:1]

X_test, y_test = create_dataset(test_set_scaled)
X_test[:1]

# Reshape features for LSTM Layer
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  #MSE because it's a regression problem

# Fitting the RNN to the Training set
if(not os.path.exists('stock_prediction.h5')):
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    model.save('stock_prediction.h5')

model = load_model('stock_prediction.h5')

predicted_stock_price = model.predict(X_test).reshape(-1, 1)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

af = df['Open'].values
af = af.reshape(-1,1)   #converting to numpy for pyplot

fig, ax = plt.subplots(figsize=(8,4))
plt.plot(af, color='red',  label="True Price")
ax.plot(range(len(y_train)+50,len(y_train)+50+len(predicted_stock_price)),predicted_stock_price, color='blue', label='Predicted Testing Price')
plt.legend()

y_test_scaled = sc.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(8,4))
plt.plot(y_test_scaled, color='red', label='True Testing Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Testing Price')
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

        ##trying to output some metrics
# from sklearn.metrics import classification_report
# y_pred = model.predict(X_test, batch_size=64, verbose=1)
# y_pred = (y_pred>0.5)
# y_pred_bool = np.argmax(y_pred, axis=1)

# y_pred=model.predict(X_test).reshape(-1, 1)
# y_pred =(y_pred>0.5)
# y_pred_bool = np.argmax(y_pred, axis=1)
# y_pred = np.hstack((1-y_pred,y_pred))
# print(classification_report(y_pred, y_pred_bool))


#LIME-ing the model
newX_train, newy_train = create_dataset(training_set)
newX_test, newy_test = create_dataset(test_set)
# training the random forest model
rf_model = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=-1, random_state=10)
rf_model.fit(newX_train, newy_train)

# print(newX_test)

# creating the explainer function
explainer = lime.lime_tabular.LimeTabularExplainer(newX_train, verbose=True, mode="regression")

# storing a new observation
i = 6
# X_observation = newX_test[[i], :]
X_observation = newX_test[i]

# explanation using the random forest model
# print("observation: ")
# print(X_observation)
explanation = explainer.explain_instance(X_observation, rf_model.predict)
explanation.save_to_file("lstm.html")
print("score")
print(explanation.score)
