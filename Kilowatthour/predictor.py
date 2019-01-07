# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
#girl_dataset = pd.read_csv('data.csv')
#training_set = girl_dataset.iloc[:,2:3].values
#girl_training_set = girl_training_set.astype('float32')
dataset_train = pd.read_csv('data.csv') #pd.read_csv('Google_Stock_Price_Train.csv')
        
training_set = dataset_train.iloc[:-500, 2:3].values

for i in range(len(training_set)):
    if np.isnan(training_set[i]):
        training_set[i]=0
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(48, len(training_set)):
    X_train.append(training_set_scaled[i-48:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 128)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_price = dataset_train.iloc[-500:, 2:3].values #dataset_test.iloc[:, 1:2].values
for i in range(len(real_price)):
    if np.isnan(real_price[i]):
        real_price[i]=0
# Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train['Cena'], dataset_train['Cena']), axis = 0)
inputs = real_price #dataset_total[len(dataset_total) - len(dataset_train) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(48, 500):
    X_test.append(inputs[i-48:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = regressor.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)
x_axis  = dataset_train.iloc[-500:, 0:1]
x_axis1 = []
x = []
for i in range(0, 500-48):
    x.append(i)
for i in range(len(x_axis)):
    x_axis1.append(x_axis.iloc[i,0])
# Visualising the results
plt.figure()
#plt.plot(x_axis1[48:],real_price[48:], color = 'red', label = 'realna cena')
plt.xticks(x, x_axis1)
plt.plot(x,real_price[48:], color = 'red', label = 'realna cena')
plt.show()
plt.plot(predicted_price, color = 'blue', label = 'Przewidziana cena')
plt.title('Przewidywanie ceny')
plt.xlabel('Czas')
plt.ylabel('Cena')
plt.legend()
plt.show()
