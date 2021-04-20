import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error
import get_data
import math


look_back = 10
look_next = 1
trainX, trainY, testX, testY = get_data.get_data(look_back, is_reshape=True)

# create and fit the LSTM network
batch_size = 4
epochs = 50
model = Sequential()

model.add(LSTM(10, input_shape=(32, look_back), activation='selu'))
# model.add(LSTM(32, input_shape=(1, look_back), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(32))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='sigmoid'))

opt = Adam(learning_rate=1e-4)
model.compile(loss='binary_crossentropy', optimizer='adam')
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
# model.reset_states()
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# print(np.round(trainPredict[:5], 3))
