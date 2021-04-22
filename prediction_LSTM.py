import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error
import get_data
import math


# # in case we want to crease a new data set
# get_data.create_dataset()

look_back = 20
look_next = 1
num_numbers_chosen = 32
trainX, trainY, testX, testY = get_data.get_data(look_back, num_numbers_chosen=num_numbers_chosen, is_reshape=True)

# create and fit the LSTM network
batch_size = 4
epochs = 1
model = Sequential()

model.add(LSTM(1000, input_shape=(num_numbers_chosen, look_back), activation='elu'))
# model.add(LSTM(32, input_shape=(1, look_back), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(32))
# model.add(Dense(1000, activation='elu'))
# model.add(Dropout(0.2))
# # model.add(Dense(1000, activation='selu'))
model.add(Dense(num_numbers_chosen, activation='elu'))

opt = Adam(learning_rate=1e-2)
model.compile(loss='binary_crossentropy', optimizer=opt)
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

# get the predicted lottery numbers
num_numbers_predict = 10
if num_numbers_predict >= num_numbers_chosen: num_numbers_predict = num_numbers_chosen
num_overlap = []
for i in range(len(testY)):
    arg_sort = np.argsort(-testPredict[i])
    numberPredict = np.zeros(num_numbers_chosen)    # there are 32 numbers
    for j in range(num_numbers_predict):              # choose 5 numbers with the highest possibility
        numberPredict[arg_sort[j]] = 1

    # find how many numbers overlapped with the true winning numbers
    num_overlap.append(np.sum(numberPredict * testY[i]))

num_overlap = np.array(num_overlap)

print("\nwith 10 numbers chosen")
print("\nhow many predictions with at least 2 numbers overlapped over total %d times?" % len(testY))
print(np.sum(num_overlap >= 2))

print("\nhow many predictions with at least 3 numbers overlapped over total %d times?" % len(testY))
print(np.sum(num_overlap >= 3))

print("\nhow many predictions with at least 4 numbers overlapped over total %d times?" % len(testY))
print(np.sum(num_overlap >= 4))

print("\nhow many predictions with at least 5 numbers overlapped over total %d times?" % len(testY))
print(np.sum(num_overlap >= 5))

# predict a number for tomorrow
testPredict = model.predict(np.array([testX[-1]]))
arg_sort = np.argsort(-testPredict[0])
print("\nPrediction for tomorrow ordered by probability: ")
print(arg_sort[:num_numbers_predict]+1)
print("\nProbability:")
print(np.round(testPredict[0, arg_sort[:num_numbers_predict]], 4))

