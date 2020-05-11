import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('ggplot')
global days
days = 40

#get the stock quote for teh company

quotes = web.DataReader('MCD', data_source = 'yahoo', start = '2005-01-01', end = '2020-05-01')

#show data
quotes


#visualize the closing proce of the stock history
plt.figure(figsize = (16,8))
plt.title('close price history MCD')
plt.plot(quotes['Close'])
plt.xlabel('Date', fontsize =18)
plt.ylabel('closed USD price ($)', fontsize =18)
plt.show()


#create a new dataframe with only the close column
data =  quotes.filter(['Close'])
                      
#covert data to numpy array
dataset = data.values


#get number of rows to train the data on
train_data_len = math.ceil(len(dataset)*0.9)

print(train_data_len)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#create the training dataset
#also create the scaled training data set subsequently

train_data = scaled_data[0:train_data_len,:]
print(train_data)
print(train_data.shape)

#split into x and y training data sets

x_train=[]
y_train=[]

for i in range(days, len(train_data)):
  x_train.append(train_data[i-days:i,0])
  y_train.append(train_data[i,0])
  




#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1 ))
x_train.shape

#Build teh LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


#create testing data set
test_data = scaled_data[train_data_len-days:,:]
#create the data sets x_test and y_test
x_test=[]
y_test = dataset[train_data_len,:]
for i in range(days, len(test_data)):
  x_test.append(test_data[i-days:i,0])


#convert data to  a numpy array
x_test = np.array(x_test)
x_test.shape

#reshape the data to 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)



#get the root mean squared error (RMSE)
rmse = (np.sqrt(np.mean(((predictions- y_test)**2))))
rmse

#plot the data
train = data [:train_data_len]
valid = data[train_data_len:]
valid['Predictions']=predictions

#visualize model/data
plt.figure(figsize=(16,8))
plt.title('MOdel of predicted APPle close price in USD')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


print(valid['Close'][30] )
print(valid['Predictions'][30])
