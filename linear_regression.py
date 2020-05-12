from google.colab import files 
uploaded = files.upload()

#import appropriate packages
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.utils import shuffle
import math
import pickle

#read  data from the file
data = pd.read_csv('student-mat.csv', sep=';')
data.head()  # to check if data is uploading


#trim our data, by only taking in the columns that we need
#lol lets check if having a relationship affects teh students marks in this case
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

#set data that we will be predicting
predicting = 'G3'     #makes code more dynamic

X = np.array(data.drop([predicting], 1))
y = np.array(data[predicting])

#checking shape of arrays, has to be equal in rows
print(X.shape)
print()
print(len(y))

#set aside testing data, 0.1 means settign aside 10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.1)


#implement the algorithm
#make object
model = linear_model.LinearRegression()

#fit model to arrays
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
score

#getting coeff and intercept
print('Coefficient is : ', model.coef_)
print('Intercept is :', model.intercept_)


#compare predictions
#make list of all teh predictions
predictions = model.predict(x_test)
for i in range(len(predictions)):
  print(round(predictions[i],1), y_test[i])


#visualize the data
plt.style.use('ggplot')
plot = "G1"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot, fontsize=18)
plt.ylabel("Final Grade", fontsize=18)
plt.show()


#load model into a file using pickle
with open('student_scores.pickle', 'wb') as f:
  pickle.dump(model, f)


#read model from pickle
pickle_in = open('student_scores.pickle', 'rb')
model = pickle.load(pickle_in)


#get model intercepts and coeff
print("-------------------------")
print('Coefficient: ', model.coef_)
print('Intercept: ', model.intercept_)
print("-------------------------")

