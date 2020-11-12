# Random Forest: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor # Random Forest
from sklearn import metrics

def RandomTree(dataset):
  # -----------Preparing Data For Training---------#
  X = dataset.iloc[:, 0:4].values
  y = dataset.iloc[:, 4].values

  # ------divide the data into training and testing sets------#
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  # ------Feature Scaling------#
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  # ------Training the Algorithm------#
  regressor = RandomForestRegressor(n_estimators=20, random_state=0)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)


  # ------Evaluating the Algorithm------#
  print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
  print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
  print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


dataset = pd.read_csv("petrol_consumption.csv")  # import the dataset
print(dataset.head())  # To get a high-level view of what the dataset looks like


