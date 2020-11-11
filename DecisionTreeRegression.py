import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

dataset = pd.read_csv('resources/phone_cost.csv')

dataset.head()

dataset.describe()

X = dataset.drop('price_range', axis=1)  #This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).
y = dataset['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
