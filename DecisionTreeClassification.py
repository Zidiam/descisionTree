import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Importing our dataset
dataset = pd.read_csv("resources/phone_cost.csv")

# Shows rows and columns of our dataset
dataset.shape

# Shows first 5 records in dataset
dataset.head()

# Dividing the data into attrivuted and labels
X = dataset.drop('price_range', axis=1)  # Holds all the columns except Class
y = dataset['price_range']  # Holds all the values for class dolumn

# Getting training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train the data using a classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Now predict data on the trained data
y_pred = classifier.predict(X_test)

# Calculate how accurate the predictions are
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
