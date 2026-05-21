# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the placement dataset and remove unnecessary columns such as serial number and salary.
2. Convert all categorical attributes into numerical values using label encoding.
3. Split the dataset into training and testing sets using train–test split.
4. Train a Logistic Regression model using the training data.
5.Predict placement status and evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MAHAGANAPATHI R
RegisterNumber:  212225040221
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("Placement_Data (1).csv")
data = data.drop("salary", axis=1)

data = pd.get_dummies(data, drop_first=True)
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

X1 = X.iloc[:, 0].values.reshape(-1, 1)

model_plot = LogisticRegression(max_iter=1000)
model_plot.fit(X1, y)

plt.scatter(X1, y, color='blue')
x_values = np.linspace(X1.min(), X1.max(), 100)
y_values = model_plot.predict_proba(x_values.reshape(-1,1))[:,1]

plt.plot(x_values, y_values)

plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Curve")
plt.show()

```

## Output:

<img width="1918" height="914" alt="image" src="https://github.com/user-attachments/assets/834f11af-13ef-4d01-bce5-e5dae7acddae" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
