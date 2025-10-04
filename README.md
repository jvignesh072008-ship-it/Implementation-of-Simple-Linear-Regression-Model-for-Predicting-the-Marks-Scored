# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data collection and preparation First, you gather a dataset with pairs of data, such as the number of hours studied and the corresponding marks scored. The data is then cleaned and formatted. In Python, a common practice is to use libraries like Pandas and NumPy to load the data and reshape it, with the hours studied becoming the independent variable (X) and the marks scored becoming the dependent variable (Y).

2. Splitting the dataset To evaluate the model's predictive power, the dataset is divided into two parts: a training set and a testing set. A standard practice is to use 80% of the data to train the model and the remaining 20% to test it. Libraries like Scikit-learn's train_test_split function are used to perform this split.
  
3. Training the model Using the training data, the linear regression model is trained to find the line of best fit. This line minimizes the sum of the squared differences between the actual marks and the marks predicted by the line, a process known as the least squares method. The model learns the optimal values for the slope and intercept (b) in the linear equation
 
4. Prediction After the model has been trained, it can make predictions on the unseen test data. The predict() function from Scikit-learn can be applied to the test set of study hours (X_test) to forecast the corresponding marks (y_pred). This step demonstrates how well the model generalizes to new data points.

5. Evaluation and visualization The final step is to evaluate the performance of the model and visualize its predictions. Metrics such as the Mean Squared Error (MSE) and the R-squared score are used to quantify the model's accuracy. A scatter plot can then be created to visually compare the actual test data points against the regression line, which represents the model's predictions. A strong model will show the regression line closely following the trend of the data points. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VIGNESH J
RegisterNumber:  25014705
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv('student_scores.csv')
df.head()
df.tail()

# Extract features and target
X = df.iloc[:, 0].values.reshape(-1,1)
print(*X)  #unpacks and displays only the elements in X 
Y = df.iloc[:, 1].values
print(*Y)  #unpacks and displays only the elements in Y

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Make predictions
Y_pred = regressor.predict(X_test)
print(*Y_pred)
print(*Y_test)

# Visualize training set results
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Visualize testing set results
plt.scatter(X_test, Y_test, color="blue")  # Fixed variable names x_test to X_test and y_test to Y_test
plt.plot(X_test, regressor.predict(X_test), color="green")  # Fixed variable names x_test to X_test and reg to regressor
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Calculate metrics
mae = mean_absolute_error(Y_test, Y_pred)  # Fixed y_test to Y_test
mse = mean_squared_error(Y_test, Y_pred)  # Fixed y_test to Y_test
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="968" height="762" alt="Screenshot 2025-10-04 211248" src="https://github.com/user-attachments/assets/4a517cc1-df5e-4e1a-9f68-a287a018eca2" />
<img width="949" height="693" alt="Screenshot 2025-10-04 211301" src="https://github.com/user-attachments/assets/749bbfe1-aaca-4064-a683-3c3266c22d63" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
