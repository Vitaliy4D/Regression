from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# loading data and splitting to train and test

dataset = pd.read_csv('C:/ML/Housing_Prices_Competition/train.csv')

x_i = dataset['OverallQual']
y_i = dataset['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(x_i, y_i, train_size=.68, random_state=15)


from sklearn.isotonic import IsotonicRegression

iso_reg = IsotonicRegression().fit(X_train, y_train)

y_iso_pred = iso_reg.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_error,r2_score 
from math import sqrt

print('R squared train: {:.2f}'.format(iso_reg.score(X_train, y_train)*100))
print('R squared test: {:.2f}'.format(iso_reg.score(X_test, y_test)*100))

# printing the mean absolute error
print(f'mean_absolute_error train: {mean_absolute_error(X_train, iso_reg.predict(X_train)):.2f}') # actual_values, predicted_values

# printing the mean absolute error
print(f'mean_absolute_error test: {mean_absolute_error(y_test, y_iso_pred):.2f}')

# printing the mean squared error
print(f'mean_squared_error train: {mean_squared_error(X_train, iso_reg.predict(X_train)):.2f}') # actual_values, predicted_values

# printing the mean squared error
print(f'mean_squared_error test: {mean_squared_error(y_test, y_iso_pred):.2f}')

# printing the root mean absolute error
print(f'Root Mean Square Error train: {sqrt(mean_squared_error(X_train, iso_reg.predict(X_train))):.2f}') # actual_values, predicted_values

# printing the mean squared error
print(f'Root Mean Square Error test: {sqrt(mean_squared_error(y_test, y_iso_pred)):.2f}')


# ploting the training dataset in scattered graph
plt.scatter(X_train, y_train, color='blue')

# ploting the testing dataset in line 
plt.plot(X_train, iso_reg.predict(X_train), color='red')
plt.title('Prices vs OverallQual')

# labeling the input and outputs
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')

# showing the graph
plt.show()

#####
# ploting the training dataset in scattered graph
plt.scatter(X_test, y_test, color='blue')

# ploting the testing dataset in line line
plt.plot(X_test, iso_reg.predict(X_test), color='red')
plt.title('Prices vs OverallQual')

# labeling the input and outputs
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')

# showing the graph
plt.show()

# IsotonicRegression performs better that LinearRegression but tend to overfit