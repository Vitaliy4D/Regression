from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# X_train and test are from LinearRegression split

poly = PolynomialFeatures(degree=2, include_bias=False)

poly_features = poly.fit_transform(X_train.array.reshape(-1, 1))

poly_features_test = poly.fit_transform(X_test.array.reshape(-1, 1))



poly_reg_model = LinearRegression()

# fit the model
poly_reg_model.fit(poly_features, y_train)

# prediction of train set
y_predicted = poly_reg_model.predict(poly_features)
print(y_predicted)


# prediction of test set
y_predicted_test = poly_reg_model.predict(poly_features_test)
print(y_predicted_test)


# ploting the training dataset in scattered graph
plt.scatter(X_train, y_train, color='blue')

# ploting the training dataset in line line
plt.plot(X_train, y_predicted, color='red')
plt.title('Prices vs OverallQual')




# ploting the test dataset in scattered graph
plt.scatter(X_test, y_test, color='blue')

# ploting the testing dataset in line line
plt.plot(X_test, y_predicted_test, color='red')
plt.title('Prices vs OverallQual')



# Importing metrics from sklearn module to evaluate our model

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_error,r2_score 
from math import sqrt

print('R squared train: {:.2f}'.format(poly_reg_model.score(poly_features, y_train.array.reshape(-1, 1))*100))
print('R squared test: {:.2f}'.format(poly_reg_model.score(poly_features_test, y_test.array.reshape(-1, 1))*100))

# printing the mean absolute error
print(f'mean_absolute_error train: {mean_absolute_error(X_train, y_predicted):.2f}') # actual_values, predicted_values

# printing the mean absolute error
print(f'mean_absolute_error test: {mean_absolute_error(y_test, y_predicted_test):.2f}')

# printing the mean squared error
print(f'mean_squared_error train: {mean_squared_error(X_train, y_predicted):.2f}') # actual_values, predicted_values

# printing the mean squared error
print(f'mean_squared_error test: {mean_squared_error(y_test, y_predicted_test):.2f}')

# printing the root mean absolute error
print(f'Root Mean Square Error train: {sqrt(mean_squared_error(X_train, y_predicted)):.2f}') # actual_values, predicted_values

# printing the mean squared error
print(f'Root Mean Square Error test: {sqrt(mean_squared_error(y_test, y_predicted_test)):.2f}')

## Polinomial model better fit with 5% better in train and test than LinearRegression but mean_absolute_error still the same
# model better predict but overall not so good explain all dataset

# R squared train: 65.12
# R squared test: 72.83
# mean_absolute_error train: 181529.31
# mean_absolute_error test: 30824.99
# mean_squared_error train: 36824187238.86
# mean_squared_error test: 1941495797.69
# Root Mean Square Error train: 191896.29
# Root Mean Square Error test: 44062.41
