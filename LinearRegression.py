import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# load and inspect data
dataset = pd.read_csv('C:/ML/Housing_Prices_Competition/train.csv')

dataset.head()


# plot features correlation in order to choose most correlated feature for analysis
# SalePrice is our dependable variable

dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)
d_corr = dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

heatmap.set_title('Features Correlating with SalePrice', fontdict={'fontsize':18}, pad=16)


# inspect if our choosen feature has no NA

dataset['OverallQual'].isna().sum()

# explore our feature for ourliers and distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
plt.subplots_adjust(wspace=0.5) 

boxplot = dataset.boxplot(['OverallQual'],ax=ax[0], color='blue',)
ax[0].set_xlabel('OverallQual')
sns.histplot(dataset['OverallQual'],ax=ax[1], color='g',)
ax[1].set_xlabel('OverallQual')

# we can replace outliers with na, and na replace with mean(mode)

x_percentile = dataset['OverallQual']

q75,q25 = np.percentile(x_percentile,[75,25])
intr_qr = q75-q25

max = q75+(1.5*intr_qr)
min = q25-(1.5*intr_qr)

x_percentile[x_percentile < min] = np.nan # .cout - to count how many
x_percentile[x_percentile > max] = np.nan


# vizualise our x and y

x = dataset['OverallQual']
y = dataset['SalePrice']
plt.scatter(x, y)
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.title('OverallQual vs SalePrice')
plt.show() 


# Scipy: Calculate a linear least-squares regression for two sets of measurements

slope, intercept, r, p, std_err = stats.linregress(x, y)

res = stats.linregress(x, y)
print(f"R-squared: {res.rvalue**2*100:.4f}")

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.plot(x, y, 'o', label='original data')
plt.plot(x, mymodel, 'r', label='fitted line')
plt.legend()
plt.show() 


# Calculate LinearRegression with sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# loading data and splitting to train and test

dataset = pd.read_csv('C:/ML/Housing Prices Competition/train.csv')

x = dataset['OverallQual'].array.reshape(-1, 1) 
y = dataset['SalePrice'].array.reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.7, random_state=1)

regr = LinearRegression()


# Fitting Simple Linear Regression to the Training set
regr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regr.predict(X_test)


# ploting the training dataset in scattered graph
plt.scatter(X_train, y_train, color='blue')

# ploting the testing dataset in line line
plt.plot(X_train, regr.predict(X_train), color='red')
plt.title('Prices vs OverallQual')

# labeling the input and outputs
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')

# showing the graph
plt.show()


# Visualysing the Test set results
viz_test = plt

# red dot colors for actual values
viz_test.scatter(X_test, y_test, color='blue')

# Blue line for the predicted values
viz_test.plot(X_test, regr.predict(X_test), color='red')

# defining the title
viz_test.title('OverallQual vs SalePrice')

# x lable
viz_test.xlabel('OverallQual')

# y label
viz_test.ylabel('SalePrice')

# showing the graph
viz_test.show()


# Importing metrics from sklearn module to evaluate our model

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_error,r2_score 
from math import sqrt

print('R squared train: {:.2f}'.format(regr.score(X_train, y_train)*100))
print('R squared test: {:.2f}'.format(regr.score(X_test, y_test)*100))

# printing the mean absolute error
print(f'mean_absolute_error train: {mean_absolute_error(X_train, regr.predict(X_train)):.2f}') # actual_values, predicted_values

# printing the mean absolute error
print(f'mean_absolute_error test: {mean_absolute_error(y_test, y_pred):.2f}')

# printing the mean squared error
print(f'mean_squared_error train: {mean_squared_error(X_train, regr.predict(X_train)):.2f}') # actual_values, predicted_values

# printing the mean squared error
print(f'mean_squared_error test: {mean_squared_error(y_test, y_pred):.2f}')

# printing the root mean absolute error
print(f'Root Mean Square Error train: {sqrt(mean_squared_error(X_train, regr.predict(X_train))):.2f}') # actual_values, predicted_values

# printing the mean squared error
print(f'Root Mean Square Error test: {sqrt(mean_squared_error(y_test, y_pred)):.2f}')


# applying r square error
# The R-Squared Error method is also known as 
# the coefficient of determination. 
# This metric indicates how well a model fits a given dataset. 
# Or in simple words, it indicates how close the regression line 
# is to the actual data values.
R_square = r2_score(X_train, regr.predict(X_train)) # actual_values, predicted_values
print(R_square)


# applying r square error, % how close our model predict 
R_square_train = r2_score(y_train, regr.predict(X_train))*100
R_square = r2_score(y_test, y_pred)*100

print(R_square_train)
print(R_square)