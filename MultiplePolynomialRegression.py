import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

# load and inspect data
dataset = pd.read_csv('C:/ML/Housing_Prices_Competition/train.csv')

dataset.head()

# let me see what is the features correlation
dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)
d_corr = dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)

col_corr = d_corr.loc[d_corr['SalePrice'] >= 0.05]
col_corr_names= col_corr.index[1:].tolist()
    
print(col_corr_names)

# inspect the data of choosen correlated columns
pd.set_option('display.max_columns', None)
dataset[col_corr_names].head()

# choose few columns from dataset
dataset[col_corr_names].iloc[:,:9]


# create multiple scatter plot of features
x_p = dataset[col_corr_names].iloc[:,:9]
y_p = dataset['SalePrice']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 4))
plt.subplots_adjust(wspace=0.6,hspace=0.7) 

for i,n in enumerate(x_p):
    if i<3:
        axes[0,i].scatter(dataset[n], y_p)
        axes[0,i].set_title(n)
    elif 3<=i<=5:
        axes[1,i-3].scatter(dataset[n], y_p)
        axes[1,i-3].set_title(n)
    elif 6<=i<=8:
        axes[2,i-6].scatter(dataset[n], y_p)    
        axes[2,i-6].set_title(n) 
plt.show()

# create multiple histograms of features
x_p = dataset[col_corr_names].iloc[:,:9]
y_p = dataset['SalePrice']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 4))
plt.subplots_adjust(wspace=0.6,hspace=0.7) 

for i,n in enumerate(x_p):
    if i<3:
        axes[0,i].hist(dataset[n])
        axes[0,i].set_title(n)
    elif 3<=i<=5:
        axes[1,i-3].hist(dataset[n])
        axes[1,i-3].set_title(n)
    elif 6<=i<=8:
        axes[2,i-6].hist(dataset[n])    
        axes[2,i-6].set_title(n) 
plt.show()

# have a look at pairplot
x_y_p=pd.concat([x_p,y_p],axis=1)
sns.pairplot(x_y_p, hue = 'SalePrice')
# sns.pairplot(x_y_p, hue = 'SalePrice', diag_kind = 'kde',
#              plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
#              height = 4)

# check na
dataset[col_corr_names].describe(include=[None])

# fill na with mean
dataset['GarageYrBlt']=dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean())
dataset['MasVnrArea']=dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())
dataset['LotFrontage']=dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

# fill na in all columns
# for m in dataset[col_corr_names]:
#    dataset[m]=dataset[m].fillna(dataset[m].mean())

# explore if job is done
dataset[col_corr_names].isna().sum()

# poly
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

X, y = dataset[col_corr_names], dataset['SalePrice']

poly = PolynomialFeatures(degree=2, include_bias=False)

poly_features = poly.fit_transform(X)

Xp_train, Xp_test, yp_train, yp_test = train_test_split(poly_features, y, test_size=0.33, random_state=24)

# fit
poly_reg_model = LinearRegression()
poly_reg_model.fit(Xp_train, yp_train)

poly_reg_y_predicted = poly_reg_model.predict(Xp_test)

from sklearn.metrics import mean_squared_error

poly_reg_rmse = np.sqrt(mean_squared_error(yp_test, poly_reg_y_predicted))
poly_reg_rmse

# for this model rmse is worse than multilinearRegression

# RMSE shows how far the values your model predicts (poly_reg_y_predicted) 
# are from the true values (y_test), on average. Roughly speaking: 
# the smaller the RMSE, the better the model.

#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(yp_test, poly_reg_y_predicted)
meanSqErr = metrics.mean_squared_error(yp_test, poly_reg_y_predicted)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(yp_test, poly_reg_y_predicted))
print('R squared train: {:.2f}'.format(poly_reg_model.score(Xp_train, yp_train)*100))
print('R squared test: {:.2f}'.format(poly_reg_model.score(Xp_test, yp_test)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

# overfitted for train data

# R2 is negative only when the chosen model does not follow the trend of the data, 
# so fits worse than a horizontal line.
