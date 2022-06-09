import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
dataset = pd.read_csv("C:/ML/Housing Prices Competition/train.csv")

# let do some exploration of dataset
dataset.head()

dataset.describe()

#### let see and draw some barplot how many features with NA
col_na=dataset.columns[dataset.isna().any()].tolist()

col_na_sum=[]
for i in col_na:
    col_na_sum.append(dataset[i].isna().sum())
    
df_col_na = ({
    'colnames':col_na,
    'na_quantity':col_na_sum
})

print(df_col_na)

import matplotlib.pyplot as plt

x_na=col_na
y_na=col_na_sum

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(x_na, sorted(y_na), color='green')
ax.bar_label(bars)
plt.title('Columns with NA')
plt.xlabel('quantity of NA')
plt.ylabel('Columns names')
plt.show
####


# have a look at dependent variable
sns.histplot(dataset['SalePrice'])

# let me see what is the features correlation
dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)
d_corr = dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(dataset.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

heatmap.set_title('Features Correlating with SalePrice', fontdict={'fontsize':18}, pad=16)



# let us choose columns where correlation greater that 0,05 to make a regression
col_corr = d_corr.loc[d_corr['SalePrice'] >= 0.05]
col_corr_names= col_corr.index[1:].tolist()
    
print(col_corr_names)

# explore data for outliers with boxplot
fig, ax = plt.subplots(1, 4, figsize=(10, 6))
plt.subplots_adjust(wspace=0.5) 

boxplot = dataset.boxplot(['OverallQual'],ax=ax[0], color='brown',)
ax[0].set_xlabel('OverallQual')
boxplot = dataset.boxplot(['GrLivArea'],ax=ax[1], color='g',)
ax[1].set_xlabel('OverallQual')
boxplot = dataset.boxplot(['GarageCars'],ax=ax[2], color='y',)
ax[2].set_xlabel('OverallQual')
boxplot = dataset.boxplot(['GarageArea'],ax=ax[3],)
ax[3].set_xlabel('OverallQual')

# alternative for box plot, to iterate 
# create boxplot with a different y scale for different rows
selection = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']
fig, axes = plt.subplots(1, len(selection),figsize=(10, 6))
plt.subplots_adjust(wspace=1) 
for i, col in enumerate(selection):
    ax = sns.boxplot(y=dataset[col], ax=axes.flatten()[i])
    ax.set_ylim(dataset[col].min(), dataset[col].max())
    ax.set_ylabel(col + ' / Unit')
plt.show()

# let us see how many outliers are in our data
# count how many outliers in our features
for x in dataset[col_corr_names]:
    q75,q25 = np.percentile(dataset[col_corr_names].loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    print(x,dataset[col_corr_names].loc[dataset[col_corr_names][x] < min,x].count())
    print(x,dataset[col_corr_names].loc[dataset[col_corr_names][x] > max,x].count())

# in this case removing outliers and replacing it with mean did not improved our model
# it made our model overfitted
# to replace outliers change code above '.count' to '= np.nan'

# inspect the data of choosen correlated columns
pd.set_option('display.max_columns', None)
dataset[col_corr_names].head()

# let's check what type of data in the columns? 
dataset[col_corr_names].dtypes

# Linear regression doesn't work on date data. Therefore we need to convert it 
# into numerical value.The following code will convert the date into numerical value:

# import datetime as dt
# data_df['Date'] = pd.to_datetime(data_df['Date'])
# data_df['Date']=data_df['Date'].map(dt.datetime.toordinal)

# df.date = df.date.map(datetime.toordinal)

# convert back to datetime
# pd.to_datetime('20220101', format='%Y%m%d', errors='coerce')

# checking if choosen columns have NA,  include='all', exclude=[np.number], include = [np.number]
dataset[col_corr_names].describe(include=[None])

# alternative check if correlated columns have NA
# dataset[col_corr_names].columns[dataset[col_corr_names].isna().any()].tolist()

# few columns have NA, let's count na in columns
[dataset[i].isna().sum() for i in ['GarageYrBlt','MasVnrArea','LotFrontage']]

# now let see how many % of total of na in the column
[(dataset[i].isna().sum()/dataset[i].count())*100 for i in ['GarageYrBlt','MasVnrArea','LotFrontage']]

# define what is correlation of these columns, 
# is it important to include it into our mult.regression
d_corr.loc[['GarageYrBlt', 'MasVnrArea', 'LotFrontage']]

# for feature GarageYrBlt - would use mode(The most common value) to impute NA, 
# hist is not equally distrib and amount of NA is not so big
sns.histplot(dataset['GarageYrBlt']) 

# for MasVnrArea - would also use mode(The most common value) to impute NA, hist is not equally distrib
sns.histplot(dataset['MasVnrArea']) 

# column LotFrontage - I should drop it, because 21% of NA is too many and it can influence the final result
# but it seems hist looks like normal so I will use mean for experimental purposes
sns.histplot(dataset['LotFrontage']) 

# let's fill na in three columns
dataset['GarageYrBlt']=dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mode().iloc[0])
dataset['MasVnrArea']=dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mode().iloc[0])
dataset['LotFrontage']=dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

# count na in columns one more time to be sure fill na is done
[dataset[i].isna().sum() for i in ['GarageYrBlt','MasVnrArea','LotFrontage']]

# as alternative to imputation, we can drop columns with NA from our dataset, 
# if it's correlation small (less than 0.05)

# list_corr_na = dataset[col_corr_names].columns[dataset[col_corr_names].isna().any()].tolist()
# fin_corr_list = [x for x in col_corr_names if not x in list_corr_na or list_corr_na.remove(x)]
# print(fin_corr_list)

# set variables for our regression
X_t = dataset[col_corr_names]
y_t=dataset['SalePrice'].values

# let's make test/train split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_t, y_t, test_size = 0.3, random_state = 100)

#Fitting the Multiple Linear Regression model
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()  
mlr.fit(x_train, y_train)

# let check Intercept and Coefficients for our selected features
from operator import itemgetter

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(sorted(zip(X_t, mlr.coef_),key=itemgetter(1),reverse=True))

#Prediction of test set
y_pred_mlr= mlr.predict(x_test)

#Predicted values
print(f"Prediction for test set: {y_pred_mlr}")

#Actual value and the predicted value, lists are flattened from list of lists
# mlr_diff = pd.DataFrame({'Actual value': [x for xs in y_test.values for x in xs], 'Predicted value': [x for xs in y_pred_mlr for x in xs]})
# mlr_diff.head()
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
print(mlr_diff)

#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared train: {:.2f}'.format(mlr.score(X_t, y_t)*100))
print('R squared test: {:.2f}'.format(mlr.score(x_test, y_test)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

# R Squared: R Square is the coefficient of determination. 
# It tells us how many points fall on the regression line. 
# The value of R Square is 79.72 for train data amd 80.67 for our test data
# which indicates that 79.72% of the data fit the regression model.

# Mean Absolute Error: Mean Absolute Error is the absolute difference 
# between the actual or true values and the predicted values. 
# The lower the value, the better is the model’s performance. 
# A mean absolute error of 0 means that your model is 
# a perfect predictor of the outputs. 
# The mean absolute error obtained for this particular model is 22960, 
# which is not so good as it is not really close to 0.

# Mean Square Error: Mean Square Error 
# is calculated by taking the average of the square of the difference 
# between the original and predicted values of the data. 
# The lower the value, the better is the model’s performance. 
# The mean square error obtained for this particular model is 1244157729, 
# which is not really good

# Root Mean Square Error: Root Mean Square Error 
# is the standard deviation of the errors which occur 
# when a prediction is made on a dataset. 
# This is the same as Mean Squared Error, 
# but the root of the value is considered 
# while determining the accuracy of the model. 
# The lower the value, the better is the model’s performance. 
# The root mean square error obtained for this particular model 
# is 35272, which is not so good.

# for this particular set it is better to try some other ML models