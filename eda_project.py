import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
import calmap
from pandas_profiling import ProfileReport
# TASK 1
# step 1: guide python to read csv file named 'supermarket_sales.cvs'
df = pd.read_csv('supermarket_sales.csv')
# step 2: we can learn about data frame such as quick glance, first n number of row, last n number of row,
# and name of the columns
print(df.head())
# if we want to see more row we can use df.head(number of rows we want to see)
print(df.head(10))
# to see last 10 rows change head to tail
print(df.tail(3))
# print name of all the columns
print(df.columns)

# step 3: read data type:
print(df.dtypes)

# change the data type, for instance
# date in the file were initially 'object' type which is string
# we want to change it into datetime type
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'])
print(df.dtypes)

# step 4: to set date column as index column:
df.set_index('Date', inplace=True)
print(df.head())

# for all the numeric column, we can have a quick calculation on mean, std, etc.:
print(df.describe())
print(df.describe(['gross margin percentage']))
# TASK 2: UNIVARIATE ANALYSIS
# the distribution of customer rating? is it skewed?
sns.distplot(df['Rating'])
plt.show()

# QUESTION 1:
# Add a kernel density
# estimate to smooth the histogram, providing complementary information
# about the shape of the distribution:
sns.histplot(df['Rating'], kde= True)
plt.show()

# plot the mean rating in the graph
sns.histplot(df['Rating'], kde= True)
plt.axvline(x=np.mean(df['Rating']))
plt.show()

# percentile plot
sns.histplot(df['Rating'], kde= True)
plt.axvline(x=np.mean(df['Rating']),c='red',ls=':')
plt.axvline(x=np.percentile(df['Rating'],25),c='blue',ls=':')
plt.show()

# adding label to demonstrate line function
sns.histplot(df['Rating'], kde= True)
plt.axvline(x=np.mean(df['Rating']),c='red',ls=':',label='mean')
plt.axvline(x=np.percentile(df['Rating'],25),c='blue',ls=':',label='25 -75th percentile')
plt.axvline(x=np.percentile(df['Rating'],75),c='blue',ls=':')
plt.legend()
plt.show()

# get all the plots available for the dataset,
# noted that we need to add figure size to avoid mess up result
df.hist(figsize=(10,10))
plt.show()

# QUESTION 2: do aggregate numbers differ by much between branches?
# plot number of user in branches:
sns.countplot(df['Branch'])
plt.show()
# show the precise number
df['Branch'].value_counts()

# plot and show the precise number of customer for each type of payment
sns.countplot(df['Payment'])
plt.show()
df['Payment'].value_counts()

# TASK 3: BIVARIATE ANALYSIS
# Q1: Is there a relationship between gross margin and customer rating?
sns.scatterplot(df['Rating'],df['gross income'])
plt.show()
# to show the trend line of this relationship:
sns.regplot(df['Rating'],df['gross income'])
sns.relplot
# optional question: different branches shows different income?
sns.boxplot(x=df['Branch'],y=df['gross income'])
plt.show()
# respectively with gender VS gross income
sns.boxplot(x=df['Gender'],y=df['gross income'])
plt.show()

# Q2: is there a noticeable time trend in gross income?
# in the given data frame, date are repeated because there might be different customer at the same date
# aggregate the data
df.groupby(df.index).mean()
# put this into code for plotting
plt.figure(figsize=(15,8))
sns.lineplot(x=df.groupby(df.index).mean().index,
             y=df.groupby(df.index).mean()['gross income'])
plt.show()

# Q3: Dealing with duplicate rows and missing values
# count duplicated data
df.duplicated().sum()
#deal with duplicated data
df.drop_duplicates(inplace=True)
# demonstrate missing data
df.isna().sum()
# missing data and dataset ratio i.e. there is 7.9% of missing data in customer type
df.isna().sum()/len(df)
# demonstrate missing data better with heat map
sns.heatmap(df.isnull(),cbar=False) # cbar is the color indicator
plt.show()
# dealing with missing data
df.fillna(0) # fill missing data with 0
df.fillna(df.mean()) # fill missing data with mean of the column
df.fillna(df.mean(), inplace=True) # inplace means changing the value permanently
# fillna can only fill missing data for numeric column
# to fill all the missing data we use
df.fillna(df.mode().iloc[0])

# package 'Pandas profilling report'
# noted: can ONLY USED for small dataset
dataset = pd.read_csv('supermarket_sales.csv')
prof = ProfileReport(dataset)
prof

# TASK 5: CORRELATION ANALYSIS
np.corrcoef(df['gross income'],df['Rating'])
round(np.corrcoef(df['gross income'],df['Rating']) [1][0],2) # pick and rounded to 2nd decimal
# correlation matrix
round(df.corr(),2)

plt.figure(figsize=(10,10)) # resize the figure size in plots
sns.heatmap(round(df.corr(),2),annot=True)
plt.show()