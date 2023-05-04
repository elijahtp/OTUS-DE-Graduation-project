#!/usr/bin/env python
# coding: utf-8

#import the necessary Libraries
from pyspark.sql import DataFrameWriter

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from datetime import timedelta
from datetime import datetime

import sys

import findspark
findspark.init()

import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col, avg
import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from global_variables import PSQL_SERVERNAME,PSQL_PORTNUMBER,PSQL_USERNAME,PSQL_PASSWORD,PSQL_DBNAME_STG,TABLE_STG,PSQL_DBNAME_ODS,TABLE_ODS,PSQL_DBNAME_FORECAST,TABLE_FORECAST

pd.set_option('max_columns', 7)

#↓ БЛОК С загрузкой ODS из POSTRGRESQL для создания прогноза ↓
#Connection details
URL_ODS = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_ODS}"

Df_ODS = spark.read \
    .format("jdbc") \
    .option("url", URL_ODS)\
    .option("dbtable", TABLE_ODS)\
    .option("user", PSQL_USERNAME) \
    .option("password", PSQL_PASSWORD)\
    .load()
 #↑ БЛОК С загрузкой ODS из POSTRGRESQL для создания прогноза ↑

#↓СОЗДАНИЕ ВИТРИН↓
df=Df_ODS.toPandas()


#↓БЛОК Time Series Analysis↓

#In this notebook, we will perform time series analysis to get the sales for forecasting for next 7 days.

# Understanding the distribution of the concerned data. This will display information about numeric columns only.
print('describe')
df.describe()


#Understanding the type of data in every columns of the data set that we will be dealing with.
print('info')
df.info()


# Dropping the column 'Row ID', as it does not help much in the process of data analysis of the dataset.
#df.drop('Row ID',axis = 1, inplace = True)

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%Y-%m-%d') #converting the data type of 'Order Date' column to date time format
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%Y-%m-%d') #converting the data type of 'Ship Date' column to date time format
df.info()


print('Order Date Description\n')
print(df['Order Date'].describe()) #Displays the distribution of dates in 'Order Data' column
print('\nShip Date Description\n')
print(df['Ship Date'].describe()) #Displays the distribution of dates in 'Ship Data' column

#sorting data by order date
#df.sort_values(by=['Order Date'], inplace=True, ascending=True) #Sorting data by  ascending order of the coloumn values 'Order Date'
df.set_index("Order Date", inplace = True) #Setting 'Order Date' as index of the dataframe 'df' for ease of Time Series Analysis

# To forecast sales seven days later of the order date, let us create a new dataframe with only the target column i.e,
# the 'Sales' column and 'Order Date' as the index

new_data = pd.DataFrame(df['Sales'])
print('new_data')
print(new_data)


#Plotting the data to understand the sales distribution from the year 2015-2018
new_data.plot();

#A series is said to be stationary when its mean and variance do not change over time. From the above distribution of the sales it is not clear whether the sales distribution is stationary or not. Let us perform some stationarity tests to check whether the time series is stationary or not.

# # Checkting for Stationarity
new_data =  pd.DataFrame(new_data['Sales'].resample('D').mean())
new_data = new_data.interpolate(method='linear') #The interpolate() function is used to interpolate values according to
#different methods. It ignore the index and treats the values as equally spaced.

# Method 1
# To check for stationarity by comparing the change in mean and variance over time, let us split the data into train, test and validate.
train, test, validate = np.split(new_data['Sales'].sample(frac=1), [int(.6*len(new_data['Sales'])),int(.8*len(new_data['Sales']))])

print('Train Dataset')
print(train)
print('Test Dataset')
print(test)
print('Validate Dataset')
print(validate)


mean1, mean2, mean3 = train.mean(), test.mean(), validate.mean() #taking mean of train, test and validate data
var1, var2, var3 = train.var(), test.var(), validate.var() #taking variance of train, test and validate data

print('Mean:')
print(mean1, mean2, mean3)
print('Variance:')
print(var1, var2, var3)


#From the above values of mean and variance, it can be inferred that their is not much difference in the three values of mean and variance, indicating that the series is stationary. However, to verify our observations, let us perform a standard stationarity test, called Augmented Dicky Fuller test.

"""
Augmented Dicky Fuller test

    The Augmented Dickey-Fuller test is a type of statistical test alsocalled a unit root test.The base of unit root test is that it helps in determining how strongly a time series is defined by a trend.

    The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary. The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
        Null Hypothesis(H0): Time series is not stationary
        Alternate Hypothesis (H1): Time series is stationary

    This result is interpreted using the p-value from the test.
        p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
        p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
"""
# Method 2
# Augmented Dicky Fuller Test

from statsmodels.tsa.stattools import adfuller #importing adfuller tool from statsmodels
#statsmodels provide adfuller() fucntion to implement stationarity test of a time series

adf = adfuller(new_data)

print(adf)
print('\nADF = ', str(adf[0])) #more towards negative value the better
print('\np-value = ', str(adf[1]))
print('\nCritical Values: ')

for key, val in adf[4].items(): #for loop to print the p-value (1%, 5% and 10%) and their respective values
    print(key,':',val)


    if adf[0] < val:
        print('Null Hypothesis Rejected. Time Series is Stationary')
    else:
        print('Null Hypothesis Accepted. Time Series is not Stationary')


from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(new_data, model='additive') #function used to decompose Time Series Data into Trend and Seasonality
fig = decomposition.plot()
plt.show();

"""
Now that we know our time series is data is stationary. Let us begin with model training for forecasting the sales. We have chosen SARIMA model to forecast the sales.

Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that supports univariate time series data with a seasonal component.

SARIMA requires selecting hyperparameters for both the trend and seasonal elements of the series.

    Trend Elements There are three trend elements that require configuration.

p: Trend autoregression order. d: Trend difference order. q: Trend moving average order.

    Seasonal Elements There are four seasonal elements:

P: Seasonal autoregressive order. D: Seasonal difference order. Q: Seasonal moving average order. m: The number of time steps for a single seasonal period.

The notation for a SARIMA model is specified as: SARIMA(p,d,q)(P,D,Q)m
"""

import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq_comb = [(i[0], i[1], i[2], 12) for i in list(itertools.product(p, d, q))] #for loop for creating combinations of seasonal parameters of SARIMA
print('Examples of parameter combinations for Seasonal ARIMA:')
print('SARIMA: {} x {}'.format(pdq[1], seasonal_pdq_comb[1]))
print('SARIMA: {} x {}'.format(pdq[1], seasonal_pdq_comb[2]))
print('SARIMA: {} x {}'.format(pdq[2], seasonal_pdq_comb[3]))
print('SARIMA: {} x {}'.format(pdq[2], seasonal_pdq_comb[4]))

#Examples of parameter combinations for Seasonal ARIMA:
#SARIMA: (0, 0, 1) x (0, 0, 1, 12)
#SARIMA: (0, 0, 1) x (0, 1, 0, 12)
#SARIMA: (0, 1, 0) x (0, 1, 1, 12)
#SARIMA: (0, 1, 0) x (1, 0, 0, 12)

for parameters in pdq: #for loop for determining the best combination of seasonal parameters for SARIMA
    for seasonal_param in seasonal_pdq_comb:
        try:
            mod = sm.tsa.statespace.SARIMAX(new_data,
                                            order=parameters,
                                            seasonal_param_order=seasonal_param,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False) #determines the AIC value of the model**
            results = mod.fit()
            print('SARIMA{}x{}12 - AIC:{}'.format(parameters, seasonal_param, results.aic))
        except:
            continue

# **The Akaike information criterion (AIC) is an estimator of out-of-sample prediction error and thereby relative
# quality of statistical models for a given set of data. AIC estimates the relative amount of information lost
# by a given model. The less information a model loses, the higher the quality of that model.



# After choosing the combination of seasonal parameters with least AIC value, let us train the SARIMA model
mod = sm.tsa.statespace.SARIMAX(new_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False) #model defintion
results = mod.fit() #model fitting
print(results.summary().tables[1]) # displaying the result

"""
results.plot_diagnostics(figsize=(16, 8)) #Produces a plot grid of: 1. Standardized residuals over time
# 2. Histogram plus estimated density of standardized residulas and along with a Normal(0,1) density plotted for reference.
# 3. Normal Q-Q plot, with Normal reference line and, 4. Correlogram.
plt.show()
"""

pred = results.get_prediction(start=pd.to_datetime('2020-01-03'), dynamic=False) # variable to display plot for predicted values
pred_val = pred.conf_int()
"""
ax = new_data['2014':].plot(label='observed') # displays plot for original values
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7)) # displays plot for predicted values
ax.fill_between(pred_val.index,
                pred_val.iloc[:, 0],
                pred_val.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show()
"""
y_forecasted = pred.predicted_mean
y_truth = new_data['Sales']

from sklearn.metrics import mean_squared_error
from math import sqrt

mse = mean_squared_error(y_forecasted, y_truth)
rmse = sqrt(mse)
print(rmse)
#print(round(rmse, 2))
#print('The Mean Squared Error of the forecasts is {}'.format(round(rmse, 2))) # displays the root mean squared error of the forecast with rounding it up to 2 decimals

#The Mean Squared Error of the forecasts is 267.66

#Out of Sample forecast:

#To forecast sales values after some time period of the given data. In our case, we have to forecast sales with time period of 7 days.

# mod = sm.tsa.statespace.SARIMAX(new_data,
#                                 order=(1, 1, 1),
#                                 seasonal_order=(1, 1, 1, 12),
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False) #model defintion
# results = mod.fit() #model fitting

forecast = results.forecast(steps=7) # making a forecast of 7 days later of the last date in the 'Order Date' column
print('forecast')
print(forecast.astype('int')) #displays the sales forecast as type integer

forecast = forecast.astype('int') #saving the sales values as type integer

forecast_df = forecast.to_frame() # forecast is in Series form, converting it to DataFrame
forecast_df.reset_index(level=0, inplace=True) # converting the index to column
forecast_df.columns = ['Prediction Date', 'Predicted Sales'] # giving appropriate names to the output columns
#prediction = pd.DataFrame(forecast_df).to_csv('prediction.csv',index=False) # saving the output as a csv file with name 'prediction.csv'

                                #↓ БЛОК С ОТПРАВКОЙ FORECAST В POSTGRESQL ↓

"""
This moudle is to write data into PostgresSQL db
"""
forecast_df_to_postgress=spark.createDataFrame(forecast_df)
#forecast_df_ch1 = forecast_df_to_postgress.withColumn("Prediction Date", to_timestamp("Order Date","dd.MM.yyyy")).withColumn("Ship Date_formated", to_timestamp("Ship Date","dd.MM.yyyy"))





#Df_ch = forecast_df_to_postgress
#Df_ch = forecast_df_to_postgress.select(date_format(col("Prediction Date"),"dd.MM.yyy").alias("Prediction Date"),"Predicted Sales")
Df_ch = forecast_df_to_postgress.select(col("Prediction Date").cast("date").alias("Prediction Date"),"Predicted Sales")
Df_ch = Df_ch.orderBy(F.col("Prediction Date").asc());

#Connection details

URL_FORECAST = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_FORECAST}"


def get_forecast(_spark):

    _df_forecast = Df_ch
    return _df_forecast


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("PostgrsSQL demo") \
        .getOrCreate()

    df_forecast = get_forecast(spark)

    df_forecast.write\
        .format("jdbc")\
        .option("url", URL_FORECAST)\
        .option("dbtable", TABLE_FORECAST)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()