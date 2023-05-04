#!/usr/bin/env python
# coding: utf-8

import sys
from datetime import timedelta,datetime
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import findspark

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col, avg, to_timestamp
from pyspark.sql.types import DateType
from pyspark.sql import DataFrameWriter

pd.set_option('max_columns', 7)

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from global_variables import (PSQL_SERVERNAME,
PSQL_PORTNUMBER,
PSQL_USERNAME,
PSQL_PASSWORD,
PSQL_DBNAME_STG,
TABLE_STG,
PSQL_DBNAME_ODS,
TABLE_ODS)

#↓ БЛОК С загрузкой слоя STG из POSTRGRESQL для ETL и создания слоя ODS↓
#Connection details
URL_STG = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_STG}"

Df_STG = spark.read \
    .format("jdbc") \
    .option("url", URL_STG)\
    .option("dbtable", TABLE_STG)\
    .option("user", PSQL_USERNAME) \
    .option("password", PSQL_PASSWORD)\
    .load()
#↑ БЛОК С загрузкой слоя STG из POSTRGRESQL для ETL и создания слоя ODS ↑

###↓ EDA $ Data Quality↓
#Df_STG.show()
#print(Df_STG.dtypes)
Df_ch = Df_STG.withColumn("Order Date_formated",
to_timestamp("Order Date","dd.MM.yyyy")) \
    .withColumn("Ship Date_formated", to_timestamp("Ship Date","dd.MM.yyyy"))

Df_ch2 = Df_ch.orderBy(F.col("Order Date_formated").asc())

Df_ch3 = Df_ch2.select("Order ID",col("Order Date_formated").cast("date").alias("Order Date"),
col("Ship Date_formated").cast("date").alias("Ship Date"),
"Ship Mode",
"Customer ID",
"Customer Name",
"Segment",
"Country",
"City",
"State",
"Postal Code",
"Region",
"Product ID",
"Category",
"Sub-Category",
"Product Name",
"Sales")

###↓ EDA $ Data Quality↓
###Df_ch3.show()
###Df_ch3.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in Df_ch3.columns)).show()


  #  we need to find the cities for which the postal code is not mentioned.
  #  Fill the postal code of the respective city into the postal code column.</b>

###↓ EDA $ Data Quality↓
###Df_ch3.where(F.col("Postal Code").isNull()).show()

#print(Df_Hyst_raw_pandas[Df_Hyst_raw_pandas['Postal Code'].isnull()])

#We can see that the postal code is not mentioned only for Burlington city in Vermont state.
#So, we need to fill the postal code of that city.

Df_ch4 = Df_ch3.na.fill({'Postal Code': '5401'})

###↓ EDA $ Data Quality↓
###Df_ch4.where(F.col("Postal Code").isNull()).show()
###Df_ch4.filter(Df_ch4['Postal Code'] == '5401').show()

#↓ БЛОК С ОТПРАВКОЙ ODS В POSTRGRESQL ↓
#Connection details
URL_ODS = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_ODS}"

def get_ods(_spark):

    _df_ods = Df_ch4
    return _df_ods

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("final_ods") \
        .getOrCreate()

    df_ods = get_ods(spark)

    df_ods.write\
        .format("jdbc")\
        .option("url", URL_ODS)\
        .option("dbtable", TABLE_ODS)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
