#!/usr/bin/env python
# coding: utf-8
from pyspark.sql import DataFrameWriter

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
from datetime import datetime

pd.set_option('max_columns', 7)
import sys

import findspark
findspark.init()

import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col, avg
import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

import argparse
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('sales_stg_path', type=str, help='path to 0_sales_stg.csv')
args = parser.parse_args()

from global_variables import (PSQL_SERVERNAME,
PSQL_PORTNUMBER,
PSQL_DBNAME_STG,
PSQL_USERNAME,
PSQL_PASSWORD,
TABLE_STG)

#↓ БЛОК С загрузкой датафрейма из.csv:
def get_stg(_spark):

    #This method is to read .scv into a dataframe and return to the caller
    #    .option("path",args.sales_stg_path) \

    _df_stg = spark.read.format("csv") \
        .option("sep",";") \
        .option("header","true") \
        .option("inferschema", "true") \
        .option("path",args.sales_stg_path) \
        .load()
    return _df_stg

#↓ БЛОК С ОТПРАВКОЙ слоя STG В POSTRGRESQL ↓
#Connection details
URL_STG = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_STG}"

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("final_stg") \
        .getOrCreate()

    df_stg = get_stg(spark)

    df_stg.write\
        .format("jdbc")\
        .option("url", URL_STG)\
        .option("dbtable", TABLE_STG)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()

#↑ БЛОК С ОТПРАВКОЙ STG В POSTRGRESQL ↑
