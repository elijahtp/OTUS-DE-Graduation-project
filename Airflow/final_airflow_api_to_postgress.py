# coding: utf8

import logging as _log
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator

import pydantic
import requests
import json

import datetime

from typing import Optional
from typing import List
from pydantic import BaseModel, ValidationError

# Global variables
PG_CONN_ID = 'postgres_final_api'
PG_TABLE = 'ods'
URL_RESPONSE_SALES_TODAY = requests.get('https://retoolapi.dev/ZyBkfL/superstore_api')
DT = datetime.datetime.now()- datetime.timedelta(days=1)
DT_TODAY = DT.strftime('%Y-%m-%d')

# Check connection to api
def conn_sales_today():
    try:
        URL_RESPONSE_SALES_TODAY.raise_for_status()
    except Exception as ex:
        print(ex)
    _log.info('connecting to superstore api')

# Getting string with sales of past day
def get_sales_today():
    dict_of_values = json.loads(URL_RESPONSE_SALES_TODAY.text)
    l = 0
    table_string = {}
    table_str_result = ''
    for i in range(len(dict_of_values)):
        if dict_of_values[i]['Order Date'] == DT_TODAY:
            table_string[l] = []
            table_string[l].extend((dict_of_values[i]['Order ID'],
            dict_of_values[i]['Order Date'],
            dict_of_values[i]['Ship Date'],
            dict_of_values[i]['Ship Mode'],
            dict_of_values[i]['Customer ID'],
            dict_of_values[i]['Customer Name'],
            dict_of_values[i]['Segment'],
            dict_of_values[i]['Country'],
            dict_of_values[i]['City'],
            dict_of_values[i]['State'],
            dict_of_values[i]['Postal Code'],
            dict_of_values[i]['Region'],
            dict_of_values[i]['Product ID'],
            dict_of_values[i]['Category'],
            dict_of_values[i]['Sub-Category'],
            dict_of_values[i]['Product Name'],
            dict_of_values[i]['Sales']))
            l = l + 1
    for n in range(len(table_string)):
        table_str_list = []
        for k in range(len(table_string[n])):
            table_str_obg = str(table_string[n][k])
            table_str_list.append("{}".format(table_str_obg))
        table_str = str(table_str_list)[1:-1]
        table_str_redact = """({0})""".format(table_str)
        table_str_result += (table_str_redact + ',')
    _log.info('getting strings values')
    return table_str_result[0:-1]

# Push to postgresql database to the ODS layer
def put_to_psql():
    population_string = """ CREATE TABLE IF NOT EXISTS {0}
                            ("Order ID" TEXT,"Order Date" DATE,"Ship Date" DATE,
                            "Ship Mode" TEXT,"Customer ID" TEXT,"Customer Name" TEXT,
                            "Segment" TEXT,"Country" TEXT,"City" TEXT,"State" TEXT,
                            "Postal Code" INT,"Region" TEXT,"Product ID" TEXT,
                            "Category" TEXT,"Sub-Category" TEXT,"Product Name" TEXT,"Sales" FLOAT8);
                            INSERT INTO {0}
                            ("Order ID","Order Date","Ship Date","Ship Mode",
                            "Customer ID","Customer Name","Segment","Country",
                            "City","State","Postal Code","Region","Product ID",
                            "Category","Sub-Category","Product Name","Sales")
                            VALUES {1};
                        """ \
                        .format(PG_TABLE, get_sales_today())
    _log.info('table created and populated successfully')
    return population_string


args = {
    'owner': 'airflow',
    'catchup': 'False',
    'depends_on_past': False,
    'email': ['krasilnikoviy@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id='final_sales_today_to_postgress',
    default_args=args,
    description='connect to api, get today sales and push to postgress',
    schedule_interval='0 16 * * *',
    start_date=days_ago(2),
    max_active_runs=1,
    tags=['4otus', 'API'],
) as dag:
    task_connect_sales_today = PythonOperator(
        task_id='connect_to_api_sales_today',
        python_callable=conn_sales_today,
        dag=dag
    )

    task_get_sales_today = PythonOperator(
        task_id='get_sales_today',
        python_callable=get_sales_today,
        dag=dag
    )

    task_populate = PostgresOperator(
        task_id="put_to_psql_sales_today",
        postgres_conn_id=PG_CONN_ID,
        sql=put_to_psql(),
        dag=dag
    )

#Task Dependencies
task_connect_sales_today >> task_get_sales_today >> task_populate
