#!/usr/bin/env python
# coding: utf-8
from pyspark.sql import DataFrameWriter
from pyspark.sql.functions import *

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

from global_variables import (PSQL_SERVERNAME,
PSQL_PORTNUMBER,PSQL_USERNAME,PSQL_PASSWORD,
PSQL_DBNAME_STG,TABLE_STG,PSQL_DBNAME_ODS,TABLE_ODS,
PSQL_DBNAME_DATAMART,
TABLE_TOP_CUSTOMERS,
TABLE_TOP_CITIES,
TABLE_TOP_CATEGORIES,
TABLE_TOP_PRODUCTS,
TABLE_TOP_SUB_CATEGORIES,
TABLE_TOP_SUB_CATEGORIES_ALL,
TABLE_TOP_SEGMENTS,
TABLE_TOP_REGIONS,
TABLE_TOP_SHIPPINGS)

#↓ БЛОК С загрузкой ODS из POSTRGRESQL для создания витрин ↓
#Connection details
URL_ODS = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_ODS}"

Df_ODS = spark.read \
    .format("jdbc") \
    .option("url", URL_ODS)\
    .option("dbtable", TABLE_ODS)\
    .option("user", PSQL_USERNAME) \
    .option("password", PSQL_PASSWORD)\
    .load()

#↑ БЛОК С загрузкой ODS из POSTRGRESQL для создания витрин ↑

#↓СОЗДАНИЕ ВИТРИН↓

Df_ODS_pandas=Df_ODS.toPandas()
#print(Df_ODS_pandas.describe())

#Let's find out who are the Most Valuable customers!

#The Most Valuable Customers are the customers who are the most profitable for a company.
#These customers buy more or higher-value products than the average customer.
Top_customers = Df_ODS_pandas.groupby(["Customer Name"]).sum().sort_values("Sales", ascending=False).head(20) # Sort the Customers as per the sales
Top_customers = Top_customers[["Sales"]].round(2) # Round off the Sales Value up to 2 decimal places
Top_customers.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the customer name into dataframe

"""
plt.figure(figsize = (15,5)) # width and height of figure is defined in inches
plt.title("Most Valuable Customers (2015-2019)", fontsize=18)
plt.bar(Top_customers["Customer Name"], Top_customers["Sales"],color= '#99ff99',edgecolor='green', linewidth = 1)
plt.xlabel("Customers",fontsize=15) # x axis shows the customers
plt.ylabel("Revenue",fontsize=15) # y axis shows the Revenue
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
for k,v in Top_customers["Sales"].items(): #To show the exact revenue generated on the figure
    plt.text(k,v-8000,'$'+ str(v), fontsize=12,rotation=90,color='k', horizontalalignment='center')
plt.savefig('plot.png')
"""
Top_customers_to_postgress=spark.createDataFrame(Top_customers)


#Let's find out which cities generated highest revenue!
#Here are the top 10 cities which generated the highest revenue

Top_cities = Df_ODS_pandas.groupby(["City"]).sum().sort_values("Sales", ascending=False).head(20) # Sort the States as per the sales
Top_cities = Top_cities[["Sales"]].round(2) # Round off the Sales Value up to 2 decimal places
Top_cities.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the cities into the dataframe

"""
plt.figure(figsize = (15,5)) # width and height of figure is defined in inches
plt.title("Cities which generated Highest Revenue (2015-2019)", fontsize=18)
plt.bar(Top_cities["City"], Top_cities["Sales"],color= '#95DEE3',edgecolor='blue', linewidth = 1)
plt.xlabel("Cities",fontsize=15)  # x axis shows the States
plt.ylabel("Revenue",fontsize=15)  # y axis shows the Revenue
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
for k,v in Top_cities["Sales"].items(): #To show the exact revenue generated on the figure
    if v>250000:
        plt.text(k,v-75000,'$'+ str(v), fontsize=12,rotation=90,color='k', horizontalalignment='center');
    else:
        plt.text(k,v+15000,'$'+ str(v), fontsize=12,rotation=90,color='k', horizontalalignment='center');
plt.savefig('plot_cities.png')
"""
Top_cities_to_postgress=spark.createDataFrame(Top_cities)


#Let's look at the revenue generated by each category!
Top_category = Df_ODS_pandas.groupby(["Category"]).sum().sort_values("Sales", ascending=False)  # Sort the Categories as per the sales
Top_category = Top_category[["Sales"]] # keep only the sales column in the dataframe
#total_revenue_category = Top_category["Sales"].sum() # To find the total revenue generated as per category
#total_revenue_category = str(int(total_revenue_category)) # Convert the total_revenue_category from float to int and then to string
#total_revenue_category = '$' + total_revenue_category # Adding '$' sign before the Value
Top_category.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the category into the dataframe

"""
plt.rcParams["figure.figsize"] = (13,5) # width and height of figure is defined in inches
plt.rcParams['font.size'] = 12.0 # Font size is defined
plt.rcParams['font.weight'] = 6 # Font weight is defined
# we don't want to look at the percentage distribution in the pie chart. Instead, we want to look at the exact revenue generated by the categories.
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return ' ${v:d}'.format(v=val)
    return my_format
colors = ['#BC243C','#FE840E','#C62168'] # Colors are defined for the pie chart
explode = (0.05,0.05,0.05)
fig1, ax1 = plt.subplots()
ax1.pie(Top_category['Sales'], colors = colors, labels=Top_category['Category'], autopct= autopct_format(Top_category['Sales']), startangle=90,explode=explode)
centre_circle = plt.Circle((0,0),0.82,fc='white') # drawing a circle on the pie chart to make it look better
fig = plt.gcf()
fig.gca().add_artist(centre_circle) # Add the circle on the pie chart
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
# we can look the total revenue generated by all the categories at the center
label = ax1.annotate('Total Revenue \n'+str(total_revenue_category),color = 'red', xy=(0, 0), fontsize=12, ha="center")
plt.tight_layout()
plt.savefig('plot_categories.png')
"""
Top_categories_to_postgress=spark.createDataFrame(Top_category)
#Top_revenue_categories_to_postgress=spark.createDataFrame(total_revenue_category)

#We can see that Category - Technology generated the highest revenue of about $827426!

#The Total Revenue generated by all the categories - $2261536!

#Which products contributed most to the revenue?
Top_products = Df_ODS_pandas.groupby(["Product Name"]).sum().sort_values("Sales",ascending=False).head(8) # Sort the product names as per the sales
Top_products = Top_products[["Sales"]].round(2) # Round off the Sales Value up to 2 decimal places
Top_products.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the product names into the dataframe
#total_revenue_products = Top_products["Sales"].sum() # To find the total revenue generated by all the top products
#total_revenue_products = str(int(total_revenue_products)) # Convert the total_revenue_products from float to int and then to string
#total_revenue_products = '$' + total_revenue_products # Adding '$' sign before the Value
"""
plt.rcParams["figure.figsize"] = (13,7) # width and height of figure is defined in inches
plt.rcParams['font.size'] = 12.0 # Font size is defined for the figure
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#55B4B0','#E15D44','#009B77','#B565A7'] # colors are defined for the pie chart
explode = (0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05)
fig1, ax1 = plt.subplots()
ax1.pie(Top_products['Sales'], colors = colors, labels=Top_products['Product Name'], autopct= autopct_format(Top_products['Sales']), startangle=90,explode=explode)
centre_circle = plt.Circle((0,0),0.80,fc='white') # Draw a circle on the pie chart
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
label = ax1.annotate('Total Revenue \n of these products \n'+str(total_revenue_products),color = 'red', xy=(0, 0), fontsize=12, ha="center")
plt.tight_layout()
plt.savefig('plot_products.png')
"""
Top_products_to_postgress=spark.createDataFrame(Top_products)
#Top_revenue_products_to_postgress=spark.createDataFrame(total_revenue_products)

#We can see that Product - Canon imageCLASS 2200 Advanced Copier generated the highest revenue of about $61600!

#The Total Revenue generated by all these products - $209624!

#Let's look at the revenue generated by each Sub-Category!<h3**>

# Sort both category and  sub category as per the sales
Top_subcat = Df_ODS_pandas.groupby(['Category','Sub-Category']).sum().sort_values("Sales", ascending=False).head(10)
Top_subcat = Top_subcat[["Sales"]].astype(int) # Cast Sales column to integer data type
Top_subcat = Top_subcat.sort_values("Category") # Sort the values as per Category
Top_subcat.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add both columns into data frame
Top_subcat_1 = Top_subcat.groupby(['Category']).sum() # Calculated the total Sales of all the categories
Top_subcat_1.reset_index(inplace=True) # Reset the index
"""
fig, ax = plt.subplots()
ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
width = 0.1
outer_colors = ['#FE840E','#009B77','#BC243C'] # Outer colors of the pie chart
inner_colors = ['Orangered','tomato','coral',"darkturquoise","mediumturquoise","paleturquoise","lightpink","pink","hotpink","deeppink"] # inner colors of the pie chart
pie = ax.pie(Top_subcat_1['Sales'], radius=1, labels=Top_subcat_1['Category'],colors=outer_colors,wedgeprops=dict(edgecolor='w'))
pie2 = ax.pie(Top_subcat['Sales'], radius=1-width, labels=Top_subcat['Sub-Category'],autopct= autopct_format(Top_subcat['Sales']),labeldistance=0.7,colors=inner_colors,wedgeprops=dict(edgecolor='w'), pctdistance=0.53,rotatelabels =True)
# Rotate fractions
# [0] = wedges, [1] = labels, [2] = fractions
fraction_text_list = pie2[2]
for text in fraction_text_list:
    text.set_rotation(315) # rotate the autopct values
centre_circle = plt.Circle((0,0),0.6,fc='white') # Draw a circle on the pie chart
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()

plt.savefig('plot_categories_sorted.png')
"""
Top_sub_categories_to_postgress=spark.createDataFrame(Top_subcat)
Top_sub_categories_all_to_postgress=spark.createDataFrame(Top_subcat_1)

#We can see that Sub-Category - Phones generated the highest revenue of about $327782!

#Which Segment has the highest sales?

Top_segment = Df_ODS_pandas.groupby(["Segment"]).sum().sort_values("Sales", ascending=False) # Sort the segment as per the sales
Top_segment = Top_segment[["Sales"]] # keep only the sales column in the dataframe
Top_segment.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the segment column into the data frame
#total_revenue_segement = Top_segment["Sales"].sum() # To find the total revenue generated as per segment
#total_revenue_segement = str(int(total_revenue_segement)) # Convert the total_revenue_segment from float to int and then to string
#total_revenue_segement= '$' + total_revenue_segement # Adding '$' sign before the Value
"""
plt.rcParams["figure.figsize"] = (13,5) # width and height of figure is defined in inches
plt.rcParams['font.size'] = 12.0 # Font size is defined
plt.rcParams['font.weight'] = 6 # Font weight is defined
colors = ['#BC243C','#FE840E','#C62168'] # Colors are defined for the pie chart
explode = (0.05,0.05,0.05)
fig1, ax1 = plt.subplots()
ax1.pie(Top_segment['Sales'], colors = colors, labels=Top_segment['Segment'], autopct= autopct_format(Top_segment['Sales']),startangle=90,explode=explode)
centre_circle = plt.Circle((0,0),0.85,fc='white') # Draw a circle on the pie chart
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
label = ax1.annotate('Total Revenue \n'+str(total_revenue_segement),color = 'red', xy=(0, 0), fontsize=12, ha="center")
plt.tight_layout()
plt.savefig('plot_highest_sales_segment.png')
"""
Top_segments_to_postgress=spark.createDataFrame(Top_segment)
#Top_revenue_segments_to_postgress=spark.createDataFrame(total_revenue_segement)


#We can see that Segment - Consumer generated the highest revenue of about $1148061!

#The Total Revenue generated by all the segments - $209624!
#Which Region has the highest sales?

Top_region = Df_ODS_pandas.groupby(["Region"]).sum().sort_values("Sales", ascending=False) # Sort the Region as per the sales
Top_region = Top_region[["Sales"]].astype(int) # Cast Sales column to integer data type
Top_region.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the Region column into the data frame
"""
plt.figure(figsize = (10,5)) # width and height of figure is defined in inches
plt.title("Region-wise Revenue Generation", fontsize=18)
plt.bar(Top_region["Region"], Top_region["Sales"],color= '#FF6F61',edgecolor='Red', linewidth = 1)
plt.xlabel("Region",fontsize=15) # x axis shows the Region
plt.ylabel("Revenue",fontsize=15) # y axis show the Revenue generated
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
for k,v in Top_region["Sales"].items(): #To show the exact revenue generated on the figure
        plt.text(k,v-150000,'$'+ str(v), fontsize=12,color='k', horizontalalignment='center');

plt.savefig('plot_highest_sales_region.png')
"""
Top_regions_to_postgress=spark.createDataFrame(Top_region)


#Which shipping mode has the highest sales?

Top_shipping = Df_ODS_pandas.groupby(["Ship Mode"]).sum().sort_values("Sales", ascending=False) # Sort the Shipping modes as per the sales
Top_shipping = Top_shipping[["Sales"]] # keep only the sales column in the dataframe
Top_shipping.reset_index(inplace=True) # Since we have used groupby, we will have to reset the index to add the Ship Mode column into the data frame
#total_revenue_ship = Top_segment["Sales"].sum() # To find the total revenue generated as per shipping mode
#total_revenue_ship = str(int(total_revenue_ship)) # Convert the total_revenue_ship from float to int and then to string
#total_revenue_ship = '$' + total_revenue_ship # Adding '$' sign before the Value
"""
plt.rcParams["figure.figsize"] = (13,5) # width and height of figure is defined in inches
plt.rcParams['font.size'] = 12.0 # Font size is defined
plt.rcParams['font.weight'] = 6 # Font weight is defined
colors = ['#BC243C','#FE840E','#C62168',"limegreen"] # define colors for the pie chart
fig1, ax1 = plt.subplots()
ax1.pie(Top_shipping['Sales'], colors = colors, labels=Top_shipping['Ship Mode'], autopct= autopct_format(Top_shipping['Sales']), startangle=90)
centre_circle = plt.Circle((0,0),0.82,fc='white') # Draw a circle on the pie chart
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
label = ax1.annotate('Total Revenue \n'+str(total_revenue_ship),color = 'red', xy=(0, 0), fontsize=12, ha="center")
plt.tight_layout()
plt.savefig('plot_shipping_mode_sales_region.png')
"""
Top_shippings_to_postgress=spark.createDataFrame(Top_shipping)
#Top_revenue_shippings_to_postgress=spark.createDataFrame(total_revenue_ship)


#We can see that Shipping mode - Standard Class generated the highest revenue of about $1340831!

#The Total Revenue generated by all the shipping modes - $2261536!

#↓ БЛОК С ОТПРАВКОЙ ВИТРИН DATAMART В POSTRGRESQL ↓

#Connection details

URL_DATAMART = f"jdbc:postgresql://{PSQL_SERVERNAME}/{PSQL_DBNAME_DATAMART}"

#↓Table details

def get_TOP_CUSTOMERS(_spark):
    _df_TOP_CUSTOMERS = Top_customers_to_postgress
    return _df_TOP_CUSTOMERS
def get_TOP_CITIES(_spark):
    _df_TOP_CITIES = Top_cities_to_postgress
    return _df_TOP_CITIES
def get_TOP_CATEGORIES(_spark):
    _df_TOP_CATEGORIES = Top_categories_to_postgress
    return _df_TOP_CATEGORIES
def get_TOP_PRODUCTS(_spark):
    _df_TOP_PRODUCTS = Top_products_to_postgress
    return _df_TOP_PRODUCTS
def get_TOP_SUB_CATEGORIES(_spark):
    _df_TOP_SUB_CATEGORIES = Top_sub_categories_to_postgress
    return _df_TOP_SUB_CATEGORIES
def get_TOP_SUB_CATEGORIES_ALL(_spark):
    _df_TOP_SUB_CATEGORIES_ALL = Top_sub_categories_all_to_postgress
    return _df_TOP_SUB_CATEGORIES_ALL
def get_TOP_SEGMENTS(_spark):
    _df_TOP_SEGMENTS = Top_segments_to_postgress
    return _df_TOP_SEGMENTS
def get_TOP_REGIONS(_spark):
    _df_TOP_REGIONS = Top_regions_to_postgress
    return _df_TOP_REGIONS
def get_TOP_SHIPPINGS(_spark):
    _df_TOP_SHIPPINGS = Top_shippings_to_postgress
    return _df_TOP_SHIPPINGS

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("PostgrsSQL demo") \
        .getOrCreate()

    df_TOP_CUSTOMERS = get_TOP_CUSTOMERS(spark)
    df_TOP_CITIES = get_TOP_CITIES(spark)
    df_TOP_CATEGORIES = get_TOP_CATEGORIES(spark)
    df_TOP_PRODUCTS = get_TOP_PRODUCTS(spark)
    df_TOP_SUB_CATEGORIES = get_TOP_SUB_CATEGORIES(spark)
    df_TOP_SUB_CATEGORIES_ALL = get_TOP_SUB_CATEGORIES_ALL(spark)
    df_TOP_SEGMENTS = get_TOP_SEGMENTS(spark)
    df_TOP_REGIONS = get_TOP_REGIONS(spark)
    df_TOP_SHIPPINGS = get_TOP_SHIPPINGS(spark)

    df_TOP_CUSTOMERS.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_CUSTOMERS)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_CITIES.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_CITIES)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_CATEGORIES.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_CATEGORIES)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_PRODUCTS.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_PRODUCTS)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_SUB_CATEGORIES.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_SUB_CATEGORIES)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_SUB_CATEGORIES_ALL.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_SUB_CATEGORIES_ALL)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_SEGMENTS.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_SEGMENTS)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_REGIONS.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_REGIONS)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()
    df_TOP_SHIPPINGS.write\
        .format("jdbc")\
        .option("url", URL_DATAMART)\
        .option("dbtable", TABLE_TOP_SHIPPINGS)\
        .option("user", PSQL_USERNAME)\
        .option("password", PSQL_PASSWORD)\
        .mode("overwrite")\
        .save()

#↑ БЛОК С ОТПРАВКОЙ ВИТРИН DATAMART В POSTRGRESQL ↑

#Correlation of Features

#By plotting a correlation matrix, we have a very nice overview of how the features are related to one another.
#For a Pandas dataframe, we can conveniently use the call .corr which by default provides the Pearson Correlation
#values of the columns pairwise in that dataframe.
df1 = Df_ODS_pandas[['Sales','Segment']]
df_cat = pd.get_dummies(df1)
cor_mat = df_cat.corr()
print(cor_mat)
mask = np.array(cor_mat)
print(mask)
mask[np.tril_indices_from(mask)]=False
print(mask)
"""
fig = plt.gcf()
fig.set_size_inches(20,5)
sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True);
fig = sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True).get_figure()
fig.savefig("corr.png")
#sns.plt.show()
"""

df1 = Df_ODS_pandas[['Category','Sales']]
df_cat = pd.get_dummies(df1)
cor_mat = df_cat.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig = plt.gcf()
fig.set_size_inches(20,5)
sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)

df1 = Df_ODS_pandas[['Ship Mode','Sales']]
df_cat = pd.get_dummies(df1)
cor_mat = df_cat.corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig = plt.gcf()
fig.set_size_inches(20,5)
sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)

#Choropleth map

#Since the state abbreviation or the latitude and longitude are not given, it is difficult to plot a map.
#So, the state abbreviations are added to the respective states and a choropleth map is plotted.
state = ['Alabama', 'Arizona' ,'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
         'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
         'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana','Nebraska', 'Nevada', 'New Hampshire',
         'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
         'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
         'West Virginia', 'Wisconsin','Wyoming']
state_code = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
              'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
              'TX','UT','VT','VA','WA','WV','WI','WY']

state_df = pd.DataFrame(state, state_code) # Create a dataframe
state_df.reset_index(level=0, inplace=True)
state_df.columns = ['State Code','State']
sales = Df_ODS_pandas.groupby(["State"]).sum().sort_values("Sales", ascending=False)
sales.reset_index(level=0, inplace=True)
sales.drop('Postal Code',1, inplace = True)
sales= sales.sort_values('State', ascending=True)
sales.reset_index(inplace = True)
sales.drop('index',1,inplace = True)
sales.insert(1, 'State Code', state_df['State Code'])

pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go

sales['text'] = sales['State']
fig = go.Figure(data=go.Choropleth(
    locations=sales['State Code'], # Spatial coordinates
    text=sales['text'],
    z = sales['Sales'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Sales",

))

fig.update_layout(
    title_text = 'Sales',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
#fig.savefig("map.png")
