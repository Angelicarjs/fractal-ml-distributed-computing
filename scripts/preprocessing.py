#!/usr/bin/env python
# coding: utf-8

# In[24]:


from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, sum

import argparse

# In[25]:


spark = (
    SparkSession.builder \
    .appName("S3ParquetReader") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()
    )

# In[34]:


#parq_cols=["xyz","Intensity","Classification","Red","Green","Blue","Infrared"]
parq_cols=["xyz","Withheld","Synthetic","EdgeOfFlightLine"]

# In[35]:


default_parq_file="s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"

# read and prune unwanted columns and persist it
df = spark.read.parquet(default_parq_file).select(*parq_cols)
#df = spark.read.parquet(default_parq_file)
df.cache()

# In[36]:


# the schema of with the selected columns
df.printSchema()


# In[37]:


# print some rows
df.show(10,truncate=False)

# In[38]:


#Check the description of the dataset
df.describe().show()


# In[23]:


print("Unique values - Withheld:", [row['Withheld'] for row in df.select("Withheld").distinct().collect()], 
      "| Synthetic:", [row['Synthetic'] for row in df.select("Synthetic").distinct().collect()], 
      "| EdgeOfFlightLine:", [row['EdgeOfFlightLine'] for row in df.select("EdgeOfFlightLine").distinct().collect()])

# In[40]:


parq_cols=["xyz","Intensity","Classification","Red","Green","Blue","Infrared"]
# read wanted columns
df = spark.read.parquet(default_parq_file).select(*parq_cols)
# print some rows
df.show(10,truncate=False)

# In[51]:


#Normalize height (by subtracting minimum z per patch)
from pyspark.sql import functions as F

#separate the x,y,z coordinates
df = df.withColumn("z", F.element_at("xyz", 3))

#get the minimum z per patch
min_z = df.agg(F.min("z").alias("z_min")).collect()[0]["z_min"]
print(f"Minimum Z (global) = {min_z}")


# In[52]:


#subtract the minimum z per patch to the z coordinates
df = df.withColumn("z_norm", F.col("z") - F.lit(min_z))

# In[54]:


#reconstruct the xyz column with z normalized
df = df.withColumn("xyz_norm",
                   F.array(
                       F.element_at("xyz", 1),
                       F.element_at("xyz", 2),
                       F.col("z_norm")
                   ))

# In[57]:


df.select("xyz", "xyz_norm").show(10, truncate=False)


# In[ ]:



