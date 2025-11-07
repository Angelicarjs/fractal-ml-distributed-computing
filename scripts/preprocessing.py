import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, sum
from sparkmeasure import TaskMetrics
from pyspark.sql import functions as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=None)
    parser.add_argument("--executor-memory", default="4g")
    parser.add_argument("--driver-memory", default="2g")
    parser.add_argument("--num-executors", type=int, default=2)
    parser.add_argument("--data-path", default="s3://ubs-datasets/FRACTAL/data/test/")
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    parser.add_argument("--output-file", default="results.json")
    return parser.parse_args()


#create a srk session
def create_spark_session(args):
    builder = SparkSession.builder.appName("fractal-cv-rf")

    if args.master:
        builder = builder.master(args.master)

    return (
        builder.config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.instances", str(args.num_executors))
        .getOrCreate()
    )


def main ():

    args = parse_args()
    spark = create_spark_session(args)

    #parq_cols=["xyz","Intensity","Classification","Red","Green","Blue","Infrared"]
    # default_parq_file="s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"
    default_parq_file=args.data_path

    taskmetrics = TaskMetrics(spark)
    taskmetrics.begin()

    parq_cols=["xyz","Intensity","Classification","Red","Green","Blue","Infrared"]
    # read wanted columns
    df = spark.read.parquet(default_parq_file).select(*parq_cols)
    # print some rows
    df.show(10,truncate=False)

    
    #separate the x,y,z coordinates
    df = df.withColumn("z", F.element_at("xyz", 3))

    #get the minimum z per patch
    min_z = df.agg(F.min("z").alias("z_min")).collect()[0]["z_min"]
    print(f"Minimum Z (global) = {min_z}")

    #subtract the minimum z per patch to the z coordinates
    df = df.withColumn("z_norm", F.col("z") - F.lit(min_z))

    #reconstruct the xyz column with z normalized
    df = df.withColumn("xyz_norm",
                    F.array(
                        F.element_at("xyz", 1),
                        F.element_at("xyz", 2),
                        F.col("z_norm")
                    ))

    df.select("xyz", "xyz_norm").show(10, truncate=False)

    taskmetrics.end()
    print("\n============< read.parquet() statistics >============\n")
    taskmetrics.print_report()
    print("\n=====================================================\n")


if __name__ == "__main__":
    main()