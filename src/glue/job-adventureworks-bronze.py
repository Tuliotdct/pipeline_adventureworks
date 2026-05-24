import sys
import boto3
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import current_date, date_format

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'DATABASE_NAME', 'TARGET_S3_BASE'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

glue_client = boto3.client("glue")

database_name = args['DATABASE_NAME']
target_s3_base = args['TARGET_S3_BASE']

tables = []
paginator = glue_client.get_paginator("get_tables")

for page in paginator.paginate(DatabaseName=database_name):
    tables.extend(page["TableList"])

for table in tables:
    table_name = table["Name"].lower()
    print(f"Processing table: {table_name}")

    if "_v" in table_name:
        print(f"Skipping view: {table_name}")
        continue

    dyf = glueContext.create_dynamic_frame.from_catalog(
        database=database_name,
        table_name=table_name,
        transformation_ctx=f"src_{table_name}"
    )

    df = dyf.toDF()

    df = df.withColumn("dt", date_format(current_date(), "yyyy-MM-dd"))

    out_dyf = DynamicFrame.fromDF(df, glueContext, f"out_{table_name}")

    glueContext.write_dynamic_frame.from_options(
        frame=out_dyf,
        connection_type="s3",
        format="glueparquet",
        connection_options={
            "path": f"{target_s3_base}{table_name}/",
            "partitionKeys": ["dt"]
        },
        format_options={"compression": "snappy"},
        transformation_ctx=f"s3_{table_name}"
    )

job.commit()