import sys
import json
import boto3
from urllib.parse import urlparse
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import current_date, date_format, col, lit, max as spark_max


args = getResolvedOptions(
    sys.argv,
    ['JOB_NAME', 'DATABASE_NAME', 'TARGET_S3_BASE', 'CONTROL_S3_BASE']
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

glue_client = boto3.client("glue")
s3 = boto3.client("s3")

database_name = args['DATABASE_NAME']
target_s3_base = args['TARGET_S3_BASE']
control_s3_base = args['CONTROL_S3_BASE']

if not target_s3_base.endswith("/"):
    target_s3_base += "/"

if not control_s3_base.endswith("/"):
    control_s3_base += "/"

SKIP_TABLES = {"databaselog", "errorlog"}


def split_s3_uri(s3_uri):
    p = urlparse(s3_uri)
    return p.netloc, p.path.lstrip("/")


def read_watermark(table_name):
    bucket, prefix = split_s3_uri(control_s3_base)
    key = f"{prefix}{table_name}.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data.get("last_watermark")
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if error_code in ["NoSuchKey", "404", "NoSuchBucket"]:
            return None
        raise


def write_watermark(table_name, watermark):
    bucket, prefix = split_s3_uri(control_s3_base)
    key = f"{prefix}{table_name}.json"
    body = json.dumps({"last_watermark": str(watermark)})
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json"
    )


tables = []
paginator = glue_client.get_paginator("get_tables")

for page in paginator.paginate(DatabaseName=database_name):
    tables.extend(page["TableList"])


for table in tables:
    table_name = table["Name"].lower()
    table_type = table.get("TableType", "")
    base_name = table_name.split("_")[-1]

    print(f"Processing table: {table_name}")

    if table_type == "VIRTUAL_VIEW" or base_name.startswith("v"):
        print(f"Skipping view: {table_name}")
        continue

    if base_name in SKIP_TABLES:
        print(f"Skipping table: {table_name}")
        continue

    dyf = glueContext.create_dynamic_frame.from_catalog(
        database=database_name,
        table_name=table_name,
        transformation_ctx=f"src_{table_name}"
    )

    df = dyf.toDF()

    modified_col = None
    for c in df.columns:
        if c.lower() == "modifieddate":
            modified_col = c
            break

    if modified_col:
        last_watermark = read_watermark(table_name)

        if last_watermark:
            print(f"Incremental load: {modified_col} > {last_watermark}")
            df = df.filter(col(modified_col) > lit(last_watermark))
        else:
            print("First run: full load")

        if df.rdd.isEmpty():
            print(f"No new data for {table_name}")
            continue

        max_modified = df.select(
            spark_max(col(modified_col)).alias("max_modified")
        ).collect()[0]["max_modified"]
    else:
        print("No ModifiedDate found: snapshot load")
        max_modified = None

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

    if max_modified is not None:
        write_watermark(table_name, max_modified)
        print(f"Updated watermark for {table_name}: {max_modified}")

job.commit()