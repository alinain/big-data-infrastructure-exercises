import gzip
import json
import logging
from datetime import datetime, timedelta
from io import StringIO

import botocore.exceptions
import pandas as pd
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Configuration
S3_BUCKET = "alina-bdi-aircraft"
S3_RAW_PREFIX = "raw/aircraft_db/"
S3_PREPARED_PREFIX = "prepared/aircraft_db/"
POSTGRES_CONN_ID = "postgres_default"
POSTGRES_TABLE = "aircraft_db"
FILE_URL = "http://downloads.adsbexchange.com/downloads/basic-ac-db.json.gz"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="my_dag3",
    start_date=datetime(2023, 11, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
) as dag:

    @task
    def download_file():
        try:
            s3_hook = S3Hook(aws_conn_id="aws_default")
        except Exception as e:
            logging.error(
                "Failed to initialize S3Hook. Please configure the 'aws_default' connection in Airflow UI "
                "(Admin > Connections) with your AWS credentials (Access Key, Secret Key, Region). "
                "Alternatively, set AWS credentials as environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY). "
                f"Error: {e}"
            )
            raise

        s3_key = f"{S3_RAW_PREFIX}basic-ac-db.json.gz"

        # Check idempotency
        try:
            if s3_hook.check_for_key(s3_key, S3_BUCKET):
                logging.info(f"File {s3_key} already exists in S3, skipping download.")
                return s3_key
        except botocore.exceptions.NoCredentialsError:
            logging.error(
                "No AWS credentials found. Please configure the 'aws_default' connection in Airflow UI "
                "or set AWS credentials as environment variables."
            )
            raise

        response = requests.get(FILE_URL)
        if response.status_code == 200:
            s3_hook.load_bytes(
                response.content,
                key=s3_key,
                bucket_name=S3_BUCKET,
                replace=True
            )
            return s3_key
        else:
            raise Exception(f"Failed to download file: {response.status_code}")

    @task
    def prepare_file(s3_key):
        s3_hook = S3Hook(aws_conn_id="aws_default")
        prepared_key = f"{S3_PREPARED_PREFIX}basic-ac-db.csv"

        # Check idempotency
        if s3_hook.check_for_key(prepared_key, S3_BUCKET):
            logging.info(f"Prepared file {prepared_key} already exists in S3, skipping preparation.")
            return prepared_key

        # Read raw file
        obj = s3_hook.get_key(s3_key, S3_BUCKET)
        compressed_data = obj.get()['Body'].read()
        json_data = gzip.decompress(compressed_data).decode('utf-8')
        records = json.loads(json_data).get("aircraft", [])

        # Transform
        df = pd.DataFrame(records)
        df = df[["icao", "type", "registration"]].rename(columns={"type": "aircraft_type"})

        # Save to S3
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_hook.load_string(
            csv_buffer.getvalue(),
            key=prepared_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        return df.to_dict('records')

    @task
    def load_to_postgres(prepared_data):
        if not prepared_data:
            logging.info("No data to load into PostgreSQL.")
            return

        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        df = pd.DataFrame(prepared_data)

        # Create table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {POSTGRES_TABLE} (
            icao VARCHAR(6) PRIMARY KEY,
            aircraft_type VARCHAR(50),
            registration VARCHAR(20)
        );
        """
        pg_hook.run(create_table_sql)

        # Truncate for full refresh
        pg_hook.run(f"TRUNCATE TABLE {POSTGRES_TABLE};")

        # Load data
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, header=False)
        csv_buffer.seek(0)

        pg_hook.bulk_load(
            table=POSTGRES_TABLE,
            tmp_file=csv_buffer,
            delimiter=',',
            is_tmp_file=False
        )

    # Task dependencies
    s3_key = download_file()
    prepared_key = prepare_file(s3_key)
    load_to_postgres(prepared_key)
