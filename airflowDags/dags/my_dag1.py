import gzip
import json
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Configuration
S3_BUCKET = "alina-bdi-aircraft"
S3_RAW_PREFIX = "raw/readsb/"
S3_PREPARED_PREFIX = "prepared/readsb/"
POSTGRES_CONN_ID = "postgres_default"
POSTGRES_TABLE = "readsb_hist"
BASE_URL = "https://data.adsbexchange.com/readsb-hist/"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# Generate dates: 1st day of each month from 2023-11-01 to 2024-11-01
def get_dates():
    dates = []
    current = datetime(2023, 11, 1)
    end = datetime(2024, 11, 1)
    while current <= end:
        dates.append(current.strftime("%Y/%m/%d/%Y%m%d"))
        current = (current + timedelta(days=32)).replace(day=1)
    return dates

with DAG(
    dag_id="my_dag1",
    start_date=datetime(2023, 11, 1),
    schedule_interval="@monthly",
    catchup=True,
    default_args=default_args,
    max_active_runs=1,
) as dag:

    @task
    def download_files(**kwargs):
        execution_date = kwargs["execution_date"]
        s3_hook = S3Hook(aws_conn_id="aws_default")
        date_str = execution_date.strftime("%Y/%m/%d/%Y%m%d")
        files_downloaded = []

        for i in range(100):  # 100 files per day
            file_num = f"{i:06d}"
            file_name = f"{date_str}-{file_num}.json.gz"
            s3_key = f"{S3_RAW_PREFIX}{file_name}"

            # Check idempotency: skip if file exists in S3
            if s3_hook.check_for_key(s3_key, S3_BUCKET):
                logging.info(f"File {s3_key} already exists in S3, skipping download.")
                continue

            url = f"{BASE_URL}{file_name}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    s3_hook.load_bytes(
                        response.content,
                        key=s3_key,
                        bucket_name=S3_BUCKET,
                        replace=True
                    )
                    files_downloaded.append(file_name)
                else:
                    logging.warning(f"File {file_name} not available (status: {response.status_code}).")
            except Exception as e:
                logging.error(f"Failed to download {file_name}: {e}")

        return files_downloaded

    @task
    def prepare_files(files_downloaded, **kwargs):
        execution_date = kwargs["execution_date"]
        s3_hook = S3Hook(aws_conn_id="aws_default")
        date_str = execution_date.strftime("%Y/%m/%d/%Y%m%d")
        prepared_data = []

        for file_name in files_downloaded:
            s3_key = f"{S3_RAW_PREFIX}{file_name}"
            prepared_key = f"{S3_PREPARED_PREFIX}{file_name.replace('.gz', '')}"

            # Check idempotency: skip if prepared file exists
            if s3_hook.check_for_key(prepared_key, S3_BUCKET):
                logging.info(f"Prepared file {prepared_key} already exists in S3, skipping preparation.")
                continue

            # Read raw file from S3
            try:
                obj = s3_hook.get_key(s3_key, S3_BUCKET)
                compressed_data = obj.get()['Body'].read()
                json_data = gzip.decompress(compressed_data).decode('utf-8')
                records = json.loads(json_data).get("aircraft", [])

                # Transform: extract relevant fields
                for record in records:
                    prepared_data.append({
                        "icao": record.get("hex", "").upper(),
                        "lat": record.get("lat"),
                        "lon": record.get("lon"),
                        "alt": record.get("alt"),
                        "timestamp": record.get("now"),
                        "date": execution_date.strftime("%Y-%m-%d")
                    })

                # Save prepared data to S3 as CSV
                if prepared_data:
                    df = pd.DataFrame(prepared_data)
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    s3_hook.load_string(
                        csv_buffer.getvalue(),
                        key=prepared_key,
                        bucket_name=S3_BUCKET,
                        replace=True
                    )
            except Exception as e:
                logging.error(f"Failed to prepare {file_name}: {e}")

        return prepared_data

    @task
    def load_to_postgres(prepared_data):
        if not prepared_data:
            logging.info("No data to load into PostgreSQL.")
            return

        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        df = pd.DataFrame(prepared_data)

        # Create table if not exists
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {POSTGRES_TABLE} (
            icao VARCHAR(6),
            lat FLOAT,
            lon FLOAT,
            alt INTEGER,
            timestamp FLOAT,
            date DATE,
            PRIMARY KEY (icao, timestamp)
        );
        """
        pg_hook.run(create_table_sql)

        # Load data with upsert to avoid duplicates
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
    files = download_files()
    prepared = prepare_files(files)
    load_to_postgres(prepared)
