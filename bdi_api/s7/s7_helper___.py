import json
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import boto3
from tqdm import tqdm

s3_client = boto3.client("s3")
thread_local = threading.local()


def get_json_from_s3(s3_bucket: str, file_key: str) -> dict:
    file_response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    return json.load(file_response["Body"])


def list_s3_files(s3_bucket: str, s3_prefix_path: str) -> list:
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix_path)
    return [obj["Key"] for obj in response.get("Contents", [])]


def process_batch_s7(batch, aircraft_timestamp):
    """
    Processes a batch of aircraft data, ensuring data cleaning & deduplication.
    Returns a cleaned list of aircraft records.
    """
    processed_batch = []
    unique_aircraft_data = set()

    for ac in batch:
        if not all(ac.get(field) for field in ["hex", "lat", "lon"]):
            continue
        if ac.get("r") == "TWR":
            continue

        # Normalize emergency status
        if ac.get("emergency") in {None, "none", "None", "null"}:
            ac["emergency"] = False
        else:
            ac["emergency"] = ac["emergency"]

        # Normalize altitude (barometric altitude)
        alt_baro = ac.get("alt_baro")
        if alt_baro in {"ground", None} or (isinstance(alt_baro, (int, float)) and alt_baro < 0):
            ac["alt_baro"] = 0
        else:
            ac["alt_baro"] = alt_baro

        # Normalize ground speed
        gs = ac.get("gs")
        if gs in {None, "none", "None", "null"}:
            ac["gs"] = 0.0
        else:
            try:
                ac["gs"] = float(gs)
            except ValueError:
                ac["gs"] = 0.0  # Fallback in case of unexpected values

        # Create Unique Key
        unique_key = (
            str(ac.get("hex")),
            str(ac.get("r")),
            str(ac.get("t")),
            round(float(ac["lat"]), 6),
            round(float(ac["lon"]), 6),
            int(ac.get("alt_baro")),
            int(ac.get("gs")),  # Ensure integer consistency
            bool(ac.get("emergency")),
        )

        # Deduplication within batch
        if unique_key not in unique_aircraft_data:
            unique_aircraft_data.add(unique_key)
            processed_batch.append(
                {
                    "icao": unique_key[0],
                    "registration": unique_key[1],
                    "type": unique_key[2],
                    "lat": unique_key[3],
                    "lon": unique_key[4],
                    "max_altitude_baro": unique_key[5],
                    "max_ground_speed": unique_key[6],
                    "had_emergency": unique_key[7],
                    "timestamp": aircraft_timestamp,
                },
            )

    return processed_batch


def process_aircraft_data(data: dict, batch_size: int = 100) -> list:
    """
    Processes a batch of aircraft data in parallel.
    Returns a cleaned & deduplicated list of records.
    """
    aircraft_data = data.get("aircraft", [])
    aircraft_timestamp = data.get("now")

    if not aircraft_data:
        return []

    total_records = len(aircraft_data)
    batches = [aircraft_data[i : i + batch_size] for i in range(0, total_records, batch_size)]

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = executor.map(process_batch_s7, batches, [aircraft_timestamp] * len(batches))

    return [item for result in results for item in result]  # Flatten results


def process_s3_files(s3_bucket: str, file_keys: list, batch_size: int = 100) -> list:
    all_aircraft_data = []
    max_threads = min(len(file_keys), 10)  # Avoid excessive threads
    seen_aircraft = set()  # Global deduplication set

    def download_and_process(file_key):
        try:
            file_data = get_json_from_s3(s3_bucket, file_key)
            processed_data = process_aircraft_data(file_data, batch_size)

            unique_cleaned_data = []
            for record in processed_data:
                unique_key = (
                    record["icao"],
                    record["registration"],
                    record["type"],
                    record["lat"],
                    record["lon"],
                    record["max_altitude_baro"],
                    record["max_ground_speed"],
                    record["had_emergency"],
                )

                if unique_key not in seen_aircraft:
                    seen_aircraft.add(unique_key)
                    unique_cleaned_data.append(record)

            return unique_cleaned_data

        except Exception as e:
            print(f"Error processing {file_key}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_key = {executor.submit(download_and_process, file_key): file_key for file_key in file_keys}

        for future in tqdm(as_completed(future_to_key), total=len(file_keys), desc="Processing Files", unit="file"):
            all_aircraft_data.extend(future.result())

    return all_aircraft_data
