import io
import json
import os
import boto3
import requests
s3_client = boto3.client("s3")
def remove_gz_extension(file_name):
    return file_name[:-3] if file_name.endswith(".gz") else file_name
def delete_files_in_s3_folder(bucket_name, prefix):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        print(":white_tick: No files to delete in S3 folder.")
        return
    keys_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
    s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": keys_to_delete})
    print(f":white_tick: Deleted {len(keys_to_delete)} files from s3://{bucket_name}/{prefix}")
def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    files_in_directory = os.listdir(directory_path)
    if files_in_directory:
        for file_name in files_in_directory:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):  # Check if it's a file
                os.remove(file_path)
    else:
        print(f"No files found in {directory_path}.")
def list_s3_files(s3_bucket: str, s3_prefix_path: str) -> list:
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix_path)
    return [obj["Key"] for obj in response.get("Contents", [])]
def get_json_from_s3(s3_bucket: str, file_key: str) -> dict:
    file_response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    return json.load(file_response["Body"])
def process_aircraft_data(data: dict, unique_aircraft_data: set[str]) -> list:
    aircraft_data = data.get("aircraft", [])
    aircraft_timestamp = data.get("now")
    cleaned_data = []
    for ac in aircraft_data:
        if not all(ac.get(field) for field in ["hex", "lat", "lon"]):
            continue
        if ac.get("r") == "TWR":
            continue
        ac["emergency"] = False if ac.get("emergency") in [None, "none", "None", "null"] else ac.get("emergency")
        if ac.get("alt_baro") == "ground" or (isinstance(ac.get("alt_baro"), (int, float)) and ac["alt_baro"] < 0):
            ac["alt_baro"] = 0
        string_record = json.dumps(ac, sort_keys=True)
        if string_record not in unique_aircraft_data:
            unique_aircraft_data.add(string_record)
            cleaned_data.append(
                {
                    "icao": ac.get("hex"),
                    "registration": ac.get("r"),
                    "type": ac.get("t"),
                    "lat": ac.get("lat"),
                    "lon": ac.get("lon"),
                    "max_altitude_baro": ac.get("alt_baro"),
                    "max_ground_speed": ac.get("gs"),
                    "had_emergency": ac.get("emergency"),
                    "timestamp": aircraft_timestamp,
                },
            )
    return cleaned_data
def process_s3_files(s3_bucket: str, file_keys: list) -> list:
    unique_aircraft_data: set[str] = set()
    all_aircraft_data = []
    for file_key in file_keys:
        try:
            file_data = get_json_from_s3(s3_bucket, file_key)
            all_aircraft_data.extend(process_aircraft_data(file_data, unique_aircraft_data))
            print("file process: " + file_key)
        except Exception as e:
            print(f":x: Error processing {file_key}: {e}")
    return all_aircraft_data
def upload_to_s3(data: list, s3_bucket: str, s3_output_prefix: str):
    output_key = f"{s3_output_prefix}prepared.json"
    print("prepared json file: " + output_key)
    with io.BytesIO(json.dumps(data, indent=4).encode()) as file_buffer:
        s3_client.upload_fileobj(file_buffer, s3_bucket, output_key)
    print(f":white_tick: Upload complete: s3://{s3_bucket}/{output_key}")
def download_file(link, base_url, s3_bucket, s3_prefix_path):
    file_url = f"{base_url}{link}"
    s3_object_key = f"{s3_prefix_path}{remove_gz_extension(link)}"
    try:
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()
        file_content = file_response.content
        file_size = len(file_content)
        with io.BytesIO(file_content) as file_buffer:
            s3_client.upload_fileobj(file_buffer, s3_bucket, s3_object_key)
        print(f":package: File: {link} | Size: {file_size} bytes | :white_tick: Uploaded to S3: s3://{s3_bucket}/{s3_object_key}")
    except requests.RequestException as e:
        print(f":x: Download Error: {file_url} - {str(e)}")