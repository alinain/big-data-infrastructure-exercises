import os
import requests
from typing import Annotated
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import shutil
import json
from fastapi import APIRouter, status, HTTPException
from fastapi.params import Query

from bdi_api.settings import Settings
from bs4 import BeautifulSoup

settings = Settings()

s1 = APIRouter(
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Something is wrong with the request"},
    },
    prefix="/api/s1",
    tags=["s1"],
)

@s1.post("/aircraft/download")
def download_data(
        file_limit: Annotated[
            int,
            Query(
                ..., 
                description="""
    Limits the number of files to download.
    You must always start from the first the page returns and
    go in ascending order in order to correctly obtain the results.
    I'll test with increasing number of files starting from 100.""",
            ),
        ] = 100,
) -> str:

    download_dir = os.path.join(settings.raw_dir, "day=20231101")
    base_url = settings.source_url + "/2023/11/01/"

    if file_limit < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="file_limit must be greater than 0",
        )

    os.makedirs(download_dir, exist_ok=True)
    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        file_links = []
        for file_row in soup.find_all("tr", class_="file"):
            link = file_row.find("a", href=True)
            if link:
                file_links.append(link["href"])

        if not file_links:
            return "No files found to download."

        def download_file(link):
            file_url = f"{base_url}{link}"
            local_file_path = os.path.join(download_dir, link)
            try:
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status()

                with open(local_file_path, "wb") as file:
                    for chunk in file_response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print(f"Downloaded: {local_file_path}")
            except Exception as e:
                print(f"Failed to download {file_url}: {e}")

        def execute_concurrent_downloads():
            print("Starting concurrent downloads...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(download_file, file_links[:file_limit])
            print("All downloads completed.")

        execute_concurrent_downloads()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch or download files: {str(e)}"
        )
    return "OK"


@s1.post("/aircraft/prepare")
def prepare_data() -> str:
    download_dir = os.path.join(settings.raw_dir, "day=20231101")
    output_dir = os.path.join(settings.raw_dir, "prepared_day=20231101")

    # Inline clear_directory functionality
    files_in_directory = []
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist.")
    else:
        files_in_directory = os.listdir(output_dir)
        if files_in_directory:
            for file_name in files_in_directory:
                file_path = os.path.join(output_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            print(f"No files found in {output_dir}.")

    # Inline copy_and_rename_files functionality
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(download_dir):
        source_path = os.path.join(download_dir, file_name)
        if os.path.isfile(source_path):
            new_name = remove_gz_extension(file_name)
            destination_path = os.path.join(output_dir, new_name)

            shutil.copy(str(source_path), str(destination_path))
            print(f"Copied and renamed: {source_path} -> {destination_path}")

    prep_data(output_dir)

    return "OK"



@s1.get("/aircraft/")
def list_aircraft(num_results: int = 100, page: int = 0) -> list[dict]:

    prepared_file = os.path.join(settings.raw_dir, "prepared_day=20231101/prepared.json")
    if not os.path.exists(prepared_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prepared data file not found."
        )
    try:
        # Load the data into a Pandas DataFrame
        df = pd.read_json(prepared_file)
        df = df[["icao", "registration", "type"]]
        df = df.sort_values(by="icao", ascending=True)

        start_index = page * num_results
        end_index = start_index + num_results
        paginated_data = df.iloc[start_index:end_index].to_dict(orient="records")
        print(paginated_data)
        return paginated_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list aircraft: {str(e)}"
        )


@s1.get("/aircraft/{icao}/positions")
def get_aircraft_position(icao: str, num_results: int = 1000, page: int = 0) -> list[dict]:
    prepared_file = os.path.join(settings.raw_dir, "prepared_day=20231101/prepared.json")
    if not os.path.exists(prepared_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prepared data file not found."
        )
    try:
        df = pd.read_json(prepared_file)
        df = df[["icao","timestamp", "lat", "lon"]]
        df_filtered = df[df["icao"] == icao]

        if df_filtered.empty:
            return []

        df_filtered = df_filtered.sort_values(by="timestamp", ascending=True)

        start_index = page * num_results
        end_index = start_index + num_results
        paginated_data = df_filtered.iloc[start_index:end_index].to_dict(orient="records")

        return paginated_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve aircraft positions: {str(e)}"
        )

@s1.get("/aircraft/{icao}/stats")
def get_aircraft_statistics(icao: str) -> dict:
    # TODO Gather and return the correct statistics for the requested aircraft
    prepared_file = os.path.join(settings.raw_dir, "prepared_day=20231101/prepared.json")
    if not os.path.exists(prepared_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prepared data file not found."
        )
    try:
        df = pd.read_json(prepared_file)
        df_filtered = df[df["icao"] == icao]

        if df_filtered.empty:
            return {
                "max_altitude_baro": None,
                "max_ground_speed": None,
                "had_emergency": None
            }

        max_altitude_baro = df_filtered["max_altitude_baro"].max()
        max_ground_speed = df_filtered["max_ground_speed"].max()
        had_emergency = bool((df_filtered["had_emergency"] == 1).any())

        return {
            "max_altitude_baro": max_altitude_baro,
            "max_ground_speed": max_ground_speed,
            "had_emergency": had_emergency,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve aircraft statistics: {str(e)}"
        )
    
def remove_gz_extension(file_name):
    if file_name.endswith(".gz"):
        return file_name[:-3]
    return file_name

def copy_and_rename_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        if os.path.isfile(source_path):
            new_name = remove_gz_extension(file_name)
            destination_path = os.path.join(destination_dir, new_name)

            shutil.copy(str(source_path), str(destination_path))
            print(f"Copied and renamed: {source_path} -> {destination_path}")


def prep_data(directory, output_file="prepared.json", batch_size=100):
    output_path = os.path.join(directory, output_file)
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")

    all_aircraft_data = []

    for file_name in os.listdir(directory):
        if file_name.endswith(".json") and file_name != output_file:
            file_path = os.path.join(directory, file_name)

            try:
                with open(file_path, "r") as file:
                    data = json.load(file)

                aircraft_data = data.get("aircraft", [])
                aircraft_timestamp = data.get("now")
                total_records = len(aircraft_data)

                for i in range(0, total_records,batch_size):
                    batch = aircraft_data[i:i+batch_size]

                    for ac in batch:
                        aircraft = {
                            "icao": ac.get("hex"),
                            "registration": ac.get("r"),
                            "type": ac.get("t"),
                            "lat" : ac.get("lat"),
                            "lon" : ac.get("lon"),
                            "max_altitude_baro" : ac.get("alt_baro"),
                            "max_ground_speed" : ac.get("gs"),
                            "had_emergency" : ac.get("emergency"),
                            "timestamp" : aircraft_timestamp
                        }
                        all_aircraft_data.append(aircraft)

                    print(f"Processed {len(batch)} records from {file_name}")
                os.remove(file_path)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    with open(output_path, "w") as output_file:
        json.dump(all_aircraft_data, output_file, indent=4)
    print(f"Data saved to: {output_path}")
