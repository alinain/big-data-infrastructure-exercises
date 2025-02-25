from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import requests
from botocore.exceptions import BotoCoreError, NoCredentialsError
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, status
from fastapi.params import Query

from bdi_api.settings import Settings

from .helper import delete_files_in_s3_folder, download_file, list_s3_files, process_s3_files, upload_to_s3

settings = Settings()

s4 = APIRouter(
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Something is wrong with the request"},
    },
    prefix="/api/s4",
    tags=["s4"],
)


@s4.post("/aircraft/download")
def download_data(
    file_limit: Annotated[
        int,
        Query(..., description="Limits the number of files to download from the source."),
    ] = 100,
) -> str:
    """Download files from external source and upload to S3."""
    base_url = settings.source_url + "/2023/11/01/"
    s3_bucket = settings.s3_bucket
    s3_prefix_path = "raw/day=20231101/"
    if file_limit < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="file_limit must be greater than 0",
        ) from None
    try:
        delete_files_in_s3_folder(s3_bucket, s3_prefix_path)
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
        file_links = file_links[:file_limit]
        print(":rocket: Starting concurrent downloads...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda link: download_file(link, base_url, s3_bucket, s3_prefix_path), file_links)
        print(":white_tick: All downloads and uploads completed.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch or download files: {str(e)}",
        ) from e
    return "OK"


@s4.post("/aircraft/prepare")
def prepare_data() -> str:
    s3_bucket = settings.s3_bucket
    s3_prefix_path = "raw/day=20231101/"
    s3_output_prefix = "prepared_day=20231101/"
    try:
        file_keys = list_s3_files(s3_bucket, s3_prefix_path)
        if not file_keys:
            return ":warning: No files found to process."
        all_aircraft_data = process_s3_files(s3_bucket, file_keys)
        if all_aircraft_data:
            upload_to_s3(all_aircraft_data, s3_bucket, s3_output_prefix)
        else:
            return ":warning: No valid aircraft data found for preparation."
        return ":white_tick: Aircraft data prepared and uploaded to S3."
    except (NoCredentialsError, BotoCoreError) as e:
        raise HTTPException(status_code=500, detail=f":x: S3 Error: {str(e)}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f":x: Processing Error: {str(e)}") from e
